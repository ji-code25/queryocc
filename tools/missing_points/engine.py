"""Offline generation of the "missing points" (free-ray) caches.

A spinning lidar fires a fixed grid of rays (one per (channel, azimuth) cell of the sensor
spec). Rays that hit nothing produce no return, so they are simply absent from the recorded
sweep -- yet they certify that the space along them is empty. This tool reconstructs those
dropped rays: for every channel it finds the azimuth bins with no measurement, interpolates
the elevation/azimuth of the missing ray from its neighbours, and places a synthetic point
at a fixed dummy range along it. The result feeds QueryOcc's free-space query supervision
(``add_missing_points=True`` in every preset) and is read back at training time by
``_load_missing_points`` in ``queryocc/data/datasets/{nuscenes_dataset,argoverse2}.py``.

Output contract -- ``float32 [N, 6]`` = ``[x, y, z, intensity, time, channel]``, where
``time`` is the per-point offset in seconds relative to the sweep and ``channel`` is the
laser index:

* nuScenes: LIDAR_TOP **sensor** frame, flat layout ``<out_dir>/<lidar_stem>_missing.npy``
  (``<lidar>.pcd.bin`` -> ``<lidar>_missing.npy``).
* Argoverse 2: **ego** frame, ``<out_dir>/<split>/<log_id>/<timestamp_ns>_missing.npy``.
  Channels span both lidars: 0-31 for ``up_lidar``, 32-63 for ``down_lidar``.

Run with::

    python -m tools.missing_points.engine --dataset nuscenes --version v1.0-trainval
    python -m tools.missing_points.engine --dataset argoverse2 --split train

NOTE ON FIDELITY: the two ``_get_missing_points_*`` functions below are deliberately kept
as two near-identical copies. They differ only in their sensor tables (elevation mapping,
azimuth resolution, skipped channels, dummy range) but they produced the published caches,
so their arithmetic, ordering and dtypes are reproduced verbatim from the original scripts.
Do not unify or "clean up" the numeric path.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from tqdm import tqdm

if __package__ in (None, ""):  # allow `python tools/missing_points/engine.py` as well as `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from queryocc.common import paths
from queryocc.utils.geometry import inverse as pose_inverse, to4x4, transform_points
from tools.missing_points.utils import Lidars, _remove_ego_motion_compensation, _rotation_translation_to_pose

# --------------------------------------------------------------------------------------
# nuScenes sensor spec (Velodyne HDL-32E)
# --------------------------------------------------------------------------------------

NUSC_DUMMY_DISTANCE_VALUE = 150.0
NUSC_MAX_RELECTANCE_VALUE = 255.0
NUSC_LIDAR_FREQUENCY = 20.0  # Hz
NUSC_LIDAR_CHANNELS = 32  # number of vertical channels

VELODYNE_HDL32E_ELEVATION_MAPPING = dict(zip(np.arange(32), tuple(np.linspace(-30.67, 10.67, 32))))

NUSCENES_ELEVATION_MAPPING = {
    "LIDAR_TOP": VELODYNE_HDL32E_ELEVATION_MAPPING,
}
NUSCENES_AZIMUTH_RESOLUTION = {
    "LIDAR_TOP": 1 / 3.0,
}
NUSCENES_SKIP_ELEVATION_CHANNELS = {
    "LIDAR_TOP": (
        0,
        1,
    )
}

# --------------------------------------------------------------------------------------
# Argoverse 2 sensor spec (two Velodyne VLP-32C, one up-facing and one down-facing)
# --------------------------------------------------------------------------------------

AV2_DUMMY_DISTANCE_VALUE = 500.0
AV2_MAX_REFLECTANCE_VALUE = 255.0

ARGOVERSE2_VELODYNE_VLP32C_ELEVATION_MAPPING = {
    31: -25.0,
    13: -1.0,
    1: -1.667,
    17: -15.639,
    29: -11.31,
    9: 0.0,
    3: -0.667,
    21: -8.843,
    27: -7.254,
    12: 0.333,
    5: -0.333,
    16: -6.148,
    25: -5.333,
    8: 1.333,
    7: 0.667,
    20: -4.0,
    22: -4.667,
    2: 1.667,
    10: 1.0,
    28: -3.667,
    23: -3.333,
    6: 3.333,
    11: 2.333,
    24: -2.667,
    18: -3.0,
    0: 7.0,
    14: 4.667,
    30: -2.333,
    19: -2.0,
    4: 15.0,
    15: 10.333,
    26: -1.333,
}

AV2_VELO_VLP32C_ELEVATION_MAPPING = ARGOVERSE2_VELODYNE_VLP32C_ELEVATION_MAPPING
AV2_ELEVATION_MAPPING = {
    "lidar_down": AV2_VELO_VLP32C_ELEVATION_MAPPING,
    "lidar_up": AV2_VELO_VLP32C_ELEVATION_MAPPING,
    "lidar": AV2_VELO_VLP32C_ELEVATION_MAPPING,
}

AV2_AZIMUTH_RESOLUTION = {
    "lidar_down": 0.2,
    "lidar_up": 0.2,
    "lidar": 0.2,
}  # degrees
AV2_SKIP_ELEVATION_CHANNELS = {
    "lidar": (0, 1, 2),
    "lidar_down": (4, 15, 0),  # corresponding to 15,10.333,7.0 degrees as lidar_down is upside down
    "lidar_up": (0, 3, 4),  # corresponding to -25,-15.639,-11.31 degrees
}
AV2_IGNORE_REGIONS = {
    "lidar_down": [[140, 200, -30.0, 20.0], [-200.0, -170.0, -30.0, 20.0]],
    "lidar_up": [],
    "lidar": [],
}


# --------------------------------------------------------------------------------------
# nuScenes numeric path (verbatim from extract_missing_points_nusc.py)
# --------------------------------------------------------------------------------------


def _find_all_sample_data(nusc, sample_data_token: str):
    """Finds all sample data from a given sample token."""
    curr_token = sample_data_token
    sd = nusc.get("sample_data", curr_token)
    # Rewind to first sample data
    while sd["prev"]:
        curr_token = sd["prev"]
        sd = nusc.get("sample_data", curr_token)
    # Forward to last sample data
    all_sample_data = [sd]
    while sd["next"]:
        curr_token = sd["next"]
        sd = nusc.get("sample_data", curr_token)
        all_sample_data.append(sd)
    return all_sample_data


def _get_lidars_nusc(nusc, scene: Dict) -> Tuple[Lidars, List[Path]]:
    lidar_filenames, times, poses, lid2egos, idxs = [], [], [], [], []
    first_sample = nusc.get("sample", scene["first_sample_token"])
    is_key_frame = []
    for lidar_idx, lidar in enumerate(["LIDAR_TOP"]):
        for lidar_data in _find_all_sample_data(nusc, first_sample["data"][lidar]):
            calibrated_sensor_data = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            ego_pose_data = nusc.get("ego_pose", lidar_data["ego_pose_token"])
            ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
            lidar_pose = _rotation_translation_to_pose(
                calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
            )
            pose = ego_pose @ lidar_pose

            lidar_filenames.append(Path(nusc.dataroot) / lidar_data["filename"])

            poses.append(pose)
            lid2egos.append(lidar_pose)
            times.append(lidar_data["timestamp"] / 1e6)
            idxs.append(lidar_idx)
            is_key_frame.append(lidar_data["is_key_frame"])

    poses = torch.tensor(np.array(poses), dtype=torch.float64)  # will be changed to float32 later
    lid2egos = torch.tensor(np.array(lid2egos), dtype=torch.float32)
    times = torch.tensor(times, dtype=torch.float64)  # need higher precision
    idxs = torch.tensor(idxs).int().unsqueeze(-1)

    is_key_frame = torch.tensor(is_key_frame).reshape(-1, 1).bool()
    lidars = Lidars(
        lidar_to_worlds=poses[:, :3, :4],
        assume_ego_compensated=True,
        times=times,
        metadata={"sensor_idxs": idxs, "is_key_frame": is_key_frame},
    )
    return lidars, lidar_filenames


def _get_missing_points_nusc(
    point_cloud: torch.Tensor,
    l2ws: torch.Tensor,
    lidar_name: str,
    dist_cutoff: float = 1.0,
    ignore_regions: list = [],
    outlier_thresh: float = 0.2,
) -> torch.Tensor:
    """Finds missing points in the point cloud according to sensor spec (self.config.lidar_elevation_mapping)

    Args:
        point_cloud: Point cloud to find missing points in (in sensor frame). Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
        l2ws: Poses of the lidar. Shape: [num_points, 4, 4]
        lidar_name: Name of the lidar
        dist_cutoff: Distance cutoff for points to consider. Points closer than this will be ignored.
        ignore_regions: List of regions to ignore. Each region is a list of [min_azimuth, max_azimuth, min_elevation, max_elevation]
        outlier_thresh: Threshold for outlier elevation values. If the median elevation of the missing points is more than this value away from the median elevation of the points in the channel, we ignore the missing points.

    Returns:
        Missing points in the point cloud, in world_frame. Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
    """
    dist = torch.norm(point_cloud[:, :3], dim=-1)
    dist_mask = dist > dist_cutoff
    dist = dist[dist_mask]
    point_cloud = point_cloud[dist_mask]
    l2ws = l2ws[dist_mask]
    elevation = torch.arcsin(point_cloud[:, 2] / dist)
    elevation = torch.rad2deg(elevation)
    azimuth = torch.atan2(point_cloud[:, 1], point_cloud[:, 0])
    azimuth = torch.rad2deg(azimuth)

    # find missing points
    missing_points = []

    for channel_id, expected_elevation in NUSCENES_ELEVATION_MAPPING[lidar_name].items():
        if channel_id in NUSCENES_SKIP_ELEVATION_CHANNELS[lidar_name]:
            continue
        channel_mask = (point_cloud[:, 3] - channel_id).abs() < 0.1  # handle floats
        curr_azimuth = azimuth[channel_mask]
        curr_l2ws = l2ws[channel_mask]
        curr_elev = elevation[channel_mask]
        if not channel_mask.any():
            continue
        curr_azimuth, sort_idx = curr_azimuth.sort()
        curr_l2ws = curr_l2ws[sort_idx]
        curr_elev = curr_elev[sort_idx]

        # find missing azimuths, we should have 360 / lidar_azimuth_resolution azimuths
        num_expected_azimuths = int(360 / NUSCENES_AZIMUTH_RESOLUTION[lidar_name]) + 1
        expected_idx = torch.arange(num_expected_azimuths, device=curr_azimuth.device)
        # find offset
        offset = curr_azimuth[0] % NUSCENES_AZIMUTH_RESOLUTION[lidar_name]
        current_idx = ((curr_azimuth - offset + 180) / NUSCENES_AZIMUTH_RESOLUTION[lidar_name]).round().int()
        missing_idx = expected_idx[torch.isin(expected_idx, current_idx, invert=True)]
        # interpolate missing azimuths
        missing_azimuth = (
            torch.from_numpy(
                np.interp(
                    missing_idx,
                    torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                    torch.cat(
                        [
                            (-180 + offset - NUSCENES_AZIMUTH_RESOLUTION[lidar_name]).view(1),
                            curr_azimuth,
                            (180 + offset + NUSCENES_AZIMUTH_RESOLUTION[lidar_name]).view(1),
                        ]
                    ),
                )
            )
            .float()
            .view(-1, 1)
        )
        missing_elevation = (
            torch.from_numpy(
                np.interp(
                    missing_idx,
                    torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                    torch.cat(
                        [
                            torch.tensor(expected_elevation).view(1),
                            curr_elev.view(-1),
                            torch.tensor(expected_elevation).view(1),
                        ]
                    ),
                )
            )
            .float()
            .view(-1, 1)
        )
        elevation_outlier_mask = (missing_elevation - curr_elev.median()).abs() > outlier_thresh
        missing_elevation[elevation_outlier_mask] = curr_elev.median()

        ignore_mask = torch.zeros_like(missing_azimuth.squeeze(-1)).bool()
        for ignore_region in ignore_regions:
            ignore_mask = (
                (missing_azimuth > ignore_region[0])
                & (missing_azimuth < ignore_region[1])
                & (missing_elevation > ignore_region[2])
                & (missing_elevation < ignore_region[3])
            ).squeeze(-1) | ignore_mask
        missing_azimuth = missing_azimuth[~ignore_mask]
        missing_elevation = missing_elevation[~ignore_mask]

        missing_distance = torch.ones_like(missing_azimuth) * NUSC_DUMMY_DISTANCE_VALUE
        x = torch.cos(torch.deg2rad(missing_azimuth)) * torch.cos(torch.deg2rad(missing_elevation)) * missing_distance
        y = torch.sin(torch.deg2rad(missing_azimuth)) * torch.cos(torch.deg2rad(missing_elevation)) * missing_distance
        z = torch.sin(torch.deg2rad(missing_elevation)) * missing_distance
        points = torch.cat([x, y, z], dim=-1)
        # transform points from sensor space to world space
        points = torch.cat([points, torch.ones_like(points[:, -1:])], dim=-1)
        # find closest pose idx
        closest_pose_idx = torch.searchsorted(curr_azimuth, missing_azimuth.squeeze())
        closest_pose_idx = closest_pose_idx.clamp(max=len(curr_azimuth) - 1)
        if closest_pose_idx.shape == ():
            closest_pose_idx = closest_pose_idx.unsqueeze(0)
        closest_l2w = curr_l2ws[closest_pose_idx]
        points = torch.matmul(closest_l2w.float(), points.unsqueeze(-1))[:, :, 0]
        closest_pc_values = point_cloud[channel_mask][sort_idx][closest_pose_idx, 3:]
        points = torch.cat([points, closest_pc_values], dim=-1)
        missing_points.append(points)

    missing_points = torch.cat(missing_points, dim=0)
    return missing_points


def _load_nusc_sweep(filepath: Path) -> torch.Tensor:
    """Reads one ``.pcd.bin`` sweep as [x, y, z, intensity, time_offset, channel]."""
    pc = np.fromfile(str(filepath), dtype=np.float32).reshape([-1, 5])
    pc[..., 3] = pc[..., 3] / NUSC_MAX_RELECTANCE_VALUE  # normalize reflectance
    # nuscenes lidar time is the time of the end of the sweep. Add estimated times per point
    offsets = np.repeat(
        np.linspace(-1 / NUSC_LIDAR_FREQUENCY, 0, pc.shape[0] // NUSC_LIDAR_CHANNELS), NUSC_LIDAR_CHANNELS
    )
    # warning: here we overwrite the beam index channel with time offsets, since we assume x,y,z,r,t format
    channel_id = pc[..., 4].astype(np.int32)
    pc[..., 4] = offsets
    # concatenate pc with channel id
    pc = np.concatenate([pc, channel_id[:, None]], axis=-1).astype(np.float32)
    return torch.from_numpy(pc)


def _get_missing_points_from_lidars_nusc(
    lidars: Lidars, filepaths: List[Path], todo: List[bool]
) -> Dict[int, torch.Tensor]:
    """Missing points for the sweeps flagged in ``todo``, keyed by their index in ``filepaths``.

    The per-sweep body is unchanged from the original; only the sweep loading was folded into
    the same loop so that already-cached sweeps can be skipped without touching the numerics.
    Each sweep is independent of the others *except* through the full-scene ``poses``/``times``
    trajectory, which is always built from every sweep regardless of what is skipped.
    """
    poses = lidars.lidar_to_worlds
    times = lidars.times.squeeze(-1)
    missing_points = {}
    for idx, (filepath, l2w, time) in enumerate(zip(filepaths, poses, times)):
        if not todo[idx]:
            continue
        point_cloud = _load_nusc_sweep(filepath)
        # make sure we have double for avoid overflow when adding time
        pc = point_cloud.clone().double()
        # absolute time
        pc[:, 4] = pc[:, 4] + time
        # project to world frame
        pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
        # remove ego motion compensation
        pc, interpolated_poses = _remove_ego_motion_compensation(pc, poses, times)
        # reset time
        pc[:, 4] = point_cloud[:, 4].clone()
        # transform to common lidar frame again
        interpolated_poses = torch.matmul(pose_inverse(l2w.unsqueeze(0)).float(), to4x4(interpolated_poses).float())
        # move channel from index 5 to 3
        pc = pc[..., [0, 1, 2, 5, 3, 4]]
        # get missing points
        miss_pc = _get_missing_points_nusc(pc, interpolated_poses, "LIDAR_TOP", dist_cutoff=0.05)
        # move channel from index 3 to 5
        miss_pc = miss_pc[..., [0, 1, 2, 4, 5, 3]]
        # add missing points
        missing_points[idx] = miss_pc.float()
    return missing_points


# --------------------------------------------------------------------------------------
# Argoverse 2 numeric path (verbatim from extract_missing_points_av2.py)
# --------------------------------------------------------------------------------------


def _get_lidars_av2(av2, sequence: str) -> Tuple[Lidars, List[Path]]:
    """Returns lidar info and loaded point clouds."""
    from av2.utils.io import read_ego_SE3_sensor

    (
        lidar_filenames,
        times,
        poses_up,
        poses_down,
        idxs,
    ) = ([], [], [], [], [])

    lidar_times = av2.get_ordered_log_lidar_timestamps(sequence)

    for t in lidar_times:
        # get the path to the lidar point cloud
        pc_path = av2.get_lidar_fpath(sequence, t)
        lidar_filenames.append(pc_path)

        log_dir = pc_path.parent.parent.parent
        sensor_name_to_pose = read_ego_SE3_sensor(log_dir=log_dir)
        ego_SE3_up_lidar = sensor_name_to_pose["up_lidar"]
        ego_SE3_down_lidar = sensor_name_to_pose["down_lidar"]

        # get the ego-vehicle pose at the lidar timestamp
        ego_pose = av2.get_city_SE3_ego(sequence, t)
        # move to the actual lidar position
        ego_pose_up = ego_pose.compose(ego_SE3_up_lidar)
        ego_pose_down = ego_pose.compose(ego_SE3_down_lidar)
        # add the pose
        poses_up.append(ego_pose_up.transform_matrix)
        poses_down.append(ego_pose_down.transform_matrix)
        idxs.append(0)  # add the lidar index (there is only one so lets use 0)
        times.append(t / 1e9)  # convert to seconds

    poses_up = torch.tensor(np.array(poses_up), dtype=torch.float32)
    poses_down = torch.tensor(np.array(poses_down), dtype=torch.float32)
    down2up = pose_inverse(poses_up[0, ...]) @ poses_down[0, ...]
    times = torch.tensor(np.array(times, dtype=np.float64), dtype=torch.float64)  # need higher precision
    idxs = torch.tensor(np.array(idxs)).int().unsqueeze(-1)
    # we will store the down2up transformation in the metadata, as we will need it later
    down2up = down2up.repeat(len(poses_up), 1, 1).view(len(poses_up), -1)

    lidars = Lidars(
        lidar_to_worlds=torch.cat([poses_up, poses_down], dim=0)[:, :3, :4],
        times=torch.cat([times, times], dim=0),
        metadata={
            "sensor_idxs": torch.cat([idxs, idxs + (idxs.max() + 1)], dim=0),
            "down2up": torch.cat([down2up, down2up], dim=0),  # will be removed later, here for compatibility
            "poses_down": torch.cat(
                [poses_down.view(len(poses_up), -1), poses_down.view(len(poses_up), -1)], dim=0
            ),  # will be removed later, here for compatibility
        },
    )

    return lidars, lidar_filenames


def _read_lidars_av2(
    lidars: Lidars, filepaths: List[Path], data_dir: Path, sequence: str, todo: List[bool]
) -> Tuple[Dict[int, np.ndarray], List[Path]]:
    """Missing points (ego frame) for the sweeps flagged in ``todo``, keyed by sorted-filepath index.

    Only the sweep loading is made conditional on ``todo``; the calibration is taken from the
    last sorted sweep exactly as the original did ("calibration is constant, just use the last
    one"), and the pose stream is always read in full, so skipping changes nothing numerically.
    """
    import av2.utils.io as io_utils
    from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3
    from av2.structures.sweep import Sweep

    point_clouds_up, point_clouds_down = {}, {}

    sorted_filepaths = sorted(filepaths)
    for idx, pc_path in enumerate(sorted_filepaths):
        if not todo[idx]:
            continue
        sweep = Sweep.from_feather(pc_path)
        # get laser_number, 0-31 is up_lidar, 32-63 is down_lidar according to
        laser_number = sweep.laser_number
        # transform the points back to the sensor position
        xyz_up = sweep.ego_SE3_up_lidar.inverse().transform_point_cloud(sweep.xyz[laser_number < 32])
        xyz_down = sweep.ego_SE3_down_lidar.inverse().transform_point_cloud(sweep.xyz[laser_number >= 32])
        # normalize the intensity values to be in [0, 1]
        intensity = sweep.intensity / AV2_MAX_REFLECTANCE_VALUE
        # Add relative time
        time = sweep.offset_ns / 1e9
        # add the intensity as the last channel
        xyz_intensity_up = np.concatenate(
            [
                xyz_up,
                intensity[laser_number < 32, None],
                time[laser_number < 32, None],
                laser_number[laser_number < 32, None],
            ],
            axis=1,
        )
        xyz_intensity_down = np.concatenate(
            [
                xyz_down,
                intensity[laser_number >= 32, None],
                time[laser_number >= 32, None],
                laser_number[laser_number >= 32, None],
            ],
            axis=1,
        )
        # add the point cloud
        point_clouds_up[idx] = torch.from_numpy(xyz_intensity_up)
        point_clouds_down[idx] = torch.from_numpy(xyz_intensity_down)

    missing_points_up = {}
    missing_points_down = {}

    assert lidars.times is not None  # typehints
    assert lidars.metadata is not None  # typehints

    poses_down = lidars.metadata["poses_down"].view(len(lidars.times), 4, 4)
    poses_up = lidars.lidar_to_worlds
    times = lidars.times
    log_pose_df = io_utils.read_feather(data_dir / sequence / "city_SE3_egovehicle.feather")
    all_ego2w = [convert_pose_dataframe_to_SE3(pose_df) for _, pose_df in log_pose_df.iterrows()]
    # calibration is constant, just use the last one
    calibration_sweep = Sweep.from_feather(sorted_filepaths[-1])
    uplidar2ego = calibration_sweep.ego_SE3_up_lidar
    all_lup2w = torch.tensor(
        np.array([e2w.compose(uplidar2ego).transform_matrix for e2w in all_ego2w]), dtype=torch.float32
    )
    downlidar2ego = calibration_sweep.ego_SE3_down_lidar
    all_ldown2w = torch.tensor(
        np.array([e2w.compose(downlidar2ego).transform_matrix for e2w in all_ego2w]), dtype=torch.float32
    )
    all_times = torch.from_numpy(log_pose_df["timestamp_ns"].to_numpy() / 1e9)

    for i in range(len(sorted_filepaths)):
        if not todo[i]:
            continue
        pc_up, pc_down = point_clouds_up[i], point_clouds_down[i]
        lup2w, ldown2w, time = poses_up[i], poses_down[i], times[i]
        for j, (pc_, l2w) in enumerate(zip((pc_up, pc_down), (lup2w, ldown2w))):
            pc = pc_.clone().to(torch.float64)
            # relative -> absolute time
            pc[:, -2] += time
            # project to world frame
            pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
            # remove ego motion compensation and move to sensor frame
            if j == 0:  # up_lidar
                l2ws = all_lup2w
                lidar_name = "lidar_up"
            else:  # down_lidar
                l2ws = all_ldown2w
                lidar_name = "lidar_down"
            pc, interpolated_poses = _remove_ego_motion_compensation(pc, l2ws, all_times)
            # reset time to relative
            pc[:, 4] = pc_[:, 4].clone()
            # transform to common lidar frame again, interpolated_poses is l2w for each point at its timestamp
            # NOTE: `poses_down` is stored 4x4 while `lidar_to_worlds` (up) is 3x4, so the inverse is
            # sliced back to 3x4 here. Without it the down-lidar composition stays 4x4, the transform
            # in _get_missing_points_av2 emits a 4th homogeneous column, and the subsequent
            # `[..., 3] += 32` / reorder land on that `w` instead of the laser index -- which drops the
            # `time` column and writes a constant 33 into the channel column. The up-lidar branch is
            # 3x4 already, so this only changes the down lidar, and it reproduces the published cache.
            interpolated_poses = torch.matmul(
                pose_inverse(l2w.unsqueeze(0))[..., :3, :4].float(), to4x4(interpolated_poses).float()
            )
            # move channel from index 5 to 3
            pc = pc[..., [0, 1, 2, 5, 3, 4]]
            if j == 1:  # down_lidar
                pc[..., 3] -= 32
            # add missing points
            miss_points = _get_missing_points_av2(
                pc,
                interpolated_poses,
                lidar_name,
                dist_cutoff=0.0,
                outlier_thresh=0.2,
                ignore_regions=AV2_IGNORE_REGIONS[lidar_name],
            )
            if j == 1:  # down_lidar
                miss_points[..., 3] += 32

            # move channel back from index 3 to 5
            miss_points = miss_points[..., [0, 1, 2, 4, 5, 3]]
            if j == 0:  # up_lidar
                missing_points_up[i] = miss_points.float()
            else:  # down_lidar
                missing_points_down[i] = miss_points.float()

    missing_points_in_ego = {}
    for i, pc_up in missing_points_up.items():
        pc_down = missing_points_down[i]
        pc_up_in_ego = torch.cat([uplidar2ego.transform_point_cloud(pc_up[:, :3]), pc_up[:, 3:]], dim=-1)
        pc_down_in_ego = torch.cat([downlidar2ego.transform_point_cloud(pc_down[:, :3]), pc_down[:, 3:]], dim=-1)
        missing_points_in_ego[i] = torch.cat([pc_up_in_ego, pc_down_in_ego], dim=0).cpu().numpy()
    return missing_points_in_ego, sorted_filepaths


def _get_missing_points_av2(
    point_cloud: torch.Tensor,
    l2ws: torch.Tensor,
    lidar_name: str,
    dist_cutoff: float = 1.0,
    ignore_regions: list = [],
    outlier_thresh: float = 0.2,
) -> torch.Tensor:
    """Finds missing points in the point cloud according to sensor spec (self.config.lidar_elevation_mapping)

    Args:
        point_cloud: Point cloud to find missing points in (in sensor frame). Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
        l2ws: Poses of the lidar. Shape: [num_points, 4, 4]
        lidar_name: Name of the lidar
        dist_cutoff: Distance cutoff for points to consider. Points closer than this will be ignored.
        ignore_regions: List of regions to ignore. Each region is a list of [min_azimuth, max_azimuth, min_elevation, max_elevation]
        outlier_thresh: Threshold for outlier elevation values. If the median elevation of the missing points is more than this value away from the median elevation of the points in the channel, we ignore the missing points.

    Returns:
        Missing points in the point cloud, in world_frame. Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
    """
    dist = torch.norm(point_cloud[:, :3], dim=-1)
    dist_mask = dist > dist_cutoff
    dist = dist[dist_mask]
    point_cloud = point_cloud[dist_mask]
    l2ws = l2ws[dist_mask]
    elevation = torch.arcsin(point_cloud[:, 2] / dist)
    elevation = torch.rad2deg(elevation)
    azimuth = torch.atan2(point_cloud[:, 1], point_cloud[:, 0])
    azimuth = torch.rad2deg(azimuth)

    # find missing points
    missing_points = []
    for channel_id, expected_elevation in AV2_ELEVATION_MAPPING[lidar_name].items():
        if channel_id in AV2_SKIP_ELEVATION_CHANNELS[lidar_name]:
            continue
        channel_mask = (point_cloud[:, 3] - channel_id).abs() < 0.1  # handle floats
        curr_azimuth = azimuth[channel_mask]
        curr_l2ws = l2ws[channel_mask]
        curr_elev = elevation[channel_mask]
        if not channel_mask.any():
            continue
        curr_azimuth, sort_idx = curr_azimuth.sort()
        curr_l2ws = curr_l2ws[sort_idx]
        curr_elev = curr_elev[sort_idx]

        # find missing azimuths, we should have 360 / lidar_azimuth_resolution azimuths
        num_expected_azimuths = int(360 / AV2_AZIMUTH_RESOLUTION[lidar_name]) + 1
        expected_idx = torch.arange(num_expected_azimuths, device=curr_azimuth.device)
        # find offset
        offset = curr_azimuth[0] % AV2_AZIMUTH_RESOLUTION[lidar_name]
        current_idx = ((curr_azimuth - offset + 180) / AV2_AZIMUTH_RESOLUTION[lidar_name]).round().int()
        missing_idx = expected_idx[torch.isin(expected_idx, current_idx, invert=True)]
        # interpolate missing azimuths
        missing_azimuth = (
            torch.from_numpy(
                np.interp(
                    missing_idx,
                    torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                    torch.cat(
                        [
                            (-180 + offset - AV2_AZIMUTH_RESOLUTION[lidar_name]).view(1),
                            curr_azimuth,
                            (180 + offset + AV2_AZIMUTH_RESOLUTION[lidar_name]).view(1),
                        ]
                    ),
                )
            )
            .float()
            .view(-1, 1)
        )
        missing_elevation = (
            torch.from_numpy(
                np.interp(
                    missing_idx,
                    torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                    torch.cat(
                        [
                            torch.tensor(expected_elevation).view(1),
                            curr_elev.view(-1),
                            torch.tensor(expected_elevation).view(1),
                        ]
                    ),
                )
            )
            .float()
            .view(-1, 1)
        )
        elevation_outlier_mask = (missing_elevation - curr_elev.median()).abs() > outlier_thresh
        missing_elevation[elevation_outlier_mask] = curr_elev.median()

        ignore_mask = torch.zeros_like(missing_azimuth.squeeze(-1)).bool()
        for ignore_region in ignore_regions:
            ignore_mask = (
                (missing_azimuth > ignore_region[0])
                & (missing_azimuth < ignore_region[1])
                & (missing_elevation > ignore_region[2])
                & (missing_elevation < ignore_region[3])
            ).squeeze(-1) | ignore_mask
        missing_azimuth = missing_azimuth[~ignore_mask]
        missing_elevation = missing_elevation[~ignore_mask]

        missing_distance = torch.ones_like(missing_azimuth) * AV2_DUMMY_DISTANCE_VALUE
        x = torch.cos(torch.deg2rad(missing_azimuth)) * torch.cos(torch.deg2rad(missing_elevation)) * missing_distance
        y = torch.sin(torch.deg2rad(missing_azimuth)) * torch.cos(torch.deg2rad(missing_elevation)) * missing_distance
        z = torch.sin(torch.deg2rad(missing_elevation)) * missing_distance
        points = torch.cat([x, y, z], dim=-1)
        # transform points from sensor space to world space
        points = torch.cat([points, torch.ones_like(points[:, -1:])], dim=-1)
        # find closest pose idx
        closest_pose_idx = torch.searchsorted(curr_azimuth, missing_azimuth.squeeze())
        closest_pose_idx = closest_pose_idx.clamp(max=len(curr_azimuth) - 1)
        if closest_pose_idx.shape == ():
            closest_pose_idx = closest_pose_idx.unsqueeze(0)
        closest_l2w = curr_l2ws[closest_pose_idx]
        points = torch.matmul(closest_l2w.float(), points.unsqueeze(-1))[:, :, 0]
        closest_pc_values = point_cloud[channel_mask][sort_idx][closest_pose_idx, 3:]
        points = torch.cat([points, closest_pc_values], dim=-1)
        missing_points.append(points)

    missing_points = torch.cat(missing_points, dim=0)
    return missing_points


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def find_split_dir(root_path: Path, split: str) -> Path:
    """Locate the directory holding the logs of `split`.

    Handles both layouts in the wild: the released tarballs extract to
    ``<root>/<split>/<log>``, while the AV2 devkit's SensorDataloader (and the
    queryocc AV2Dataset, via ``subset``) expects ``<root>/sensor/<split>/<log>``.
    Same resolution as ``generate_pseudo_labels/create_infos_av2.py``.
    """
    candidates = [root_path / split, root_path / "sensor" / split]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No {split} split under {root_path} (tried: {', '.join(str(c) for c in candidates)})")


@dataclass
class MissingPointsConfig:
    dataset: Literal["nuscenes", "argoverse2"] = "nuscenes"
    """The dataset to generate missing points for."""

    out_dir: Path = paths.NUSC_MISSING_POINTS_ROOT
    """Root of the missing-points cache that training reads back."""

    nuscenes_data: Path = paths.NUSC_ROOT
    """The path to the nuscenes data."""

    argoverse_data: Path = paths.AV2_ROOT
    """The path to the argoverse 2 data."""

    version: str = "v1.0-trainval"
    """nuScenes devkit version, e.g. v1.0-trainval or v1.0-mini."""

    split: Literal["train", "val", "test"] = "train"
    """Argoverse 2 split."""

    override: bool = False
    """Whether to regenerate sweeps whose output file already exists."""

    num_workers: int = 8
    """Number of worker threads (one scene / log per worker)."""


@dataclass
class Counts:
    """Per-sweep tallies, plus scenes/logs that raised."""

    written: int = 0
    skipped: int = 0
    failed_sweeps: int = 0
    failed_scenes: int = 0

    def add(self, other: "Counts") -> None:
        self.written += other.written
        self.skipped += other.skipped
        self.failed_sweeps += other.failed_sweeps
        self.failed_scenes += other.failed_scenes


class MissingPointsEngine:
    def __init__(self, config: MissingPointsConfig):
        self.config = config

    # ------------------------------------------------------------------ nuScenes

    def _run_nuscenes(self) -> Counts:
        from nuscenes.nuscenes import NuScenes

        # The NuScenes devkit object holds the whole database plus its reverse indices and is
        # not cheaply picklable, so scenes are farmed out to *threads* (which share it) rather
        # than processes. The heavy lifting is numpy/torch, which releases the GIL.
        nusc = NuScenes(version=self.config.version, dataroot=str(self.config.nuscenes_data), verbose=False)
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {len(nusc.scene)} scenes")
        print(f"Output root: {self.config.out_dir}")

        totals = Counts()
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {executor.submit(self._extract_nuscenes_scene, nusc, scene): scene for scene in nusc.scene}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
                totals.add(future.result())
        return totals

    def _extract_nuscenes_scene(self, nusc, scene: Dict) -> Counts:
        counts = Counts()
        try:
            lidars, lidar_filenames = _get_lidars_nusc(nusc, scene)
            out_paths = [self.config.out_dir / f.name.replace(".pcd.bin", "_missing.npy") for f in lidar_filenames]
            todo = [self.config.override or not p.exists() for p in out_paths]
            counts.skipped = sum(1 for t in todo if not t)
            if not any(todo):
                return counts

            missing_points = _get_missing_points_from_lidars_nusc(lidars, lidar_filenames, todo)
            for idx, mps in missing_points.items():
                np.save(str(out_paths[idx]), mps.numpy())
                counts.written += 1
        except Exception as e:  # noqa: BLE001 - one bad scene must not kill a 331k-file run
            print(f"Error processing scene {scene['name']}: {str(e)}")
            counts.failed_scenes = 1
        return counts

    # ------------------------------------------------------------------ Argoverse 2

    def _run_argoverse2(self) -> Counts:
        from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

        datapath = find_split_dir(self.config.argoverse_data, self.config.split)
        print(f"Setting up data loader for {datapath} ...")
        av2 = AV2SensorDataLoader(datapath, datapath)

        out_dir = self.config.out_dir / self.config.split
        # NOTE: the original script hardcoded `av2.get_log_ids()[350:]` as a crude resume hack,
        # silently skipping 350 logs. Resumability is now per-sweep, see `todo` below.
        sequences = av2.get_log_ids()
        print(f"Processing {len(sequences)} sequences in {self.config.split}-split")
        print(f"Output root: {out_dir}")

        totals = Counts()
        # Threads for the same reason as nuScenes: the dataloader keeps a big in-memory index.
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {
                executor.submit(self._extract_argoverse_sequence, av2, out_dir, sequence): sequence
                for sequence in sequences
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing logs"):
                totals.add(future.result())
        return totals

    def _extract_argoverse_sequence(self, av2, out_dir: Path, sequence: str) -> Counts:
        counts = Counts()
        try:
            sequence_out_dir = out_dir / sequence
            # Same ordering as _read_lidars_av2 (which sorts the same filepaths), so `todo`
            # indices line up with the sweeps it returns.
            sorted_pc_paths = sorted(
                av2.get_lidar_fpath(sequence, t) for t in av2.get_ordered_log_lidar_timestamps(sequence)
            )
            todo = [
                self.config.override or not (sequence_out_dir / f"{p.stem}_missing.npy").exists()
                for p in sorted_pc_paths
            ]
            counts.skipped = sum(1 for t in todo if not t)
            if not any(todo):
                return counts

            lidars, pc_filenames = _get_lidars_av2(av2, sequence)
            missing_points_in_ego, returned_pc_paths = _read_lidars_av2(
                lidars, pc_filenames, av2._data_dir, sequence, todo
            )
            assert returned_pc_paths == sorted_pc_paths

            os.makedirs(sequence_out_dir, exist_ok=True)
            for idx, missing_points in missing_points_in_ego.items():
                pc_name = sorted_pc_paths[idx].stem + "_missing.npy"
                np.save(sequence_out_dir / pc_name, missing_points)
                counts.written += 1
        except Exception as e:  # noqa: BLE001 - one bad log must not kill the run
            print(f"Error processing sequence {sequence}: {str(e)}")
            counts.failed_scenes = 1
        return counts

    # ------------------------------------------------------------------ entry point

    def run(self) -> Counts:
        if self.config.dataset == "nuscenes":
            totals = self._run_nuscenes()
        elif self.config.dataset == "argoverse2":
            totals = self._run_argoverse2()
        else:
            raise ValueError(f"Invalid dataset: {self.config.dataset}")

        unit = "scenes" if self.config.dataset == "nuscenes" else "logs"
        print(
            f"Missing points: {totals.written} written, {totals.skipped} skipped (already existed), "
            f"{totals.failed_scenes} {unit} failed"
        )
        return totals


def parse_args():
    parser = argparse.ArgumentParser(description="Missing points (free-ray) generator")

    parser.add_argument(
        "--dataset", type=str, choices=["nuscenes", "argoverse2"], default="nuscenes", help="The dataset to use"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Root of the missing-points cache. Defaults to paths.NUSC_MISSING_POINTS_ROOT "
        "($QUERYOCC_NUSC_MISSING_POINTS_ROOT) for nuscenes and paths.AV2_MISSING_POINTS_ROOT "
        "($QUERYOCC_AV2_MISSING_POINTS_ROOT) for argoverse2",
    )

    parser.add_argument(
        "--nuscenes_data",
        type=str,
        default=None,
        help="The path to the nuscenes data directory (default: paths.NUSC_ROOT, $NUSC_ROOT)",
    )

    parser.add_argument(
        "--argoverse_data",
        type=str,
        default=None,
        help="The path to the argoverse 2 data directory (default: paths.AV2_ROOT, $AV2_ROOT). Both layouts are "
        "accepted: <root>/<split>/<log> and <root>/sensor/<split>/<log>",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes devkit version, e.g. v1.0-trainval or v1.0-mini (nuscenes only)",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="The split to use (argoverse2 only)",
    )

    parser.add_argument(
        "--override",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to regenerate sweeps whose output file already exists (true/false)",
    )

    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads (one scene/log each)")

    return parser.parse_args()


def main():
    args = parse_args()

    default_out_dir = paths.NUSC_MISSING_POINTS_ROOT if args.dataset == "nuscenes" else paths.AV2_MISSING_POINTS_ROOT

    config = MissingPointsConfig(
        dataset=args.dataset,
        out_dir=Path(args.out_dir) if args.out_dir else default_out_dir,
        nuscenes_data=Path(args.nuscenes_data) if args.nuscenes_data else paths.NUSC_ROOT,
        argoverse_data=Path(args.argoverse_data) if args.argoverse_data else paths.AV2_ROOT,
        version=args.version,
        split=args.split,
        override=args.override.lower() == "true",
        num_workers=args.num_workers,
    )

    print("Running Missing Points generator with config:")
    print(f"  dataset: {config.dataset}")
    print(f"  out_dir: {config.out_dir}")
    print(f"  nuscenes_data: {config.nuscenes_data}")
    print(f"  argoverse_data: {config.argoverse_data}")
    print(f"  version: {config.version}")
    print(f"  split: {config.split}")
    print(f"  override: {config.override}")
    print(f"  num_workers: {config.num_workers}")

    engine = MissingPointsEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
