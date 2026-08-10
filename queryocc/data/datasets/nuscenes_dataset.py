import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import torch
from av2.utils.io import read_img
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from zod.data_classes.ego_motion import EgoMotion

import queryocc.config.base_config as cfg
from queryocc.common import paths
from queryocc.data.datasets.nuscenes_infos import NUSCENES_CAMERAS, load_or_build_infos
from queryocc.data.datasets.nuscenes_mappings import LabelMapper
from queryocc.data.datasets.nuscenes_utils import (
    EGO_POINTS_RANGE,
    ego_motion_compensate_scanwise,
    fast_uncompensate_pointwise,
    get_ego_from_lidar,
    get_global_pose,
    points_in_target_bev,
)
from queryocc.data.datasets.utils import (
    crop_images,
    crop_intrinsics,
    pad_offset_data_for_batching,
    resize_images_with_scale,
    resize_intrinsics,
    resize_intrinsics_with_scale,
)
from queryocc.data.datasets.zod_base import ZODBaseDatasetConfig
from queryocc.data.vfm.metric3d_reader import Metric3DReader
from queryocc.data.vfm.vfm_reader import VFMReader
from queryocc.utils.geometry import inverse_np

MAX_RELECTANCE_VALUE = 255.0
NUSC_LIDAR_FREQUENCY = 20.0  # Hz
NUSC_LIDAR_CHANNELS = 32
NUSC_IMAGE_FREQUENCY = 10.0  # Hz


# (top, right, bottom, left)
NUSCENES_CAM_CROP = {
    "CAM_FRONT": (0, 0, 0, 0),
    "CAM_FRONT_LEFT": (0, 0, 0, 0),
    "CAM_FRONT_RIGHT": (0, 0, 0, 0),
    "CAM_BACK_LEFT": (0, 0, 0, 0),
    "CAM_BACK_RIGHT": (0, 0, 0, 0),
    "CAM_BACK": (0, 0, 150, 0),
}

warnings.filterwarnings(action="ignore", category=UserWarning)

# Native nuScenes camera resolution, used to shape the depth placeholder for frames
# that have no stored pseudo-label depth.
DEFAULT_IMAGE_SHAPE = (900, 1600)

# Emitted once per process: with add_missing_points=True but no cache, training silently
# proceeds on far weaker free-space supervision (only out-of-range points), which is
# invisible in the loss and easy to mistake for a modelling result.
_MISSING_POINTS_WARNED = False


def _warn_missing_points_cache_absent(cache_dir) -> None:
    global _MISSING_POINTS_WARNED
    if _MISSING_POINTS_WARNED:
        return
    _MISSING_POINTS_WARNED = True
    warnings.warn(
        f"add_missing_points=True but no cached free-ray files were found under {cache_dir}. "
        "Training will continue with much weaker free-space supervision. Generate the cache "
        "with `python -m tools.missing_points.engine` (see tools/missing_points/README.md) "
        "or set --datamodule.dataset.no-add_missing_points to silence this.",
        stacklevel=2,
    )


@dataclass
class NuScenesDatasetConfig(ZODBaseDatasetConfig, cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: NuScenesDataset)
    """The target class for this configuration."""

    data: Path = Path("/datasets/nuscenes/")
    """The path to the dataset."""

    occ3d_data: Path = paths.OCC3D_ROOT
    """The path to the occ3d dataset."""

    nusc_cached_missing_points_dir: Path = paths.NUSC_MISSING_POINTS_ROOT
    """The path to the cached missing points."""

    only_tokens: bool = False

    nuscenes_cameras_to_load: Tuple[str, ...] = NUSCENES_CAMERAS
    """The cameras to load images from."""
    nuscenes_camera_downsample_factor: float = 2.0
    """The factor to downsample the camera images by."""

    n_max_offset_points_to_load_per_scan: int = 30_000
    """The maximum number of offset points to load per scan."""
    n_max_missing_rays_to_load_per_scan: int = 10_000
    """The maximum number of missing rays to load per scan."""

    load_4d_lidar_origins: bool = True


# Coordinate pipeline for nuScenes:
#   sensor frame (lidar points in LIDAR_TOP frame)
#   → ego frame at sweep timestamp  (via lidar-to-ego calibration extrinsics from nuscenes-devkit)
#   → global frame                  (via ego-to-global pose)
#   → ego frame at bev_oxts_pose    (via inverse of bev_oxts_pose)
#   → BEV frame                     (axis-aligned, centered at bev_oxts_pose)
#
# bev_oxts_pose: 4×4 SE(3) matrix, ego vehicle pose in world frame at current timestamp.
# Transformed from nuscenes ego-to-global pose using pyquaternion + translation.


class NuScenesDataset(Dataset):
    _BLACKLIST: List[str] = []

    def __init__(
        self,
        config: NuScenesDatasetConfig,
        train_or_val: Literal["train", "val"],
        split: Literal["original", "geo_near"] = "original",
    ):
        super().__init__()
        assert config.point_cloud_range != (0, 0, 0, 0, 0, 0), "Model did not set point_cloud_range."

        self.config = config
        self.train_or_val = train_or_val
        self.split = split
        self.version = "v1.0-trainval" if self.config.version == "full" else "v1.0-mini"
        self.label_mapper = LabelMapper(config.panoptic_labels_classes)

        self._init_queryocc_infos(config, train_or_val, split)

        self.dataset_idx_to_scene_and_idx: List[Tuple[str, int]] = []
        # check how many previous frames we need to be able to load
        start_frames_buffer, end_frames_buffer = self.get_start_and_end_frames_buffer()

        for scene, sweeps in self.sweeps_per_scene.items():
            samples_to_select = len(sweeps) - start_frames_buffer - end_frames_buffer
            assert samples_to_select > 0, f"Not enough sweeps in {scene}"
            if self.only_key_frames or self.is_val:
                self.dataset_idx_to_scene_and_idx += [
                    (str(scene), i + start_frames_buffer)
                    for i in range(samples_to_select)
                    if sweeps[i + start_frames_buffer]["sample_data_token"] in self.sample_data_token_to_sample_token
                ]
            else:
                self.dataset_idx_to_scene_and_idx += [
                    (str(scene), i + start_frames_buffer) for i in range(samples_to_select)
                ]

        # Shuffle with a fixed local RNG so dataset indices map to the same samples in
        # every process (the train dataloader reshuffles anyway; this only interleaves scenes).
        if train_or_val == "train":
            np.random.default_rng(0).shuffle(self.dataset_idx_to_scene_and_idx)

        if config.load_vfm_stuff:
            self.vfm_reader: VFMReader = config.vfm_cfg.setup(
                dataset_name="nuscenes",
                data_root=config.data,
                label_mapper=self.label_mapper,
                point_cloud_range=config.point_cloud_range,
            )

        if self.config.load_depth_maps and self.config.depth_maps_source == "metric3d":
            self.metric3d_depth_reader: Metric3DReader = self.config.metric3d_depth_reader.setup(
                dataset_name="nuscenes"
            )

    def _init_queryocc_infos(self, config, train_or_val, split):
        queryocc_infos = load_or_build_infos(config, train_or_val, split, self.version)

        if len(config.ids_to_load) > 0:
            queryocc_infos["sweeps_per_scene"] = {
                scene: sweeps
                for scene, sweeps in queryocc_infos["sweeps_per_scene"].items()
                if scene in config.ids_to_load
            }
            print(
                f"[{train_or_val}:{split}] Loaded {len(queryocc_infos['sweeps_per_scene'])} scenes from {config.ids_to_load}"
            )

        # sort scenes by name
        sweeps_per_scene = dict(sorted(queryocc_infos["sweeps_per_scene"].items(), key=lambda x: x[0]))
        self.sample_data_token_to_sample_token = queryocc_infos["sample_data_token_to_sample_token"]
        self.sweeps_per_scene = sweeps_per_scene
        self.scene_token_to_real_scene_name = queryocc_infos["scene_token_to_real_scene_name"]
        self.cameras_per_scene = queryocc_infos["cameras_per_scene"]
        self.camera_timestamps_per_scene = queryocc_infos["camera_timestamps_per_scene"]
        self.camera_tokens_per_scene = queryocc_infos["camera_tokens_per_scene"]
        self.camera_calib_per_scene = queryocc_infos["camera_calib_per_scene"]
        self.scene_to_location = queryocc_infos["scene_to_location"]

    @property
    def is_train(self) -> bool:
        return "train" in self.train_or_val

    @property
    def is_val(self) -> bool:
        return "val" in self.train_or_val

    @property
    def any_augmentation(self) -> bool:
        return self.config.rotation_augmentation or self.config.translation_augmentation

    @property
    def lidar_frequency(self) -> float:
        return NUSC_LIDAR_FREQUENCY

    @property
    def image_frequency(self) -> float:
        return NUSC_IMAGE_FREQUENCY

    @property
    def max_prev_images_frames(self) -> int:
        return int(np.ceil((self.config.prev_images_max_time) * self.image_frequency))

    @property
    def max_future_images_frames(self) -> int:
        return int(np.ceil(self.config.future_images_max_time * self.image_frequency))

    @property
    def max_prev_lidar_frames(self) -> int:
        return int(np.ceil(self.config.prev_lidar_max_time * self.lidar_frequency))

    @property
    def max_future_lidar_frames(self) -> int:
        return int(np.ceil(self.config.future_lidar_max_time * self.lidar_frequency))

    @property
    def load_future_offset_frames(self) -> bool:
        return self.config.num_future_frames_to_sample > 0

    @property
    def load_past_offset_frames(self) -> bool:
        return self.config.num_past_frames_to_sample > 0

    @property
    def min_future_offset(self) -> float:
        min_ = 1 if self.config.min_future_frame_s > 0.0 else 0
        return max(int(self.config.min_future_frame_s * self.lidar_frequency), min_)

    @property
    def max_future_offset(self) -> float:
        return int(self.config.max_future_frame_s * self.lidar_frequency)

    @property
    def min_past_offset(self) -> float:
        min_ = 1 if self.config.min_past_frame_s > 0.0 else 0
        return max(int(self.config.min_past_frame_s * self.lidar_frequency), min_)

    @property
    def max_past_offset(self) -> float:
        return int(self.config.max_past_frame_s * self.lidar_frequency)

    @property
    def load_occ3d(self) -> bool:
        if self.is_train:
            return self.config.load_occ3d_train
        else:
            return self.config.load_occ3d_val

    @property
    def only_key_frames(self) -> bool:
        return self.config.only_tokens or self.is_val

    @property
    def only_key_frames_for_offsets(self) -> bool:
        return self.config.only_tokens_for_offsets or self.is_val

    def get_start_and_end_frames_buffer(self) -> Tuple[int, int]:
        start_frames_buffer_alternatives = []
        end_frames_buffer_alternatives = []
        if self.config.load_image:
            start_frames_buffer_alternatives.append(self.max_prev_images_frames)
            end_frames_buffer_alternatives.append(self.max_future_images_frames)
        if self.config.load_lidar:
            start_frames_buffer_alternatives.append(self.max_prev_lidar_frames)
            end_frames_buffer_alternatives.append(self.max_future_lidar_frames)
        if len(start_frames_buffer_alternatives) == 0:
            start_frames_buffer_alternatives.append(self.max_prev_lidar_frames)
            end_frames_buffer_alternatives.append(self.max_future_lidar_frames)

        start_frames_buffer = max([2] + start_frames_buffer_alternatives)
        end_frames_buffer = max([1] + end_frames_buffer_alternatives)
        return start_frames_buffer, end_frames_buffer

    def __len__(self):
        return len(self.dataset_idx_to_scene_and_idx)

    def _get_ego_motion(self, scene_name: str) -> EgoMotion:
        sweeps = self.sweeps_per_scene[scene_name]
        n_sweeps = len(sweeps)

        poses = np.zeros((n_sweeps, 4, 4))
        for i in range(n_sweeps):
            poses[i, :3, :3] = Quaternion(sweeps[i]["ego2global_rotation"]).rotation_matrix
            poses[i, :3, 3] = sweeps[i]["ego2global_translation"]
            poses[i, 3, 3] = 1

        # make all poses relative first pose
        origin = poses[0].copy()
        poses = inverse_np(origin) @ poses
        poses[0] = np.eye(4)

        timestamps = np.array([sweep["timestamp"] / 1e6 for sweep in sweeps], dtype=np.float64)

        # order poses by timestamps
        order = np.argsort(timestamps)
        poses = poses[order]
        timestamps = timestamps[order]

        return EgoMotion(
            poses=poses,
            timestamps=timestamps,
            velocities=np.zeros((n_sweeps, 3)),  # we dont have velocities
            accelerations=np.zeros((n_sweeps, 3)),  # we dont have accelerations
            angular_rates=np.zeros((n_sweeps, 3)),  # we dont have angular rates
            origin_lat_lon=origin,  # This is used for extracting map data
        )

    def _rotation_augmentation(
        self, ego_motion: EgoMotion, current_timestamp: float
    ) -> Tuple[EgoMotion, np.ndarray, float, np.ndarray, np.ndarray]:
        if self.is_train and self.any_augmentation:
            rotation_range = self.config.rotation_range
            translation_range = self.config.translation_range

            if self.config.rotation_augmentation:
                angle = np.random.uniform(rotation_range[0], rotation_range[1])
                rotation_matrix = np.array(
                    [
                        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
                    ]
                )
            else:
                rotation_matrix = np.eye(2)

            if self.config.translation_augmentation:
                translation = np.random.uniform(translation_range[0], translation_range[1], size=(2,))
            else:
                translation = np.zeros(2)

            transform = np.eye(4)
            transform[:2, :2] = rotation_matrix[:2, :2]
            transform[:2, 3] = translation

            # we want to transform the oxts to the current time first and then apply the transformation
            current_oxts_pose = ego_motion.get_poses(current_timestamp)
            bev_oxts_pose = current_oxts_pose.copy()
            oxts_poses = ego_motion.poses.copy()
            oxts_poses = current_oxts_pose @ transform @ inverse_np(current_oxts_pose) @ oxts_poses
            ego_motion.poses = oxts_poses
        else:
            bev_oxts_pose = ego_motion.get_poses(current_timestamp)
            transform = np.eye(4)
            angle = 0
            translation = np.zeros(2)

        return ego_motion, bev_oxts_pose, angle, translation, transform

    def _crop_pointclouds(self, point_clouds: List[np.ndarray]):
        min_range = np.array(self.config.point_cloud_range[:3])
        max_range = np.array(self.config.point_cloud_range[3:])
        for i in range(len(point_clouds)):
            in_range = np.all((min_range <= point_clouds[i][:, :3]) & (point_clouds[i][:, :3] <= max_range), axis=-1)
            point_clouds[i] = point_clouds[i][in_range]
        return point_clouds

    def _load_pc(
        self, filepath: str, pc_timestamp: float, get_mask: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # loads the point cloud in lidar frame

        pc = np.fromfile(filepath, dtype=np.float32).reshape([-1, 5]).astype(np.float64)
        pc[..., 3] = pc[..., 3] / MAX_RELECTANCE_VALUE
        # nuscenes lidar time is the time of the end of the sweep. Add estimated times per point
        n_sweeps = pc.shape[0] // NUSC_LIDAR_CHANNELS
        remainder = pc.shape[0] % NUSC_LIDAR_CHANNELS

        # Create time offsets for complete sweeps
        main_offsets = np.repeat(np.linspace(-1 / NUSC_LIDAR_FREQUENCY, 0, n_sweeps), NUSC_LIDAR_CHANNELS)

        # Handle remaining points if any
        if remainder > 0:
            last_sweep_time = -1 / NUSC_LIDAR_FREQUENCY
            remaining_offsets = np.linspace(last_sweep_time, 0, remainder)
            offsets = np.concatenate([main_offsets, remaining_offsets])
        else:
            offsets = main_offsets
        # we assume xyzrt so we overwrite the last column which is channel index
        pc[..., -1] = offsets + pc_timestamp
        # remove ego points
        mask = (abs(pc[..., :3]) >= EGO_POINTS_RANGE).any(axis=-1)
        pc = pc[mask]

        if get_mask:
            return pc, mask
        else:
            return pc

    def _get_timestamp(self, scene_name: str, current_idx: int) -> float:
        return self.sweeps_per_scene[scene_name][current_idx]["timestamp"] / 1e6

    def _aggregate_lidar_frames(
        self,
        scene_name: str,
        current_idx: int,
        ego_motion: EgoMotion,
        target_pose: np.ndarray,
        lidar_extrinsics: np.ndarray,
    ) -> Dict[str, Any]:
        # Calculate max number of previous frames based on time and frequency
        n_prev_max = self.max_prev_lidar_frames
        n_prev_lidar = self.config.n_prev_lidar

        if n_prev_max <= 0 or n_prev_lidar <= 0:
            prev_indices: List[int] = []
        else:
            # Generate uniformly spaced indices from max previous time to current
            prev_indices = np.linspace(current_idx - n_prev_max, current_idx, n_prev_lidar + 1)[:-1]
            prev_indices = np.floor(prev_indices).astype(np.int32).tolist()

        n_future_max = self.max_future_lidar_frames
        n_future_lidar = self.config.n_future_lidar

        if n_future_max <= 0 or n_future_lidar <= 0:
            future_indices: List[int] = []
        else:
            future_indices = np.linspace(current_idx, current_idx + n_future_max, n_future_lidar + 1)[1:]
            future_indices = np.floor(future_indices).astype(np.int32).tolist()

        used_indices = [current_idx] + prev_indices + future_indices
        used_indices = list(set(used_indices))
        used_indices.sort()

        lidar_filepaths = [self.sweeps_per_scene[scene_name][i]["data_path"] for i in used_indices]

        pcs = [
            self._load_pc(fp, self.sweeps_per_scene[scene_name][i]["timestamp"] / 1e6)
            for i, fp in zip(used_indices, lidar_filepaths)
        ]

        current_timestamp = self._get_timestamp(scene_name, current_idx)
        comp_pcs = []
        for pc in pcs:
            comp_pc = ego_motion_compensate_scanwise(
                pc, ego_motion, target_pose=target_pose, lidar_extrinsics=lidar_extrinsics
            )
            # remove the current timestamp from the time
            comp_pc[..., -1] -= current_timestamp
            # float64 is only needed while the time column holds absolute unix timestamps;
            # after re-basing to the current frame, float32 is exact enough. Casting per
            # sweep keeps the float64 footprint transient.
            comp_pcs.append(comp_pc.astype(np.float32))

        if comp_pcs:
            aggregated = [np.concatenate(comp_pcs, axis=0)]
        else:
            aggregated = [np.empty((0, 5), dtype=np.float32)]
        # # remove points outside the range if specified,
        if self.config.point_cloud_range is not None:
            aggregated = self._crop_pointclouds(aggregated)

        return {
            "lidars": aggregated,
            "lidar_timestamps": [self.sweeps_per_scene[scene_name][current_idx]["timestamp"]],  # ms
        }

    def _select_offset_frames(self, all_lidar_frames: List, frame_idx: int) -> List:
        offset_frames = []
        n_frames = len(all_lidar_frames)

        if self.load_future_offset_frames:
            start_idx = min(frame_idx + self.min_future_offset, n_frames)
            end_idx = min(frame_idx + self.max_future_offset, n_frames)
            if start_idx <= end_idx:
                lidar_frames = all_lidar_frames[start_idx : end_idx + 1]

                if self.only_key_frames_for_offsets:
                    lidar_frames = [frame for frame in lidar_frames if "/samples/" in frame["data_path"]]

                lidar_frames_weights = np.ones(len(lidar_frames))
                if self.config.upsample_key_frames:
                    key_frames = [True if "/samples/" in frame["data_path"] else False for frame in lidar_frames]
                    lidar_frames_weights[key_frames] = self.config.key_frames_weight
                lidar_frames_weights /= lidar_frames_weights.sum()

                num_frames_to_select = min(self.config.num_future_frames_to_sample, len(lidar_frames))
                if num_frames_to_select > 0:
                    if self.is_train:
                        select_frame_idx: List[int] = np.random.choice(
                            len(lidar_frames), num_frames_to_select, replace=False, p=lidar_frames_weights
                        ).tolist()
                    else:
                        # Uniformly select frames
                        select_frame_idx = (
                            np.linspace(0, len(lidar_frames) - 1, num_frames_to_select).astype(np.int32).tolist()
                        )

                    offset_frames += [lidar_frames[i] for i in select_frame_idx]

        if self.load_past_offset_frames:
            start_idx = max(frame_idx - self.max_past_offset, 0)
            end_idx = max(frame_idx - self.min_past_offset, 0)
            if start_idx <= end_idx:
                lidar_frames = all_lidar_frames[start_idx : end_idx + 1]

                if self.only_key_frames_for_offsets:
                    lidar_frames = [frame for frame in lidar_frames if "/samples/" in frame["data_path"]]

                lidar_frames_weights = np.ones(len(lidar_frames))
                if self.config.upsample_key_frames:
                    key_frames = [True if "/samples/" in frame["data_path"] else False for frame in lidar_frames]
                    lidar_frames_weights[key_frames] = self.config.key_frames_weight
                lidar_frames_weights /= lidar_frames_weights.sum()

                num_frames_to_select = min(self.config.num_past_frames_to_sample, len(lidar_frames))
                if num_frames_to_select > 0:
                    if self.is_train:
                        select_frame_idx: List[int] = np.random.choice(
                            len(lidar_frames), num_frames_to_select, replace=False, p=lidar_frames_weights
                        ).tolist()
                    else:
                        # Uniformly select frames
                        select_frame_idx = (
                            np.linspace(0, len(lidar_frames) - 1, num_frames_to_select).astype(np.int32).tolist()
                        )

                    offset_frames += [lidar_frames[i] for i in select_frame_idx]

        if self.config.always_add_current_frame_to_offsets:
            offset_frames.append(all_lidar_frames[frame_idx])

        return offset_frames

    def _get_offset_lidar_frames(
        self,
        scene_name: str,
        scene_idx: int,
    ) -> List[dict]:
        lidar_sweeps = self.sweeps_per_scene[scene_name]
        selected_sweeps = self._select_offset_frames(lidar_sweeps, scene_idx)
        selected_sweeps = sorted(selected_sweeps, key=lambda x: x["timestamp"])
        return selected_sweeps

    def _get_offset_pointclouds(
        self,
        scene_name: str,
        scene_idx: int,
        bev_oxts_pose: np.ndarray,
        ego_motion: EgoMotion,
        lidar_extrinsics: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], List[float]]:
        # load the future lidar scans
        future_lidar_sweeps = self._get_offset_lidar_frames(scene_name, scene_idx)
        future_lidar_filepaths = [sweep["data_path"] for sweep in future_lidar_sweeps]
        lidar_timestamps = [sweep["timestamp"] / 1e6 for sweep in future_lidar_sweeps]

        lidar_pcs_and_masks = [
            self._load_pc(lidar_file, t, get_mask=True)
            for lidar_file, t in zip(future_lidar_filepaths, lidar_timestamps)
        ]
        lidar_pcs = [pc for pc, _ in lidar_pcs_and_masks]
        lidar_masks = [mask for _, mask in lidar_pcs_and_masks]

        # slice relevant part of oxts (ego motion) and store for use on gpu
        current_timestamp = self._get_timestamp(scene_name, scene_idx)  # this is the current time
        sweep_eps = 0.2  # 0.2 seconds, sweep is roughly 0.1 seconds so this has margin
        max_timestamp = max(lidar_timestamps + [current_timestamp]) + sweep_eps
        min_timestamp = min(lidar_timestamps + [current_timestamp]) - sweep_eps
        lower_mask = min_timestamp < ego_motion.timestamps
        upper_mask = ego_motion.timestamps < max_timestamp
        oxts_range_mask = lower_mask & upper_mask
        sliced_oxts = EgoMotion(  # dummy for use with compensation code
            ego_motion.poses[oxts_range_mask],
            None,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
            ego_motion.timestamps[oxts_range_mask],
            None,  # type: ignore
        )

        labels = {}
        if self.config.load_offset_panoptic_labels:
            panoptic_labels = self._get_panoptic_labels(lidar_pcs, future_lidar_sweeps, lidar_masks)
            labels.update(panoptic_labels)

        # downsample the pointclouds if needed
        if (max_per_scan := self.config.n_max_offset_points_to_load_per_scan) >= 0:
            for i, lidar_sweep in enumerate(lidar_pcs):
                if len(lidar_sweep) > max_per_scan:
                    idx = np.random.permutation(len(lidar_sweep))[:max_per_scan]
                    lidar_pcs[i] = lidar_pcs[i][idx]
                    # Filter all labels with the same indices
                    for key, value in labels.items():
                        labels[key][i] = value[i][idx]

        # compute in-range mask
        in_range_mask = [
            points_in_target_bev(
                lidar_pc, sliced_oxts, self.config.point_cloud_range, bev_oxts_pose, lidar_extrinsics, t
            )
            for lidar_pc, t in zip(lidar_pcs, lidar_timestamps)
        ]

        # uncompensate clouds if needed
        lidar_pcs = [
            fast_uncompensate_pointwise(pc, sliced_oxts, lidar_extrinsics, t)
            for pc, t in zip(lidar_pcs, lidar_timestamps)
        ]

        # fetch these before we mess up the timestamps
        if self.config.add_missing_points:
            missing_points = self._load_missing_points(
                future_lidar_filepaths, future_lidar_sweeps, sliced_oxts, lidar_extrinsics, current_timestamp
            )

        # set everything in reference to the current frame
        for pc in lidar_pcs:
            pc[..., -1] -= current_timestamp

        sliced_oxts.timestamps -= current_timestamp
        sliced_oxts.poses = inverse_np(bev_oxts_pose) @ sliced_oxts.poses

        # NOTE: If we want to move the pointclouds to their correct position we have to also include the laser number
        # here to use for the transformation later.
        offset_pointclouds = [np.hstack((pc[..., :3], pc[..., 4:5])) for pc in lidar_pcs]
        offset_pcds_in_range = [pc[mask] for pc, mask in zip(offset_pointclouds, in_range_mask)]
        offset_pointclouds_indices = np.concatenate(
            [np.full((in_range_mask[i].sum(),), i, dtype=np.uint8) for i in range(len(in_range_mask))]
        )

        # Mask and Concatenate all labels
        for key, value in labels.items():
            labels[key] = np.concatenate([value[mask] for value, mask in zip(value, in_range_mask)])

        # points outside the range become missing rays
        missing_rays = [pc[~mask] for pc, mask in zip(offset_pointclouds, in_range_mask)]

        if self.config.add_missing_points:
            missing_rays += missing_points

        missing_rays = np.vstack(missing_rays).astype(np.float32)
        offset_pcds_in_range = np.vstack(offset_pcds_in_range).astype(np.float32)

        # get the ratio of points inside the range to be used to samples queries in occupancy sampler
        ratio_inside = len(offset_pcds_in_range) / (len(offset_pcds_in_range) + len(missing_rays))

        if self.config.n_max_offset_points >= 0:
            idx = np.random.permutation(len(offset_pcds_in_range))[: self.config.n_max_offset_points]
            offset_pcds_in_range = offset_pcds_in_range[idx]
            offset_pointclouds_indices = offset_pointclouds_indices[idx]
            for key, value in labels.items():
                labels[key] = value[idx]

        if self.config.n_max_missing_rays >= 0:
            idx = np.random.permutation(len(missing_rays))[: self.config.n_max_missing_rays]
            missing_rays = missing_rays[idx]

        if self.config.make_batchable:
            (
                offset_pcds_in_range,
                offset_pointclouds_indices,
                missing_rays,
                sliced_oxts,
                labels,
            ) = pad_offset_data_for_batching(
                offset_pcds_in_range,
                offset_pointclouds_indices,
                missing_rays,
                sliced_oxts,
                labels,
                n_max_offset_points=self.config.n_max_offset_points,
                n_max_missing_rays=self.config.n_max_missing_rays,
                n_max_poses=self.config.n_max_poses,
            )

        data = {
            "ego_poses": sliced_oxts.poses.astype(np.float64),
            "ego_pose_times": sliced_oxts.timestamps.astype(np.float32),
            "lidar_extrinsics": lidar_extrinsics,
            "offset_pointclouds": offset_pcds_in_range,
            "offset_pointclouds_indices": offset_pointclouds_indices,
            "missing_rays": missing_rays,
            "future_points_ratio_inside": ratio_inside,
        }

        if labels:
            data.update({"offset_pointclouds_labels": labels})

        return data, lidar_timestamps

    def _get_ground_truth_panoptic_labels(
        self, data_token: str, lidar_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lidarseg_fpath = self.config.data / "lidarseg" / self.version / f"{data_token}_lidarseg.bin"
        lidarseg = np.fromfile(lidarseg_fpath, dtype=np.uint8).reshape(-1, 1)
        panoptic_fpath = self.config.data / "panoptic" / self.version / f"{data_token}_panoptic.npz"
        panoptic = np.load(panoptic_fpath)["data"].reshape(-1, 1).astype(np.int32)
        lidarseg = lidarseg[lidar_mask]
        lidarseg = self.label_mapper.lidarseg_to_learning(lidarseg)
        panoptic = panoptic[lidar_mask]
        valid = np.ones((len(lidarseg), 1), dtype=np.bool_)
        return lidarseg, panoptic, valid

    def _get_pseudo_panoptic_labels(self, data_token: str, lidar_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pseudoseg_fpath = str(
            paths.NUSC_PSEUDOSEG_ROOT / self.config.panoptic_pseudo_label_name / f"{data_token}_pseudoseg.bin"
        )
        num_points = lidar_mask.sum()
        if os.path.exists(pseudoseg_fpath):
            pseudoseg = np.fromfile(pseudoseg_fpath, dtype=np.uint8).reshape(-1, 1)
            pseudoseg = self.label_mapper.nusc_to_learning(pseudoseg)
            pseudoseg = pseudoseg[lidar_mask]
            valid = np.ones((num_points, 1), dtype=np.bool_)
            invalid = pseudoseg == self.label_mapper.free_idx
            valid[invalid] = False
        else:
            print(f"Pseudoseg file not found for {data_token}")
            pseudoseg = np.ones((num_points, 1), dtype=np.int32) * -1
            valid = np.zeros((num_points, 1), dtype=np.bool_)
        panoptic = np.zeros((num_points, 1), dtype=np.int32)
        return pseudoseg, panoptic, valid

    def _get_panoptic_labels(
        self, point_clouds: List[np.ndarray], lidar_sweeps: List[dict], lidar_masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        lidarseg_labels = []
        panoptic_labels = []
        valid_labels = []
        for pc, lidar_sweep, lidar_mask in zip(point_clouds, lidar_sweeps, lidar_masks):
            num_points = pc.shape[0]
            data_token = lidar_sweep["sample_data_token"]
            is_key_frame = "sample" in lidar_sweep["data_path"]
            if is_key_frame:
                if self.config.use_pseudo_labels_for_panoptic and self.is_train:
                    lidarseg, panoptic, valid = self._get_pseudo_panoptic_labels(data_token, lidar_mask)
                else:
                    lidarseg, panoptic, valid = self._get_ground_truth_panoptic_labels(data_token, lidar_mask)
                lidarseg_labels.append(lidarseg)
                panoptic_labels.append(panoptic)
                valid_labels.append(valid)
            else:
                lidarseg_labels.append(np.zeros((num_points, 1), dtype=np.uint8))
                panoptic_labels.append(np.zeros((num_points, 1), dtype=np.int32))
                valid_labels.append(np.zeros((num_points, 1), dtype=np.bool_))

        return {
            "panoptic_lidarseg": lidarseg_labels,
            "panoptic_cluster": panoptic_labels,
            "panoptic_valid": valid_labels,
        }

    def _load_missing_points(
        self,
        lidar_filepaths: List[str],
        lidar_sweeps: List,
        sliced_oxts: EgoMotion,
        lidar_extrinsics: np.ndarray,
        current_timestamp: float,
    ) -> List[np.ndarray]:
        mps_fnames = [Path(f).name.replace(".pcd.bin", "_missing.npy") for f in lidar_filepaths]
        mps_fpaths = [self.config.nusc_cached_missing_points_dir / f for f in mps_fnames]

        existing = [f.exists() for f in mps_fpaths]
        if not any(existing):
            _warn_missing_points_cache_absent(self.config.nusc_cached_missing_points_dir)

        xyzts = [np.load(f) for f, e in zip(mps_fpaths, existing) if e]
        xyzts = [np.concatenate((pc[:, :3], pc[:, 4:5]), axis=1) for pc in xyzts]
        sweeps = [s for s, e in zip(lidar_sweeps, existing) if e]
        sweeps_timestamps = [s["timestamp"] / 1e6 for s in sweeps]

        # Create sweeps of missing points
        missing_points_sweeps = []
        for xyzt in xyzts:
            z_mask = xyzt[:, 2] > 0.0
            points = xyzt[z_mask, :]
            missing_points_sweeps.append(points)

        missing_points_sweeps = [
            fast_uncompensate_pointwise(pc, sliced_oxts, lidar_extrinsics, t)
            for pc, t in zip(missing_points_sweeps, sweeps_timestamps)
        ]

        # Set time relative to current frame
        for sweep, sweep_timestamp in zip(missing_points_sweeps, sweeps_timestamps):
            sweep[..., -1] += sweep_timestamp - current_timestamp

        if self.config.missing_points_downsample_factor > 0.0:
            subsampled_missing_sweeps = []
            for sweep in missing_points_sweeps:
                selected_indeces = np.random.choice(
                    range(sweep.shape[0]),
                    size=int(sweep.shape[0] * self.config.missing_points_downsample_factor),
                    replace=False,
                )
                subsampled_missing_sweeps.append(sweep[selected_indeces])
            missing_points_sweeps = subsampled_missing_sweeps

        return missing_points_sweeps

    def _keyframe_camera_indices(self, scene_name: str, camera: str) -> np.ndarray:
        """Indices into cameras_per_scene[scene][camera] that are KEYFRAME images (samples/, not
        sweeps/). With only_tokens_for_offsets=True the pool is keyframes-only and this is
        arange(n); with 20 Hz retention it filters once per (scene, camera) and caches (per
        dataloader worker)."""
        cache = getattr(self, "_kf_cam_idx_cache", None)
        if cache is None:
            cache = self._kf_cam_idx_cache = {}
        key = (scene_name, camera)
        if key not in cache:
            files = self.cameras_per_scene[scene_name][camera]
            idx = np.array([i for i, f in enumerate(files) if "sweeps" not in str(f)], dtype=np.int64)
            # A scene with no keyframe camera entries would be an infos bug; fall back to the full
            # pool rather than crash (matches the pre-filter behavior).
            cache[key] = idx if len(idx) else np.arange(len(files), dtype=np.int64)
        return cache[key]

    def _load_images(
        self, scene_name: str, current_idx: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load the configured images for the current timestep and the previous ones.

        Returns
            a dict containing one numpy array per loaded camera where each array has shape (t, h, w, c).
                note that the newest image is first in the array (t, t-1, t-2, ...)
            a dict containinig the timestamps of the images in seconds.
        """

        images = {}
        times = {}
        filepaths = {}
        # Sample token of the current keyframe per camera: pseudo labels are stored per
        # (scene, token, camera), so the depth reader needs it rather than the image path.
        tokens = {}
        current_timestamp = self._get_timestamp(scene_name, current_idx)
        current_timestamp_ms = int(current_timestamp * 1e6)

        prev_max_time = self.config.prev_images_max_time
        n_prev = self.config.n_prev_images
        # If any of the two are 0, we just load the current image
        if prev_max_time <= 0 or n_prev <= 0:
            target_prev_timestamps = np.array([])
        else:
            target_prev_timestamps = np.linspace(current_timestamp - prev_max_time, current_timestamp, n_prev + 1)[:-1]

        future_max_time = self.config.future_images_max_time
        n_future = self.config.n_future_images
        if future_max_time <= 0 or n_future <= 0:
            target_future_timestamps = np.array([])
        else:
            target_future_timestamps = np.linspace(
                current_timestamp, current_timestamp + future_max_time, n_future + 1
            )[1:]

        target_timestamps = np.concatenate([target_prev_timestamps, target_future_timestamps])
        target_timestamps_ms = np.round(target_timestamps * 1e6).astype(np.int64)
        n_target_timestamps = target_timestamps_ms.shape[0]

        for camera in self.config.nuscenes_cameras_to_load:
            available_timestamps_ms = np.array(self.camera_timestamps_per_scene[scene_name][camera])
            # The CURRENT frame must be the keyframe image: with 20 Hz camera entries in the infos
            # (only_tokens_for_offsets=False), the time-nearest file is often a sweeps/ image —
            # which has no Metric3D depth (loses depth supervision; the int8 placeholder also broke
            # batch collation, 6797355) and pairs a sweep exposure with keyframe calibration.
            # Restricting the pick to samples/ entries keeps camera input IDENTICAL to
            # keyframe-offset configs; the prev/future target picks below keep the full pool.
            kf_indices = self._keyframe_camera_indices(scene_name, camera)
            current_camera_idx = kf_indices[
                np.argmin(np.abs(available_timestamps_ms[kf_indices] - current_timestamp_ms))
            ]
            closest_image = self.config.data / self.cameras_per_scene[scene_name][camera][current_camera_idx]
            closest_timestamp_ms = available_timestamps_ms[current_camera_idx]
            tokens[camera] = self.camera_tokens_per_scene[scene_name][camera][current_camera_idx]
            # load the closest image
            image = read_img(closest_image)  # RGB, HWC, uint8
            filepaths[camera] = [closest_image]
            images[camera] = np.expand_dims(image, axis=0)  # n-times x h x w x c
            times[camera] = np.array([closest_timestamp_ms / 1e6], dtype=np.float64)  # s shape: (n-times,)

            if n_target_timestamps > 0:
                # get the closest timestamps to the target_prev_timestamps
                delta_timestamps = np.abs(available_timestamps_ms[None, :] - target_timestamps_ms[:, None])
                min_delta_indices = np.argmin(delta_timestamps, axis=1)
                # load the images, the newest image first, hence reverse
                for ind in min_delta_indices[::-1]:
                    img_file = self.config.data / self.cameras_per_scene[scene_name][camera][ind]
                    image = read_img(img_file)  # RGB, HWC, uint8
                    # put the most newest image first
                    images[camera] = np.concatenate((images[camera], np.expand_dims(image, axis=0)), axis=0)
                    times[camera] = np.append(times[camera], available_timestamps_ms[ind] / 1e6)
                    filepaths[camera].append(img_file)
            # convert to (t, c, h, w)
            images[camera] = np.transpose(images[camera], (0, 3, 1, 2))  # (t, c, h, w)

        delta_times = {k: (v - current_timestamp).astype(np.float32) for k, v in times.items()}

        return images, times, delta_times, filepaths, tokens

    def _resize_images(
        self, images: Dict[str, np.ndarray], calibration: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        if self.config.image_resize_method == "none":
            for cam_name, calib in calibration.items():
                calib["intrinsics_modified"] = calib["intrinsics"].copy()
            return images, calibration

        images, calibration = self._crop_images_and_intrinsics(images, calibration)

        if self.config.image_resize_method == "resize":
            images, calibration = self._resize_images_and_intrinsics(images, calibration)

        elif self.config.image_resize_method == "downsample":
            images, calibration = self._downsample_images_and_intrinsics(images, calibration)
            if self.config.pad_images_to_same_size:
                images, calibration = self._pad_images_to_same_size(images, calibration)
            if self.config.pad_images_to_multiple_of > 0:
                images, calibration = self._pad_images_to_multiple_of(images, calibration)
        else:
            raise ValueError(f"Invalid image resize method: {self.config.image_resize_method}")

        return images, calibration

    def _get_camera_calibration(self, scene_name: str) -> Dict[str, Dict[str, np.ndarray]]:
        calib: Dict[str, Dict[str, np.ndarray]] = {}

        for cam_name in self.config.nuscenes_cameras_to_load:
            # we have the inverse here as that is how we've defined it in other datasets. maybe look into this?
            extrinsics = np.linalg.inv(self.camera_calib_per_scene[scene_name][cam_name]["extrinsics"])  # 4x4
            intrinsics = np.array(self.camera_calib_per_scene[scene_name][cam_name]["intrinsics"])  # 3x3

            width = int(intrinsics[0, 2] * 2)
            height = int(intrinsics[1, 2] * 2)

            original_image_size = np.array([width, height])

            field_of_view = np.array(
                2 * np.arctan(0.5 * intrinsics[0, 2] * 2 / intrinsics[0, 0]),
                2 * np.arctan(0.5 * intrinsics[1, 2] * 2 / intrinsics[1, 1]),
            )

            calib[cam_name] = {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "distortion": np.zeros(4),  # cameras are undistorted
                "original_camera_intrinsics": intrinsics.copy(),
                "original_camera_extrinsics": extrinsics.copy(),
                "original_camera_size": original_image_size.copy(),
                "T_cam_ego": extrinsics.copy(),
                "field_of_view": field_of_view,
            }

        return calib

    def _crop_images_and_intrinsics(
        self, images: Dict[str, np.ndarray], calibration: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """images are (t, h, w, c)"""
        for cam_name, img in images.items():
            img = crop_images(img, NUSCENES_CAM_CROP[cam_name])
            intrinsics_modified = calibration[cam_name]["intrinsics"].copy()
            intrinsics_modified = crop_intrinsics(intrinsics_modified, NUSCENES_CAM_CROP[cam_name])
            images[cam_name] = img
            calibration[cam_name]["intrinsics_modified"] = intrinsics_modified

        return images, calibration

    def _downsample_images_and_intrinsics(
        self, images: Dict[str, np.ndarray], calibration: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        for cam_name, img in images.items():
            img = resize_images_with_scale(img, 1 / self.config.nuscenes_camera_downsample_factor)
            images[cam_name] = img
            calibration[cam_name]["intrinsics_modified"] = resize_intrinsics_with_scale(
                calibration[cam_name]["intrinsics_modified"], 1 / self.config.nuscenes_camera_downsample_factor
            )
        return images, calibration

    def _resize_images_and_intrinsics(
        self, images: Dict[str, np.ndarray], calibration: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        assert self.config.image_size is not None, "Image size must be set for resize"
        new_h, new_w = self.config.image_size
        for cam_name, img in images.items():
            old_h, old_w = img.shape[-2:]
            img = img.transpose(0, 2, 3, 1)
            img = [cv2.resize(i, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for i in img]
            img = np.stack(img, axis=0)
            img = img.transpose(0, 3, 1, 2)

            images[cam_name] = img
            calibration[cam_name]["intrinsics_modified"] = resize_intrinsics(
                calibration[cam_name]["intrinsics_modified"], (new_h, new_w), (old_w, old_h)
            )
        return images, calibration

    def _pad_images_to_same_size(self, images, calibration):
        max_h = max(v.shape[2] for v in images.values())
        max_w = max(v.shape[3] for v in images.values())
        for cam_name, img in images.items():
            h, w = img.shape[2], img.shape[3]
            pad_h = max_h - h
            pad_w = max_w - w
            pad_h_top = pad_h // 2
            pad_h_bottom = pad_h - pad_h_top
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
            images[cam_name] = np.pad(
                img,
                ((0, 0), (0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                mode="constant",
                constant_values=0,
            )
            calibration[cam_name]["intrinsics_modified"][0, 2] += pad_w_left
            calibration[cam_name]["intrinsics_modified"][1, 2] += pad_h_top

        return images, calibration

    def _pad_images_to_multiple_of(self, images, calibration):
        for cam_name, img in images.items():
            h, w = img.shape[2], img.shape[3]
            pad_h = (
                self.config.pad_images_to_multiple_of - h % self.config.pad_images_to_multiple_of
            ) % self.config.pad_images_to_multiple_of
            pad_w = (
                self.config.pad_images_to_multiple_of - w % self.config.pad_images_to_multiple_of
            ) % self.config.pad_images_to_multiple_of
            pad_h_top = pad_h // 2
            pad_h_bottom = pad_h - pad_h_top
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
            images[cam_name] = np.pad(
                img,
                ((0, 0), (0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                mode="constant",
                constant_values=0,
            )
            calibration[cam_name]["intrinsics_modified"][0, 2] += pad_w_left
            calibration[cam_name]["intrinsics_modified"][1, 2] += pad_h_top

        return images, calibration

    def _motion_compensate_extrinsics(
        self,
        calibration: Dict[str, Dict[str, np.ndarray]],
        ego_motion: EgoMotion,
        image_times: Dict[str, np.ndarray],
        bev_oxts_pose: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        for cam_name, calib in calibration.items():
            p_then = ego_motion.get_poses(image_times[cam_name])
            T_then_nows = inverse_np(p_then) @ bev_oxts_pose  # t x 4 x 4
            extrinsics = calib["extrinsics"].copy()  # 4x4
            t_cam_ego = np.expand_dims(extrinsics, axis=0) @ T_then_nows  # t x 4 x 4
            calibration[cam_name]["T_cam_ego"] = t_cam_ego

        return calibration

    def _get_occ3d_data(self, scene_name: str, scene_idx: int) -> Dict[str, Any]:
        occ3d_data = {}
        if self.sweeps_per_scene[scene_name][scene_idx]["sample_data_token"] in self.sample_data_token_to_sample_token:
            sample_token = self.sample_data_token_to_sample_token[
                self.sweeps_per_scene[scene_name][scene_idx]["sample_data_token"]
            ]

            occ3d_data["sample_token"] = sample_token

            if self.load_occ3d:
                real_scene_name = self.scene_token_to_real_scene_name[scene_name]
                occ3d_scene_samples_path = self.config.occ3d_data / real_scene_name / sample_token / "labels.npz"
                occ3d_info = np.load(occ3d_scene_samples_path)
                occ3d_data["occ3d_labels"] = self.label_mapper.nusc_to_learning(occ3d_info["semantics"])
                occ3d_data["occ3d_mask_lidar"] = occ3d_info["mask_lidar"]
                occ3d_data["occ3d_mask_camera"] = occ3d_info["mask_camera"]
        else:
            occ3d_data["sample_token"] = "n/a"
            if self.load_occ3d:
                occ3d_data["occ3d_labels"] = np.zeros((200, 200, 16), dtype=np.uint8)
                occ3d_data["occ3d_mask_lidar"] = np.zeros((200, 200, 16), dtype=np.uint8)
                occ3d_data["occ3d_mask_camera"] = np.zeros((200, 200, 16), dtype=np.uint8)

        return occ3d_data

    def __getitem__(self, idx: int):
        scene_name, scene_idx = self.dataset_idx_to_scene_and_idx[idx]
        current_timestamp = self._get_timestamp(scene_name, scene_idx)
        data: Dict[str, Any] = {
            "id": scene_name,
            "frame_idx": scene_idx,
        }

        data.update(self._get_occ3d_data(scene_name, scene_idx))

        ego_motion = self._get_ego_motion(scene_name)

        data["timestamp"] = current_timestamp

        ego_motion, bev_oxts_pose, aug_angle, aug_translation, _ = self._rotation_augmentation(
            ego_motion, current_timestamp
        )

        data["augmentation_angle"] = aug_angle
        data["augmentation_translation"] = aug_translation
        # Ego->world pose of the (augmented) BEV frame: everything per-frame is expressed relative
        # to this, so it lifts predictions/lidar/boxes to world (vizkit aggregation, chase cameras).
        data["bev_oxts_pose"] = bev_oxts_pose.astype(np.float64)

        lidar_extrinsics = np.eye(4)
        lidar_extrinsics[:3, -1] = self.sweeps_per_scene[scene_name][0]["sensor2ego_translation"]
        lidar_extrinsics[:3, :3] = Quaternion(
            self.sweeps_per_scene[scene_name][0]["sensor2ego_rotation"]
        ).rotation_matrix

        if self.config.load_lidar:
            # get the lidar frames
            if self.config.aggregate_lidar:
                data.update(
                    self._aggregate_lidar_frames(
                        scene_name, scene_idx, ego_motion, target_pose=bev_oxts_pose, lidar_extrinsics=lidar_extrinsics
                    )
                )
            else:
                raise NotImplementedError(
                    "Non-aggregated lidar loading for nuScenes is not implemented. Set config.aggregate_lidar=True."
                )

        if self.config.load_image:
            # get the images for each camera
            images, image_times, delta_times, _, image_tokens = self._load_images(scene_name, scene_idx)
            calibration = self._get_camera_calibration(scene_name)

            images, calibration = self._resize_images(images, calibration)

            # # update extrinsics based on odometry
            calibration = self._motion_compensate_extrinsics(calibration, ego_motion, image_times, bev_oxts_pose)
            ni = images[list(images.keys())[0]].shape[0]

            # make it imgs -> cam-name -> tensor, and similar for the rest
            data.update(
                {
                    # uint8; converted to float in [0, 1] on the GPU (BaseModel._get_camera_data)
                    "imgs": images,
                    "imgs_timestamps": delta_times,
                    "imgs_num_timesteps": ni,
                    "imgs_h": {k: v.shape[2] for k, v in images.items()},
                    "imgs_w": {k: v.shape[3] for k, v in images.items()},
                    "extrinsics": {k: np.array([v["extrinsics"]] * ni) for k, v in calibration.items()},
                    "intrinsics": {k: np.array([v["intrinsics"]] * ni) for k, v in calibration.items()},
                    "intrinsics_modified": {
                        k: np.array([v["intrinsics_modified"]] * ni) for k, v in calibration.items()
                    },
                    "distortion": {k: np.array([v["distortion"]] * ni) for k, v in calibration.items()},
                    "T_cam_ego": {k: v["T_cam_ego"] for k, v in calibration.items()},
                    "original_camera_intrinsics": {k: v["original_camera_intrinsics"] for k, v in calibration.items()},
                    "original_camera_extrinsics": {k: v["original_camera_extrinsics"] for k, v in calibration.items()},
                    "original_camera_size": {k: v["original_camera_size"] for k, v in calibration.items()},
                    "field_of_view": {k: v["field_of_view"] for k, v in calibration.items()},
                }
            )

            if self.config.load_depth_maps:
                if self.config.depth_maps_source == "metric3d":
                    depth_maps, valid_depth_maps = self._get_depth_maps_metric3d(scene_name, image_tokens)
                elif self.config.depth_maps_source == "lidar":
                    depth_maps, valid_depth_maps = self._get_depth_maps(
                        data, lidar_extrinsics, scene_name, scene_idx, current_timestamp
                    )
                data.update(
                    {
                        "depth_maps": depth_maps,
                        "depth_maps_valid": valid_depth_maps,
                    }
                )

        offset_lidar_timestamps = None
        if self.config.load_offset_pointclouds:
            offset_lidar_data, offset_lidar_timestamps = self._get_offset_pointclouds(
                scene_name, scene_idx, bev_oxts_pose, ego_motion, lidar_extrinsics
            )
            data.update(offset_lidar_data)

        if self.config.load_vfm_stuff:
            if offset_lidar_timestamps is None:
                future_lidar_sweeps = self._get_offset_lidar_frames(scene_name, scene_idx)
                offset_lidar_timestamps = [sweep["timestamp"] / 1e6 for sweep in future_lidar_sweeps]

            vfm_data = self.vfm_reader.read(
                cameras=self.cameras_per_scene[scene_name],
                cameras_timestamps=self.camera_timestamps_per_scene[scene_name],
                cameras_calibrations=self.camera_calib_per_scene[scene_name],
                cameras_tokens=self.camera_tokens_per_scene[scene_name],
                scene_name=self.scene_token_to_real_scene_name[scene_name],
                current_timestamp=current_timestamp,
                lidar_timestamps=offset_lidar_timestamps,
                ego_motion=ego_motion,
                bev_oxts_pose=bev_oxts_pose,
                is_train=self.is_train,
            )
            data.update(vfm_data)

        # prepare data for computing rayiou
        if self.config.load_4d_lidar_origins and self.load_occ3d:
            scene_info = self.sweeps_per_scene[scene_name]
            sweep_info = scene_info[scene_idx]

            ref_lidar_from_global = get_global_pose(sweep_info, inverse=True)
            ref_ego_from_lidar = get_ego_from_lidar(sweep_info)
            ref_index = scene_idx  # current (lidar) frame idx

            # NOTE: getting output frames
            output_origin_list = []
            for curr_index in range(len(scene_info)):
                # if this exists a valid target
                if curr_index == ref_index:
                    origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    # transform from the current lidar frame to global and then to the reference lidar frame
                    global_from_curr = get_global_pose(scene_info[curr_index], inverse=False)
                    ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                    origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

                origin_tf_pad = np.ones([4])
                origin_tf_pad[:3] = origin_tf  # pad to [4]
                origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T  # [3]

                # origin. ensure lidar positions at other times are within the bev map at the current time
                if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                    output_origin_list.append(origin_tf)

            # select 8 origins
            if len(output_origin_list) > 8:
                select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
                output_origin_list = [output_origin_list[i] for i in select_idx]

            data.update(
                {
                    "lidar_origins": np.array(output_origin_list),
                }
            )

        if self.config.load_current_lidar_panoptic_labels:
            panoptic_lidar_labels = self._get_current_panoptic_lidar_labels(
                scene_name, scene_idx, ego_motion, lidar_extrinsics, bev_oxts_pose, current_timestamp
            )
            data.update(panoptic_lidar_labels)

        def _to_tensor(data: Union[np.ndarray, list, str]) -> Union[str, torch.Tensor]:
            if isinstance(data, (str, int, float, dict)):
                return data
            data = np.stack(data, axis=0) if isinstance(data, list) else data
            return torch.from_numpy(data)

        return {k: _to_tensor(v) for k, v in data.items()}

    def _get_current_panoptic_lidar_labels(
        self,
        scene_name: str,
        scene_idx: int,
        ego_motion: EgoMotion,
        lidar_extrinsics: np.ndarray,
        bev_oxts_pose: np.ndarray,
        current_timestamp: float,
    ) -> Dict[str, np.ndarray]:
        lidar_sweep = self.sweeps_per_scene[scene_name][scene_idx]
        lidar_timestamp = lidar_sweep["timestamp"] / 1e6
        lidar_pc, lidar_mask = self._load_pc(lidar_sweep["data_path"], lidar_timestamp, get_mask=True)
        panoptic_lidar_labels = self._get_panoptic_labels([lidar_pc], [lidar_sweep], [lidar_mask])
        lidar_pc = fast_uncompensate_pointwise(lidar_pc, ego_motion, lidar_extrinsics, lidar_timestamp)
        in_range_mask, xyz = points_in_target_bev(
            lidar_pc,
            ego_motion,
            self.config.point_cloud_range,
            bev_oxts_pose,
            lidar_extrinsics,
            lidar_timestamp,
            output_xyz=True,
        )

        for key in panoptic_lidar_labels.keys():
            panoptic_lidar_labels[key] = panoptic_lidar_labels[key][0][in_range_mask]

        lidar_pcs_times = (lidar_pc[in_range_mask, -1] - current_timestamp).reshape(-1, 1)
        panoptic_lidar_queries = np.concatenate([xyz, lidar_pcs_times], axis=-1).astype(np.float32)

        return {"panoptic_lidar_labels": panoptic_lidar_labels, "panoptic_lidar_queries": panoptic_lidar_queries}

    def _reshape_depth_map(self, depth_map: np.ndarray, cam_name: str) -> np.ndarray:
        if self.config.image_resize_method == "none":
            return depth_map

        depth_map = crop_images(depth_map, NUSCENES_CAM_CROP[cam_name])

        if self.config.image_resize_method == "resize":
            new_h, new_w = self.config.image_size
            depth_map = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST_EXACT)

        elif self.config.image_resize_method == "downsample":
            scale = float(1.0 / self.config.nuscenes_camera_downsample_factor)
            depth_map = cv2.resize(depth_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST_EXACT)
            if self.config.pad_images_to_same_size:
                raise NotImplementedError(
                    "pad_images_to_same_size with downsample resize is not implemented for nuScenes."
                )
            if self.config.pad_images_to_multiple_of > 0:
                h, w = depth_map.shape[0], depth_map.shape[1]
                pad_h = (
                    self.config.pad_images_to_multiple_of - h % self.config.pad_images_to_multiple_of
                ) % self.config.pad_images_to_multiple_of
                pad_w = (
                    self.config.pad_images_to_multiple_of - w % self.config.pad_images_to_multiple_of
                ) % self.config.pad_images_to_multiple_of
                pad_h_top = pad_h // 2
                pad_h_bottom = pad_h - pad_h_top
                pad_w_left = pad_w // 2
                pad_w_right = pad_w - pad_w_left
                depth_map = np.pad(
                    depth_map,
                    ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                    mode="constant",
                    constant_values=-1,
                )
        else:
            raise ValueError(f"Invalid image resize method: {self.config.image_resize_method}")

        scale = float(1.0 / self.config.depth_downsample_factor)
        depth_map = cv2.resize(depth_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST_EXACT)

        return depth_map

    def _get_depth_maps_metric3d(
        self, scene_name: str, image_tokens: Dict[str, Optional[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Metric3D depth for the current keyframe of each camera, from the pseudo-label npz.

        Frames without a stored depth map (non-keyframes, or samples the generator skipped)
        get the -1 placeholder, which `valid_depth_maps` masks out — the same convention the
        old per-image reader used, so the depth loss ignores them.
        """
        real_scene_name = self.scene_token_to_real_scene_name[scene_name]
        depth_maps = {}
        valid_depth_maps = {}
        for camera, token in image_tokens.items():
            depth_map = self.metric3d_depth_reader.read_camera(real_scene_name, token, camera)
            if depth_map is None:
                depth_map = np.full(DEFAULT_IMAGE_SHAPE, -1.0, dtype=np.float32)
            depth_map = self._reshape_depth_map(depth_map, camera)
            depth_maps[camera] = depth_map
            valid_depth_maps[camera] = depth_map != -1
        return depth_maps, valid_depth_maps

    def _get_depth_maps(
        self,
        data: Dict[str, Any],
        lidar_extrinsics: np.ndarray,
        scene_name: str,
        scene_idx: int,
        current_timestamp: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ego_motion = self._get_ego_motion(scene_name)
        current_pose = ego_motion.get_poses(current_timestamp).copy()

        images = data["imgs"]
        num_cams = len(images)

        img_h = int(list(data["imgs_h"].values())[0] / self.config.depth_downsample_factor)
        img_w = int(list(data["imgs_w"].values())[0] / self.config.depth_downsample_factor)

        lidar_filepath = self.sweeps_per_scene[scene_name][scene_idx]["data_path"]

        points = self._load_pc(lidar_filepath, self.sweeps_per_scene[scene_name][scene_idx]["timestamp"] / 1e6)

        points = ego_motion_compensate_scanwise(
            points, ego_motion, target_pose=current_pose, lidar_extrinsics=lidar_extrinsics
        )

        # Only get the xyz coordinates
        points = points[:, :3]
        num_points = points.shape[0]

        # Create intrinsics for each camera
        intrinsics_t0 = [intr[0] for intr in data["intrinsics_modified"].values()]
        intr = np.stack(intrinsics_t0, axis=0)
        # Downsample intrinsics
        intr[:, 0, 0] /= self.config.depth_downsample_factor
        intr[:, 1, 1] /= self.config.depth_downsample_factor
        intr[:, 0, 2] /= self.config.depth_downsample_factor
        intr[:, 1, 2] /= self.config.depth_downsample_factor

        # Create homogeneous intrinsics
        intr_homo = np.eye(4)  # 4x4
        intr_homo = np.repeat(intr_homo[None, :, :], num_cams, axis=0)
        intr_homo[:, :3, :3] = intr[:, :3, :3]
        intr_homo = np.repeat(intr_homo[:, None, :, :], num_points, axis=1)

        # Create extrinsics for each camera
        extrinsics_t0 = [extr[0] for extr in data["extrinsics"].values()]
        extr = np.stack(extrinsics_t0, axis=0)
        extr_homo = np.repeat(extr[:, None, :, :], num_points, axis=1)

        # Prepare the points
        points_homo = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        points_homo = np.repeat(points_homo[None, :, :], num_cams, axis=0)
        points_homo = np.expand_dims(points_homo, axis=-1)

        # Transform to image coordinates
        points_cam = extr_homo @ points_homo
        points_img = intr_homo @ points_cam
        points_img = points_img.squeeze(-1)

        # Normalize
        eps = 1e-9
        points_img[..., 0:2] = points_img[..., 0:2] / (points_img[..., 2:3] + eps)

        # Mask out relevant points
        mask_h = (points_img[..., 1] < img_h) & (points_img[..., 1] > 0)
        mask_w = (points_img[..., 0] < img_w) & (points_img[..., 0] > 0)
        mask_d = points_img[..., 2] > eps
        mask = mask_h & mask_w & mask_d

        # Create indeces for scattering
        flat_image_index, _ = torch.where(torch.from_numpy(mask))
        flat_points = torch.from_numpy(points_img[..., :3])[mask, :]

        flat_xs = flat_points[:, 0].long()
        flat_ys = flat_points[:, 1].long()
        flat_depths = flat_points[:, 2]

        # 1. Initialize Depth Maps Tensor
        # Use float('inf') as initial value because 'amin' finds the minimum.
        # Any value smaller than infinity will replace it.
        # Shape: (num_cams, img_h, img_w)
        depth_maps_tensor = torch.full((num_cams, img_h, img_w), fill_value=float("inf"), dtype=torch.float32)

        # Flatten the tensor for 1D scattering
        # Shape: (num_cams * img_h * img_w)
        flat_depth_maps_tensor = depth_maps_tensor.view(-1)

        # 2. Prepare Indices for Scattering (Same as before)
        # Calculate the flattened index for each point
        # index = cam_idx * (img_h * img_w) + y * img_w + x
        flat_indices = flat_image_index * (img_h * img_w) + flat_ys * img_w + flat_xs

        # Ensure types are correct
        flat_indices = flat_indices.long()
        flat_depths = flat_depths.float()

        # 3. Scatter Minimum Depth using scatter_reduce_
        # Apply the scatter operation in-place
        flat_depth_maps_tensor.scatter_reduce_(
            dim=0,
            index=flat_indices,
            src=flat_depths,
            reduce="amin",
            include_self=False,  # Important: Don't include the initial 'inf' values in the min calculation unless it's the only value
        )

        # Reshape the flat tensor back to (num_cams, img_h, img_w)
        depth_maps_tensor = flat_depth_maps_tensor.view(num_cams, img_h, img_w)

        # 4. Post-processing
        available_depth_mask = depth_maps_tensor != float("inf")
        depth_maps_tensor[~available_depth_mask] = 0.0

        # 5. Store in Dictionary (Optional, if needed)
        cam_keys = list(images.keys())
        depth_maps = {}
        valid_depth_maps = {}
        for i in range(num_cams):
            cam_key = cam_keys[i]
            depth_maps[cam_key] = depth_maps_tensor[i]
            valid_depth_maps[cam_key] = available_depth_mask[i]

        return depth_maps, valid_depth_maps
