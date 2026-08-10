import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import av2
import av2.geometry
import av2.geometry.geometry
import av2.utils
import av2.utils.io
import cv2
import numpy as np
import torch
from av2.datasets.sensor.constants import RingCameras as Cam
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.structures.sweep import Sweep
from torch.utils.data import Dataset
from zod.data_classes.ego_motion import EgoMotion

import queryocc.config.base_config as cfg
from queryocc.common import paths
from queryocc.common.types import PointCloudRange
from queryocc.data.datasets.nuscenes_dataset import _warn_missing_points_cache_absent
from queryocc.data.datasets.nuscenes_mappings import LabelMapper
from queryocc.data.datasets.utils import (
    crop_images,
    crop_intrinsics,
    pad_offset_data_for_batching,
    resize_images_with_scale,
    resize_intrinsics,
    resize_intrinsics_with_scale,
)
from queryocc.data.datasets.zod_base import ZODBaseDatasetConfig
from queryocc.data.vfm.vfm_reader import VFMReader
from queryocc.utils.geometry import fast_ego_motion_poses, inverse_np

AV2_MAX_REFLECTANCE: float = 255.0
AV2_LIDAR_FREQUENCY: int = 10
AV2_IMAGE_FREQUENCY: int = 10
# (top, right, bottom, left)
AV2_CAM_CROP = {
    Cam.RING_FRONT_CENTER.value: (0, 0, 250, 0),
    Cam.RING_FRONT_LEFT.value: (0, 0, 0, 0),
    Cam.RING_FRONT_RIGHT.value: (0, 0, 0, 0),
    Cam.RING_REAR_LEFT.value: (0, 0, 250, 0),
    Cam.RING_REAR_RIGHT.value: (0, 0, 250, 0),
    Cam.RING_SIDE_LEFT.value: (0, 0, 0, 0),
    Cam.RING_SIDE_RIGHT.value: (0, 0, 0, 0),
}

AV2_SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits" / "argoverse2"

warnings.filterwarnings(action="ignore", category=UserWarning)


@dataclass
class AV2DatasetConfig(ZODBaseDatasetConfig, cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: AV2Dataset)
    """The target class for this configuration."""

    data: Path = Path("/datasets/argoverse/")
    """The path to the dataset."""
    subset: Literal["sensor", "lidar"] = "sensor"
    """The subset of the dataset to use."""

    av2_cached_missing_points_dir: Path = paths.AV2_MISSING_POINTS_ROOT

    av2_cameras_to_load: Tuple[str, ...] = (
        Cam.RING_FRONT_CENTER.value,
        Cam.RING_FRONT_LEFT.value,
        Cam.RING_FRONT_RIGHT.value,
        Cam.RING_REAR_LEFT.value,
        Cam.RING_REAR_RIGHT.value,
        Cam.RING_SIDE_LEFT.value,
        Cam.RING_SIDE_RIGHT.value,
    )
    """The cameras to load images from."""
    av2_camera_downsample_factor: float = 4.0

    lidar_sweep_subsample_factor: int = 1
    """The subsample factor for the lidar sweeps. (1 out of every n sweeps)"""


# Coordinate pipeline for AV2:
#   sensor frame (raw Sweep points at capture time, in ego frame via AV2 SDK)
#   → ego frame at bev_oxts_pose  (via _ego_motion_compensate_scanwise using SE3 poses from AV2 SDK)
#   → BEV frame                   (axis-aligned, centered at bev_oxts_pose)
#
# bev_oxts_pose: 4×4 SE(3) matrix, ego vehicle pose in world frame at current timestamp.
# AV2 lidar sweeps are already in city frame; the AV2 SDK converts them to ego frame via Sweep.ego_SE3_lidar.


class AV2Dataset(Dataset):
    _BLACKLIST: List[str] = [
        "75e8adad-50a6-3245-8726-5e612db3d165",
        "54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca",
        "af170aac-8465-3d7b-82c5-64147e94af7d",
        "6e106cf8-f6dd-38f6-89c8-9be7a71e7275",
    ]

    def __init__(
        self,
        config: AV2DatasetConfig,
        train_or_val: Literal["train", "val"],
        split: Literal[
            "original",
            "geo_near",
            "geo_near_050",
            "geo_near_100",
            "geo_near_500",
            "geo_far_A",
            "geo_far_B",
            "geo_far_C",
        ],
    ):
        super().__init__()
        assert config.point_cloud_range != (0, 0, 0, 0, 0, 0), "Model did not set point_cloud_range."

        if config.subset == "lidar":
            assert not config.load_image, "Cannot load images in lidar subset."

        self.config = config
        self.train_or_val = train_or_val
        self.split = split

        # get all sequences for current split
        all_train = sorted([d for d in (config.data / config.subset / "train").iterdir() if d.is_dir()])
        all_val = sorted([d for d in (config.data / config.subset / "val").iterdir() if d.is_dir()])

        # read splits
        split_name = train_or_val

        with open(AV2_SPLITS_DIR / self.split / f"{split_name}.txt", "r") as f:
            sequences_in_split = [line.strip() for line in f]

        # Get the sequences from the train or val folder
        sequences = [seq for seq in all_train if seq.stem in sequences_in_split]
        sequences += [seq for seq in all_val if seq.stem in sequences_in_split]
        # remove blacklisted sequences
        sequences = [seq for seq in sequences if seq.stem not in self._BLACKLIST]
        # only use specified ids if provided
        if len(config.ids_to_load) > 0:
            sequences = [seq for seq in sequences if seq.stem in config.ids_to_load]
            print(f"Using {len(sequences)} sequences from {split_name}")
        else:
            assert len(sequences) > 0, f"No sequences found in {config.data / config.subset / split_name}"

        if self.config.version == "mini":
            sequences = sequences[:10]

        # we want to roll out the sequences to individual samples
        self.dataset_idx_to_seq_and_frame: List[Tuple[str, str]] = []
        # ensure that we have enough frames to predict all the way forward during validation
        stop_frames_before = max(1, int(self.config.max_future_frame_s * AV2_LIDAR_FREQUENCY))

        # check how many previous frames we need to be able to load
        n_prevs = []
        if config.load_image:
            n_prevs.append(config.prev_images_max_time * AV2_IMAGE_FREQUENCY)
        if config.load_lidar:
            n_prevs.append(config.prev_lidar_max_time * AV2_LIDAR_FREQUENCY)
        if len(n_prevs) == 0:
            n_prevs.append(config.prev_lidar_max_time * AV2_LIDAR_FREQUENCY)

        n_prev = int(max([2] + n_prevs))

        for seq in sequences:
            lidar_files = sorted((seq / "sensors" / "lidar").iterdir())
            samples_to_select = len(lidar_files) - n_prev - stop_frames_before
            assert samples_to_select > 0, (
                f"No lidar files found in {config.data / config.subset / train_or_val / seq / 'sensors' / 'lidar'}"
            )
            self.dataset_idx_to_seq_and_frame += [
                (str(seq), str(lidar_files[i + n_prev]))
                for i in range(0, samples_to_select, config.lidar_sweep_subsample_factor)
            ]

        if len(config.frames_to_load) > 0:
            self.dataset_idx_to_seq_and_frame = [
                (seq, frame)
                for seq, frame in self.dataset_idx_to_seq_and_frame
                if Path(frame).stem in config.frames_to_load
            ]

        if config.load_vfm_stuff:
            self.vfm_reader: VFMReader = config.vfm_cfg.setup(
                dataset_name="av2",
                data_root=config.data,
                point_cloud_range=config.point_cloud_range,
            )

        self.label_mapper = LabelMapper(config.panoptic_labels_classes)

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
        return AV2_LIDAR_FREQUENCY

    @property
    def image_frequency(self) -> float:
        return AV2_IMAGE_FREQUENCY

    @property
    def max_prev_images_frames(self) -> int:
        return int(np.ceil(self.config.prev_images_max_time * self.image_frequency))

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
        return len(self.dataset_idx_to_seq_and_frame)

    def _get_ego_motion(self, seq_root: Path) -> EgoMotion:
        df = av2.utils.io.read_feather(
            seq_root / "city_SE3_egovehicle.feather",
        )
        quats = df[["qw", "qx", "qy", "qz"]].to_numpy()
        trans = df[["tx_m", "ty_m", "tz_m"]].to_numpy()
        rot = av2.geometry.geometry.quat_to_mat(quats)
        poses = np.zeros((len(quats), 4, 4))
        poses[:, :3, :3] = rot
        poses[:, :3, 3] = trans
        poses[:, 3, 3] = 1

        # make all poses relative first pose
        origin = poses[0].copy()
        poses = inverse_np(origin) @ poses
        poses[0] = np.eye(4)

        # interpolate speeds from poses
        d_s = trans[1:] - trans[:-1]
        timestamps = df["timestamp_ns"].to_numpy().astype(np.float64) / 1e9
        d_t = timestamps[1:] - timestamps[:-1]
        d_t[d_t < 1e-9] = 1e-9
        velocities = d_s / d_t[:, np.newaxis].repeat(3, axis=-1)
        velocities = np.vstack(
            [velocities, np.zeros((1, 3))]
        )  # assume constant velocity between final pose and proceeding

        return EgoMotion(
            poses=poses,
            timestamps=timestamps,
            velocities=velocities,  # we dont have velocities, so we interpolate from poses
            accelerations=np.zeros((len(quats), 3)),  # we dont have accelerations
            angular_rates=np.zeros((len(quats), 3)),  # we dont have angular rates
            origin_lat_lon=origin,  # This is used for extracting map data
        )

    def _get_pseudo_panoptic_labels(self, lidar_filepath: str, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        sequence = lidar_filepath.parent.parent.parent.name
        frame = lidar_filepath.stem
        filename = f"{sequence}/{frame}_pseudoseg.bin"
        pseudoseg_fpath = str(paths.AV2_PSEUDOSEG_ROOT / filename)
        assert "delta" in self.config.panoptic_pseudo_label_name, "Pseudo label name must be delta for now"
        if os.path.exists(pseudoseg_fpath):
            pseudoseg = np.fromfile(pseudoseg_fpath, dtype=np.uint8).reshape(-1, 1)
            pseudoseg = self.label_mapper.nusc_to_learning(pseudoseg)
            valid = np.ones((num_points, 1), dtype=np.bool_)
            invalid = pseudoseg == self.label_mapper.free_idx
            valid[invalid] = False
        else:
            print(f"Pseudoseg file not found for {filename}")
            pseudoseg = np.ones((num_points, 1), dtype=np.int32) * -1
            valid = np.zeros((num_points, 1), dtype=np.bool_)
        panoptic = np.zeros((num_points, 1), dtype=np.int32)
        return pseudoseg, panoptic, valid

    def _get_panoptic_labels(self, lidar_sweeps: List[Sweep], lidar_filepaths: List[Path]) -> List[np.ndarray]:
        lidarseg_labels = []
        panoptic_labels = []
        valid_labels = []
        for lidar_sweep, lidar_filepath in zip(lidar_sweeps, lidar_filepaths):
            num_points = lidar_sweep.xyz.shape[0]

            if self.config.use_pseudo_labels_for_panoptic and self.is_train:
                lidarseg, panoptic, valid = self._get_pseudo_panoptic_labels(lidar_filepath, num_points)
            else:
                lidarseg = np.zeros((num_points, 1), dtype=np.uint8)
                panoptic = np.zeros((num_points, 1), dtype=np.int32)
                valid = np.zeros((num_points, 1), dtype=np.bool_)
            lidarseg_labels.append(lidarseg)
            panoptic_labels.append(panoptic)
            valid_labels.append(valid)

        return {
            "panoptic_lidarseg": lidarseg_labels,
            "panoptic_cluster": panoptic_labels,
            "panoptic_valid": valid_labels,
        }

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
                angle = 0
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

    def _aggregate_lidar_frames(
        self, root_folder: Path, current_nstimestamp_str: str, ego_motion: EgoMotion, target_pose: np.ndarray
    ) -> Dict[str, Any]:
        lidar_nstimestamps = sorted(f.stem for f in (root_folder / "sensors" / "lidar").iterdir() if f.is_file())
        # find the index of the current lidar frame
        current_idx = lidar_nstimestamps.index(current_nstimestamp_str)
        current_nstimestamp = int(current_nstimestamp_str)
        # get the previous lidar frames
        # used indices

        n_prev_max = self.max_prev_lidar_frames
        n_prev_lidar = self.config.n_prev_lidar

        if n_prev_max <= 0 or n_prev_lidar <= 0:
            used_indices: List[int] = []
        else:
            # Generate uniformly spaced indices from max previous time to current
            indices = np.linspace(current_idx - n_prev_max, current_idx, n_prev_lidar + 1)[:-1]
            used_indices = np.floor(indices).astype(np.int32).tolist()

        n_future_max = self.max_future_lidar_frames
        n_future_lidar = self.config.n_future_lidar

        if n_future_max <= 0 or n_future_lidar <= 0:
            future_indices: List[int] = []
        else:
            future_indices = np.linspace(current_idx, current_idx + n_future_max, n_future_lidar + 1)[1:]
            future_indices = np.floor(future_indices).astype(np.int32).tolist()

        used_indices = [current_idx] + used_indices + future_indices
        used_indices = list(set(used_indices))
        used_indices.sort()

        lidar_nstimestamps = [lidar_nstimestamps[i] for i in used_indices]
        sweeps = [
            Sweep.from_feather(root_folder / "sensors" / "lidar" / f"{frame}.feather") for frame in lidar_nstimestamps
        ]
        aggregated = np.empty((0, 5), dtype=np.float32)
        for sweep in sweeps:
            comp_sweep = _ego_motion_compensate_scanwise(sweep, ego_motion, target_pose=target_pose)
            times = (comp_sweep.offset_ns + comp_sweep.timestamp_ns - current_nstimestamp) / 1e9  # seconds
            intensity = comp_sweep.intensity / AV2_MAX_REFLECTANCE
            points = np.concatenate([comp_sweep.xyz, intensity[:, None], times[:, None]], axis=1)
            aggregated = np.concatenate([aggregated, points], axis=0)

        aggregated = [aggregated.astype(np.float32)]
        # # remove points outside the range if specified,
        if self.config.point_cloud_range is not None:
            aggregated = self._crop_pointclouds(aggregated)

        return {
            "lidars": aggregated,
            "lidar_timestamps": [int(current_nstimestamp / 1e3)],  # ms
        }

    def _select_offset_frames(self, all_lidar_frames: List, frame_idx: int) -> List:
        offset_frames = []
        n_frames = len(all_lidar_frames)

        if self.load_future_offset_frames:
            start_idx = min(frame_idx + self.min_future_offset, n_frames)
            end_idx = min(frame_idx + self.max_future_offset, n_frames)
            if start_idx <= end_idx:
                lidar_frames = all_lidar_frames[start_idx : end_idx + 1]
                num_frames_to_select = min(self.config.num_future_frames_to_sample, len(lidar_frames))
                if num_frames_to_select > 0:
                    if self.is_train:
                        select_frame_idx: List[int] = np.random.choice(
                            len(lidar_frames), num_frames_to_select, replace=False
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
                num_frames_to_select = min(self.config.num_past_frames_to_sample, len(lidar_frames))
                if num_frames_to_select > 0:
                    if self.is_train:
                        select_frame_idx: List[int] = np.random.choice(
                            len(lidar_frames), num_frames_to_select, replace=False
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

    def _get_offset_lidar_frames(self, root_folder: Path, current_nstimestamp_str: str) -> List[Path]:
        lidar_nstimestamps = sorted(f.stem for f in (root_folder / "sensors" / "lidar").iterdir() if f.is_file())
        # find the index of the current lidar frame
        current_idx = lidar_nstimestamps.index(current_nstimestamp_str)
        # get the offset lidar frames
        selected = self._select_offset_frames(lidar_nstimestamps, current_idx)
        selected = [root_folder / "sensors" / "lidar" / f"{name}.feather" for name in selected]
        selected = sorted(selected, key=lambda x: int(x.stem))
        return selected

    def _get_offset_pointclouds(
        self, root_folder: Path, current_nstimestamp_str: str, bev_oxts_pose: np.ndarray, ego_motion: EgoMotion
    ) -> Tuple[Dict[str, np.ndarray], List[float]]:
        # load the future lidar scans
        lidar_filepaths = self._get_offset_lidar_frames(root_folder, current_nstimestamp_str)
        lidar_sweeps = [Sweep.from_feather(lidar_file) for lidar_file in lidar_filepaths]

        lidar_timestamps = [int(f.stem) / 1e9 for f in lidar_filepaths]

        # slice relevant part of oxts (ego motion) and store for use on gpu
        current_nstimestamp = int(current_nstimestamp_str)  # ns to s
        current_timestamp = current_nstimestamp / 1e9  # ns to s
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
            panoptic_labels = self._get_panoptic_labels(lidar_sweeps, lidar_filepaths)
            labels.update(panoptic_labels)

        # downsample the pointclouds if needed
        if (max_per_scan := self.config.n_max_offset_points_to_load_per_scan) >= 0:
            for i, lidar_sweep in enumerate(lidar_sweeps):
                if len(lidar_sweeps) > max_per_scan:
                    idx = np.random.permutation(len(lidar_sweeps))[:max_per_scan]
                    lidar_sweeps[i] = Sweep(
                        xyz=lidar_sweep.xyz[idx],
                        intensity=lidar_sweep.intensity[idx],
                        laser_number=lidar_sweep.laser_number[idx],
                        offset_ns=lidar_sweep.offset_ns[idx],
                        timestamp_ns=lidar_sweep.timestamp_ns,
                        ego_SE3_up_lidar=lidar_sweep.ego_SE3_up_lidar,
                        ego_SE3_down_lidar=lidar_sweep.ego_SE3_down_lidar,
                    )
                    for key, value in labels.items():
                        labels[key][i] = value[i][idx]

        # compute in-range mask
        in_range_mask = [
            _points_in_target_bev(
                lidar_sweep,
                sliced_oxts,
                self.config.point_cloud_range,
                bev_oxts_pose,
            )
            for lidar_sweep in lidar_sweeps
        ]

        # uncompensate clouds if needed
        lidar_sweeps = [fast_uncompensate_pointwise(s, sliced_oxts) for s in lidar_sweeps]

        # NOTE: This is a hack to make compatabile with current code where we assume we only have one lidar pose
        # and that the lidar pose is the same for all points.
        for sweep in lidar_sweeps:
            sweep.xyz = sweep.ego_SE3_up_lidar.inverse().transform_point_cloud(sweep.xyz)
        lidar_extrinsics = sweep.ego_SE3_up_lidar.transform_matrix.astype(np.float64)

        # fetch these before we mess up the timestamps
        if self.config.add_missing_points:
            missing_points = self._load_missing_points(lidar_filepaths, lidar_sweeps, sliced_oxts, current_nstimestamp)

        # set everything in reference to the current frame
        for sweep in lidar_sweeps:
            t_ = (sweep.offset_ns.astype(np.int64) + sweep.timestamp_ns) - current_nstimestamp
            sweep.timestamp_ns = 0
            sweep.offset_ns = t_

        # NOTE: If we want to move the pointclouds to their correct position we have to also include the laser number
        # here to use for the transformation later.
        offset_pointclouds = [np.hstack((sweep.xyz, sweep.offset_ns[:, None] / 1e9)) for sweep in lidar_sweeps]
        future_pcs_in_range = np.vstack([pc[mask] for pc, mask in zip(offset_pointclouds, in_range_mask)])
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

        # get the ratio of points inside the range to be used to samples queries in occupancy sampler
        ratio_inside = len(future_pcs_in_range) / (len(future_pcs_in_range) + len(missing_rays))

        if self.config.n_max_offset_points >= 0:
            idx = np.random.permutation(len(future_pcs_in_range))[: self.config.n_max_offset_points]
            future_pcs_in_range = future_pcs_in_range[idx]
            offset_pointclouds_indices = offset_pointclouds_indices[idx]
            for key, value in labels.items():
                labels[key] = value[idx]

        if self.config.n_max_missing_rays >= 0:
            idx = np.random.permutation(len(missing_rays))[: self.config.n_max_missing_rays]
            missing_rays = missing_rays[idx]

        sliced_oxts.timestamps -= current_timestamp
        sliced_oxts.poses = inverse_np(bev_oxts_pose) @ sliced_oxts.poses

        if self.config.make_batchable:
            (
                future_pcs_in_range,
                offset_pointclouds_indices,
                missing_rays,
                sliced_oxts,
                labels,
            ) = pad_offset_data_for_batching(
                future_pcs_in_range,
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
            "offset_pointclouds": future_pcs_in_range.astype(np.float32),
            "offset_pointclouds_indices": offset_pointclouds_indices,
            "missing_rays": missing_rays,
            "future_points_ratio_inside": ratio_inside,
        }

        if labels:
            data.update({"offset_pointclouds_labels": labels})

        return data, lidar_timestamps

    def _load_missing_points(
        self, lidar_filepaths: List[Path], lidar_sweeps: List[Sweep], sliced_oxts: EgoMotion, current_nstimestamp: int
    ) -> List[np.ndarray]:
        split_names = []
        for f in lidar_filepaths:
            if "train" in f.parts:
                split_names.append("train")
            elif "val" in f.parts:
                split_names.append("val")
            elif "test" in f.parts:
                split_names.append("test")
        mps_dir = [str(f).split("/sensors")[0].split("/")[-1] for f in lidar_filepaths]
        mps_fnames = [f"{s}/{d}/{f.stem}_missing.npy" for s, d, f in zip(split_names, mps_dir, lidar_filepaths)]
        mps_fpaths = [self.config.av2_cached_missing_points_dir / f for f in mps_fnames]

        existing = [f.exists() for f in mps_fpaths]
        if not any(existing):
            _warn_missing_points_cache_absent(self.config.av2_cached_missing_points_dir)

        xyzts = [np.load(f) for f, e in zip(mps_fpaths, existing) if e]
        xyzts = [np.concatenate((pc[:, :3], pc[:, 4:5]), axis=1) for pc in xyzts]
        swps = [s for s, e in zip(lidar_sweeps, existing) if e]

        # Create sweeps of missing points
        missing_sweeps = []
        for xyzt, swp in zip(xyzts, swps):
            points = xyzt[:, :3]
            offsets = xyzt[:, 3]
            z_mask = points[:, 2] > 0.0
            points = points[z_mask]
            offsets_ns = (offsets[z_mask] * 1e9).astype(np.int64)
            mps = Sweep(
                xyz=points,
                intensity=np.zeros((len(points),), dtype=swp.intensity.dtype),
                laser_number=np.zeros((len(points),), dtype=np.uint8),
                offset_ns=offsets_ns,
                timestamp_ns=swp.timestamp_ns,
                ego_SE3_up_lidar=swp.ego_SE3_up_lidar,
                ego_SE3_down_lidar=swp.ego_SE3_down_lidar,
            )
            missing_sweeps.append(mps)

        # uncompensate clouds if needed
        missing_sweeps = [fast_uncompensate_pointwise(s, sliced_oxts) for s in missing_sweeps]

        # NOTE: This is a hack to make compatabile with current code where we assume we only have one lidar pose
        # and that the lidar pose is the same for all points.
        for sweep in missing_sweeps:
            sweep.xyz = sweep.ego_SE3_up_lidar.inverse().transform_point_cloud(sweep.xyz)

        # set everything in reference to the current frame
        for sweep in missing_sweeps:
            t_ = (sweep.offset_ns.astype(np.int64) + sweep.timestamp_ns) - current_nstimestamp
            sweep.timestamp_ns = 0
            sweep.offset_ns = t_

        # NOTE: If we want to move the pointclouds to their correct position we have to also include the laser number
        # here to use for the transformation later.
        missing_points_cloud = [np.hstack((sweep.xyz, sweep.offset_ns[:, None] / 1e9)) for sweep in missing_sweeps]

        # Subsample points randomly
        if self.config.missing_points_downsample_factor > 0.0:
            subsampled_missing_points_cloud = []
            for pc in missing_points_cloud:
                selected_indeces = np.random.choice(
                    range(pc.shape[0]),
                    size=int(pc.shape[0] * self.config.missing_points_downsample_factor),
                    replace=False,
                )
                subsampled_missing_points_cloud.append(pc[selected_indeces])
            missing_points_cloud = subsampled_missing_points_cloud

        return missing_points_cloud

    def _load_images(
        self, root_folder: Path, current_nstimestamp_str: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load the configured images for the current timestep and the previous ones.

        Returns
            a dict containing one numpy array per loaded camera where each array has shape (t, h, w, c).
                note that the newest image is first in the array (t, t-1, t-2, ...)
            a dict containinig the timestamps of the images in seconds.
        """
        images = {}
        times = {}
        current_timestamp_ns = int(current_nstimestamp_str)

        prev_max_time = self.config.prev_images_max_time
        n_prev = self.config.n_prev_images
        # If any of the two are 0, we just load the current image
        if prev_max_time <= 0 or n_prev <= 0:
            target_prev_timestamps_ns = np.array([])
        else:
            target_prev_timestamps_ns = np.linspace(
                current_timestamp_ns - prev_max_time * 1e9, current_timestamp_ns, n_prev + 1
            )[:-1]

        future_max_time = self.config.future_images_max_time
        n_future = self.config.n_future_images
        if future_max_time <= 0 or n_future <= 0:
            target_future_timestamps_ns = np.array([])
        else:
            target_future_timestamps_ns = np.linspace(
                current_timestamp_ns, current_timestamp_ns + future_max_time * 1e9, n_future + 1
            )[1:]

        target_timestamps_ns = np.concatenate([target_prev_timestamps_ns, target_future_timestamps_ns])
        n_target_timestamps = target_timestamps_ns.shape[0]

        for camera in self.config.av2_cameras_to_load:
            image_folder = root_folder / "sensors" / "cameras" / camera
            # find the closest image
            image_files = sorted(image_folder.iterdir())
            available_timestamps = np.array([int(x.stem) for x in image_files])
            closest_index = np.argmin(np.abs(available_timestamps - current_timestamp_ns))
            closest_image = image_files[closest_index]
            # load the closest image
            image = av2.utils.io.read_img(closest_image)  # RGB, HWC, uint8
            images[camera] = np.expand_dims(image, axis=0)  # n-times x h x w x c
            times[camera] = np.array([int(closest_image.stem) / 1e9])  # s shape: (n-times,)
            if n_target_timestamps > 0:
                delta_timestamps = np.abs(available_timestamps[None, :] - target_timestamps_ns[:, None])
                prev_indices = np.argmin(delta_timestamps, axis=1)
                for ind in prev_indices[::-1]:
                    img_file = image_files[ind]
                    image = av2.utils.io.read_img(img_file)  # RGB, HWC, uint8
                    # put the most newest image first
                    images[camera] = np.concatenate((images[camera], np.expand_dims(image, axis=0)), axis=0)
                    times[camera] = np.append(times[camera], int(img_file.stem) / 1e9)
            # convert to (t, c, h, w); kept as uint8, converted to float in [0, 1]
            # on the GPU (BaseModel._get_camera_data)
            images[camera] = np.transpose(images[camera], (0, 3, 1, 2))  # (t, c, h, w)

        return images, times

    def _get_camera_calibration(self, root_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
        calib: Dict[str, Dict[str, np.ndarray]] = {}
        for cam_name in self.config.av2_cameras_to_load:
            camera = PinholeCamera.from_feather(log_dir=root_dir, cam_name=cam_name)
            extrinsics = camera.extrinsics.copy()
            intrinsics = camera.intrinsics.K.copy()
            original_image_size = np.array([camera.intrinsics.width_px, camera.intrinsics.height_px])

            field_of_view = np.array(
                2 * np.arctan(0.5 * camera.intrinsics.width_px / camera.intrinsics.fx_px),
                2 * np.arctan(0.5 * camera.intrinsics.height_px / camera.intrinsics.fy_px),
            )

            calib[cam_name] = {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "distortion": np.zeros(4),  # cameras are undistorted
                "original_camera_intrinsics": camera.intrinsics.K.copy(),
                "original_camera_extrinsics": camera.extrinsics.copy(),
                "original_camera_size": original_image_size.copy(),
                "T_cam_ego": camera.extrinsics.copy(),
                "field_of_view": field_of_view,
            }

        return calib

    def _crop_images_and_intrinsics(
        self, images: Dict[str, np.ndarray], calibration: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """images are (t, h, w, c)"""
        for cam_name, img in images.items():
            img = crop_images(img, AV2_CAM_CROP[cam_name])
            intrinsics_modified = calibration[cam_name]["intrinsics"].copy()
            intrinsics_modified = crop_intrinsics(intrinsics_modified, AV2_CAM_CROP[cam_name])
            images[cam_name] = img
            calibration[cam_name]["intrinsics_modified"] = intrinsics_modified

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

    def _downsample_images_and_intrinsics(
        self, images: Dict[str, np.ndarray], calibration: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        for cam_name, img in images.items():
            img = resize_images_with_scale(img, 1 / self.config.av2_camera_downsample_factor)
            images[cam_name] = img
            calibration[cam_name]["intrinsics_modified"] = resize_intrinsics_with_scale(
                calibration[cam_name]["intrinsics_modified"], 1 / self.config.av2_camera_downsample_factor
            )
        return images, calibration

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

    def _get_depth_maps(
        self, data, images, root_folder, current_nstimestamp_str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if not self.config.pad_images_to_same_size:
            raise ValueError("config.pad_images_to_same_size must be True when loading depth maps for AV2.")
        num_cams = len(images)
        img_h = int(list(data["imgs_h"].values())[0] / self.config.depth_downsample_factor)
        img_w = int(list(data["imgs_w"].values())[0] / self.config.depth_downsample_factor)

        # Load the lidar sweep
        lidar_sweep = Sweep.from_feather(root_folder / "sensors" / "lidar" / f"{current_nstimestamp_str}.feather")
        points = lidar_sweep.xyz
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

    def __getitem__(self, idx: int):
        root_folder, lidar_file = self.dataset_idx_to_seq_and_frame[idx]
        current_nstimestamp_str = lidar_file.split("/")[-1].split(".")[0]
        current_timestamp = float(current_nstimestamp_str) / 1e9  # ns to s
        sequence_id = root_folder.split("/")[-1]

        data: Dict[str, Any] = {
            "id": sequence_id,
            "frame_idx": current_nstimestamp_str,
        }

        root_folder = Path(root_folder)
        ego_motion = self._get_ego_motion(root_folder)

        ego_motion, bev_oxts_pose, aug_angle, aug_translation, _ = self._rotation_augmentation(
            ego_motion, current_timestamp
        )

        if self.config.load_full_ego_motion:
            data["full_ego_motion"] = inverse_np(bev_oxts_pose) @ ego_motion.poses.copy()

        data["augmentation_angle"] = aug_angle
        data["augmentation_translation"] = aug_translation

        if self.config.load_lidar:
            # get the lidar frames
            if self.config.aggregate_lidar:
                data.update(
                    self._aggregate_lidar_frames(
                        root_folder, current_nstimestamp_str, ego_motion, target_pose=bev_oxts_pose
                    )
                )
            else:
                raise NotImplementedError

        if self.config.load_image:
            # get the images for each camera
            images, image_times = self._load_images(root_folder, current_nstimestamp_str)
            calibration = self._get_camera_calibration(root_folder)
            delta_times = {k: (v - current_timestamp).astype(np.float32) for k, v in image_times.items()}

            images, calibration = self._resize_images(images, calibration)

            # update extrinsics based on odometry
            calibration = self._motion_compensate_extrinsics(calibration, ego_motion, image_times, bev_oxts_pose)
            ni = images[list(images.keys())[0]].shape[0]

            # make it imgs -> cam-name -> tensor, and similar for the rest
            data.update(
                {
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
                depth_maps, valid_depth_maps = self._get_depth_maps(data, images, root_folder, current_nstimestamp_str)
                data["depth_maps"] = depth_maps
                data["depth_maps_valid"] = valid_depth_maps

        future_lidar_timestamps = None
        if self.config.load_offset_pointclouds:
            future_lidar_data, future_lidar_timestamps = self._get_offset_pointclouds(
                root_folder, current_nstimestamp_str, bev_oxts_pose, ego_motion
            )
            data.update(future_lidar_data)

        if self.config.load_vfm_stuff:
            vfm_data = self.vfm_reader.read(
                root_folder=root_folder,
                current_timestamp=current_timestamp,
                lidar_timestamps=future_lidar_timestamps,
                ego_motion=ego_motion,
                bev_oxts_pose=bev_oxts_pose,
            )
            data.update(vfm_data)

        def _to_tensor(data: Union[np.ndarray, list, str]) -> Union[str, torch.Tensor]:
            if isinstance(data, (str, int, float, dict)):
                return data
            data = np.stack(data, axis=0) if isinstance(data, list) else data
            return torch.from_numpy(data)

        return {k: _to_tensor(v) for k, v in data.items()}


def _ego_motion_compensate_scanwise(
    sweep: Sweep,
    ego_motion: EgoMotion,
    target_timestamp: Optional[float] = None,
    target_pose: Optional[np.ndarray] = None,
) -> Sweep:
    assert target_timestamp is not None or target_pose is not None, (
        "Either target_timestamp or target_pose must be provided."
    )
    xyz = sweep.xyz.copy()
    sweep_pose = ego_motion.get_poses(sweep.timestamp_ns / 1e9)

    if target_pose is None:
        assert target_timestamp is not None, "Target timestamp must be provided if target_pose is not."
        target_pose = ego_motion.get_poses(target_timestamp)

    odometry = inverse_np(target_pose) @ sweep_pose

    rotations = odometry[..., :3, :3]
    translations = odometry[..., :3, 3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)

    sweep.xyz = xyz
    return sweep


def _points_in_target_bev(
    lidar_sweep: Sweep,
    ego_motion: EgoMotion,
    point_cloud_range: PointCloudRange,
    bev_oxts_pose: np.ndarray,
) -> np.ndarray:
    xyz = lidar_sweep.xyz.copy()
    # motion compensate scanwise to the target timestamp
    source_pose = ego_motion.get_poses(lidar_sweep.timestamp_ns / 1e9)
    target_pose = bev_oxts_pose
    odometry = inverse_np(target_pose) @ source_pose
    rotations = odometry[..., :3, :3]
    translations = odometry[..., :3, 3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)

    # check bev range
    min_range = np.array(point_cloud_range[:3])
    max_range = np.array(point_cloud_range[3:])
    in_range = np.all((min_range <= xyz) & (xyz <= max_range), axis=-1)
    # restore lidar_data (we don't want to overwrite)

    # remove points that are within the ego-vehicle
    mask_x = np.logical_and(xyz[..., 0] > -1.75, xyz[..., 0] < 3.75)
    mask_y = np.logical_and(xyz[..., 1] > -1.25, xyz[..., 1] < 1.25)
    not_in_vehicle = ~np.logical_and(mask_x, mask_y)

    in_bev = np.logical_and(in_range, not_in_vehicle)

    return in_bev


def fast_uncompensate_pointwise(sweep: Sweep, ego_motion: EgoMotion) -> Sweep:
    """Uncompensate an already scan-compensated point cloud. (inverse of fast_compensate_pointwise)

    Given a point cloud that has been compensated to the scan timestamp, un-compensate each point to its original
    pose given the delta-time encoded in each point (x,y,z,r,t)
    """
    # get source (scan) and target (pointwise) poses and compute relative transform (odometry)
    point_pose = ego_motion.get_poses(sweep.timestamp_ns / 1e9)
    target_times = (sweep.timestamp_ns + sweep.offset_ns) / 1e9  # should this be + or -?
    target_poses = fast_ego_motion_poses(ego_motion, target_times)
    odometry = inverse_np(target_poses) @ point_pose
    # Project ego -> new-ego
    sweep = _transform_sweep(sweep, odometry)
    return sweep


def _transform_sweep(sweep: Sweep, transform: np.ndarray) -> Sweep:
    xyz = sweep.xyz.copy()
    rotations = transform[..., :3, :3]
    translations = transform[..., :3, 3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)
    sweep.xyz = xyz
    return sweep
