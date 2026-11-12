import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import cv2
import numpy as np
import torch
from av2.geometry.camera.pinhole_camera import PinholeCamera
from zod.data_classes.ego_motion import EgoMotion

import queryocc.config.base_config as cfg
from queryocc.common.types import PointCloudRange
from queryocc.data.datasets.nuscenes_mappings import LabelMapper
from queryocc.data.datasets.utils import (
    crop_images,
    crop_intrinsics,
    resize_images_with_scale,
    resize_intrinsics_with_scale,
)
from queryocc.data.vfm.grounding_sam_reader import GroundingSAMReaderConfig
from queryocc.data.vfm.metric3d_reader import Metric3DReaderConfig
from queryocc.data.vfm.pseudo_label_npz import DEFAULT_SHAPES
from queryocc.utils.geometry import inverse_np

DELETE_STUFF = True

# Depth (m) assigned to sky pixels when sky_as_free_rays is on: far beyond any point-cloud range so the
# lifted point is always out of range and becomes a free (missing-ray) sample rather than an occupied point.
SKY_FREE_RAY_DEPTH = 1000.0

# Oversample factor for the uniform pseudo-point subsample fast path: we draw cap * this many candidate
# pixels, drop the noise-labelled ones, and keep the first `cap`. 1.5 covers up to a 33% local noise
# fraction (measured nuScenes noise is ~9%); rarer high-noise draws fall back to the exact path.
SUBSAMPLE_OVERSAMPLE = 1.5

RGB_MEAN = (123.675, 116.28, 103.53)  # (0.485, 0.456, 0.406)
RGB_STD = (58.395, 57.12, 57.375)  # (0.229, 0.224, 0.225)

AV2_CENTER_CROP: Tuple[int, int, int, int] = (314, 0, 212, 0)  # (top, right, bottom, left)
AV2_OTHERS_CROP: Tuple[int, int, int, int] = (12, 0, 2, 0)  # (top, right, bottom, left)
AV2_SCALE = 2.0

NUSCENES_CROP: Tuple[int, int, int, int] = (4, 16, 0, 16)  # (top, right, bottom, left)
NUSCENES_SCALE = 1.0

MAXIMAL_CAMERA_LIDAR_TIMEDIFF = 0.15  # seconds

DATASET_TO_CAMERAS = {
    "av2": (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    ),
    "nuscenes": (
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
    ),
}


@dataclass
class VFMReaderConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: VFMReader)

    downsample_factor: float = 1.0
    """Downsample factor to use before cropping."""

    image_final_size: Tuple[int, int] = (-224, -224)
    """Final size of the images. Set to negative to not crop."""

    extracted_features_downsample_factor: int = 16
    """Which downsample factor to use."""

    sample_center_camera_factor: float = 2.0
    """How much more to sample the center camera features."""

    sample_camera_per_timestep: bool = False
    """Whether to draw a (weighted) random camera independently for each timestep instead of one camera for the
    whole sample. Calibrations become per-timestep (T, 3, 3)/(T, 4, 4) and frames from differently-sized cameras
    are padded bottom/right to a common size. Supported for nuScenes and AV2."""

    read_grounding_sam: bool = False
    """Whether to read the grounding sam."""

    read_grounding_sam_scores: bool = False
    """Whether to read the grounding sam scores."""

    score_threshold: float = 0.2
    """The threshold to apply to the grounding sam scores to create a binary mask."""

    grounding_sam_reader: GroundingSAMReaderConfig = field(default_factory=GroundingSAMReaderConfig)
    """The grounding sam reader config."""

    read_metric3d_depth: bool = False
    """Whether to read the metric3d depth."""

    metric3d_depth_reader: Metric3DReaderConfig = field(default_factory=Metric3DReaderConfig)
    """The metric3d depth reader config."""

    create_pseudo_pointclouds: bool = False
    """Whether to create the pseudo pointclouds."""

    mask_out_noise_pseudo_pointclouds: bool = True
    """Whether to mask out noise points in the pseudo pointclouds."""

    sample_points_based_on_label_frequency: Literal["none", "class_frequency", "class_frequency_pseudo"] = "none"
    """Whether to upsample based on the label frequency in the pseudo pointclouds."""

    sample_points_based_on_label_frequency_alpha: float = 1.0
    """The alpha parameter for the label frequency sampling."""

    remove_sky_points_pseudo_pointclouds: bool = True
    """Whether to remove sky points in the pseudo pointclouds."""

    sky_as_free_rays: bool = False
    """Whether to turn grounding-SAM sky pixels into free-space (missing-ray) supervision instead of
    discarding them. Sky is empty space along the camera ray, so it certifies free space. Requires the
    grounding-SAM reader to tag sky as free_idx (see GroundingSAMReader.read_individual_cam_nuscenes);
    such pixels are pushed past the point-cloud range here so they become free rays rather than (mis-depth)
    occupied points. Baseline (False) drops sky via the noise filter (sky -> noise) and the depth>=160 cut."""

    pseudo_pointclouds_max_points: int = 30_000
    """The maximum number of points to sample in the pseudo pointclouds."""

    pseudo_pointclouds_labels: bool = True
    """Whether to include labels in the pseudo pointclouds."""

    pad_pseudo_data_for_batching: bool = False
    """Pad/cap the variable-length pseudo pointcloud outputs to the fixed sizes below so that samples can be
    stacked by the default collate function at batch_size > 1. Points and rays are padded by duplicating random
    entries (harmless: the GPU samplers draw from them with replacement, and the point/ray balance comes from the
    separately passed ratio_inside) and capped by random subsampling. Enabled automatically by JointDataModule
    when the per-device batch size is > 1."""

    n_max_pseudo_points: int = 240_000
    """Fixed pseudo_pointcloud size when pad_pseudo_data_for_batching is set. The per-timestep cap
    (pseudo_pointclouds_max_points) bounds points + missing rays by T * cap, so with the default 7 offset
    frames x 30k the in-range points never exceed 210k and this default is lossless."""

    n_max_pseudo_missing_rays: int = 80_000
    """Fixed pseudo_missing_rays size when pad_pseudo_data_for_batching is set. Measured nuScenes samples
    peak below 40k; rays beyond the cap are randomly subsampled."""

    n_max_pseudo_ego_poses: int = 192
    """Fixed pseudo_pointcloud_ego_poses size when pad_pseudo_data_for_batching is set. Padded by repeating
    the last pose; subsampled uniformly when over (poses are only interpolated downstream). Measured nuScenes
    samples stay below 130."""

    pad_to_num_timesteps: int = -1
    """Fixed number of timesteps T for the per-timestep camera outputs (vfm_* and pseudo_pointcloud_extrinsics
    become (T, ...)). Samples near scene edges have fewer offset lidar frames, so without this the T dimension
    varies and samples cannot be stacked at batch_size > 1. Kept timesteps are duplicated (with their own
    timestamps/calibrations) to reach T. Set by JointDataModule from the dataset frame-sampling config when the
    per-device batch size is > 1; -1 keeps the natural per-sample count."""

    add_noise_depths: bool = False
    """Whether to add noise to the depths."""
    noise_depths_std: float = 0.003
    """The standard deviation of the noise to add to the inverse depths.
    $\sigma = 0.001$ (Very Conservative): At 50m, you get 47.6m to 52.6m.
    $\sigma = 0.003$ (Realistic): At 50m, you get 43.4m to 58.8m.
    $\sigma = 0.005$ (Stress Test): This is the 40m to 66m range.
    """

    add_noise_semantic_labels: bool = False
    """Whether to add noise to the semantic labels."""
    noise_semantic_labels_ratio: float = 0.05
    """The ratio of pixels to corrupt in the semantic labels."""


def get_sample_probability_per_camera(cameras: Tuple[str, ...], sample_center_camera_factor: float):
    sample_probability_per_cam = {cam: 1.0 / len(cameras) for cam in cameras}
    if "ring_front_center" in cameras:
        sample_probability_per_cam["ring_front_center"] *= sample_center_camera_factor
    if "CAM_FRONT" in cameras:
        sample_probability_per_cam["CAM_FRONT"] *= sample_center_camera_factor
    if "CAM_BACK" in cameras:
        sample_probability_per_cam["CAM_BACK"] *= sample_center_camera_factor
    # normalize the probabilities
    sample_probability_per_cam = np.array(list(sample_probability_per_cam.values()))
    sample_probability_per_cam = sample_probability_per_cam / np.sum(sample_probability_per_cam)
    return sample_probability_per_cam


def get_crop_per_camera(dataset_name: str) -> Dict[str, Tuple[int, int, int, int]]:
    if dataset_name == "av2":
        return {
            "ring_front_center": AV2_CENTER_CROP,
            "ring_front_left": AV2_OTHERS_CROP,
            "ring_front_right": AV2_OTHERS_CROP,
            "ring_rear_left": AV2_OTHERS_CROP,
            "ring_rear_right": AV2_OTHERS_CROP,
            "ring_side_left": AV2_OTHERS_CROP,
            "ring_side_right": AV2_OTHERS_CROP,
        }
    elif dataset_name == "nuscenes":
        return {
            "CAM_FRONT": NUSCENES_CROP,
            "CAM_FRONT_LEFT": NUSCENES_CROP,
            "CAM_FRONT_RIGHT": NUSCENES_CROP,
            "CAM_BACK_LEFT": NUSCENES_CROP,
            "CAM_BACK_RIGHT": NUSCENES_CROP,
            "CAM_BACK": tuple(np.array(NUSCENES_CROP) + np.array((0, 0, 160, 0))),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def crop_images_and_intrinsics(
    images: np.ndarray, calibrations: np.ndarray, crop_per_camera: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    images = crop_images(images, crop_per_camera)
    intrinsics = crop_intrinsics(calibrations["intrinsics"], crop_per_camera)
    calibrations["intrinsics"] = intrinsics
    return images, calibrations


def random_crop_images_and_intrinsics(
    images: np.ndarray, calibrations: np.ndarray, image_final_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if image_final_size[0] < 0 or image_final_size[1] < 0:
        return images, calibrations
    h, w = images.shape[-2:]
    h_crop, w_crop = image_final_size
    h_start = np.random.randint(0, h - h_crop)
    w_start = np.random.randint(0, w - w_crop)
    h_end = h - (h_start + h_crop)
    w_end = w - (w_start + w_crop)
    crop = (h_start, w_end, h_end, w_start)
    images, calibrations = crop_images_and_intrinsics(images, calibrations, crop)
    return images, calibrations


def get_relative_timestamps(timestamps: np.ndarray, current_timestamp: float) -> np.ndarray:
    """Get the relative timestamps to the current timestamp."""
    return timestamps - float(current_timestamp)


def get_intrinsics_for_feature_extraction(intrinsics: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Get the intrinsics for feature extraction."""
    intrinsics = resize_intrinsics_with_scale(intrinsics, 1 / float(downsample_factor))
    return intrinsics


def get_downsample_factor(dataset_name: str, downsample_factor: float) -> float:
    if dataset_name == "nuscenes":
        return float(downsample_factor * NUSCENES_SCALE)
    elif dataset_name == "av2":
        return float(downsample_factor * AV2_SCALE)
    else:
        return float(downsample_factor)


def pad_frames_bottom_right(
    frames: List[np.ndarray], target_h: int, target_w: int, pad_value: float
) -> List[np.ndarray]:
    """Pad (n, c, h, w) frames at the bottom/right to a common size; this keeps the intrinsics valid."""
    padded = []
    for frame in frames:
        pad_h = target_h - frame.shape[-2]
        pad_w = target_w - frame.shape[-1]
        if pad_h or pad_w:
            frame = np.pad(frame, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), constant_values=pad_value)
        padded.append(frame)
    return padded


def resize_first_dim_indices(n_current: int, n_target: int) -> Optional[np.ndarray]:
    """Index array that pads (duplicates random rows) or randomly subsamples the first dim to n_target.

    Returns None when no resize is needed. Requires n_current > 0.
    """
    if n_current == n_target:
        return None
    if n_current > n_target:
        return np.random.permutation(n_current)[:n_target]
    return np.concatenate([np.arange(n_current), np.random.choice(n_current, n_target - n_current)])


def concat_camera_frames(per_timestep: List[tuple], mask_pad_value: float) -> tuple:
    """Concatenate per-timestep single-camera frames (from `VFMReader._load_camera_frames`) along the time axis.

    Frames from differently-sized cameras are padded bottom/right to the largest size; padded depth pixels are 0
    and therefore never produce pseudo points. Calibrations are stacked to (T, 3, 3)/(T, 4, 4)/(T, D).
    """
    images_l, masks_l, scores_l, depths_l, timestamps_l, calibrations_l = map(list, zip(*per_timestep))

    reference = [
        next(a for a in (im, mk, sc, dp) if a is not None)
        for im, mk, sc, dp in zip(images_l, masks_l, scores_l, depths_l)
    ]
    target_h = max(r.shape[-2] for r in reference)
    target_w = max(r.shape[-1] for r in reference)

    def pad_and_concat(arrays: List[Optional[np.ndarray]], pad_value: float) -> Optional[np.ndarray]:
        if arrays[0] is None:
            return None
        return np.concatenate(pad_frames_bottom_right(arrays, target_h, target_w, pad_value), axis=0)

    images = pad_and_concat(images_l, 0.0)
    masks = pad_and_concat(masks_l, mask_pad_value)
    scores = pad_and_concat(scores_l, 0.0)  # padded scores are 0 -> always below score_threshold
    depths = pad_and_concat(depths_l, 0.0)  # padded depths are 0 -> excluded by the depth > 0 mask
    timestamps = np.concatenate(timestamps_l, axis=0)
    calibrations = {key: np.stack([c[key] for c in calibrations_l], axis=0) for key in calibrations_l[0]}
    return images, masks, scores, depths, timestamps, calibrations


class VFMReader:
    def __init__(
        self,
        config: VFMReaderConfig,
        dataset_name: str,
        data_root: Path,
        point_cloud_range: PointCloudRange,
        label_mapper: Optional[LabelMapper] = None,
    ):
        self.config = config
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.cameras = DATASET_TO_CAMERAS[dataset_name]
        self.sample_probability_per_cam = get_sample_probability_per_camera(
            self.cameras, self.config.sample_center_camera_factor
        )
        self.crop_per_camera = get_crop_per_camera(dataset_name)

        # Per-camera bottom/right padding that equalizes the cropped frame sizes across cameras
        # (e.g. nuScenes CAM_BACK is cropped 160 px shorter). Applied right after cropping, before
        # downsampling, so the padded frame size is identical for every camera and every sample —
        # required for stacking at batch_size > 1 with per-timestep camera sampling. Padding
        # bottom/right keeps the intrinsics valid; padded depths are 0 and padded mask pixels are
        # the noise index, so the dead rows generate no supervision. Only valid when the raw image
        # size is the same for every camera, which holds for all datasets here except AV2 (its
        # center camera is portrait); AV2 keeps per-sample padding in concat_camera_frames.
        if dataset_name == "av2":
            self._pad_to_uniform_crop: Optional[Dict[str, Tuple[int, int]]] = None
        else:
            crop_h = {cam: crop[0] + crop[2] for cam, crop in self.crop_per_camera.items()}  # top + bottom
            crop_w = {cam: crop[1] + crop[3] for cam, crop in self.crop_per_camera.items()}  # right + left
            min_h, min_w = min(crop_h.values()), min(crop_w.values())
            self._pad_to_uniform_crop = {
                cam: (crop_h[cam] - min_h, crop_w[cam] - min_w) for cam in self.crop_per_camera
            }

        if self.config.read_grounding_sam and self.dataset_name == "nuscenes":
            self.grounding_sam_reader = self.config.grounding_sam_reader.setup(dataset_name=self.dataset_name)
            self.label_mapper = label_mapper
        else:
            self.grounding_sam_reader = None
            self.label_mapper = None

        if self.config.read_metric3d_depth and self.dataset_name == "nuscenes":
            self.metric3d_depth_reader = self.config.metric3d_depth_reader.setup(dataset_name=self.dataset_name)
        else:
            self.metric3d_depth_reader = None

        self.point_cloud_range = point_cloud_range

    def _get_camera_info(
        self,
        camera_name: str,
        root_folder: Optional[Path] = None,
        cameras: Optional[Dict[str, List[str]]] = None,
        cameras_timestamps: Optional[Dict[str, List[int]]] = None,
        cameras_tokens: Optional[Dict[str, List[Optional[str]]]] = None,
    ) -> Tuple[List[Path], np.ndarray, List[Optional[str]]]:
        """Get the camera info for the given camera name.
        Args:
            camera_name: The name of the camera to get the info for.
            root_folder: The root folder to get the info from.
            cameras: The cameras to get the info from.
            cameras_timestamps: The camera timestamps to get the info from.
            cameras_tokens: Sample token per camera frame (None for non-keyframes), used to
                locate that frame's pseudo-label npz.
        Returns:
            files: The files to get the info from.
            timestamps: The timestamps to get the info from.
            tokens: The sample token per frame (None where the frame is not a keyframe).
        """
        if self.dataset_name == "av2":
            assert root_folder is not None, "Need to provide root_folder if dataset is av2"
            files: List[Path] = sorted((root_folder / "sensors/cameras" / camera_name).glob("*.jpg"))
            timestamps = np.array([int(f.stem) for f in files], dtype=np.float64) / 1e9  # Convert to seconds
            tokens: List[Optional[str]] = [None] * len(files)
        elif self.dataset_name == "nuscenes":
            assert cameras is not None, "Need to provide cameras if dataset is nuscenes"
            assert cameras_timestamps is not None, "Need to provide camera_timestamps if dataset is nuscenes"
            files = [Path(fp) for fp in cameras[camera_name]]
            timestamps = np.array(cameras_timestamps[camera_name], dtype=np.float64) / 1e6  # Convert to seconds
            tokens = list(cameras_tokens[camera_name]) if cameras_tokens is not None else [None] * len(files)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        return files, np.array(timestamps), tokens

    def _filter_closest(
        self,
        files: List[Path],
        timestamps: np.ndarray,
        tokens: List[Optional[str]],
        lidar_timestamps: np.ndarray,
    ):
        min_useful_timestamp = min(lidar_timestamps) - MAXIMAL_CAMERA_LIDAR_TIMEDIFF
        max_useful_timestamp = max(lidar_timestamps) + MAXIMAL_CAMERA_LIDAR_TIMEDIFF

        filtered = [
            (f, t, tok)
            for f, t, tok in zip(files, timestamps, tokens)
            if min_useful_timestamp <= t <= max_useful_timestamp
        ]
        if filtered:
            files, timestamps, tokens = zip(*filtered)
            files = list(files)
            timestamps = np.array(timestamps)
            tokens = list(tokens)
        else:
            # No frame of this camera within the threshold (scene edges, dropped frames in the
            # raw data). Callers fall back to another camera or raise.
            return [], [], []

        # Find closest file to each lidar timestamp
        closest_file = []
        closest_timestamp = []
        closest_token = []
        for lidar_timestamp in lidar_timestamps:
            time_diffs = np.abs(timestamps - lidar_timestamp)
            closest_idx = np.argmin(time_diffs)
            closest_file.append(files[closest_idx])
            closest_timestamp.append(timestamps[closest_idx])
            closest_token.append(tokens[closest_idx])

        return closest_file, np.array(closest_timestamp), closest_token

    def _get_camera_calibrations(
        self,
        camera_name: str,
        root_folder: Optional[Path] = None,
        cameras_calibrations: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        if self.dataset_name == "av2":
            assert root_folder is not None, "Need to provide root_folder if dataset is av2"
            pinhole_cam = PinholeCamera.from_feather(root_folder, camera_name)
            intrinsics = pinhole_cam.intrinsics.K.copy()
            extrinsics = inverse_np(pinhole_cam.extrinsics.copy())
            distortion = np.zeros(4)
            return {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "extrinsics_inverse": inverse_np(extrinsics),
                "distortion": distortion,
            }
        elif self.dataset_name == "nuscenes":
            assert cameras_calibrations is not None, "Need to provide calibrations if dataset is nuscenes"
            intrinsics = np.array(cameras_calibrations[camera_name]["intrinsics"].copy())
            extrinsics = cameras_calibrations[camera_name]["extrinsics"].copy()
            distortion = np.zeros(4)
            return {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "extrinsics_inverse": inverse_np(extrinsics),
                "distortion": distortion,
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _stack_camera_reads(self, reads: List[Optional[np.ndarray]], missing_fill: float, dtype) -> np.ndarray:
        """Stack per-frame reads, substituting a constant placeholder for frames without labels.

        Pseudo labels exist for keyframes only, so sweep frames legitimately have none. The
        placeholder must be the same shape and dtype as a real read, otherwise samples in a batch
        end up with mixed dtypes/shapes and default_collate falls back to a per-sample list.
        """
        shape = next((r.shape for r in reads if r is not None), None)
        if shape is None:
            shape = DEFAULT_SHAPES[self.dataset_name]
        return np.stack(
            [np.full(shape, missing_fill, dtype=dtype) if r is None else r.astype(dtype) for r in reads], axis=0
        )

    def _read_masks(self, scene_name: str, tokens: List[Optional[str]], camera_name: str) -> Optional[np.ndarray]:
        if not (self.config.read_grounding_sam and self.dataset_name == "nuscenes"):
            return None
        masks = [
            self.grounding_sam_reader.read_camera(
                scene_name, token, camera_name, self.label_mapper, sky_to_free=self.config.sky_as_free_rays
            )
            for token in tokens
        ]
        # -1 marks "no pseudo label here"; create_pseudo_pointclouds drops those pixels.
        return self._stack_camera_reads(masks, missing_fill=-1, dtype=np.int64)

    def _read_gs_scores(self, scene_name: str, tokens: List[Optional[str]], camera_name: str) -> Optional[np.ndarray]:
        if not (self.config.read_grounding_sam_scores and self.dataset_name == "nuscenes"):
            return None
        scores = [self.grounding_sam_reader.read_camera_scores(scene_name, token, camera_name) for token in tokens]
        if all(s is None for s in scores):
            return None
        # 0 is always below score_threshold.
        return self._stack_camera_reads(scores, missing_fill=0.0, dtype=np.float32)

    def _read_depths(self, scene_name: str, tokens: List[Optional[str]], camera_name: str) -> Optional[np.ndarray]:
        if not (self.config.read_metric3d_depth and self.dataset_name == "nuscenes"):
            return None
        depths = [self.metric3d_depth_reader.read_camera(scene_name, token, camera_name) for token in tokens]
        # -1 is excluded by the depth > 0 mask in create_pseudo_pointclouds.
        return self._stack_camera_reads(depths, missing_fill=-1.0, dtype=np.float32)

    def _downsample_vfm_data(self, vfm_data: np.ndarray, scale: float) -> np.ndarray:
        scalex = scale
        scaley = scale
        vfm_data = vfm_data.transpose(0, 2, 3, 1)
        vfm_data = np.array(
            [
                cv2.resize(data, (0, 0), fx=scalex, fy=scaley or scalex, interpolation=cv2.INTER_NEAREST_EXACT)
                for data in vfm_data
            ]
        )
        return np.expand_dims(vfm_data, axis=1)

    def _load_camera_frames(
        self,
        camera_name: str,
        lidar_timestamps: np.ndarray,
        root_folder: Optional[Path] = None,
        cameras: Optional[Dict[str, List[str]]] = None,
        cameras_timestamps: Optional[Dict[str, List[int]]] = None,
        cameras_calibrations: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        cameras_tokens: Optional[Dict[str, List[Optional[str]]]] = None,
        scene_name: Optional[str] = None,
        is_train: bool = False,
        camera_info_cache: Optional[Dict[str, tuple]] = None,
        camera_calibration_cache: Optional[Dict[str, dict]] = None,
    ) -> Optional[tuple]:
        """Read, crop and downsample the frames of one camera closest to the given lidar timestamps.

        Returns (images, masks, scores, depths, closest_timestamps, calibrations), each of the first four
        being None when its read flag is off, or None altogether if the camera has no frame close enough
        to any of the timestamps. The caches avoid repeated camera info/calibration lookups (e.g. AV2
        reads calibrations from disk) when called once per timestep.
        """
        if camera_info_cache is not None and camera_name in camera_info_cache:
            files, timestamps, tokens = camera_info_cache[camera_name]
        else:
            files, timestamps, tokens = self._get_camera_info(
                camera_name, root_folder, cameras, cameras_timestamps, cameras_tokens
            )
            if camera_info_cache is not None:
                camera_info_cache[camera_name] = (files, timestamps, tokens)

        # Filter the closest files and timestamps
        closest_files, closest_timestamps, closest_tokens = self._filter_closest(
            files, timestamps, tokens, lidar_timestamps
        )
        if len(closest_files) == 0:
            return None

        images = None
        masks = None
        scores = None
        if self.config.read_grounding_sam:
            masks = self._read_masks(scene_name, closest_tokens, camera_name)
            if masks is not None:
                masks = np.expand_dims(masks, axis=1)

            if self.config.read_grounding_sam_scores and is_train:
                scores = self._read_gs_scores(scene_name, closest_tokens, camera_name)
                if scores is not None:
                    scores = np.expand_dims(scores, axis=1)

        depths = None
        if self.config.read_metric3d_depth:
            depths = self._read_depths(scene_name, closest_tokens, camera_name)
            if depths is not None:
                depths = np.expand_dims(depths, axis=1)

        # Get the camera calibrations (deepcopy cached entries since "intrinsics" is overwritten below)
        if camera_calibration_cache is not None and camera_name in camera_calibration_cache:
            calibrations = copy.deepcopy(camera_calibration_cache[camera_name])
        else:
            calibrations = self._get_camera_calibrations(camera_name, root_folder, cameras_calibrations)
            if camera_calibration_cache is not None:
                camera_calibration_cache[camera_name] = copy.deepcopy(calibrations)

        # Crop the images to remove useless pixels
        crop = self.crop_per_camera[camera_name]
        calibrations["intrinsics"] = crop_intrinsics(calibrations["intrinsics"], crop)

        if images is not None:
            images = crop_images(images, crop)
        if masks is not None:
            masks = crop_images(masks, crop)
        if scores is not None:
            scores = crop_images(scores, crop)
        if depths is not None:
            depths = crop_images(depths, crop)

        # Equalize frame sizes across cameras (see _pad_to_uniform_crop in __init__). Done before
        # the downsample so the crop deltas are exact raw-pixel counts, free of resize rounding.
        if self._pad_to_uniform_crop is not None:
            pad_h, pad_w = self._pad_to_uniform_crop[camera_name]
            if pad_h or pad_w:
                pad_spec = ((0, 0), (0, 0), (0, pad_h), (0, pad_w))
                if images is not None:
                    images = np.pad(images, pad_spec, constant_values=0.0)
                if masks is not None:
                    mask_pad = float(self.label_mapper.noise_idx) if self.label_mapper is not None else 0.0
                    masks = np.pad(masks, pad_spec, constant_values=mask_pad)
                if scores is not None:
                    scores = np.pad(scores, pad_spec, constant_values=0.0)  # always below score_threshold
                if depths is not None:
                    depths = np.pad(depths, pad_spec, constant_values=0.0)  # excluded by the depth > 0 mask

        # Downsample
        scale = 1 / get_downsample_factor(self.dataset_name, self.config.downsample_factor)
        calibrations["intrinsics"] = resize_intrinsics_with_scale(calibrations["intrinsics"], scale)
        if images is not None:
            images = resize_images_with_scale(images, scale)
        if masks is not None:
            masks = self._downsample_vfm_data(masks, scale)
        if scores is not None:
            scores = self._downsample_vfm_data(scores, scale)
        if depths is not None:
            depths = self._downsample_vfm_data(depths, scale)

        return images, masks, scores, depths, closest_timestamps, calibrations

    def read(
        self,
        root_folder: Optional[Path] = None,
        cameras: Optional[Dict[str, List[str]]] = None,
        cameras_timestamps: Optional[Dict[str, List[int]]] = None,
        lidar_timestamps: Optional[List[float]] = None,
        current_timestamp: Optional[float] = None,
        cameras_calibrations: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        cameras_tokens: Optional[Dict[str, List[Optional[str]]]] = None,
        scene_name: Optional[str] = None,
        ego_motion: Optional[EgoMotion] = None,
        bev_oxts_pose: Optional[np.ndarray] = None,
        is_train: bool = False,
    ) -> Dict[str, np.ndarray]:
        lidar_timestamps = np.array(lidar_timestamps, dtype=np.float64)

        if self.config.sample_camera_per_timestep:
            # Draw a (weighted) random camera independently for each timestep
            camera_names = list(
                np.random.choice(self.cameras, size=len(lidar_timestamps), p=self.sample_probability_per_cam)
            )
            camera_info_cache: Dict[str, tuple] = {}
            camera_calibration_cache: Dict[str, dict] = {}
            per_timestep = []
            for camera_name, lidar_timestamp in zip(camera_names, lidar_timestamps):
                # Try the drawn camera first; when it has no frame within the time threshold
                # (scene edges, dropped frames in the raw data), redraw from the remaining cameras
                # so the timestep is not lost (a constant T is required at batch_size > 1).
                fallbacks = [cam for cam in self.cameras if cam != camera_name]
                np.random.shuffle(fallbacks)
                loaded = None
                for candidate in (camera_name, *fallbacks):
                    loaded = self._load_camera_frames(
                        candidate,
                        np.array([lidar_timestamp], dtype=np.float64),
                        root_folder=root_folder,
                        cameras=cameras,
                        cameras_timestamps=cameras_timestamps,
                        cameras_calibrations=cameras_calibrations,
                        cameras_tokens=cameras_tokens,
                        scene_name=scene_name,
                        is_train=is_train,
                        camera_info_cache=camera_info_cache,
                        camera_calibration_cache=camera_calibration_cache,
                    )
                    if loaded is not None:
                        break
                if loaded is not None:
                    per_timestep.append(loaded)
            if len(per_timestep) == 0:
                raise RuntimeError("No camera frames found close to any of the lidar timestamps")
            target_t = (
                self.config.pad_to_num_timesteps if self.config.pad_to_num_timesteps > 0 else len(lidar_timestamps)
            )
            if len(per_timestep) < target_t:
                # Fewer timesteps than the target: the sample is near a scene edge (fewer offset
                # lidar frames available) and/or no camera had a close-enough frame for some
                # timestep. Duplicate random kept timesteps to restore a constant T; duplicates
                # carry their own timestamps and calibrations, so supervision stays consistent.
                extra = np.random.choice(len(per_timestep), target_t - len(per_timestep))
                per_timestep += [per_timestep[i] for i in extra]
            mask_pad_value = float(self.label_mapper.noise_idx) if self.label_mapper is not None else 0.0
            images, masks, scores, depths, closest_timestamps, calibrations = concat_camera_frames(
                per_timestep, mask_pad_value
            )
        else:
            # Select only one camera per sample for efficiency
            camera_name = np.random.choice(self.cameras, p=self.sample_probability_per_cam)
            loaded = self._load_camera_frames(
                camera_name,
                lidar_timestamps,
                root_folder=root_folder,
                cameras=cameras,
                cameras_timestamps=cameras_timestamps,
                cameras_calibrations=cameras_calibrations,
                cameras_tokens=cameras_tokens,
                scene_name=scene_name,
                is_train=is_train,
            )
            if loaded is None:
                raise RuntimeError("No camera frames found close to any of the lidar timestamps")
            images, masks, scores, depths, closest_timestamps, calibrations = loaded

        # Reconstruct images with properly downsampled masks and depths
        contents = []
        stack = []
        if self.config.read_grounding_sam:
            contents.append(["mask"])
            stack.append(masks)
            del masks

        if self.config.read_grounding_sam_scores and is_train:
            contents.append(["score"])
            stack.append(scores)
            del scores

        if self.config.read_metric3d_depth:
            contents.append(["depth"])
            stack.append(depths)
            del depths

        if len(stack) > 0:
            stack = np.concatenate(stack, axis=1)
            contents = np.concatenate(contents, axis=0)

        # Randomly crop the images if needed
        stack, calibrations = random_crop_images_and_intrinsics(stack, calibrations, self.config.image_final_size)

        # Get the relative timestamps
        closest_timestamps_relative = get_relative_timestamps(closest_timestamps, current_timestamp)

        # Change the intrinsics to match the feature extraction
        intrinsics = get_intrinsics_for_feature_extraction(
            calibrations["intrinsics"], self.config.extracted_features_downsample_factor
        )

        # Final extraction before output:
        if self.config.read_grounding_sam:
            masks = stack[:, contents == "mask", :, :]
            if self.config.read_grounding_sam_scores and is_train:
                scores = stack[:, contents == "score", :, :]
        if self.config.read_metric3d_depth:
            depths = stack[:, contents == "depth", :, :]

        p_then = ego_motion.get_poses(closest_timestamps_relative + current_timestamp)
        T_cam_then = inverse_np(calibrations["extrinsics"].copy())  # 4 x 4, or t x 4 x 4 when per-timestep
        if T_cam_then.ndim == 2:
            T_cam_then = np.expand_dims(T_cam_then, axis=0)
        T_then_nows = inverse_np(p_then) @ bev_oxts_pose  # t x 4 x 4
        T_cam_ego = T_cam_then @ T_then_nows  # t x 4 x 4

        out = {
            "vfm_timestamps": closest_timestamps_relative,
            "vfm_extrinsics": calibrations["extrinsics"],
            "vfm_extrinsics_inverse": calibrations["extrinsics_inverse"],
            "vfm_distortion": calibrations["distortion"],
            "vfm_intrinsics": intrinsics,
            "rgb_intrinsics": calibrations["intrinsics"],
            "rgb_mean": np.array(RGB_MEAN, dtype=np.float32),
            "rgb_std": np.array(RGB_STD, dtype=np.float32),
            "vfm_T_cam_ego": T_cam_ego,
        }

        # Add noise to the depths and semantic labels if needed and training
        if self.config.add_noise_depths and depths is not None and is_train:
            depths = self.add_noise_depths(depths)

        # Add noise to the semantic labels if needed and training
        if self.config.add_noise_semantic_labels and masks is not None and is_train:
            masks = self.add_noise_semantic_labels(masks)

        # Add noise to the grounding sam scores if needed and training: Not implemented yet.

        if self.config.create_pseudo_pointclouds:
            pseudo_pointclouds = self.create_pseudo_pointclouds(
                calibrations["intrinsics"],
                calibrations["extrinsics"],
                T_cam_ego,
                images,
                closest_timestamps,
                depths,
                masks,
                scores,
                ego_motion,
                bev_oxts_pose,
                current_timestamp,
            )
            out.update(pseudo_pointclouds)
            if pseudo_pointclouds["pseudo_pointcloud"].shape[0] == 0:
                print(
                    "WARNING: Empty pseudo_pointcloud produced for "
                    f"scene={scene_name}, camera={camera_name}, "
                    f"timestamps={np.asarray(closest_timestamps).tolist()}",
                    flush=True,
                )

        if self.config.read_grounding_sam:
            # int8: class indices (< 20 classes) plus the -1 invalid sentinel. Quarters the
            # collate/H2D payload vs float32 (~10 MPix per sample); consumers cast on the GPU.
            out["vfm_masks"] = masks.astype(np.int8)
        if self.config.read_metric3d_depth:
            # fp16 is exact to ~0.1% for depths < 160 m, far below the supervision noise floor;
            # halves the payload, consumers cast on the GPU.
            out["vfm_depths"] = depths.astype(np.float16)

        return out

    def add_noise_depths(self, depths: np.ndarray) -> np.ndarray:
        """
        Add noise to the depths in inverse depth space.
        Models the quadratic increase in error relative to distance.
        """
        if not self.config.add_noise_depths or depths is None:
            return depths

        # 1. Convert to inverse depth (1/d), avoiding division by zero
        # 1e-3 is a safe epsilon for depths up to 1000m
        inv_depths = 1.0 / np.clip(depths, a_min=1e-3, a_max=None)

        # 2. Add Gaussian noise in inverse space
        noise = np.random.normal(loc=0.0, scale=self.config.noise_depths_std, size=inv_depths.shape).astype(np.float32)

        inv_depths_noisy = inv_depths + noise

        # 3. Clip to ensure inverse depth stays positive (avoid points flipping behind camera)
        # 0.01 inverse depth corresponds to a 100m max range limit
        inv_depths_noisy = np.clip(inv_depths_noisy, a_min=1e-3, a_max=None)

        # 4. Convert back to metric depth
        return 1.0 / inv_depths_noisy

    def add_noise_semantic_labels(self, masks: np.ndarray) -> np.ndarray:
        """
        Add noise to semantic labels via random label flipping.
        Models classification uncertainty/noise from Grounding-SAM.
        """
        if not self.config.add_noise_semantic_labels or masks is None or self.label_mapper is None:
            return masks

        ratio = self.config.noise_semantic_labels_ratio

        # Create a boolean mask of which pixels to 'corrupt'
        corruption_mask = np.random.random(masks.shape) < ratio

        # Determine total classes (adjust based on your label_mapper structure)
        num_classes = len(self.label_mapper.learning_classes)

        # Generate random valid labels
        random_labels = np.random.randint(0, num_classes, size=masks.shape)

        # Apply corruption: overwrite only selected pixels
        masks[corruption_mask] = random_labels[corruption_mask]

        return masks

    def create_pseudo_pointclouds(
        self,
        intrinsics,
        extrinsics,
        T_cam_ego,
        images,
        timestamps,
        depths,
        vfm_masks,
        vfm_scores,
        ego_motion,
        bev_oxts_pose,
        current_timestamp,
    ) -> np.ndarray:
        """Create the pseudo pointclouds."""
        # Get the intrinsics and extrinsics

        intrinsics_inverse = torch.inverse(torch.from_numpy(intrinsics))  # (3, 3), or (t, 3, 3) when per-timestep

        # Loop invariants hoisted out of the per-timestep loop below.
        T_ego_cam = inverse_np(T_cam_ego)  # (T, 4, 4); was re-inverting the whole stack every timestep
        min_range = np.array(self.point_cloud_range[:3])
        max_range = np.array(self.point_cloud_range[3:])

        # np.random.Generator's without-replacement sampler is ~7x faster than the legacy
        # np.random.choice(replace=False) used below for the (default, uniform-weight) point subsample,
        # which the profile showed was the single largest line (~27%). Seed it from the legacy global RNG
        # so reproducibility still follows the per-worker seed (and workers stay decorrelated). Same uniform
        # k-subset distribution as before -> quality-neutral.
        rng = np.random.default_rng(int(np.random.randint(0, 2**32)))

        n_timesteps = len(timestamps)
        # Create the pointclouds
        pointclouds = []
        missing_rays = []
        thw_image_coords = []
        pointcloud_t_idx = []
        missing_rays_t_idx = []
        if self.config.pseudo_pointclouds_labels:
            pointclouds_labels = []
            pointcloud_label_valids = []

        for i in range(n_timesteps):
            depth = depths[i].squeeze(0)
            timestamp = timestamps[i] - current_timestamp

            # Sky-as-free-rays: the grounding-SAM reader tagged sky pixels with free_idx. Push them past the
            # point-cloud range so they survive the noise filter and become free (missing-ray) supervision
            # instead of being dropped (baseline) or lifted as mis-depth occupied points. depth is a view of
            # the cached array, so copy before mutating.
            sky_free_mask = None
            if self.config.sky_as_free_rays and self.label_mapper is not None and vfm_masks is not None:
                sky_free_mask = vfm_masks[i].squeeze(0) == self.label_mapper.free_idx
                if sky_free_mask.any():
                    depth = depth.copy()
                    depth[sky_free_mask] = SKY_FREE_RAY_DEPTH
                else:
                    sky_free_mask = None

            mask = depth > 0
            if self.config.remove_sky_points_pseudo_pointclouds:
                keep_in_range = depth < 160
                if sky_free_mask is not None:
                    keep_in_range = keep_in_range | sky_free_mask  # keep forced sky rays despite the depth cut
                mask = mask & keep_in_range

            intr_inv = intrinsics_inverse if intrinsics_inverse.dim() == 2 else intrinsics_inverse[i]

            cap = self.config.pseudo_pointclouds_max_points
            width = mask.shape[1]
            uniform_subsample = self.config.sample_points_based_on_label_frequency == "none"

            # cam_coords (the pixel->camera lift) is intentionally deferred until AFTER the max-points
            # subsample below, so the per-point transforms run on at most `cap` points, not the full
            # (~1M-pixel) depth map. Going further: the output keeps at most `cap` points per timestep,
            # but typically ~1.1M pixels are in range — the exact path gathers labels/depth/coords for
            # ALL of them and only then subsamples, i.e. it touches ~1.1M-element arrays repeatedly to
            # keep 30k. When the subsample is uniform (the default
            # sample_points_based_on_label_frequency=="none"), we instead subsample the candidate *pixel
            # indices* first and gather only for the kept points, cutting per-pixel memory traffic ~20x
            # (the dominant cost, and worse under many concurrent dataloader workers contending for memory
            # bandwidth). We oversample to absorb the noise-filter drop and fall back to the exact path on
            # the rare draw with too much local noise.
            fast_done = False
            if uniform_subsample:
                flat = np.flatnonzero(mask)  # in-range candidate pixels, flat (C-order) indices
                n_cand = len(flat)
                if n_cand == 0:
                    continue
                if n_cand > cap:
                    # rng.choice(replace=False) returns a uniformly random subset in random order, so the
                    # post-noise-filter head [:cap] is a uniform random cap-subset of the non-noise
                    # candidates — same distribution as the exact path (noise-filter then uniform subsample).
                    over = min(n_cand, int(np.ceil(cap * SUBSAMPLE_OVERSAMPLE)))
                    cand = flat[rng.choice(n_cand, size=over, replace=False)]
                    vfm_labels = vfm_masks[i].squeeze(0).reshape(-1)[cand]
                    if self.config.mask_out_noise_pseudo_pointclouds:
                        good = vfm_labels != self.label_mapper.noise_idx
                        cand = cand[good]
                        vfm_labels = vfm_labels[good]
                    if len(cand) >= cap:
                        cand = cand[:cap]
                        vfm_labels = vfm_labels[:cap]
                        h_coords = cand // width
                        w_coords = cand - h_coords * width
                        depth = depth.reshape(-1)[cand]
                        fast_done = True
                    # else: too much local noise in the draw -> exact fallback below

            if not fast_done:
                # Exact path. h_coords/w_coords come from np.nonzero in C order — bit-identical to the old
                # row-major [row, col, 1] pixel-grid indexing, but skips materializing and
                # float64->long-casting the full (~1M-pixel) pixel grid.
                h_coords, w_coords = np.nonzero(mask)
                if len(h_coords) == 0:
                    continue
                depth = depth[mask]
                vfm_labels = vfm_masks[i].squeeze(0)[h_coords, w_coords]

                if self.config.mask_out_noise_pseudo_pointclouds:
                    keep = vfm_labels != self.label_mapper.noise_idx
                    h_coords = h_coords[keep]
                    w_coords = w_coords[keep]
                    depth = depth[keep]
                    vfm_labels = vfm_labels[keep]

                if self.config.sample_points_based_on_label_frequency == "class_frequency":
                    num_points = self.label_mapper.learning_to_num_points(vfm_labels.astype(np.int32))
                    frequencies = torch.from_numpy(num_points)
                    log_weights = torch.log(frequencies.sum() / frequencies)
                    semseg_weights = log_weights**self.config.sample_points_based_on_label_frequency_alpha
                    semseg_weights = (log_weights.numel() / semseg_weights.sum()) * semseg_weights
                    weights = semseg_weights.numpy()
                elif self.config.sample_points_based_on_label_frequency == "class_frequency_pseudo":
                    num_points = self.label_mapper.learning_to_num_points_pseudo(vfm_labels.astype(np.int32))
                    frequencies = torch.from_numpy(num_points)
                    log_weights = torch.log(frequencies.sum() / frequencies)
                    semseg_weights = log_weights**self.config.sample_points_based_on_label_frequency_alpha
                    semseg_weights = (log_weights.numel() / semseg_weights.sum()) * semseg_weights
                    weights = semseg_weights.numpy()
                else:
                    weights = None  # uniform

                # Subsample to the point cap *before* the expensive lift/transform below.
                if len(h_coords) > cap:
                    if weights is None:
                        # Uniform subsample without replacement via the ~7x-faster Generator sampler.
                        indices = rng.choice(len(h_coords), size=cap, replace=False)
                    else:
                        indices = np.random.choice(len(h_coords), size=cap, replace=False, p=weights / weights.sum())
                    h_coords = h_coords[indices]
                    w_coords = w_coords[indices]
                    depth = depth[indices]
                    vfm_labels = vfm_labels[indices]

            # Lift the subsampled pixels to camera coordinates: intr_inv @ [col, row, 1] * depth.
            # h_coords/w_coords are numpy int64; bridge to torch for the matmul.
            w_t = torch.from_numpy(w_coords)
            h_t = torch.from_numpy(h_coords)
            homog = torch.stack((w_t, h_t, torch.ones_like(w_t)), dim=-1).to(intr_inv.dtype)
            cam_coords = ((intr_inv[None, :, :] @ homog[..., None]).squeeze(-1) * depth[:, None]).numpy()

            valid_sem_mask = None
            if vfm_scores is not None:
                vfm_score = vfm_scores[i].squeeze(0)
                score_values = np.asarray(vfm_score[h_coords, w_coords]).reshape(-1)
                valid_sem_mask = score_values >= self.config.score_threshold
                if DELETE_STUFF:
                    del score_values, vfm_score

            pointcloud = np.concatenate(
                (cam_coords, np.ones(cam_coords.shape[0], dtype=np.float32)[:, None] * timestamp), axis=-1
            )

            rotations = T_ego_cam[i, :3, :3]
            translations = T_ego_cam[i, :3, 3]
            xyz = (cam_coords[..., None, :3] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)
            in_range = np.all((min_range <= xyz) & (xyz <= max_range), axis=-1)

            missing_rays.append(pointcloud[~in_range])
            pointclouds.append(pointcloud[in_range])
            pointcloud_t_idx.append(np.full(int(in_range.sum()), i, dtype=np.int64))
            missing_rays_t_idx.append(np.full(int((~in_range).sum()), i, dtype=np.int64))

            h_coords = h_coords[in_range]
            w_coords = w_coords[in_range]
            t_coords = np.ones_like(h_coords) * i
            thw_coords = np.concatenate((t_coords[:, None], h_coords[:, None], w_coords[:, None]), axis=-1)
            thw_image_coords.append(thw_coords)

            if self.config.pseudo_pointclouds_labels:
                pointclouds_labels.append(vfm_labels[in_range])
                if vfm_scores is not None:
                    pointcloud_label_valids.append(valid_sem_mask[in_range])

        sliced_ego_motion = self.get_sliced_ego_motion(bev_oxts_pose, ego_motion, timestamps, current_timestamp)

        # Initialize all with defaults
        pointclouds_out = np.empty((0, 4), dtype=np.float32)
        missing_rays_out = np.empty((0, 4), dtype=np.float32)
        pointclouds_labels_out = np.empty((0), dtype=np.int32)
        ratio_inside = 0.0
        thw_image_coords_out = np.empty((0, 3), dtype=np.int64)
        pointcloud_t_idx_out = np.empty((0), dtype=np.int64)
        missing_rays_t_idx_out = np.empty((0), dtype=np.int64)

        # Concatenate if we collected data
        if len(pointclouds) > 0:
            pointclouds_out = np.concatenate(pointclouds, axis=0).astype(np.float32)
            missing_rays_out = np.concatenate(missing_rays, axis=0).astype(np.float32)
            total_points = len(pointclouds_out) + len(missing_rays_out)
            if total_points > 0:
                ratio_inside = len(pointclouds_out) / total_points
            if self.config.pseudo_pointclouds_labels:
                pointclouds_labels_out = np.concatenate(pointclouds_labels, axis=0)
                if vfm_scores is not None:
                    pointcloud_label_valids = np.concatenate(pointcloud_label_valids, axis=0).astype(bool)
            thw_image_coords_out = np.concatenate(thw_image_coords, axis=0).astype(np.int64)
            pointcloud_t_idx_out = np.concatenate(pointcloud_t_idx, axis=0)
            missing_rays_t_idx_out = np.concatenate(missing_rays_t_idx, axis=0)

        pseudo_supervision_valid = len(pointclouds_out) > 0
        if self.config.pad_pseudo_data_for_batching:
            # Pad/cap every variable-length output to a fixed size so samples can be stacked at
            # batch_size > 1. Note that ratio_inside is computed from the real counts above, so
            # the point/ray balance in the occupancy sampler is unaffected by the padding.
            if not pseudo_supervision_valid:
                # The full-data preflight blacklists samples with no usable raw pixels, but a
                # particular random temporal/camera draw can still be empty. Keep the batch
                # collatable with finite non-degenerate dummy geometry and mark the entire
                # sample invalid; training_step then removes this rank's pseudo loss.
                num_points = self.config.n_max_pseudo_points
                num_rays = self.config.n_max_pseudo_missing_rays
                if num_points < 1 or num_rays < 1:
                    raise ValueError("padded pseudo data requires positive point and ray capacities")
                pointclouds_out = np.zeros((num_points, 4), dtype=np.float32)
                pointclouds_out[:, 2] = 1.0
                missing_rays_out = np.zeros((num_rays, 4), dtype=np.float32)
                missing_rays_out[:, 2] = 1.0
                thw_image_coords_out = np.zeros((num_points, 3), dtype=np.int64)
                pointcloud_t_idx_out = np.zeros(num_points, dtype=np.int64)
                missing_rays_t_idx_out = np.zeros(num_rays, dtype=np.int64)
                noise_idx = self.label_mapper.noise_idx if self.label_mapper is not None else 0
                pointclouds_labels_out = np.full(num_points, noise_idx, dtype=np.int32)
                pointcloud_label_valids = np.zeros(num_points, dtype=bool)
                ratio_inside = 1.0
            else:
                idx = resize_first_dim_indices(len(pointclouds_out), self.config.n_max_pseudo_points)
                if idx is not None:
                    pointclouds_out = pointclouds_out[idx]
                    thw_image_coords_out = thw_image_coords_out[idx]
                    pointcloud_t_idx_out = pointcloud_t_idx_out[idx]
                    if self.config.pseudo_pointclouds_labels:
                        pointclouds_labels_out = pointclouds_labels_out[idx]
                        if vfm_scores is not None:
                            pointcloud_label_valids = pointcloud_label_valids[idx]

                if len(missing_rays_out) > 0:
                    ray_idx = resize_first_dim_indices(len(missing_rays_out), self.config.n_max_pseudo_missing_rays)
                    if ray_idx is not None:
                        missing_rays_out = missing_rays_out[ray_idx]
                        missing_rays_t_idx_out = missing_rays_t_idx_out[ray_idx]
                else:
                    # Use copies of valid in-range points instead of zero rays, whose
                    # degenerate direction would produce NaNs in ray-box intersection.
                    fill = resize_first_dim_indices(len(pointclouds_out), self.config.n_max_pseudo_missing_rays)
                    fill = np.arange(len(pointclouds_out)) if fill is None else fill
                    missing_rays_out = pointclouds_out[fill].astype(np.float32)
                    missing_rays_t_idx_out = pointcloud_t_idx_out[fill]

            num_poses = sliced_ego_motion.timestamps.shape[0]
            n_max_poses = self.config.n_max_pseudo_ego_poses
            if num_poses > n_max_poses:
                # Uniform thinning is safe: the poses are only used for interpolation downstream.
                keep = np.linspace(0, num_poses - 1, n_max_poses).astype(np.int64)
                sliced_ego_motion.poses = sliced_ego_motion.poses[keep]
                sliced_ego_motion.timestamps = sliced_ego_motion.timestamps[keep]
            elif num_poses < n_max_poses:
                num_pad = n_max_poses - num_poses
                sliced_ego_motion.poses = np.concatenate(
                    [sliced_ego_motion.poses, sliced_ego_motion.poses[-1:].copy().repeat(num_pad, axis=0)], axis=0
                )
                sliced_ego_motion.timestamps = np.concatenate(
                    [sliced_ego_motion.timestamps, sliced_ego_motion.timestamps[-1:].copy().repeat(num_pad, axis=0)],
                    axis=0,
                )

        pointclouds = pointclouds_out
        missing_rays = missing_rays_out
        pointclouds_labels = pointclouds_labels_out
        thw_image_coords = thw_image_coords_out

        out = {}
        out["pseudo_pointcloud"] = pointclouds
        out["pseudo_missing_rays"] = missing_rays
        out["pseudo_points_ratio_inside"] = ratio_inside
        out["pseudo_thw_image_coords"] = thw_image_coords
        out["pseudo_pointcloud_t_idx"] = pointcloud_t_idx_out
        out["pseudo_missing_rays_t_idx"] = missing_rays_t_idx_out
        out["pseudo_pointcloud_valids"] = np.ones((pointclouds.shape[0], 1), dtype=bool)
        out["pseudo_supervision_valid"] = np.asarray(pseudo_supervision_valid, dtype=bool)
        if self.config.pseudo_pointclouds_labels:
            out["pseudo_pointcloud_labels"] = np.expand_dims(pointclouds_labels, axis=-1)
            if vfm_scores is not None:
                out["pseudo_pointcloud_valids"] = np.expand_dims(pointcloud_label_valids, axis=-1)
        out["pseudo_pointcloud_extrinsics"] = extrinsics.copy()
        out["pseudo_pointcloud_ego_poses"] = sliced_ego_motion.poses.astype(np.float64)
        out["pseudo_pointcloud_ego_timestamps"] = sliced_ego_motion.timestamps.astype(np.float32)

        return out

    def get_sliced_ego_motion(self, bev_oxts_pose, ego_motion, timestamps, current_timestamp) -> EgoMotion:
        sweep_eps = 0.2  # 0.2 seconds, sweep is roughly 0.1 seconds so this has margin
        max_timestamp = max(list(timestamps) + [current_timestamp]) + sweep_eps
        min_timestamp = min(list(timestamps) + [current_timestamp]) - sweep_eps
        lower_mask = min_timestamp < ego_motion.timestamps
        upper_mask = ego_motion.timestamps < max_timestamp
        oxts_range_mask = lower_mask & upper_mask
        sliced_ego_motion = EgoMotion(  # dummy for use with compensation code
            ego_motion.poses[oxts_range_mask],
            None,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
            ego_motion.timestamps[oxts_range_mask],
            None,  # type: ignore
        )
        sliced_ego_motion.timestamps -= current_timestamp
        sliced_ego_motion.poses = inverse_np(bev_oxts_pose) @ sliced_ego_motion.poses
        return sliced_ego_motion
