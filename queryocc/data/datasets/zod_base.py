import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from torch.utils.data.dataloader import default_collate

from queryocc.common.types import PointCloudRange
from queryocc.data.datasets.nuscenes_mappings import DEFAULT_NUSCENES_CLASSES
from queryocc.data.vfm.metric3d_reader import Metric3DReaderConfig
from queryocc.data.vfm.vfm_reader import VFMReaderConfig

warnings.filterwarnings(action="ignore", category=UserWarning)


# Keys whose per-sample shape varied and were collated as a list (warned once each).
_WARNED_VARLEN_KEYS: set = set()


@staticmethod
def new_base_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    res = {}
    keys_to_pop = ("poses", "poses_timestamps")

    # Determine if we can safely collate lidars
    collate_lidar = "lidars" in batch[0] and all(batch[0]["lidars"].shape[1] == b["lidars"].shape[1] for b in batch)
    if not collate_lidar:
        keys_to_pop += ("lidars", "lidar_timestamps")

    # Determine if we can safely collate offset pointclouds
    collate_offset_pointclouds = "offset_pointclouds" in batch[0] and all(
        batch[0]["offset_pointclouds"].shape[1] == b["offset_pointclouds"].shape[1] for b in batch
    )
    if not collate_offset_pointclouds:
        keys_to_pop += ("offset_pointclouds", "offset_pointclouds_indices")

    # Copy the batch to avoid mutating original dicts
    batch_copy = [b.copy() for b in batch]

    # Extract and store the known variable-length fields manually
    for key in keys_to_pop:
        if key in batch_copy[0]:
            res[key] = [b.pop(key) for b in batch_copy]

    # Collate each remaining key independently: try to stack it, and fall back to a per-sample list
    # for anything default_collate can't handle — a tensor/numpy array of varying length, or a nested
    # dict/list containing one (e.g. the per-point / per-object offset-panoptic labels pulled in when
    # one_head.semantic_segmentation enables load_offset_panoptic_labels). The model consumes these as
    # per-sample lists, exactly like offset_pointclouds above. Per-key collation matches default_collate's
    # dict behaviour, so keys that stack cleanly are unchanged.
    for key in list(batch_copy[0].keys()):
        column = [b[key] for b in batch_copy]
        try:
            res[key] = default_collate(column)
        except (RuntimeError, TypeError):
            if key not in _WARNED_VARLEN_KEYS:
                _WARNED_VARLEN_KEYS.add(key)
                warnings.warn(
                    f"new_base_collate_fn: '{key}' could not be stacked (per-sample-varying shape); "
                    f"collating it as a list. Add it to keys_to_pop to make this explicit.",
                    stacklevel=2,
                )
            res[key] = column

    return res


@dataclass
class ZODBaseDatasetConfig:
    data: Path = Path("/datasets/zod/")
    """The path to the dataset."""
    version: Literal["full", "mini"] = "full"
    """What version of the dataset to use."""
    infos_tag: str = ""
    """Filename tag of the dataset index pkls written by generate_pseudo_labels.create_infos
    (<tag>_infos_{train,val}.pkl in QUERYOCC_INFOS_DIR). Empty picks bevdetv2-nuscenes, or
    bevdetv2-nuscenes-mini when version="mini"."""
    ids_to_load: List[int] = field(default_factory=lambda: [])
    """The sequence ids to load. If empty, all ids are loaded."""
    frames_to_load: List[str] = field(default_factory=lambda: [])
    """The frames within the sequences to load. If empty, all frames are loaded."""

    load_queryocc_infos_if_exists: bool = True
    """Whether to load the queryocc infos if they exist."""
    save_queryocc_infos: bool = True
    """Whether to save the queryocc infos."""

    make_batchable: bool = False
    """Whether to make the dataset batchable. If False, the dataset will only be able to run in single sample mode."""
    only_tokens: bool = True
    """Whether to only load the tokens as inputs. (Only applicable for nuscenes)"""
    only_tokens_for_offsets: bool = True
    """Whether to only load the tokens for the offsets as supervision. (Only applicable for nuscenes)"""
    upsample_key_frames: bool = True
    """Whether to upsample the key frames to the same frequency as the other frames."""
    key_frames_weight: float = 5.0
    """The weight of the key frames."""

    # camera
    load_image: bool = False
    """Whether to load the image data."""
    n_prev_images: int = 3
    """How many previous images to load. (In addition to the current image)"""
    prev_images_max_time: float = 3.0
    """The maximum time difference between the current image and the previous images. (seconds)"""
    n_future_images: int = -1
    """How many future images to load AS INPUT to the model. (In addition to the current image)"""
    future_images_max_time: float = 3.0
    """The maximum time difference between the current image and the future images. (seconds)"""

    image_resize_method: Literal["none", "resize", "downsample"] = "resize"
    """The resize method for the images."""
    image_size: Optional[Tuple[int, int]] = (518, 518)  # (1204, 3836)  # [h, w]
    """The size of the images to send to the model."""
    nuscenes_camera_downsample_factor: float = 2.0
    av2_camera_downsample_factor: float = 4.0
    """The downsample factor for the camera images."""
    pad_images_to_same_size: bool = True
    """Whether to pad the images to the same size."""
    pad_images_to_multiple_of: int = 16
    """The multiple to pad the images to. Only used if pad_images_to_same_size is larger than 0."""
    load_depth_maps: bool = True
    """Whether to load the depth maps."""
    depth_maps_source: Literal["metric3d", "lidar"] = "lidar"
    """The source of the depth maps."""
    depth_downsample_factor: float = 8.0
    """The downsample factor for the depth maps. Depends on the encoder."""
    metric3d_depth_reader: Metric3DReaderConfig = field(default_factory=Metric3DReaderConfig)
    """The metric3d depth reader config."""

    # lidar
    load_lidar: bool = False
    """Whether to load the lidar data."""
    n_prev_lidar: int = 3
    """How many lidar scans to load as input. (In addition to the current scan)."""
    prev_lidar_max_time: float = 3.0
    """The maximum time difference between the current lidar scan and the previous lidar scans. (seconds)"""
    n_future_lidar: int = -1
    """How many future lidar scans to load AS INPUT to the model. (In addition to the current scan)."""
    future_lidar_max_time: float = 3.0
    """The maximum time difference between the current lidar scan and the future lidar scans. (seconds)"""
    aggregate_lidar: bool = True
    """Whether to aggregate the lidar scans."""
    point_cloud_range: PointCloudRange = (0, 0, 0, 0, 0, 0)  # will be set by model
    """The range of the point cloud (x_min, y_min, z_min, x_max, y_max, z_max)."""

    # load full ego motion
    load_full_ego_motion: bool = False
    """Whether to load the full ego motion data."""

    # Offset frames settings for both occupancy and dino features
    num_future_frames_to_sample: int = 3
    """Number of future frames to sample. If 0 (or negative), no future frames will be sampled."""
    min_future_frame_s: float = 0.05
    """Minimum time in the future to sample num_future_frames_to_sample from."""
    max_future_frame_s: float = 3.0
    """Maximum time in the future to sample num_future_frames_to_sample from."""
    num_past_frames_to_sample: int = 0
    """Number of past frames to sample. If 0 (or negative), no past frames will be sampled."""
    min_past_frame_s: float = 0.05
    """Minimum time in the past to sample num_past_frames_to_sample from."""
    max_past_frame_s: float = 3.0
    """Maximum time in the past to sample num_past_frames_to_sample from."""
    always_add_current_frame_to_offsets: bool = True
    """Whether to always add the current frame to the offsets. """

    # lidar occupancy query sampler
    load_offset_pointclouds: bool = False
    """Whether to load the offset lidar scan to use for occupancy data."""
    n_max_offset_points_to_load_per_scan: int = -1
    """The maximum number of future points to load, for dataloader efficiency (-1 = no limit)."""
    n_max_missing_rays_to_load_per_scan: int = -1
    """The maximum number of missing rays to send to gpu (-1 = no limit)."""
    n_max_poses_to_load_per_scan: int = 650
    """The maximum number of poses to send to gpu (-1 = no limit)."""

    add_missing_points: bool = False
    """Whether to complete the future point clouds with missing points data."""
    missing_points_downsample_factor: float = -1.0
    """The downsample factor for the created missing points. -1.0 for no downsampling."""

    # VFM features projection without cached features
    load_vfm_stuff: bool = False
    """Whether to load the VFM features projection data."""
    vfm_cfg: VFMReaderConfig = field(default_factory=VFMReaderConfig)
    """The VFM features projection sampler configuration."""

    # OCC-3d (nuscenes only)
    load_occ3d_train: bool = False
    load_occ3d_val: bool = False
    """Whether to load the occ-3d data."""

    # Panoptic labels for query generation (nuscenes only)
    load_offset_panoptic_labels: bool = False
    """Whether to load the panoptic labels for offset point clouds."""
    use_pseudo_labels_for_panoptic: bool = True
    """Whether to use the pseudo labels for the panoptic labels."""
    load_current_lidar_panoptic_labels: bool = False
    """Whether to load the lidar panoptic labels for the current timestep."""
    panoptic_labels_classes: Tuple[str, ...] = DEFAULT_NUSCENES_CLASSES
    """The classes to use for the panoptic labels."""
    panoptic_pseudo_label_name: str = "delta_fused"
    """The name of the pseudo label to save."""

    # Augmentations
    rotation_augmentation: bool = True
    """Whether to apply rotation augmentations."""
    rotation_range: Tuple[float, float] = (-20, 20)
    """The range of the rotation augmentation. (degrees)"""
    translation_augmentation: bool = False
    """Whether to apply rotation and translation augmentations."""
    translation_range: Tuple[float, float] = (-1.0, 1.0)
    """The range of the translation augmentation. (meters)"""

    @property
    def n_max_offset_points(self) -> int:
        if self.n_max_offset_points_to_load_per_scan < 0:
            return self.n_max_offset_points_to_load_per_scan

        num_frames = 0
        if self.num_future_frames_to_sample > 0:
            num_frames += self.num_future_frames_to_sample
        if self.num_past_frames_to_sample > 0:
            num_frames += self.num_past_frames_to_sample

        return self.n_max_offset_points_to_load_per_scan * num_frames

    @property
    def n_max_missing_rays(self) -> int:
        if self.n_max_missing_rays_to_load_per_scan < 0:
            return self.n_max_missing_rays_to_load_per_scan

        num_frames = 0
        if self.num_future_frames_to_sample > 0:
            num_frames += self.num_future_frames_to_sample
        if self.num_past_frames_to_sample > 0:
            num_frames += self.num_past_frames_to_sample
        return self.n_max_missing_rays_to_load_per_scan * num_frames

    @property
    def n_max_poses(self) -> int:
        return self.n_max_poses_to_load_per_scan * (self.num_future_frames_to_sample + self.num_past_frames_to_sample)
