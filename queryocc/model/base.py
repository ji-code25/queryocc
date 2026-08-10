from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import lightning as L
import torch
from torch.optim.lr_scheduler import OneCycleLR

from queryocc.common.types import PointCloudRange, VoxelSize
from queryocc.config.base_config import InstantiateConfig
from queryocc.config.logging_configs import LoggingConfig
from queryocc.data.augmentations.augmentation_configs import AugmentationConfig
from queryocc.data.datasets.zod_base import ZODBaseDatasetConfig
from queryocc.model.components.net import NetConfig, QueryOccNet
from queryocc.optimizer.cAdam import CautiousAdamW


def warmup_dataloader(dl, num_batches=5):
    it = iter(dl)
    for _ in range(num_batches):
        try:
            next(it)
        except StopIteration:
            break
    # Release the warmup iterator explicitly: with persistent_workers the worker pool is shared
    # with the training iterator created right after, and a lingering half-consumed iterator
    # perturbs how many batches the epoch yields.
    del it


DEFAULT_PCD_RANGE_UNO = (-100.04, -100, -2, 150.04, 100, 6)
DEFAULT_PCD_RANGE_FULL = (-69.12, -69.12, -2, 138.24, 69.12, 6)

DEFAULT_PCD_RANGE_FWD = (-2.58, -69.12, -2, 138.24, 69.12, 6)
DEFAULT_PCD_RANGE_FWD_NARROW = (-2.58, -17.92, -2, 138.24, 17.92, 6)

DEFAULT_PCD_RANGE_SURROUND = (-69.12, -69.12, -2, 69.12, 69.12, 6)
DEFAULT_PCD_RANGE_SURROUND_LARGE = (-58.88, -58.88, -2, 58.88, 58.88, 6)
DEFAULT_PCD_RANGE_SURROUND_BASE = (-51.20, -51.20, -2, 51.20, 51.20, 9)
DEFAULT_PCD_RANGE_SURROUND_LAGOM = (-40.96, -40.96, -2, 40.96, 40.96, 6)
DEFAULT_PCD_RANGE_SURROUND_MINI = (-34.56, -34.56, -2, 34.56, 34.56, 6)
DEFAULT_PCD_RANGE_SURROUND_MICRO = (-17.92, -17.92, -2, 17.92, 17.92, 6)
# Note that the range must be divisible by voxel_size*8 (eg. 0.16*8) and give an even number
# Example (69.12*2)/(0.16*8) = 1728 => OK!

def pcd_range_defaults(defaults: str):
    if defaults == "uno":
        return DEFAULT_PCD_RANGE_UNO
    elif defaults == "full":
        return DEFAULT_PCD_RANGE_FULL
    elif defaults == "fwd":
        return DEFAULT_PCD_RANGE_FWD
    elif defaults == "fwd_narrow":
        return DEFAULT_PCD_RANGE_FWD_NARROW
    elif defaults == "camonly":
        return DEFAULT_PCD_RANGE_FWD
    elif defaults == "surround":
        return DEFAULT_PCD_RANGE_SURROUND
    elif defaults == "surround_large":
        return DEFAULT_PCD_RANGE_SURROUND_LARGE
    elif defaults == "surround_base":
        return DEFAULT_PCD_RANGE_SURROUND_BASE
    elif defaults == "surround_lagom":
        return DEFAULT_PCD_RANGE_SURROUND_LAGOM
    elif defaults == "surround_mini":
        return DEFAULT_PCD_RANGE_SURROUND_MINI
    elif defaults == "surround_micro":
        return DEFAULT_PCD_RANGE_SURROUND_MICRO
    else:
        raise ValueError(f"Invalid pcd range defaults: {defaults}")


def lifter_canvas_size(pc_range, voxel_size) -> Tuple[int, int, int]:
    x_range = pc_range[3] - pc_range[0]
    y_range = pc_range[4] - pc_range[1]
    canvas_x = int(x_range / (voxel_size[0] * 2))
    canvas_y = int(y_range / (voxel_size[1] * 2))
    canvas_z = int(6)
    return (canvas_x, canvas_y, canvas_z)


def bev_features_canvas_size(pc_range, voxel_size) -> Tuple[int, int]:
    x_range = pc_range[3] - pc_range[0]
    y_range = pc_range[4] - pc_range[1]
    canvas_x = int(x_range / (voxel_size[0] * 2) / 2)
    canvas_y = int(y_range / (voxel_size[1] * 2) / 2)
    return (canvas_x, canvas_y)


@dataclass
class ModelConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: BaseModel)

    point_cloud_range: PointCloudRange = DEFAULT_PCD_RANGE_FULL
    """The range of the point cloud (x_min, y_min, z_min, x_max, y_max, z_max)."""
    voxel_size: VoxelSize = (0.16, 0.16, 8.0)
    """The size of the voxels in the point cloud (x, y, z)."""
    roi: str = "surround"  # TODO: make enum
    """The default ROI to use for the model."""

    # core network settings
    net: NetConfig = field(default_factory=NetConfig)
    """The QueryOccNet (encoders + fusion + decoder heads) configuration."""

    # augmentation settings
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    """The augmentation configuration."""

    # optimizer settings
    max_lr: float = 5e-5
    pct_start: float = 0.02
    div_factor: float = 10
    final_div_factor: float = 1e4
    optimizer_cls: Literal["adam", "adamw", "cadam"] = "adamw"
    weight_decay: float = 1e-4
    fused_optimizer: bool = False
    """Run the Adam/AdamW update as fused multi-tensor kernels (same algorithm, fewer launches). Ignored for cadam.
    Incompatible with the default trainer.gradient_clip_val > 0 under mixed precision: Lightning's AMP plugin
    rejects fused optimizers there because they unscale gradients inside .step()."""
    image_backbone_reduce_lr_factor: float = 100.0

    # logging
    compute_train_metrics: bool = False
    """ Whether to compute train metrics """
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    _min_query_time: float = -3.0
    """ The minimum query time """
    _max_query_time: float = 3.0
    """ The maximum query time """

    def __post_init__(self) -> None:
        self.point_cloud_range = pcd_range_defaults(self.roi)
        self.net.lifter_canvas_size = lifter_canvas_size(self.point_cloud_range, self.voxel_size)
        self.net.bev_features_canvas_size = bev_features_canvas_size(self.point_cloud_range, self.voxel_size)

    def configure_dataset(self, dataset: ZODBaseDatasetConfig):  # pragma: no cover
        dataset.point_cloud_range = self.point_cloud_range
        if self.net.image_encoder:
            dataset.load_image = True
        if self.net.lidar_encoder:
            dataset.load_lidar = True


class BaseModel(L.LightningModule):
    config: ModelConfig

    def __init__(self, config: ModelConfig, visualization_mode: bool = False):
        super().__init__()
        self.config = config
        self.visualization_mode = visualization_mode
        self.net: QueryOccNet = config.net.setup(
            point_cloud_range=config.point_cloud_range,
            voxel_size=config.voxel_size[:2],
            logging_config=config.logging,
            augmentation_config=config.augmentations,
            min_query_time=config._min_query_time,
            max_query_time=config._max_query_time,
        )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Checkpoints saved before the dos->net rename store the core network under "dos.*".
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for key in [k for k in sd if k.startswith("dos.")]:
                sd["net." + key[len("dos.") :]] = sd.pop(key)

    @property
    def _compute_metrics(self):
        if self.config.compute_train_metrics and self.training:
            return True
        elif not self.training:
            return True
        else:
            return False

    def on_train_start(self) -> None:
        # Every rank warms its own dataloader (each rank holds its own sampler shard, so there is
        # no duplicate I/O). This MUST be rank-symmetric: warming only rank 0 perturbed its
        # persistent-worker iterator state, so rank 0's epoch yielded a different number of
        # batches than the other ranks — the ranks then crossed the epoch/val boundary at
        # different steps and deadlocked in mismatched collectives (the NCCL watchdog kills the
        # run at the first boundary, reproduced deterministically on multiple clusters).
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n🧊 Warming up dataloaders.\n")
        warmup_dataloader(self.trainer.train_dataloader)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("🔥 Warmed up dataloaders.\n")

        # Synchronize all ranks before training starts
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def configure_optimizers(self):
        if self.config.optimizer_cls not in ("adam", "cadam", "adamw"):
            raise ValueError(f"Invalid optimizer class: {self.config.optimizer_cls}")

        if self.config.optimizer_cls == "cadam":
            optimizer_cls = CautiousAdamW
        elif self.config.optimizer_cls == "adam":
            optimizer_cls = torch.optim.Adam  # type: ignore
        elif self.config.optimizer_cls == "adamw":
            optimizer_cls = torch.optim.AdamW  # type: ignore
        else:
            raise ValueError(f"Invalid optimizer class: {self.config.optimizer_cls}")

        main_lr = self.config.max_lr
        image_backbone_lr = self.config.max_lr / self.config.image_backbone_reduce_lr_factor

        # Get the parameters for the early encoder
        try:
            image_backbone_params = list(self.net.image_encoder.image_encoder.backbone.parameters())  # type: ignore
            image_backbone_param_ids = set(map(id, image_backbone_params))
        except AttributeError:
            print(
                "Not using early_image_encoder: self.net.image_encoder.early_image_encoder not found or has no parameters."
            )
            image_backbone_params = []
            image_backbone_param_ids = set()

        # Get the parameters for the main encoder
        other_params = []
        for param in self.parameters():
            if id(param) not in image_backbone_param_ids:
                other_params.append(param)

        # Create the parameter groups list
        param_groups = []
        if image_backbone_params:  # Only add group if it has parameters
            param_groups.append({"params": image_backbone_params, "lr": image_backbone_lr})
            print(
                f"Using early_image_encoder: Applying reduced LR ({image_backbone_lr:.2e}) to {len(image_backbone_params)} params in early_image_encoder."
            )
        else:
            print("Not using early_image_encoder: No parameters found for early_image_encoder group.")

        if other_params:  # Only add group if it has parameters
            param_groups.append({"params": other_params, "lr": main_lr})
            print(f"Using main encoder: Applying main LR ({main_lr:.2e}) to {len(other_params)} other params.")
        else:
            print("Using main encoder: No parameters found for the 'other' group.")

        if not param_groups:
            raise ValueError("No parameters found to optimize!")

        # Create the optimizer
        if self.config.optimizer_cls == "cadam":
            optimizer = optimizer_cls(param_groups, weight_decay=self.config.weight_decay)
        else:
            optimizer = optimizer_cls(
                param_groups, weight_decay=self.config.weight_decay, fused=self.config.fused_optimizer
            )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=[image_backbone_lr, main_lr] if image_backbone_params else main_lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=self.config.pct_start,
            div_factor=self.config.div_factor,
            anneal_strategy="cos",
            final_div_factor=self.config.final_div_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        if hasattr(self.trainer.datamodule, "set_epoch_for_sampler"):  # type: ignore
            self.trainer.datamodule.set_epoch_for_sampler(self.current_epoch)  # type: ignore

    #### Useful shared functions ####

    def _get_lidar_data(self, batch: dict) -> Optional[Dict[str, torch.Tensor]]:
        if "lidars" in batch:
            lidar_data = {
                "pointcloud": batch["lidars"],
            }
            return lidar_data
        else:
            return {}

    def _get_camera_data(self, batch: dict) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        # key -> cam-name -> value (e.g., "images" -> "front" -> torch.Tensor)
        if "imgs" in batch:
            # Images travel as uint8 through the dataloader (4x less collate, pinning, and
            # H2D traffic than float32); convert to float in [0, 1] here, on the device.
            images = {k: v.float().div_(255) if v.dtype == torch.uint8 else v for k, v in batch["imgs"].items()}
            camera_data = {
                "images": images,
                "extrinsics": batch["extrinsics"],
                "intrinsics": batch["intrinsics"],
                "intrinsics_modified": batch["intrinsics_modified"],
                "distortion": batch["distortion"],
                "imgs_h": batch["imgs_h"],
                "imgs_w": batch["imgs_w"],
                "T_cam_ego": batch["T_cam_ego"],
                "timestamps": batch["imgs_timestamps"],
                "batch_size": len(batch["id"]),
                "num_cams": len(batch["imgs"]),
                "num_timesteps": batch["imgs_num_timesteps"][0],
                "depth_maps": batch.get("depth_maps", None),
                "depth_maps_valid": batch.get("depth_maps_valid", None),
            }
            return camera_data

        else:
            return {}
