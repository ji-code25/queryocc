from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import einops
import torch
from torch import Tensor, nn
from torch.profiler import record_function
from torchvision.transforms.v2 import RandomPhotometricDistort

from queryocc.common.types import PointCloudRange, VoxelSize
from queryocc.config.base_config import InstantiateConfig
from queryocc.config.logging_configs import LoggingConfig
from queryocc.data.augmentations.augmentation_configs import AugmentationConfig
from queryocc.data.augmentations.augmentations import GridMaskGPU
from queryocc.model.components.checkpointing import checkpoint
from queryocc.model.components.contraction.point_contraction import PointContraction
from queryocc.model.components.image_encoders.backbones.backbone import BackboneEncoder
from queryocc.model.components.image_encoders.common import stack_dict_of_tensors as _stack_dict_of_tensors
from queryocc.model.components.image_encoders.lifters.lss import LSSProConfig


@dataclass
class CervezaImageEncoderConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: CervezaImageEncoder)
    """ The target class for instantiation. """
    image_encode_time: bool = False
    """ Whether to encode the time in the image encoder """
    image_encode_time_max_freq: float = 6.0
    """ The maximum frequency for the time encoding in the image encoder """
    image_encoder_backbone: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ] = "resnet50"
    """ The backbone to use for the image encoder. Neck input channels come from the backbone
    instance (backbone.out_channels), so any name build_backbone accepts works here. """
    image_encoder_neck_type: Literal["simpleup", "simplefpn"] = "simplefpn"
    """ The neck type to use for the image encoder """
    image_encoder_pretrained: bool = True
    """ Initialize the image backbone from its published weights (DINOv3 for convnext_*, ImageNet for
    resnet*). Training needs this. Evaluation from a QueryOcc checkpoint does not — every backbone
    weight is overwritten by the checkpoint — so train.py sets it False when loading one, which is what
    keeps the released weights usable without the gated DINOv3 download. """
    image_lifter: Literal["lss"] = "lss"
    """ The lifter to use for the image encoder. """
    image_encoder_detach_previous_inputs: bool = True
    """ Whether to detach the previous inputs in the image encoder """
    image_encoder_encode_time_fusion_type: Literal["sum", "concat", "film"] = "film"
    """ The type of fusion to use for the time encoding in the image encoder """
    image_encoder_initial_features: Literal["zeros", "sinusoidal"] = "zeros"
    """ The type of initial features to use for the image encoder """
    image_encoder_points_positional_encoding: Literal["none", "sinusoidal"] = "none"
    """ The type of positional encoding to use for the image encoder points """
    initial_features_normalization: Literal["none", "batch_norm"] = "none"
    """ The type of normalization to use for the image encoder initial features """
    initial_features_activation: Literal["none", "relu", "gelu", "leaky_relu"] = "gelu"
    """ The type of activation to use for the image encoder initial features """

    lss: LSSProConfig = field(default_factory=LSSProConfig)
    """ The configuration for the LSS encoder """


class CervezaImageEncoder(nn.Module):
    def __init__(
        self,
        config: CervezaImageEncoderConfig,
        feature_dim: int,
        point_cloud_range: PointCloudRange,
        voxel_size: VoxelSize,
        lifter_canvas_size: Tuple[int, int, int],
        compile: bool = False,
        gradient_checkpointing: bool = False,
        swap_batch_norm_to_instance_norm: bool = False,
        logging_config: Optional[LoggingConfig] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
        point_contraction: Optional[PointContraction] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.detach_previous_inputs = config.image_encoder_detach_previous_inputs
        self.supervise_depth = config.lss.supervise_depth
        self.gradient_checkpointing = gradient_checkpointing

        self.image_encoder = BackboneEncoder(
            backbone_type=config.image_encoder_backbone,
            neck_type=config.image_encoder_neck_type,
            feature_dim=feature_dim,
            encode_time=config.image_encode_time,
            time_encoding_max_freq=config.image_encode_time_max_freq,
            encode_time_fusion_type=config.image_encoder_encode_time_fusion_type,
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
            pretrained=config.image_encoder_pretrained,
        )

        if config.image_lifter == "lss":
            self.lifter = config.lss.setup(
                in_channels=feature_dim,
                out_channels=feature_dim,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                initial_features=config.image_encoder_initial_features,
                points_positional_encoding=config.image_encoder_points_positional_encoding,
                gradient_checkpointing=gradient_checkpointing,
                swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
                point_contraction=point_contraction,
            )
        else:
            raise ValueError(f"Invalid lifter: {config.image_lifter}")

        norm_layers = {
            "none": nn.Identity,
            "batch_norm": nn.InstanceNorm2d if swap_batch_norm_to_instance_norm else nn.BatchNorm2d,
        }
        if config.initial_features_normalization not in norm_layers:
            raise ValueError(f"Invalid initial features normalization: {config.initial_features_normalization}")
        Norm2d = norm_layers[config.initial_features_normalization]

        activations = {"none": nn.Identity, "relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}
        if config.initial_features_activation not in activations:
            raise ValueError(f"Invalid initial features activation: {config.initial_features_activation}")
        Activation = activations[config.initial_features_activation]

        self.init_bev_processing = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 2, padding=1),
            Norm2d(feature_dim),
            Activation(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, padding=1),
            Norm2d(feature_dim),
            Activation(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, padding=1),
            Norm2d(feature_dim),
            Activation(),
        )

        self.logging = logging_config
        self._image_sizes_checked = False

        if augmentation_config is not None:
            self.grid_mask = (
                GridMaskGPU(
                    use_h=augmentation_config.grid_mask_use_h,
                    use_w=augmentation_config.grid_mask_use_w,
                    rotate=augmentation_config.grid_mask_rotate,
                    offset=augmentation_config.grid_mask_offset,
                    ratio=augmentation_config.grid_mask_ratio,
                    mode=augmentation_config.grid_mask_mode,
                    prob=augmentation_config.grid_mask_prob,
                )
                if augmentation_config.grid_mask
                else None
            )

            self.photometric_distortion = RandomPhotometricDistort(
                brightness=augmentation_config.photometric_distortion_brightness,
                contrast=augmentation_config.photometric_distortion_contrast,
                saturation=augmentation_config.photometric_distortion_saturation,
                hue=augmentation_config.photometric_distortion_hue,
            )

        else:
            self.grid_mask = None
            self.photometric_distortion = None

        if compile:
            # Compile only the fixed-shape submodules. The lifter as a whole has
            # data-dependent ops (scatter_add splatting, optional torch.unique paths)
            # that fragment the graph, but its depth network is a plain conv stack.
            # In-place compile keeps state_dict keys free of the `_orig_mod.` prefix.
            self.image_encoder.compile(dynamic=False)
            self.init_bev_processing.compile(dynamic=False)
            if hasattr(self.lifter, "depth_network"):
                self.lifter.depth_network.compile(dynamic=False)

    def apply_augmentations(self, images: Tensor) -> Tensor:
        if self.training:
            if self.grid_mask is not None:
                images = self.grid_mask(images)
            if self.photometric_distortion is not None:
                images = self.photometric_distortion(images)

        return images

    @record_function("CervezaEncoder._encode_images")
    def _encode_images(
        self, images: Tensor, timestamps: Tensor, batch_size: int, num_cams: int, num_timesteps: int
    ) -> Tensor:
        stacked_images = einops.rearrange(_stack_dict_of_tensors(images), "n b t c h w -> b t n c h w")
        stacked_timestamps = einops.rearrange(_stack_dict_of_tensors(timestamps), "n b t -> b t n")

        def encode(imgs: Tensor, ts: Tensor, maybe_checkpoint: bool) -> Tensor:
            imgs = einops.rearrange(imgs, "b t n c h w -> (b t n) c h w")
            ts = einops.rearrange(ts, "b t n -> (b t n) 1")
            imgs = self.apply_augmentations(imgs)
            feats = self.image_encoder(imgs, ts, maybe_checkpoint=maybe_checkpoint)
            return einops.rearrange(feats, "(b t n) c h w -> b t n c h w", b=batch_size, n=num_cams)

        if not self.detach_previous_inputs:
            return encode(stacked_images, stacked_timestamps, maybe_checkpoint=True)

        # Current frame keeps its autograd graph; history frames are encoded under no_grad.
        features_now = encode(stacked_images[:, :1], stacked_timestamps[:, :1], maybe_checkpoint=True)
        if num_timesteps == 1:
            return features_now
        with torch.no_grad():
            features_hist = encode(stacked_images[:, 1:], stacked_timestamps[:, 1:], maybe_checkpoint=False)
        return torch.cat([features_now, features_hist], dim=1)

    @record_function("CervezaEncoder._lift_features")
    def _lift_features(
        self,
        image_features: Tensor,
        intrinsics_modified: dict,
        extrinsics: dict,
        distortion: dict,
        T_cam_ego: dict,
        height: int,
        width: int,
        batch_size: int,
        num_cams: int,
        num_timesteps: int,
        timestamps: Tensor,
        depth_maps: Optional[dict] = None,
        depth_maps_valid: Optional[dict] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # Per-camera dicts -> (B, T, N, ...) stacks, sliced along T below.
        calib = {
            "intrinsics": einops.rearrange(_stack_dict_of_tensors(intrinsics_modified), "n b t j k -> b t n j k"),
            "extrinsics": einops.rearrange(_stack_dict_of_tensors(extrinsics), "n b t j k -> b t n j k"),
            "T_cam_ego": einops.rearrange(_stack_dict_of_tensors(T_cam_ego), "n b t j k -> b t n j k"),
            "distortion": einops.rearrange(_stack_dict_of_tensors(distortion), "n b t j -> b t n j"),
            "timestamps": einops.rearrange(_stack_dict_of_tensors(timestamps), "n b t -> b t n"),
        }
        stacked_depth_maps = _stack_dict_of_tensors(depth_maps)
        stacked_depth_maps_valid = _stack_dict_of_tensors(depth_maps_valid)
        if stacked_depth_maps is not None:
            # NOTE: We only got this for t=0 atm. TODO: Fix all timesteps.
            stacked_depth_maps = einops.rearrange(stacked_depth_maps, "n b h w -> b n h w")
        if stacked_depth_maps_valid is not None:
            stacked_depth_maps_valid = einops.rearrange(stacked_depth_maps_valid, "n b h w -> b n h w")
        time_indices = torch.arange(num_timesteps, device=image_features.device).unsqueeze(0).repeat(batch_size, 1)

        def lift(feats: Tensor, calib_slice: dict, time_idx: Optional[Tensor], with_depth_maps: bool):
            """One lifter call over a (B', N, ...) stack of frames (B' folds any timestep dim)."""
            return self.lifter(
                features=feats,
                H_in=height,
                W_in=width,
                num_cams=num_cams,
                depth_maps=stacked_depth_maps if with_depth_maps else None,
                depth_maps_valid=stacked_depth_maps_valid if with_depth_maps else None,
                time_indices=time_idx,
                **calib_slice,
            )

        if not self.detach_previous_inputs:
            joint = {key: value.flatten(0, 1) for key, value in calib.items()}  # (B*T, N, ...)
            bev_features, depth, bev_occupancy, target_infos = lift(
                image_features.flatten(0, 1), joint, time_indices.flatten(0, 1), with_depth_maps=True
            )
            bev_features = bev_features.unflatten(0, (batch_size, num_timesteps))
            depth = depth.unflatten(0, (batch_size, num_timesteps)) if depth is not None else None
            if bev_occupancy is not None:
                bev_occupancy = bev_occupancy.unflatten(0, (batch_size, num_timesteps))
            return bev_features, depth, bev_occupancy, target_infos

        # Current frame keeps its autograd graph (and gets the t=0 depth maps for target
        # generation); history frames are lifted under no_grad and re-attached along T.
        now = {key: value[:, 0] for key, value in calib.items()}
        bev_now, depth_now, bev_occupancy, target_infos = lift(
            image_features[:, 0], now, time_indices[:, 0], with_depth_maps=True
        )
        bev_now = bev_now.unsqueeze(1)  # (B, 1, C, H, W)
        depth_now = depth_now.unsqueeze(1) if depth_now is not None else None
        # History frames never have a bev_occupancy (no depth maps), so the current-frame
        # occupancy is the whole output.
        bev_occupancy = bev_occupancy.unsqueeze(1) if bev_occupancy is not None else None

        if num_timesteps == 1:
            return bev_now, depth_now, bev_occupancy, target_infos

        hist = {key: value[:, 1:].flatten(0, 1) for key, value in calib.items()}  # (B*(T-1), N, ...)
        with torch.no_grad():
            bev_hist, depth_hist, _, _ = lift(image_features[:, 1:].flatten(0, 1), hist, None, with_depth_maps=False)
        bev_hist = bev_hist.unflatten(0, (batch_size, num_timesteps - 1))
        depth_hist = depth_hist.unflatten(0, (batch_size, num_timesteps - 1)) if depth_hist is not None else None

        bev_features = torch.cat([bev_now, bev_hist], dim=1)
        depth = torch.cat([depth_now, depth_hist], dim=1) if depth_now is not None else None
        return bev_features, depth, bev_occupancy, target_infos

    @record_function("CervezaEncoder.forward")
    def forward(
        self,
        images: Dict[str, Tensor],
        intrinsics: Dict[str, Tensor],
        intrinsics_modified: Dict[str, Tensor],
        extrinsics: Dict[str, Tensor],
        distortion: Dict[str, Tensor],
        imgs_h: Dict[str, Tensor],
        imgs_w: Dict[str, Tensor],
        T_cam_ego: Dict[str, Tensor],
        timestamps: Dict[str, Tensor],
        batch_size: int,
        num_cams: int,
        num_timesteps: int,
        depth_maps: Optional[Dict] = None,
        depth_maps_valid: Optional[Dict] = None,
    ):
        info = {}

        heights = torch.stack([h[0] for h in imgs_h.values()])
        widths = torch.stack([w[0] for w in imgs_w.values()])

        if not self._image_sizes_checked:
            # Evaluating this condition forces a GPU->CPU sync, so only check the first
            # batch: image sizes are static across batches (the dataset pads all images
            # to a common size).
            if not (torch.all(heights == heights[0]) and torch.all(widths == widths[0])):
                raise AssertionError("All images must have the same height and width to be encoded at once.")
            self._image_sizes_checked = True

        height, width = heights[0], widths[0]

        image_features = self._encode_images(images, timestamps, batch_size, num_cams, num_timesteps)
        # (batch_size, num_timesteps, num_cams, feature_dim, height, width)

        bev_features, depth, _, _ = self._lift_features(
            image_features,
            intrinsics_modified,
            extrinsics,
            distortion,
            T_cam_ego,
            height,
            width,
            batch_size,
            num_cams,
            num_timesteps,
            timestamps,
            depth_maps,
            depth_maps_valid,
        )
        # (batch_size, num_timesteps, feature_dim, bev_height, bev_width)

        if self.supervise_depth:
            info["img_depths"] = depth
        else:
            info["img_depths"] = None

        # Temporal fusion: reduce the per-timestep BEV stack (B, T, C, H, W) to a single map.
        bev_features = bev_features.sum(dim=1)

        # Massage lifted bev features a little
        if self.gradient_checkpointing:
            bev_features = checkpoint(
                self.init_bev_processing,
                bev_features,
                use_reentrant=False,
            )
        else:
            bev_features = self.init_bev_processing(bev_features)

        return bev_features, info
