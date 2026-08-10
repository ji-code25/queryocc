from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import einops
import torch
from torch import nn
from torch.profiler import record_function

from queryocc.common.types import PointCloudRange, VoxelSize
from queryocc.config.base_config import InstantiateConfig
from queryocc.model.components.attention.layers import MLP
from queryocc.model.components.checkpointing import checkpoint
from queryocc.model.components.contraction.depth_contraction import get_depths
from queryocc.model.components.contraction.point_contraction import PointContraction
from queryocc.model.components.image_encoders.geometry import invert_intrinsics, invert_rigid_transform
from queryocc.model.components.image_encoders.lifters.depth_net import DepthNet
from queryocc.model.components.image_encoders.lifters.lidar_targets import compute_lidar_bev_targets
from queryocc.model.components.image_encoders.lifters.utils import (
    initialize_sinusoidal_features,
    initialize_zeros_features,
)
from queryocc.model.components.positional_encoder import CoordinateEncoder


@dataclass
class LSSProConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: LSSPro)
    depth_num_bins: int = 42
    """ The depth number of bins for the LSS encoder """
    supervise_depth: bool = True
    """ Whether to supervise the depth in the LSS encoder """
    remove_outside_points: bool = False
    """ Whether to remove points outside the image bounds in the LSS encoder """
    depth_distribution: Literal["uniform", "log", "log_linear_blend"] = "log_linear_blend"
    """ The type of depth distribution to use for the LSS encoder """
    voxel_distribution: Literal["uniform", "adaptive"] = "uniform"
    """ The type of voxel distribution to use for the LSS encoder """
    depth_probability_activation: Literal["softmax"] = "softmax"
    """ The type of probability activation to use for the depth in the LSS encoder """
    use_point_contraction: bool = True
    """ Whether to use point contraction in the LSS encoder """
    points_temporal_encoding: bool = True
    """ Whether to encode the time in the pseudo point cloud in the image encoder """
    points_transmittance_encoding: bool = True
    """ Whether to encode the transmittance in the pseudo point cloud in the image encoder """
    points_density_encoding: bool = True
    """ Whether to encode the density in the pseudo point cloud in the image encoder """
    points_encoder_max_freq: float = 10.0
    """ The maximum frequency for the pseudo point cloud encoding in the image encoder """
    scale_positional_encoding: bool = False
    """ Whether to scale the positional encoding for the pseudo point cloud in the image encoder """
    max_depth: float = 100.0
    """ The maximum depth in the LSS encoder. -1 for automatically setting it to the maximum depth in the point cloud """
    depth_gamma: float = 0.3
    """ The gamma for the depth distribution in the LSS encoder """
    add_sky_depth: bool = False
    """ Whether to add a sky depth in the LSS encoder"""
    sky_depth: float = 170.0
    """ The sky depth in the LSS encoder """
    interpret_depth_as_ray_length: bool = False
    """ Whether to interpret the depth as the ray length in the LSS encoder """

    points_mlp: bool = True
    """ Whether to use an MLP to encode the points in the LSS encoder """
    points_mlp_num_layers: int = 1
    """ The number of layers for the MLP to encode the points in the LSS encoder """
    points_mlp_bottleneck: int = 1
    """ The bottleneck for the MLP to encode the points in the LSS encoder """
    learnable_empty_feature: bool = False
    """ Whether to use a learnable occlusion feature in the LSS encoder """


class LSSPro(nn.Module):
    """Lift-splat ish lifting module."""

    def __init__(
        self,
        config: LSSProConfig,
        in_channels: int,
        out_channels: int,
        point_cloud_range: PointCloudRange,
        voxel_size: VoxelSize,
        initial_features: Literal["zeros", "sinusoidal"] = "sinusoidal",
        points_positional_encoding: Literal["none", "sinusoidal"] = "sinusoidal",
        gradient_checkpointing: bool = False,
        swap_batch_norm_to_instance_norm: bool = False,
        point_contraction: Optional[PointContraction] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        # we only consider 2D BEV voxelization
        self.register_buffer("voxel_size", torch.tensor(voxel_size[:2], dtype=torch.float32))

        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32))
        self.register_buffer("point_cloud_lb", torch.tensor(point_cloud_range[:3], dtype=torch.float32))
        self.register_buffer("point_cloud_ub", torch.tensor(point_cloud_range[3:6], dtype=torch.float32))
        self.register_buffer(
            "voxel_grid_size", ((self.point_cloud_ub[:2] - self.point_cloud_lb[:2]) / self.voxel_size).long()
        )
        self.feature_dim = out_channels

        self.point_contraction = point_contraction

        init_features = self._initialize_features(initial_features)
        self.register_buffer("initial_features", init_features)

        # Point positional encoding
        points_dim = 3
        if config.points_transmittance_encoding:
            points_dim += 1
        if config.points_temporal_encoding:
            points_dim += 1
        if config.points_density_encoding:
            points_dim += 1

        if points_positional_encoding == "sinusoidal":
            self.points_encoder = CoordinateEncoder(
                points_dim, out_channels, out_channels, 16, max_freq=config.points_encoder_max_freq
            )
        else:
            self.points_encoder = None

        if config.points_mlp:
            self.points_mlp = MLP(
                out_channels,
                out_channels // config.points_mlp_bottleneck,
                out_channels,
                activation=nn.LeakyReLU(),
                num_layers=config.points_mlp_num_layers,
            )
        else:
            self.points_mlp = None

        if config.learnable_empty_feature:
            self.empty_feature = nn.Embedding(1, self.feature_dim)
        else:
            self.empty_feature = None

        # Depth sampling
        depths = get_depths(
            config.depth_distribution,
            point_cloud_range,
            config.depth_num_bins,
            config.max_depth,
            config.depth_gamma,
        )

        if config.add_sky_depth:
            assert config.sky_depth > config.max_depth, "Sky depth must be greater than max depth"
            depths = torch.cat([depths, torch.ones(1) * config.sky_depth], dim=0)

        self.num_depths = depths.shape[0]
        self.register_buffer("depths", depths)  # [D]

        self.voxel_distribution = config.voxel_distribution
        if config.voxel_distribution == "adaptive":
            self._precompute_and_register_adaptive_grid()

        self.depth_network = DepthNet(
            in_channels,
            in_channels,
            out_channels,
            self.num_depths,
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
        )

        self.gradient_checkpointing = gradient_checkpointing

    def _initialize_features(self, initial_features: str) -> torch.Tensor:
        """Initialize the features for the voxel grid."""
        grid = [int(self.voxel_grid_size[0]), int(self.voxel_grid_size[1])]
        if initial_features == "sinusoidal":
            return initialize_sinusoidal_features(self.feature_dim, grid)
        if initial_features == "zeros":
            return initialize_zeros_features(self.feature_dim, grid)
        raise ValueError(f"Unknown initial features: {initial_features}")

    def _initial_features(self, bs, dtype) -> torch.Tensor:
        # Cast on the fly instead of reassigning the buffer, which would flip-flop
        # its dtype under mixed precision.
        return self.initial_features.to(dtype).unsqueeze(0).repeat(bs, 1, 1)

    def _image_coords(
        self, H_in: int, W_in: int, H_out: int, W_out: int, device: torch.device, dtype: torch.dtype = torch.float32
    ):
        """Generate pixel coordinates for the downsampled image grid."""
        # Downsampled image grid
        widths = torch.linspace(0, W_in - 1, W_out, device=device, dtype=dtype)  # [W_out]
        heights = torch.linspace(0, H_in - 1, H_out, device=device, dtype=dtype)  # [H_out]
        # Create a meshgrid for the pixel coordinates

        grid_widths, grid_heights = torch.meshgrid(widths, heights, indexing="xy")  # [W_out, H_out]
        # Homogeneous pixel coordinates [u, v, 1]
        pixel_coords = torch.stack(
            (grid_widths, grid_heights, torch.ones_like(grid_widths)), dim=-1
        )  # [H_out, W_out, 3]

        return pixel_coords

    def _depth_probability_activation(self, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, d, h, w = depth.shape
        depth = depth.permute(0, 2, 3, 1)
        depth = depth.reshape(-1, d)
        if self.config.depth_probability_activation == "softmax":
            weights = depth.softmax(dim=-1)
            if self.config.points_transmittance_encoding:
                # NOTE: post-bin transmittance (the current bin is included in the cumsum).
                # Kept as-is for checkpoint compatibility.
                transmittance = 1 - torch.cumsum(weights, dim=-1) + 1e-7
        else:
            raise ValueError(f"Unknown depth probability activation: {self.config.depth_probability_activation}")
        weights = weights.reshape(n, h, w, -1)
        weights = weights.permute(0, 3, 1, 2)
        if self.config.points_transmittance_encoding:
            transmittance = transmittance.reshape(n, h, w, -1)
            transmittance = transmittance.permute(0, 3, 1, 2)
        else:
            transmittance = None
        return weights, transmittance

    @record_function("LSSPro._get_ego_coords")
    @torch.no_grad()
    def _get_ego_coords(
        self, features: torch.Tensor, H_in: int, W_in: int, intrinsics: torch.Tensor, T_cam_ego: torch.Tensor
    ) -> torch.Tensor:
        """Generate ego coordinates from the features and camera intrinsics."""
        bs = intrinsics.size(0)
        num_cams = intrinsics.size(1)
        # Analytic inverses (exact for pinhole K / rigid transforms) — torch.inverse on GPU can
        # emit silent NaN and desync DDP; see image_encoders/geometry.py.
        inv_intrinsics = invert_intrinsics(intrinsics)  # [B, num_cams, 3, 3]
        T_ego_cam = invert_rigid_transform(T_cam_ego)  # [B, num_cams, 4, 4]

        H_out, W_out = features.shape[-2:]  # [B, num_cams, C, H_in, W_in]
        pixel_coords = self._image_coords(
            H_in, W_in, H_out, W_out, features.device, intrinsics.dtype
        )  # [H_out, W_out, 3]
        pixel_coords = pixel_coords.reshape(-1, 3)  # [H_out*W_out, 3]
        pixel_coords = (
            pixel_coords.unsqueeze(0).unsqueeze(0).expand(bs, num_cams, -1, -1)
        )  # [bs, num_cams, H_out*W_out, 3]

        cam_coords = torch.matmul(inv_intrinsics.unsqueeze(2), pixel_coords.unsqueeze(-1)).squeeze(
            -1
        )  # [B, num_cams, H_out*W_out, 3]

        # Normalize cam_coords
        if self.config.interpret_depth_as_ray_length:
            cam_coords = cam_coords / cam_coords.norm(dim=-1, keepdim=True)

        del pixel_coords

        # Add depth dimension
        cam_coords = cam_coords.unsqueeze(3).expand(-1, -1, -1, self.num_depths, -1)
        depths_expanded = (
            self.depths.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, num_cams, -1, -1).unsqueeze(-1)
        )  # [B, num_cams, D, 1]
        # Multiply by depth
        cam_coords = cam_coords * depths_expanded

        del depths_expanded

        # Make homogeneous coordinates
        cam_coords_hom = torch.cat((cam_coords, torch.ones_like(cam_coords[:, :, :, :, 0:1])), dim=-1)
        cam_coords_hom = cam_coords_hom.view(bs, num_cams, -1, 4)

        del cam_coords

        # Transform to ego coordinates
        ego_coords = torch.matmul(T_ego_cam.unsqueeze(2), cam_coords_hom.unsqueeze(-1)).squeeze(-1)
        ego_coords = ego_coords[..., :3]

        ego_coords = ego_coords.to(dtype=features.dtype)

        return ego_coords  # [B, num_cams, H_out*W_out, 3]

    def _mlp_input(self, extrinsics, intrinsics):
        extrinsics_flat = extrinsics.flatten(start_dim=2, end_dim=-1)[..., 0:12]  # [B, num_cams, 12]
        intrinsics_flat = intrinsics.flatten(start_dim=2, end_dim=-1)[..., [0, 2, 4, 5]] / 1000  # [B, num_cams, 4]
        mlp_input = torch.cat((extrinsics_flat, intrinsics_flat), dim=-1)  # [B, num_cams, 16]
        return mlp_input

    @record_function("LSSPro._get_context")
    def _get_context(
        self,
        features: torch.Tensor,
        ego_coords: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the context features from the depth network."""
        bs, num_cams = features.shape[0:2]

        mlp_input = self._mlp_input(extrinsics, intrinsics)  # [B, num_cams, 16]

        feat_dim, H_feats, W_feats = features.shape[2:]

        features = features.reshape(bs * num_cams, feat_dim, H_feats, W_feats)  # [B*num_cams, C, H_feats, W_feats]

        mlp_input = mlp_input.reshape(bs * num_cams, -1)  # [B*num_cams, 16]

        out_features = self.depth_network(features, mlp_input.to(dtype=features.dtype))

        del features
        del mlp_input

        depth = out_features[:, : self.num_depths, :, :]  # [B*num_cams, D, H_feats, W_feats]
        depth_weights, depth_transmittance = self._depth_probability_activation(depth)
        # Keep a reference for reuse in forward(); depth_weights is reassigned below.
        depth_probs = depth_weights  # [B*num_cams, D, H_feats, W_feats]

        context = out_features[:, self.num_depths :, :, :]  # [B*num_cams, C, H_feats, W_feats]
        del out_features

        context = torch.einsum("bchw,bdhw->bcdhw", context, depth_weights)

        if self.empty_feature is not None:
            empty = self.empty_feature(
                torch.zeros(size=(bs * num_cams, H_feats, W_feats), dtype=torch.long, device=context.device)
            )
            empty = empty.permute(0, 3, 1, 2)
            empty = torch.einsum("bchw,bdhw->bcdhw", empty, 1 - depth_weights)
            context += empty
            del empty

        context = context.view(
            bs, num_cams, feat_dim, self.num_depths, H_feats, W_feats
        )  # [B, num_cams, feat_dim, num_depths, H_feats, W_feats]
        context = context.permute(0, 1, 2, 4, 5, 3)  # [B, num_cams, feat_dim, H_feats, W_feats, num_depths]
        context = context.reshape(
            bs, num_cams, feat_dim, H_feats * W_feats * self.num_depths
        )  # [B, num_cams, C, num_points]

        context = context.permute(0, 1, 3, 2)  # [B, num_cams, num_points, C]

        if self.points_encoder is not None:
            num_points = ego_coords.shape[2]  # H_out*W_out*num_depths

            ego_coords = einops.rearrange(ego_coords, "b num_cams num_points c -> b (num_cams num_points) c")
            ego_coords_norm = [ego_coords.to(dtype=context.dtype) / self.point_cloud_ub.to(dtype=context.dtype)]
            del ego_coords

            if self.config.points_temporal_encoding:
                # Camera-major layout to match the flattened ego_coords ordering.
                timestamps = timestamps.unsqueeze(-1).unsqueeze(2)  # [B, num_cams, 1, 1]
                timestamps = timestamps.expand(-1, -1, num_points, -1)  # [B, num_cams, num_points, 1]
                timestamps = einops.rearrange(timestamps, "b num_cams num_points c -> b (num_cams num_points) c")
                ego_coords_norm += [timestamps.to(dtype=context.dtype)]
                del timestamps

            if self.config.points_transmittance_encoding:
                depth_transmittance = depth_transmittance.view(bs, num_cams, self.num_depths, H_feats, W_feats)
                depth_transmittance = depth_transmittance.permute(
                    0, 1, 3, 4, 2
                )  # [B, num_cams, H_feats, W_feats, num_depths]
                depth_transmittance = depth_transmittance.reshape(
                    bs, num_cams, H_feats * W_feats * self.num_depths
                )  # [B, num_cams, num_points]
                depth_transmittance = depth_transmittance.reshape(
                    bs, num_cams * H_feats * W_feats * self.num_depths
                )  # [B, num_cams * num_points]
                depth_transmittance = depth_transmittance.unsqueeze(-1)  # [B, num_cams * num_points, 1]
                ego_coords_norm += [depth_transmittance.to(dtype=context.dtype)]
                del depth_transmittance

            depth_weights = depth_weights.view(bs, num_cams, self.num_depths, H_feats, W_feats)
            depth_weights = depth_weights.permute(0, 1, 3, 4, 2)  # [B, num_cams, H_feats, W_feats, num_depths]
            depth_weights = depth_weights.reshape(
                bs, num_cams, H_feats * W_feats * self.num_depths
            )  # [B, num_cams, num_points]
            depth_weights = depth_weights.reshape(
                bs, num_cams * H_feats * W_feats * self.num_depths
            )  # [B, num_cams * num_points]
            depth_weights = depth_weights.unsqueeze(-1)  # [B, num_cams * num_points, 1]
            if self.config.points_density_encoding:
                ego_coords_norm += [depth_weights.to(dtype=context.dtype)]

            ego_coords_norm = torch.cat(ego_coords_norm, dim=-1)
            ego_coords_feats = self.points_encoder(ego_coords_norm)
            del ego_coords_norm
            if self.config.scale_positional_encoding:
                ego_coords_feats *= depth_weights
            del depth_weights
            ego_coords_feats = einops.rearrange(
                ego_coords_feats,
                "b (num_cams num_points) c -> b num_cams num_points c",
                b=bs,
                num_cams=num_cams,
                num_points=num_points,
            )
            context += ego_coords_feats

        depth = depth.view(bs, num_cams, self.num_depths, H_feats, W_feats)
        depth = depth.permute(0, 1, 3, 4, 2)  # [B, num_cams, H_feats, W_feats, D]

        if self.points_mlp is not None:
            context = self.points_mlp(context)

        # [B, num_cams, num_points, C], [B, num_cams, H_feats, W_feats, D], [B*num_cams, D, H_feats, W_feats]
        return context, depth, depth_probs

    def _precompute_and_register_adaptive_grid(self):
        """
        Pre-computes the cumulative sums of voxel sizes for both x and y axes
        and registers them as non-trainable buffers.
        """
        grid_x_size = self.voxel_grid_size[0].item()
        grid_y_size = self.voxel_grid_size[1].item()

        # Grid dimensions in meters
        x_range = self.point_cloud_ub[0] - self.point_cloud_lb[0]
        y_range = self.point_cloud_ub[1] - self.point_cloud_lb[1]

        # Max distance from origin
        max_dist_x = x_range / 2
        max_dist_y = y_range / 2

        # Voxel sizes increase linearly with distance from the center
        voxel_size_min = self.voxel_grid_size[0]
        voxel_size_max = self.voxel_grid_size[0] * 10

        # Create a linear ramp for the voxel sizes
        x_dist = torch.linspace(0, max_dist_x, grid_x_size // 2, dtype=torch.float32)
        y_dist = torch.linspace(0, max_dist_y, grid_y_size // 2, dtype=torch.float32)

        # Calculate voxel sizes based on distance
        voxel_size_x_half = voxel_size_min + (voxel_size_max - voxel_size_min) * (x_dist / max_dist_x)
        voxel_size_y_half = voxel_size_min + (voxel_size_max - voxel_size_min) * (y_dist / max_dist_y)

        # Full voxel size arrays, mirrored around the center
        voxel_size_x = torch.cat([voxel_size_x_half.flip(dims=[0]), voxel_size_x_half])
        voxel_size_y = torch.cat([voxel_size_y_half.flip(dims=[0]), voxel_size_y_half])

        # Compute the cumulative sum of voxel sizes to get the grid boundaries
        x_boundaries = torch.cumsum(voxel_size_x, dim=0)
        y_boundaries = torch.cumsum(voxel_size_y, dim=0)

        # Center the boundaries
        x_boundaries -= x_boundaries[-1] / 2
        y_boundaries -= y_boundaries[-1] / 2

        # Register the boundaries as buffers.
        self.register_buffer("x_boundaries", x_boundaries)
        self.register_buffer("y_boundaries", y_boundaries)

    def _voxelize_coordinates(self, xy):
        # Center the coordinates around the point cloud center
        pc_center = (self.point_cloud_lb[:2] + self.point_cloud_ub[:2]) / 2
        centered_xy = xy - pc_center

        if self.voxel_distribution == "adaptive":
            # Compute the 2D diagonal from the ego to the farthest corner
            # Find the index for each coordinate using `searchsorted`.
            # The buffers `self.x_boundaries` and `self.y_boundaries` are already
            # on the correct device.
            voxel_x_idx = torch.searchsorted(self.x_boundaries, centered_xy[..., 0], right=True)
            voxel_y_idx = torch.searchsorted(self.y_boundaries, centered_xy[..., 1], right=True)

            # Clamp the indices to the valid range
            voxel_x_idx = torch.clamp(voxel_x_idx, 0, self.voxel_grid_size[0] - 1)
            voxel_y_idx = torch.clamp(voxel_y_idx, 0, self.voxel_grid_size[1] - 1)

            # Combine and flatten the indices
            voxel_idx = torch.stack((voxel_x_idx, voxel_y_idx), dim=-1)
            voxel_idx_flat = voxel_idx[..., 0] * self.voxel_grid_size[1] + voxel_idx[..., 1]

        elif self.voxel_distribution == "uniform":
            # Calculate voxel indices with proper centering
            voxel_idx = (centered_xy + (self.point_cloud_ub[:2] - self.point_cloud_lb[:2]) / 2) / self.voxel_size
            voxel_idx = voxel_idx.long()
            # clamp the voxel index to the valid range
            voxel_idx = torch.clamp(
                voxel_idx, torch.zeros_like(self.voxel_grid_size), self.voxel_grid_size - 1
            )  # [B, num_cams, num_points, 2]

            # Calculate the flattened index with overflow protection
            max_idx = self.voxel_grid_size[0] * self.voxel_grid_size[1]
            voxel_idx_flat = voxel_idx[..., 0] * self.voxel_grid_size[1] + voxel_idx[..., 1]

            # Additional safety check for the flattened index
            voxel_idx_flat = torch.clamp(voxel_idx_flat, 0, max_idx - 1)
        else:
            raise ValueError(f"Unknown voxel distribution: {self.voxel_distribution}")

        return voxel_idx_flat

    @record_function("LSSPro._splat_to_bev")
    def _splat_to_bev(self, context, voxel_idx):
        """
        context: [B, N_points, C]
        voxel_idx: [B, N_points, 1]
        """
        # Create the BEV feature tensor
        bs = context.shape[0]
        bev_features = self._initial_features(bs, context.dtype)
        # Direct scatter to the BEV grid
        bev_features = bev_features.scatter_add(
            1,
            voxel_idx.expand(-1, -1, self.feature_dim),
            context,
        )

        return bev_features

    @record_function("LSSPro.forward")
    def forward(
        self,
        features: torch.Tensor,
        H_in: int,
        W_in: int,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        T_cam_ego: torch.Tensor,
        timestamps: torch.Tensor,
        depth_maps: Optional[torch.Tensor] = None,
        depth_maps_valid: Optional[torch.Tensor] = None,
        time_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass of the LSSPro module.
        Args:
            features: (batch_size * num_timesteps, num_cams, feature_dim, height, width)
            H_in: int
            W_in: int
            intrinsics: (batch_size * num_timesteps, num_cams, 3, 3)
            extrinsics: (batch_size * num_timesteps, num_cams, 4, 4)
            T_cam_ego: (batch_size * num_timesteps, num_cams, 4, 4)
            timestamps: (batch_size * num_timesteps, num_cams)
            depth_maps: (batch_size, num_cams, height, width) # NOTE: We only got this for t=0 atm. TODO: Fix all timesteps.
            depth_maps_valid: (batch_size, num_cams, height, width) # NOTE: We only got this for t=0 atm. TODO: Fix all timesteps.
            time_indices: (batch_size * num_timesteps,)
        """
        bs, num_cams = features.shape[0:2]
        H_bev, W_bev = self.voxel_grid_size[0], self.voxel_grid_size[1]

        # Get point wise ego coordinates
        ego_coords = self._get_ego_coords(
            features, H_in, W_in, intrinsics, T_cam_ego
        )  # [B, num_cams, H_out*W_out*num_depths, 3]
        num_points = ego_coords.shape[2]  # H_out*W_out*num_depths

        # Get point wise context from the image features
        if self.gradient_checkpointing:
            context, depth, depth_probs = checkpoint(
                self._get_context, features, ego_coords, extrinsics, intrinsics, timestamps, use_reentrant=False
            )
        else:
            context, depth, depth_probs = self._get_context(
                features, ego_coords, extrinsics, intrinsics, timestamps
            )  # [B, num_cams, num_points, C]

        if self.config.remove_outside_points:
            indeces_to_remove = (
                (ego_coords < self.point_cloud_lb.unsqueeze(0).unsqueeze(0))
                | (ego_coords > self.point_cloud_ub.unsqueeze(0).unsqueeze(0))
            ).any(dim=-1)
            context[indeces_to_remove, ...] = 0

        xy = ego_coords[..., :2].contiguous()  # [B, num_cams, num_points, 2]

        del ego_coords

        if self.config.use_point_contraction:
            xy = self.point_contraction(xy, self.point_cloud_range)

        voxel_idx = self._voxelize_coordinates(xy)

        del xy

        context = context.reshape(bs, num_cams * num_points, self.feature_dim)  # [B, num_cams * num_points, C]
        voxel_idx = voxel_idx.reshape(bs, num_cams * num_points, 1)

        bev_features = self._splat_to_bev(context, voxel_idx)

        del context

        bev_occupancy = None
        target_infos = {}
        if depth_maps is not None:
            bev_occupancy, target_infos = compute_lidar_bev_targets(
                depth_probs=depth_probs,
                depth_maps=depth_maps,
                depth_maps_valid=depth_maps_valid,
                time_indices=time_indices,
                voxel_idx=voxel_idx,
                depths=self.depths,
                voxel_grid_size=self.voxel_grid_size,
                bs=bs,
                num_cams=num_cams,
                outside_mask=indeces_to_remove if self.config.remove_outside_points else None,
            )

        # Reshape the BEV features to the original voxel grid size
        bev_features = bev_features.view(bs, H_bev, W_bev, self.feature_dim)  # [B, H_out, W_out, C]
        bev_features = bev_features.permute(0, 3, 1, 2)  # [B, C, H_bev, W_bev]

        return bev_features, depth, bev_occupancy, target_infos
