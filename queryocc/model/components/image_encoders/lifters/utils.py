"""Canvas-initialization helpers shared by the lifters and the VGGT aggregator."""

from typing import List

import torch

from queryocc.model.components.positional_encoder import build_static_grid_encoding


def initialize_sinusoidal_features(feature_dim: int, voxel_grid_size: List[int]) -> torch.Tensor:
    """Initialize the features for the voxel grid."""
    return build_static_grid_encoding(
        feature_dim, [voxel_grid_size[0], voxel_grid_size[1]], max_freq=100, perceiver=False
    )


def initialize_zeros_features(feature_dim: int, voxel_grid_size: List[int]) -> torch.Tensor:
    """Initialize the features for the voxel grid."""
    return torch.zeros(int(voxel_grid_size[0] * voxel_grid_size[1]), feature_dim)
