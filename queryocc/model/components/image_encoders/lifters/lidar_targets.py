"""Lidar-depth-derived BEV supervision targets for the LSS lifter.

Extracted from LSSPro.forward (Phase 3 of the image_encoders refactor): given the predicted
per-pixel depth probabilities and the offline (Metric3D) depth maps, build the BEV-space
occupancy / visibility / validity targets that supervise the lifted features
(CervezaImageEncoderConfig.lifted_features_lidar_supervision). Pure functions, all under
no_grad — nothing here touches the autograd graph.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def max_reduce_to_bev(values: Tensor, voxel_idx: Tensor, voxel_grid_size: Tensor) -> Tensor:
    """Scatter per-point scalars onto the BEV grid, keeping the max per cell.

    values: [B, N_points, 1]; voxel_idx: [B, N_points, 1]; returns [B, H_bev, W_bev].
    """
    bs = values.shape[0]
    h_bev, w_bev = voxel_grid_size[0], voxel_grid_size[1]
    bev = torch.zeros(int(h_bev * w_bev), 1, device=values.device, dtype=values.dtype).unsqueeze(0).repeat(bs, 1, 1)
    bev = bev.scatter_reduce(1, voxel_idx.expand(-1, -1, 1), values, reduce="max")
    return bev.reshape(bs, h_bev, w_bev)


@torch.no_grad()
def compute_lidar_bev_targets(
    depth_probs: Tensor,
    depth_maps: Tensor,
    depth_maps_valid: Tensor,
    time_indices: Tensor,
    voxel_idx: Tensor,
    depths: Tensor,
    voxel_grid_size: Tensor,
    bs: int,
    num_cams: int,
    outside_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Predicted BEV occupancy (from depth probabilities) + lidar-depth supervision targets.

    Args:
        depth_probs: [B*num_cams, D, H_feats, W_feats] depth probabilities from the depth net.
        depth_maps: [B, num_cams, H_feats, W_feats] offline metric depth (t=0 only).
        depth_maps_valid: same shape, validity of each depth pixel.
        time_indices: [B(*T)] timestep index per lifted batch row; targets use only t == 0 rows.
        voxel_idx: [B(*T), num_cams * num_points, 1] flat BEV cell index per splatted point.
        depths: [D] depth-bin centers (includes the sky bin when configured).
        voxel_grid_size: [2] long tensor (H_bev, W_bev).
        outside_mask: optional [B, num_cams, num_points] bool, True = point outside the ROI
            (config.remove_outside_points); those probabilities are zeroed like the features.

    Returns:
        bev_occupancy [B, H_bev, W_bev] and the target_infos dict
        (target_bev_occupancy / target_bev_visibility / target_bev_valids).
    """
    num_depths = depths.shape[0]
    h_bev, w_bev = voxel_grid_size[0], voxel_grid_size[1]

    occ_probs = depth_probs.permute(0, 2, 3, 1)  # [B*num_cams, H_feats, W_feats, D]
    occ_probs = occ_probs.reshape(bs, num_cams, *occ_probs.shape[1:])
    occ_probs = occ_probs.reshape(bs, num_cams, -1)  # [B, num_cams, H*W*D]
    if outside_mask is not None:
        occ_probs[outside_mask, ...] = 0
    occ_probs = occ_probs.reshape(bs, -1, 1)  # [B, num_cams*H*W*D, 1]

    bev_occupancy = (
        torch.zeros(int(h_bev * w_bev), 1, device=depth_probs.device, dtype=occ_probs.dtype)
        .unsqueeze(0)
        .repeat(bs, 1, 1)
    )
    bev_occupancy = bev_occupancy.scatter_reduce(1, voxel_idx.expand(-1, -1, 1), occ_probs, reduce="max")
    bev_occupancy = bev_occupancy.reshape(bs, h_bev, w_bev)

    depth_bs = depth_maps.shape[0]
    current_voxel_idx = voxel_idx[time_indices == 0, ...]

    min_depth = depths[0]
    max_depth = depths[-1]
    valids = torch.logical_and(depth_maps_valid, depth_maps > min_depth)
    valids = torch.logical_and(valids, depth_maps < max_depth)

    # Broadcasting against [D] avoids materializing a repeated copy of the bins.
    target_bins = torch.argmin(torch.abs(depth_maps[..., None] - depths), dim=-1)
    bin_indices = torch.arange(num_depths, device=target_bins.device)
    target_bins_one_hot = (bin_indices == target_bins[..., None]).long()

    # Visibility: 1 for bins strictly before the hit bin, 0 at and after it
    # (equivalent to 1 - cumsum(one_hot) without the one-hot/cumsum passes).
    target_cdf = (bin_indices < target_bins[..., None]).long()
    target_cdf = target_cdf * valids.unsqueeze(-1)
    target_cdf = target_cdf.reshape(depth_bs, -1, 1)

    target_bins_one_hot = target_bins_one_hot * valids.unsqueeze(-1)
    target_bins_one_hot = target_bins_one_hot.reshape(depth_bs, -1, 1)

    target_valids = valids.unsqueeze(-1)
    target_valids = target_valids.repeat((1, 1, 1, 1, num_depths))
    target_valids = target_valids.reshape(depth_bs, -1, 1).long()

    target_infos = {
        "target_bev_occupancy": max_reduce_to_bev(target_bins_one_hot, current_voxel_idx, voxel_grid_size),
        "target_bev_visibility": max_reduce_to_bev(target_cdf, current_voxel_idx, voxel_grid_size),
        "target_bev_valids": max_reduce_to_bev(target_valids, current_voxel_idx, voxel_grid_size),
    }
    return bev_occupancy, target_infos
