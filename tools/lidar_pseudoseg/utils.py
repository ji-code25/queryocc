"""Geometry / IO helpers for the offline lidar pseudo-segmentation generator.

Only what ``engine.py`` needs to project camera semantics onto lidar points:
point-cloud + calibration accessors for nuScenes and Argoverse 2, the two
projection models, the occlusion ("depth") filter and the voxel majority vote
used by the scene-wise fusion pass.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from nuscenes.utils.geometry_utils import Quaternion


def resolve_data_path(path: Union[str, Path], data_root: Path) -> Path:
    """Resolve an infos-pkl path against the dataset root.

    ``create_infos`` can write absolute paths or paths relative to the dataset root
    (``--relative-to``), and legacy pkls carry a ``./data/nuscenes/...`` prefix. Absolute
    paths are used as-is; anything else is re-rooted at ``data_root`` (dropping everything
    up to and including a path component named like the root, e.g. ``data/nuscenes/``).
    """
    path = Path(str(path))
    data_root = Path(data_root)
    if path.is_absolute():
        return path
    parts = path.parts
    if data_root.name in parts:
        parts = parts[parts.index(data_root.name) + 1 :]
    return data_root.joinpath(*parts)


def voxel_based_relabeling(
    points: torch.Tensor, labels: torch.Tensor, num_classes: int, voxel_size: float, skip_idx: Optional[int] = None
) -> torch.Tensor:
    voxel_coords = torch.floor(points / voxel_size).int()
    voxel_keys, inverse_indices = torch.unique(voxel_coords, return_inverse=True, dim=0)

    num_voxels = voxel_keys.shape[0]

    votes = torch.zeros((num_voxels, num_classes), device=points.device, dtype=torch.int32)
    votes.index_add_(0, inverse_indices, torch.nn.functional.one_hot(labels, num_classes).int())

    if skip_idx is not None:
        votes[:, skip_idx] = 0

    majority_labels = votes.argmax(dim=1)
    relabeled = majority_labels[inverse_indices]
    return relabeled


def get_nuscenes_pc(sample: Dict, data_root: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    filepath = resolve_data_path(sample["lidar_path"], data_root)
    pc = np.fromfile(filepath, dtype=np.float32).reshape([-1, 5])
    # Filter out points that are within the ego vehicle's bounding box to remove self-hits.
    mask = (abs(pc[..., :3]) >= (1.0, 2.5, 2.0)).any(axis=-1)
    return torch.tensor(pc, dtype=torch.float64), torch.tensor(mask)


def get_argoverse_pc(sample: Dict, data_root: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """AV2 sweep points. NOTE they are stored in the EGO frame, so ``lidar2ego_*`` must not be applied."""
    filepath = resolve_data_path(sample["lidar_path"], data_root)
    df = pd.read_feather(filepath)
    pc = df[["x", "y", "z"]].values
    # Filter out points that are within the ego vehicle's bounding box to remove self-hits.
    mask = (abs(pc[..., :3]) >= (1.0, 2.5, 2.0)).any(axis=-1)
    return torch.tensor(pc, dtype=torch.float64), torch.tensor(mask)


def get_lidar_extrinsics(sample: Dict) -> torch.Tensor:
    translation = torch.tensor(sample["lidar2ego_translation"], dtype=torch.float64)
    rotation = torch.tensor(Quaternion(sample["lidar2ego_rotation"]).rotation_matrix, dtype=torch.float64)
    extrinsics = torch.eye(4, dtype=torch.float64)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation
    return extrinsics


def get_camera_intrinsics(sample: Dict) -> Dict[str, torch.Tensor]:
    intrinsics = {}
    cams_data = sample["cams"]
    for cam_name, cam_sample in cams_data.items():
        intrinsics[cam_name] = torch.tensor(cam_sample["cam_intrinsic"], dtype=torch.float64)
    return intrinsics


def get_camera_extrinsics(sample: Dict) -> Dict[str, torch.Tensor]:
    """Per-camera sensor->ego transform (4x4). Invert it to map ego-frame points into the camera."""
    extrinsics = {}
    cams_data = sample["cams"]
    for cam_name, cam_sample in cams_data.items():
        translation = torch.tensor(cam_sample["sensor2ego_translation"], dtype=torch.float64)
        rotation = torch.tensor(Quaternion(cam_sample["sensor2ego_rotation"]).rotation_matrix, dtype=torch.float64)
        extrinsics[cam_name] = torch.eye(4, dtype=torch.float64)
        extrinsics[cam_name][:3, :3] = rotation
        extrinsics[cam_name][:3, 3] = translation
    return extrinsics


def get_ego_pose(sample: Dict) -> torch.Tensor:
    """Ego->global transform (4x4) at the lidar timestamp."""
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, -1] = torch.tensor(sample["ego2global_translation"], dtype=torch.float64)
    pose[:3, :3] = torch.tensor(Quaternion(sample["ego2global_rotation"]).rotation_matrix, dtype=torch.float64)
    return pose


def get_lidar_pose(sample: Dict) -> torch.Tensor:
    lidar_extrinsics = get_lidar_extrinsics(sample)
    pose = get_ego_pose(sample)
    return pose @ lidar_extrinsics


def _transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return (transform[..., :3, :3] @ points[..., :3].unsqueeze(-1)).squeeze(-1) + transform[..., :3, 3]


def batched_project_kannala(
    points_3d: torch.Tensor,
    camera_matrix: torch.Tensor,
    distortion_coefs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project 3d points to 2d using Kannala-Brandt model with batch support.

    Args:
        points_3d: 3d points in shape (B, N, 3)
        camera_matrix: Camera matrix in shape (B, 3, 4)
        distortion_coefs: Distortion coefficients in shape (B, 4)

    Returns:
        2d points in shape (B, N, 2)
    """
    if distortion_coefs is None:
        distortion_coefs = torch.zeros((points_3d.shape[0], 4), device=points_3d.device)

    device = points_3d.device

    # Batch size and number of points
    B, N, _ = points_3d.shape

    # Create homogeneous points
    homogenous_points_3d = torch.cat([points_3d, torch.ones((B, N, 1), device=device)], dim=-1)

    # Compute norm of x and y coordinates
    norm_data = homogenous_points_3d[..., :2].norm(dim=-1)

    # Compute radial distortion
    radial = torch.atan2(norm_data, homogenous_points_3d[..., 2])
    radial2 = radial**2
    radial4 = radial2**2
    radial6 = radial4 * radial2
    radial8 = radial4**2

    # Apply distortion (broadcasting distortion coefficients)
    distortion_angle = radial * (
        1
        + distortion_coefs[:, 0, None] * radial2
        + distortion_coefs[:, 1, None] * radial4
        + distortion_coefs[:, 2, None] * radial6
        + distortion_coefs[:, 3, None] * radial8
    )

    # Compute distorted coordinates
    u_dist = distortion_angle * homogenous_points_3d[..., 0] / norm_data
    v_dist = distortion_angle * homogenous_points_3d[..., 1] / norm_data

    # Apply camera matrix
    pos_u = camera_matrix[:, 0, 0, None] * u_dist + camera_matrix[:, 0, 2, None]
    pos_v = camera_matrix[:, 1, 1, None] * v_dist + camera_matrix[:, 1, 2, None]

    # Stack coordinates
    projected_points = torch.stack((pos_u, pos_v), dim=-1)

    # Remove batch dimension if input was 2D
    return projected_points


def batched_project_pinhole(
    points_3d: torch.Tensor,
    camera_matrix: torch.Tensor,
) -> torch.Tensor:
    """Project 3d points to 2d using Pinhole model with batch support.

    Args:
        points_3d: 3d points in shape (B, N, 3)
        camera_matrix: Camera matrix in shape (B, 3, 3) or (B, 3, 4)

    Returns:
        2d points in shape (B, N, 2)
    """
    # z coordinates
    z = points_3d[..., 2:3]

    # Project points to normalized image plane
    points_2d = points_3d[..., :2] / z

    # Apply camera matrix
    # pos_u = fx * (x/z) + cx
    # pos_v = fy * (y/z) + cy
    pos_u = camera_matrix[:, 0, 0, None] * points_2d[..., 0] + camera_matrix[:, 0, 2, None]
    pos_v = camera_matrix[:, 1, 1, None] * points_2d[..., 1] + camera_matrix[:, 1, 2, None]

    return torch.stack((pos_u, pos_v), dim=-1)


def _inverse(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(matrix)


def batched_depth_filtering(
    xyds: torch.Tensor, cropped_size: Tuple[int, int], depth_scaling: float, downsample_factor: int
) -> torch.Tensor:
    bs = len(xyds)
    # Convert inputs to tensors
    u_int = torch.floor(xyds[..., 0] / float(downsample_factor)).to(torch.int64)
    v_int = torch.floor(xyds[..., 1] / float(downsample_factor)).to(torch.int64)
    depth = xyds[..., 2].to(torch.float32)
    cropped_size = (cropped_size[0] // downsample_factor, cropped_size[1] // downsample_factor)
    u_int = u_int.clamp(0, cropped_size[0] - 1)
    v_int = v_int.clamp(0, cropped_size[1] - 1)

    # Create linear indices
    linear_idx = v_int * cropped_size[0] + u_int  # B, N
    batched_offsets = torch.arange(bs, device=linear_idx.device)[:, None] * cropped_size[1] * cropped_size[0]  # B, 1
    linear_idx += batched_offsets

    # Find minimum depths for each unique pixel across all batches
    min_depths = torch.zeros(bs * cropped_size[1] * cropped_size[0], dtype=torch.float32, device=xyds.device)

    # Scatter minimum depths
    min_depths.scatter_reduce_(0, linear_idx.reshape(-1), depth.reshape(-1), reduce="amin", include_self=False)

    # Create masks based on depth scaling
    foreground_masks = depth < min_depths[linear_idx.reshape(-1)].reshape(bs, -1) * depth_scaling

    return foreground_masks
