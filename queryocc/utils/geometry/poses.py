# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SE(3) pose helpers, pose interpolation, and lidar ego-motion compensation."""

from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from zod.data_classes import LidarData
from zod.data_classes.calibration import LidarCalibration
from zod.data_classes.ego_motion import EgoMotion

from queryocc.model.ops.functions.interpolate import interpolate_trajectories_np
from queryocc.utils.geometry.rotations import (
    rotmat_to_unitquat,
    unitquat_slerp,
    unitquat_slerp_fast,
    unitquat_to_rotmat,
)


def to4x4(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 4 4"]:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Camera poses with additional homogenous coordinate added.
    """
    if pose.shape[-2:] == (4, 4):
        return pose
    constants = torch.zeros_like(pose[..., :1, :], device=pose.device)
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def inverse(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Invert provided pose matrix (3x4 or 4x4).

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Inverse of pose.
    """
    # NOTE: the transposed rotation must be materialised *before* it is written back.
    # `inverse_pose[..., :3, :3] = inverse_pose[..., :3, :3].transpose(-2, -1)` assigns a
    # view of the destination into itself; that self-overlapping copy is unspecified and
    # since torch 2.x silently yields a non-orthogonal matrix (R @ R^-1 != I).
    rotation_transposed = pose[..., :3, :3].transpose(-2, -1).contiguous()
    inverse_pose = pose.clone()
    inverse_pose[..., :3, :3] = rotation_transposed
    inverse_pose[..., :3, 3] = -rotation_transposed.matmul(pose[..., :3, 3:4]).squeeze(-1)
    return inverse_pose


def inverse_np(pose: np.ndarray) -> np.ndarray:
    """Invert provided pose matrix (numpy)."""
    inverted_pose = pose.copy()
    inverted_pose[..., :3, :3] = inverted_pose[..., :3, :3].swapaxes(-1, -2)
    inverted_pose[..., :3, 3] = -np.matmul(inverted_pose[..., :3, :3], inverted_pose[..., :3, 3:4])[..., 0]
    return inverted_pose


def multiply(pose_a: Float[Tensor, "*batch 3 4"], pose_b: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Multiply two pose matrices, A @ B.

    Args:
        pose_a: Left pose matrix, usually a transformation applied to the right.
        pose_b: Right pose matrix, usually a camera pose that will be transformed by pose_a.

    Returns:
        Camera pose matrix where pose_a was applied to pose_b.
    """
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Transforms points (N x >=3, extra columns preserved) by a 3x4/4x4 transform."""
    points = points.clone()
    rotations = transform[..., :3, :3]
    translations = transform[..., :3, 3]
    points[:, :3] = points[:, :3] @ rotations.swapaxes(-2, -1) + translations
    return points


def interpolate_trajectories(poses, pose_times, query_times, pose_valid_mask=None, clamp_frac=True):
    """Interpolate trajectory poses at query times using linear interpolation.

    Args:
        poses: A collection of poses to be interpolated N_times x N_trajectories.
        pose_times: The timestamps of the poses N_times.
        query_times: The timestamps to interpolate the poses at N_queries.
        pose_valid_mask: A mask indicating which poses are valid N_times x N_trajectories.

    Returns:
        The interpolated poses at the query times (M x 3 x 4)
        The indices of the queries used for interpolation (M).
        The indices of the tractories used for interpolation (M).
    """
    assert len(poses.shape) == 4, "Poses must be of shape [num_poses, num_actors, 3, 4]"
    # bugs are crawling like crazy if we only have one query time, fix with maybe squeeze
    qt = query_times if query_times.shape[-1] == 1 else query_times.squeeze()
    right_idx = torch.searchsorted(pose_times, qt)
    right_idx = right_idx.clamp(min=1, max=len(pose_times) - 1)
    left_idx = right_idx - 1

    # Compute the fraction between the left (previous) and right (after) timestamps
    right_time = pose_times[right_idx]
    left_time = pose_times[left_idx]
    time_diff = right_time - left_time + 1e-6
    fraction = (qt - left_time) / time_diff  # 0 = all left, 1 = all right
    if clamp_frac:
        fraction = fraction.clamp(0.0, 1.0)  # clamp to handle out of bounds

    if pose_valid_mask is None:
        pose_valid_mask = torch.ones_like(poses[..., 0, 0], dtype=torch.bool)
    trajs_to_sample = pose_valid_mask[left_idx] | pose_valid_mask[right_idx]  # [num_queries, n_trajs]
    query_idxs, object_idxs = torch.where(trajs_to_sample)  # 2 x [num_queries*n_valid_trajs]

    quat = rotmat_to_unitquat(poses[..., :3, :3])
    quat_left = quat[left_idx][query_idxs, object_idxs]  # [num_queries*n_valid_objects, 4]
    quat_right = quat[right_idx][query_idxs, object_idxs]  # [num_queries*n_valid_objects, 4]
    interp_fn = unitquat_slerp if (fraction < 0).any() or (fraction > 1).any() else unitquat_slerp_fast
    interp_quat = interp_fn(quat_left, quat_right, fraction[query_idxs])
    interp_rot = unitquat_to_rotmat(interp_quat)

    pos_left = poses[left_idx][query_idxs, object_idxs, :3, 3]  # [num_queries*n_valid_objects, 3]
    pos_right = poses[right_idx][query_idxs, object_idxs, :3, 3]  # [num_queries*n_valid_objects, 3]
    interp_pos = pos_left + (pos_right - pos_left) * fraction[query_idxs].unsqueeze(-1)

    interpolated_poses = torch.cat([interp_rot, interp_pos.unsqueeze(-1)], dim=-1)
    return interpolated_poses, query_idxs, object_idxs


def fast_ego_motion_poses(ego_motion: EgoMotion, timestamps: np.ndarray):
    return interpolate_trajectories_np(ego_motion.poses, ego_motion.timestamps, timestamps)


def fast_compensate_pointwise(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: LidarCalibration,
    target_timestamp: float,
    target_pose: Optional[np.ndarray] = None,
) -> LidarData:
    """Motion compensate a lidar point cloud in a pointwise manner."""
    # get source and target poses and compute relative transform (odometry)
    point_poses = fast_ego_motion_poses(ego_motion, lidar_data.timestamps)

    if target_pose is None:
        target_pose = ego_motion.get_poses(target_timestamp)

    odometry = inverse_np(target_pose) @ point_poses
    # Project lidar -> ego -> new-ego -> lidar
    lidar_data.transform(calibration.extrinsics)
    lidar_data.transform(odometry)
    lidar_data.transform(calibration.extrinsics.inverse)
    return lidar_data


def fast_uncompensate_pointwise(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: LidarCalibration,
) -> LidarData:
    """Uncompensate an already scan-compensated point cloud. (inverse of fast_compensate_pointwise)

    Given a point cloud that has been compensated to the scan timestamp, un-compensate each point to its original
    pose given the delta-time encoded in each point (x,y,z,r,t)
    """
    # get source (scan) and target (pointwise) poses and compute relative transform (odometry)
    point_pose = ego_motion.get_poses(lidar_data.core_timestamp)
    target_poses = fast_ego_motion_poses(ego_motion, lidar_data.timestamps)
    odometry = inverse_np(target_poses) @ point_pose
    # Project lidar -> ego -> new-ego -> lidar
    lidar_data.transform(calibration.extrinsics)
    lidar_data.transform(odometry)
    lidar_data.transform(calibration.extrinsics.inverse)
    return lidar_data
