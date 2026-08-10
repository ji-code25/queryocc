"""Pose-trajectory interpolation (quaternion SLERP + translation lerp).

Pure-PyTorch replacement for the former fast_interp_cpu/fast_interp_cuda
C++/CUDA extensions: identical semantics (searchsorted segment lookup, shortest
-path SLERP with a normalized-lerp fallback for nearly parallel quaternions,
epsilon-guarded interpolation fraction), batched over all queries at once, and
device-agnostic. Kept self-contained (no queryocc imports) because
utils.geometry.poses imports this module.
"""

import numpy as np
import torch
from torch import Tensor

_EPSILON = 1e-6
_SLERP_LERP_DOT_THRESHOLD = 0.9995


def _rotmat_to_quat(matrix: Tensor) -> Tensor:
    """Convert rotation matrices (..., 3, 3) to unit quaternions (..., 4), wxyz order.

    Branch-free batched version of Shepperd's method: compute all four candidate
    quaternions and select per-element the one with the largest pivot, matching
    the branchy scalar implementation of the former C++ extension.
    """
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # Squared magnitudes (times 4) of w, x, y, z as pivot scores.
    tr = m00 + m11 + m22
    pivots = torch.stack([1.0 + tr, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1)
    case = pivots.argmax(dim=-1)
    s = torch.sqrt(torch.clamp(torch.gather(pivots, -1, case.unsqueeze(-1)).squeeze(-1), min=_EPSILON)) * 2.0

    q_w = torch.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], dim=-1)
    q_x = torch.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], dim=-1)
    q_y = torch.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], dim=-1)
    q_z = torch.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], dim=-1)

    quats = torch.stack([q_w, q_x, q_y, q_z], dim=-2)  # (..., 4 cases, 4)
    index = case[..., None, None].expand(*case.shape, 1, 4)
    return torch.gather(quats, -2, index).squeeze(-2)


def _quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert unit quaternions (..., 4), wxyz order, to rotation matrices (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    row0 = torch.stack([w2 + x2 - y2 - z2, 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1)
    row1 = torch.stack([2 * (x * y + w * z), w2 - x2 + y2 - z2, 2 * (y * z - w * x)], dim=-1)
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w2 - x2 - y2 + z2], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _quat_slerp(q0: Tensor, q1: Tensor, alpha: Tensor) -> Tensor:
    """Shortest-path SLERP between quaternion batches (..., 4) with fraction alpha (...,)."""
    dot = (q0 * q1).sum(dim=-1)
    q1 = torch.where(dot[..., None] < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * alpha
    # Guard the denominator; the lerp branch is selected where it matters.
    s0 = torch.cos(theta) - dot * torch.sin(theta) / sin_theta_0.clamp(min=_EPSILON)
    s1 = torch.sin(theta) / sin_theta_0.clamp(min=_EPSILON)
    slerped = s0[..., None] * q0 + s1[..., None] * q1

    lerped = q0 + alpha[..., None] * (q1 - q0)
    lerped = lerped / lerped.norm(dim=-1, keepdim=True).clamp(min=_EPSILON)

    return torch.where((dot > _SLERP_LERP_DOT_THRESHOLD)[..., None], lerped, slerped)


def interpolate_trajectories(poses: Tensor, pose_times: Tensor, query_times: Tensor, clamp_frac: bool = True):
    """
    Interpolate trajectories (sequences of poses) at query timestamps.

    Args:
        poses (Tensor): Shape (..., N, 4, 4) pose matrices
        pose_times (Tensor): Shape (..., N) timestamps for poses
        query_times (Tensor): Shape (..., M) query timestamps
        clamp_frac (bool): Whether to clamp interpolation fraction to [0,1]

    Returns:
        Tensor: Shape (..., M, 4, 4) interpolated poses
    """
    # Check inputs
    if not torch.is_floating_point(poses):
        raise TypeError("poses must be a floating point tensor")
    if len({poses.device, pose_times.device, query_times.device}) > 1:
        raise ValueError("All input tensors must be on the same device")
    if pose_times.shape != poses.shape[:-2]:
        raise ValueError("pose_times must match poses batch and trajectory length dimensions")
    if query_times.shape[:-1] != poses.shape[:-3]:
        raise ValueError("query_times must match poses batch dimensions")
    if poses.shape[-2:] != (4, 4):
        raise ValueError("poses must be 4x4 matrices")

    poses, pose_times, query_times = poses.contiguous(), pose_times.contiguous(), query_times.contiguous()
    n = poses.shape[-3]
    right = torch.searchsorted(pose_times, query_times).clamp(min=1, max=n - 1)
    left = right - 1
    pose_times, query_times = pose_times.to(poses), query_times.to(poses)

    left_time = torch.gather(pose_times, -1, left)
    right_time = torch.gather(pose_times, -1, right)
    alpha = (query_times - left_time) / (right_time - left_time + _EPSILON)
    if clamp_frac:
        alpha = alpha.clamp(0.0, 1.0)

    gather_idx = left[..., None, None].expand(*left.shape, 4, 4)
    pose_left = torch.gather(poses, -3, gather_idx)
    pose_right = torch.gather(poses, -3, right[..., None, None].expand(*right.shape, 4, 4))

    q_interp = _quat_slerp(_rotmat_to_quat(pose_left[..., :3, :3]), _rotmat_to_quat(pose_right[..., :3, :3]), alpha)

    pose_interp = torch.zeros((*query_times.shape, 4, 4), device=poses.device, dtype=poses.dtype)
    pose_interp[..., :3, :3] = _quat_to_rotmat(q_interp)
    pose_interp[..., :3, 3] = (1.0 - alpha[..., None]) * pose_left[..., :3, 3] + alpha[..., None] * pose_right[
        ..., :3, 3
    ]
    pose_interp[..., 3, 3] = 1.0
    return pose_interp


def interpolate_trajectories_np(
    poses: np.ndarray, pose_times: np.ndarray, query_times: np.ndarray, clamp_frac: bool = True
):
    """
    Interpolate trajectories (sequences of poses) using numpy.
    """
    return interpolate_trajectories(
        torch.from_numpy(poses), torch.from_numpy(pose_times), torch.from_numpy(query_times), clamp_frac
    ).numpy()
