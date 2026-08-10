"""Shared geometry helpers: SE(3) poses, quaternions, and camera projection."""

from queryocc.utils.geometry.cameras import batched_project_kannala
from queryocc.utils.geometry.poses import (
    fast_compensate_pointwise,
    fast_ego_motion_poses,
    fast_uncompensate_pointwise,
    interpolate_trajectories,
    inverse,
    inverse_np,
    multiply,
    to4x4,
    transform_points,
)
from queryocc.utils.geometry.rotations import (
    quat_product,
    rotmat_to_unitquat,
    rotvec_to_unitquat,
    unitquat_slerp,
    unitquat_slerp_fast,
    unitquat_to_rotmat,
    unitquat_to_rotvec,
)

__all__ = [
    "batched_project_kannala",
    "fast_compensate_pointwise",
    "fast_ego_motion_poses",
    "fast_uncompensate_pointwise",
    "interpolate_trajectories",
    "inverse",
    "inverse_np",
    "multiply",
    "quat_product",
    "rotmat_to_unitquat",
    "rotvec_to_unitquat",
    "to4x4",
    "transform_points",
    "unitquat_slerp",
    "unitquat_slerp_fast",
    "unitquat_to_rotmat",
    "unitquat_to_rotvec",
]
