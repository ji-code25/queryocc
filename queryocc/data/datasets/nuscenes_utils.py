from typing import Optional

import numpy as np
from pyquaternion import Quaternion
from zod.data_classes.ego_motion import EgoMotion

from queryocc.common.types import PointCloudRange
from queryocc.utils.geometry import fast_ego_motion_poses, inverse_np

EGO_POINTS_RANGE = (1.0, 2.5, 2.0)


def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm


def get_ego_from_lidar(info):
    ego_from_lidar = trans_matrix(
        np.array(info["sensor2ego_translation"]),
        Quaternion(info["sensor2ego_rotation"]),
    )
    return ego_from_lidar


def get_global_pose(info, inverse=False):
    global_from_ego = trans_matrix(np.array(info["ego2global_translation"]), Quaternion(info["ego2global_rotation"]))
    ego_from_lidar = get_ego_from_lidar(info)
    pose = global_from_ego.dot(ego_from_lidar)
    if inverse:
        pose = np.linalg.inv(pose)
    return pose


def ego_motion_compensate_scanwise(
    pc: np.ndarray,
    ego_motion: EgoMotion,
    lidar_extrinsics: np.ndarray,
    target_timestamp: Optional[float] = None,
    target_pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    # returns lidar in ego frame
    assert target_timestamp is not None or target_pose is not None, (
        "Either target_timestamp or target_pose must be provided."
    )
    xyz = pc[..., :3].copy()
    pc_pose = ego_motion.get_poses(pc[..., -1].max())  # all points are in the same coordinate system currently.max()

    if target_pose is None:
        assert target_timestamp is not None, "Target timestamp must be provided if target_pose is not."
        target_pose = ego_motion.get_poses(target_timestamp)

    # lets put them in the ego-frame
    odometry = inverse_np(target_pose) @ pc_pose @ lidar_extrinsics

    rotations = odometry[..., :3, :3]
    translations = odometry[..., :3, 3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)

    pc[..., :3] = xyz
    return pc


def points_in_target_bev(
    lidar_sweep: np.ndarray,
    ego_motion: EgoMotion,
    point_cloud_range: PointCloudRange,
    bev_oxts_pose: np.ndarray,
    lidar_extrinsics: np.ndarray,
    lidar_timestamp: float,
    output_xyz: bool = False,
) -> np.ndarray:
    xyz = lidar_sweep[..., :3].copy()
    # motion compensate scanwise to the target timestamp
    source_pose = ego_motion.get_poses(lidar_timestamp)
    target_pose = bev_oxts_pose
    # go to the target pose
    odometry = inverse_np(target_pose) @ source_pose @ lidar_extrinsics
    rotations = odometry[..., :3, :3]
    translations = odometry[..., :3, 3]
    # ego frame at the BEV timestamp
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)

    # check bev range
    min_range = np.array(point_cloud_range[:3])
    max_range = np.array(point_cloud_range[3:])
    in_range = np.all((min_range <= xyz) & (xyz <= max_range), axis=-1)
    # restore lidar_data (we don't want to overwrite)
    if output_xyz:
        return in_range, xyz[in_range]
    else:
        return in_range


def fast_uncompensate_pointwise(
    sweep: np.ndarray, ego_motion: EgoMotion, lidar_extrinsics: np.ndarray, sweep_timestamp: float
) -> np.ndarray:
    """Uncompensate an already scan-compensated point cloud. (inverse of fast_compensate_pointwise)

    Given a point cloud that has been compensated to the scan timestamp, un-compensate each point to its original
    pose given the delta-time encoded in each point (x,y,z,r,t)
    """
    # get source (scan) and target (pointwise) poses and compute relative transform (odometry)
    point_pose = ego_motion.get_poses(sweep_timestamp)
    target_poses = fast_ego_motion_poses(ego_motion, sweep[..., -1])
    odometry = inverse_np(lidar_extrinsics) @ inverse_np(target_poses) @ point_pose @ lidar_extrinsics
    # Project lidar -> ego -> new-ego -> lidar
    rotations = odometry[..., :3, :3]
    translations = odometry[..., :3, 3]
    xyz = sweep[..., :3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)
    sweep[..., :3] = xyz

    return sweep



def remove_ego_points_nusc(pc: np.ndarray) -> np.ndarray:
    mask = (abs(pc[..., :3]) >= EGO_POINTS_RANGE).any(axis=-1)
    pc = pc[mask]
    return pc
