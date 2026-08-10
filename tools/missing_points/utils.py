"""Shared helpers for the offline "missing points" (free-ray) generator.

Lifted verbatim (modulo import renames) from the script that produced the published
missing-points caches. ``engine.py`` needs exactly three things from here: the light
``Lidars`` container that carries a log's lidar trajectory, the nuScenes quaternion ->
4x4 pose helper, and the ego-motion *un*-compensation that puts each point back into
the sensor frame it was actually measured from.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pyquaternion
import torch
from torch import Tensor

from queryocc.utils.geometry import interpolate_trajectories, inverse as pose_inverse


@dataclass(init=False)
class Lidars:
    """Dataparser outputs for the lidar dataset and the ray generator.

    If a single value is provided, it is broadcasted to all lidars.

    Args:
        lidar_to_worlds: Lidar to world matrices. Tensor of per-image l2w matrices, in [R | t] format
        lidar_type: Type of lidar sensor. This will be an int corresponding to the LidarType enum.
        assume_ego_compensated: Whether or not to assume that points are ego-compensated.
        times: Timestamps for each lidar
        n_points: Number of points in each lidar
        metadata: Additional metadata or data needed for interpolation, will mimic shape of the lidars
            and will be broadcasted to the rays generated from any derivative RaySamples we create with this
    """

    lidar_to_worlds: Tensor
    lidar_type: Tensor
    times: Optional[Tensor]
    metadata: Optional[Dict]
    azimuths: Optional[Tensor]
    elevations: Optional[Tensor]

    def __init__(
        self,
        lidar_to_worlds: Tensor,
        assume_ego_compensated: bool = True,
        times: Optional[Tensor] = None,
        metadata: Optional[Dict] = None,
        azimuths: Optional[Tensor] = None,
        elevations: Optional[Tensor] = None,
    ) -> None:
        """Initializes the Lidars object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions Shaped[Tensor, "3 4"]
        (in the case of the c2w matrices), Shaped[Tensor, "6"] (in the case of distortion params), or
        Shaped[Tensor, "1"] (in the case of the rest of the elements). The dimensions before that are
        considered the batch dimension of that tensor (batch_c2ws, batch_fxs, etc.). We will broadcast
        all the tensors to be the same batch dimension. This means you can use any combination of the
        input types in the function signature and it won't break. Your batch size for all tensors
        must be broadcastable to the same size, and the resulting number of batch dimensions will be
        the batch dimension with the largest number of dimensions.
        """

        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"lidar_to_worlds": 2, "azimuths": 2, "elevations": 2}

        self.lidar_to_worlds = lidar_to_worlds

        # @dataclass's post_init will take care of broadcasting
        self.times = self._init_get_times(times)

        self.metadata = metadata

        self.azimuths = self._init_get_azimuths(azimuths)
        self.elevations = self._init_get_elevations(elevations)

        self.assume_ego_compensated = assume_ego_compensated

    def _init_get_times(self, times: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if times is None:
            times = None
        elif isinstance(times, torch.Tensor):
            if times.ndim == 0 or times.shape[-1] != 1:
                times = times.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"times must be None or a tensor, got {type(times)}")

        return times

    def _init_get_azimuths(self, azimuths: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if azimuths is None:
            azimuths = None
        elif isinstance(azimuths, torch.Tensor):
            if azimuths.ndim == 0 or azimuths.shape[-1] != 1:
                azimuths = azimuths.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"azimuths must be None or a tensor, got {type(azimuths)}")

        return azimuths

    def _init_get_elevations(self, elevations: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if elevations is None:
            elevations = None
        elif isinstance(elevations, torch.Tensor):
            if elevations.ndim == 0 or elevations.shape[-1] != 1:
                elevations = elevations.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"elevations must be None or a tensor, got {type(elevations)}")

        return elevations

    @property
    def device(self) -> Union[torch.device, str]:
        """Returns the device that the camera is on."""
        return self.lidar_to_worlds.device

    @property
    def width(self) -> int:
        """Returns the width of the lidar."""
        return self.azimuths.shape[1] if self.azimuths is not None else 0

    @property
    def height(self) -> int:
        """Returns the height of the lidar."""
        return self.elevations.shape[1] if self.elevations is not None else 0


def _rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose


def _remove_ego_motion_compensation(
    point_cloud: torch.Tensor, l2ws: torch.Tensor, times: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removes ego motion compensation from point cloud.

    Args:
        point_cloud: Point cloud to remove ego motion compensation from (in world frame). Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
        l2ws: Poses of the lidar. Shape: [num_poses, 4, 4]
        times: Timestamps of the lidar poses. Shape: [num_poses]

    Returns:
        Point cloud without ego motion compensation in sensor frame. Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
        Lidar pose for each point in the point cloud. Shape: [num_points, 4, 4]
    """

    interpolated_l2ws, _, _ = interpolate_trajectories(
        l2ws.unsqueeze(1), times - times.min(), point_cloud[:, 4] - times.min(), clamp_frac=False
    )
    interpolated_l2ws = interpolated_l2ws[:, :3, :4]
    interpolated_w2ls = pose_inverse(interpolated_l2ws)
    homogen_points = torch.cat([point_cloud[:, :3], torch.ones_like(point_cloud[:, -1:])], dim=-1)
    points = torch.matmul(interpolated_w2ls, homogen_points.unsqueeze(-1))[:, :, 0]
    return torch.cat([points, point_cloud[:, 3:]], dim=-1), interpolated_l2ws
