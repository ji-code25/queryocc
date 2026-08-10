from math import sqrt
from typing import Literal

import torch

from queryocc.common.types import PointCloudRange


def get_depths(
    depth_distribution: Literal["uniform", "log", "log_linear_blend"],
    point_cloud_range: PointCloudRange,
    depth_num_bins: int,
    max_depth: float = 100.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    x_low_pc, y_low_pc = point_cloud_range[:2]
    x_high_pc, y_high_pc = point_cloud_range[3:5]
    low_range_pc = sqrt(x_low_pc**2 + y_low_pc**2)
    high_range_pc = sqrt(x_high_pc**2 + y_high_pc**2)
    max_range_pc = max(low_range_pc, high_range_pc)

    near = 0.5
    if max_depth > near:
        far = max_depth
    else:
        far = max_range_pc

    if depth_distribution == "uniform":
        depths = torch.linspace(near, far, depth_num_bins, dtype=torch.float32)  # [D]

    elif depth_distribution == "log":
        t = torch.linspace(1e-6, 1, depth_num_bins, dtype=torch.float32)
        depths = near * (far / near) ** (t**gamma)

    elif depth_distribution == "log_linear_blend":
        t = torch.linspace(1e-6, 1, depth_num_bins, dtype=torch.float32)
        log_depths = near * (far / near) ** t
        lin_depths = near + (far - near) * t
        depths = (1 - gamma) * log_depths + gamma * lin_depths

    else:
        raise ValueError(f"Unknown depth distribution: {depth_distribution}")

    return depths
