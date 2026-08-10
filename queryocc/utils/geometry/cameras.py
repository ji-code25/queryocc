"""Camera projection models."""

import torch


def batched_project_kannala(
    points_3d: torch.Tensor,
    camera_matrix: torch.Tensor,
    distortion_coefs: torch.Tensor,
) -> torch.Tensor:
    """Project 3d points to 2d using Kannala-Brandt model with batch support.

    Args:
        points_3d: 3d points in shape (B, N, 3)
        camera_matrix: Camera matrix in shape (B, 3, 4)
        distortion_coefs: Distortion coefficients in shape (B, 4)

    Returns:
        2d points in shape (B, N, 2)
    """

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

    return projected_points
