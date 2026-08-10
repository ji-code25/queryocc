"""Shared camera-geometry helpers for the image encoders.

Everything here inverts calibration matrices ANALYTICALLY. Never call ``torch.inverse`` /
``torch.linalg.inv`` on GPU inside a training graph: cuSOLVER can return silent inf/NaN on a
degenerate/near-singular input, which propagates to the loss; under AMP the GradScaler then skips
the step on just that rank, desyncing DDP into a gradient-all-reduce deadlock. The closed
forms below are exact for their input classes (rigid
transforms, upper-triangular pinhole K) and also avoid a torch.compile recompile.
"""

import torch
from torch import Tensor


def invert_rigid_transform(T: Tensor) -> Tensor:
    """Closed-form inverse of a rigid (rotation + translation) transform, (..., 4, 4).

    R' = R^T, t' = -R^T t. The bottom row is copied from the input, so [0, 0, 0, 1] stays exact.
    """
    rotation = T[..., :3, :3]
    translation = T[..., :3, 3]
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -(rotation_inv @ translation.unsqueeze(-1))
    top = torch.cat([rotation_inv, translation_inv], dim=-1)  # (..., 3, 4)
    return torch.cat([top, T[..., 3:, :]], dim=-2)


def invert_intrinsics(K: Tensor) -> Tensor:
    """Closed-form inverse of an upper-triangular pinhole K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]].

    Exact for any skew s (our datasets have s = 0). fx/fy are clamped so a garbage frame cannot
    produce inf; matches torch.inverse to float precision on valid K.
    """
    fx = K[..., 0, 0]
    skew = K[..., 0, 1]
    cx = K[..., 0, 2]
    fy = K[..., 1, 1]
    cy = K[..., 1, 2]
    inv_fx = torch.where(fx < 0, -1.0, 1.0) / fx.abs().clamp(min=1e-9)
    inv_fy = torch.where(fy < 0, -1.0, 1.0) / fy.abs().clamp(min=1e-9)
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    row0 = torch.stack([inv_fx, -skew * inv_fx * inv_fy, (skew * cy - cx * fy) * inv_fx * inv_fy], dim=-1)
    row1 = torch.stack([zeros, inv_fy, -cy * inv_fy], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
