"""Unit tests for image_encoders/geometry.py — the analytic matrix inverses that replace
torch.inverse (the DDP landmine, see geometry.py's module docstring) must match it on valid inputs.

Run: pytest tests/test_image_encoders_geometry.py
"""

import torch

from queryocc.model.components.image_encoders.geometry import invert_intrinsics, invert_rigid_transform


def random_rigid(b, n, seed=0):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(b, n, 3, 3, dtype=torch.float64))
    q = q * torch.linalg.det(q)[..., None, None].sign()
    T = torch.eye(4, dtype=torch.float64).repeat(b, n, 1, 1)
    T[..., :3, :3] = q
    T[..., :3, 3] = torch.randn(b, n, 3, dtype=torch.float64) * 5
    return T


def test_invert_rigid_transform_matches_torch_inverse():
    T = random_rigid(2, 3)
    expected = torch.inverse(T)  # CPU-only test; production code never calls this
    torch.testing.assert_close(invert_rigid_transform(T), expected, atol=1e-10, rtol=1e-10)


def test_invert_rigid_transform_roundtrip():
    T = random_rigid(1, 4, seed=1)
    eye = invert_rigid_transform(T) @ T
    torch.testing.assert_close(eye, torch.eye(4, dtype=T.dtype).expand_as(eye), atol=1e-12, rtol=0.0)


def test_invert_intrinsics_matches_torch_inverse_including_skew():
    torch.manual_seed(2)
    K = torch.eye(3, dtype=torch.float64).repeat(2, 3, 1, 1)
    K[..., 0, 0] = torch.rand(2, 3, dtype=torch.float64) * 1000 + 100  # fx
    K[..., 1, 1] = torch.rand(2, 3, dtype=torch.float64) * 1000 + 100  # fy
    K[..., 0, 2] = torch.rand(2, 3, dtype=torch.float64) * 800  # cx
    K[..., 1, 2] = torch.rand(2, 3, dtype=torch.float64) * 600  # cy
    K[..., 0, 1] = torch.randn(2, 3, dtype=torch.float64) * 2  # skew (zero in our datasets, exact anyway)
    expected = torch.inverse(K)
    torch.testing.assert_close(invert_intrinsics(K), expected, atol=1e-12, rtol=1e-12)


def test_invert_intrinsics_degenerate_input_stays_finite():
    """A garbage frame (fx = fy = 0) must produce finite values, not inf/NaN — the whole point
    of the analytic path is that a bad sample cannot poison the loss and desync DDP."""
    K = torch.zeros(1, 3, 3)
    K[..., 2, 2] = 1.0
    assert torch.isfinite(invert_intrinsics(K)).all()


def test_no_torch_inverse_in_lifter_or_geometry_source():
    """Source guard, mirroring test_cross_camera_mixer: the LSS lifter used torch.inverse on GPU
    until Phase 2 of the image_encoders refactor; keep it (and geometry.py itself) clean."""
    import inspect

    from queryocc.model.components.image_encoders import geometry
    from queryocc.model.components.image_encoders.lifters import lss

    for module in (geometry, lss):
        src = inspect.getsource(module)
        assert "torch.inverse(" not in src, module.__name__
        assert "linalg.inv(" not in src, module.__name__
