"""Tests for the pure-torch pose-trajectory interpolation (former fast_interp extension)."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation, Slerp

from queryocc.model.ops.functions.interpolate import interpolate_trajectories


def _random_trajectory(rng, n_poses):
    rotations = Rotation.random(n_poses, random_state=rng)
    translations = rng.uniform(-50.0, 50.0, size=(n_poses, 3))
    poses = np.tile(np.eye(4), (n_poses, 1, 1))
    poses[:, :3, :3] = rotations.as_matrix()
    poses[:, :3, 3] = translations
    times = np.sort(rng.uniform(0.0, 10.0, size=n_poses))
    return poses, times, rotations


def _scipy_reference(poses, times, rotations, query_times):
    """Endpoint-clamped SLERP + translation lerp, matching clamp_frac=True."""
    clamped = np.clip(query_times, times[0], times[-1])
    rot = Slerp(times, rotations)(clamped).as_matrix()
    out = np.tile(np.eye(4), (len(query_times), 1, 1))
    out[:, :3, :3] = rot
    for axis in range(3):
        out[:, axis, 3] = np.interp(clamped, times, poses[:, axis, 3])
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scipy_slerp(seed):
    rng = np.random.default_rng(seed)
    poses, times, rotations = _random_trajectory(rng, n_poses=7)
    # Queries inside the range plus queries beyond both ends (exercise clamping).
    query_times = np.concatenate([rng.uniform(0.0, 10.0, size=64), [-1.0, times[0], times[-1], 12.0]])

    result = interpolate_trajectories(
        torch.from_numpy(poses), torch.from_numpy(times), torch.from_numpy(query_times), clamp_frac=True
    ).numpy()
    expected = _scipy_reference(poses, times, rotations, query_times)

    # The interpolation fraction uses an epsilon-guarded denominator (inherited
    # from the original fast_interp extension), which perturbs alpha by ~1e-6
    # relative to scipy's exact fraction — hence the loose tolerance.
    np.testing.assert_allclose(result, expected, atol=2e-3)


def test_endpoints_near_exact():
    rng = np.random.default_rng(3)
    poses, times, _ = _random_trajectory(rng, n_poses=5)
    result = interpolate_trajectories(
        torch.from_numpy(poses), torch.from_numpy(times), torch.from_numpy(times), clamp_frac=True
    ).numpy()
    # Not bit-exact at segment endpoints because of the epsilon-guarded
    # denominator (alpha slightly below 1 at the right endpoint; worst on
    # short segments).
    np.testing.assert_allclose(result, poses, atol=1e-3)


def test_output_is_rigid_transform():
    rng = np.random.default_rng(4)
    poses, times, _ = _random_trajectory(rng, n_poses=6)
    query_times = rng.uniform(-2.0, 12.0, size=128)
    result = interpolate_trajectories(
        torch.from_numpy(poses), torch.from_numpy(times), torch.from_numpy(query_times), clamp_frac=True
    ).numpy()

    rot = result[:, :3, :3]
    np.testing.assert_allclose(rot @ rot.transpose(0, 2, 1), np.tile(np.eye(3), (len(result), 1, 1)), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-6)
    np.testing.assert_allclose(result[:, 3, :], np.tile([0, 0, 0, 1], (len(result), 1)), atol=0)


def test_batched_matches_unbatched():
    rng = np.random.default_rng(5)
    batches = [_random_trajectory(rng, n_poses=6) for _ in range(3)]
    poses = torch.from_numpy(np.stack([b[0] for b in batches]))
    times = torch.from_numpy(np.stack([b[1] for b in batches]))
    query_times = torch.from_numpy(rng.uniform(0.0, 10.0, size=(3, 32)))

    batched = interpolate_trajectories(poses, times, query_times, clamp_frac=True)
    for i in range(3):
        single = interpolate_trajectories(poses[i], times[i], query_times[i], clamp_frac=True)
        torch.testing.assert_close(batched[i], single)


def test_extrapolation_without_clamp():
    # With clamp_frac=False the last segment extrapolates linearly in translation.
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, 0, 3] = 1.0  # move 1m in x between t=0 and t=1
    times = np.array([0.0, 1.0])
    result = interpolate_trajectories(
        torch.from_numpy(poses), torch.from_numpy(times), torch.tensor([2.0], dtype=torch.float64), clamp_frac=False
    ).numpy()
    np.testing.assert_allclose(result[0, 0, 3], 2.0, atol=1e-5)


def test_float32_dtype():
    rng = np.random.default_rng(6)
    poses, times, rotations = _random_trajectory(rng, n_poses=5)
    query_times = rng.uniform(0.0, 10.0, size=16)
    result = interpolate_trajectories(
        torch.from_numpy(poses).float(), torch.from_numpy(times).float(), torch.from_numpy(query_times).float()
    )
    assert result.dtype == torch.float32
    expected = _scipy_reference(poses, times, rotations, query_times)
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-4)
