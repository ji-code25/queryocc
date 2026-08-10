"""Characterization tests for queryocc/model/components/image_encoders (refactor safety net).

Two guards against refactors of queryocc/model/components/image_encoders:

1. Forward-shape/finiteness tests on tiny CPU configs for each encoder family, so behavior
   changes surface as failures while modules move between files.
2. A state_dict key+shape snapshot per family (tests/snapshots/image_encoders_state_dict.json):
   checkpoint keys come from module *attribute paths*, so this file must stay green through
   every phase — a diff here means old checkpoints would no longer load.

The snapshot file is auto-created on first run (commit it); afterwards any mismatch fails.

Run: pytest tests/test_image_encoders_characterization.py
"""

import json
from pathlib import Path

import pytest
import torch

from queryocc.model.components.image_encoders.cerveza.encoder import CervezaImageEncoder, CervezaImageEncoderConfig
from queryocc.model.components.image_encoders.lifters.lss import LSSProConfig

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "image_encoders_state_dict.json"

FEATURE_DIM = 32
POINT_CLOUD_RANGE = (-8.0, -8.0, -2.0, 8.0, 8.0, 2.0)
VOXEL_SIZE = (1.0, 1.0, 1.0)  # -> 16x16 BEV grid
CANVAS_SIZE = (16, 16, 4)
IMG_H = IMG_W = 64  # resnet18 stage strides /8 /16 /32 -> neck output 8x8
NUM_DEPTH_BINS = 6


def make_cerveza_config(**overrides) -> CervezaImageEncoderConfig:
    kwargs = dict(
        image_encoder_backbone="resnet18",
        image_encoder_neck_type="simpleup",
        lss=LSSProConfig(depth_num_bins=NUM_DEPTH_BINS, max_depth=20.0, use_point_contraction=False),
    )
    kwargs.update(overrides)
    return CervezaImageEncoderConfig(**kwargs)


def make_cerveza(config=None) -> CervezaImageEncoder:
    return (config or make_cerveza_config()).setup(
        feature_dim=FEATURE_DIM,
        point_cloud_range=POINT_CLOUD_RANGE,
        voxel_size=VOXEL_SIZE,
        lifter_canvas_size=CANVAS_SIZE,
        point_contraction=None,
    )


def make_camera_batch(b=2, t=1, n_cams=2, h=IMG_H, w=IMG_W, seed=0):
    """Per-camera dicts in the imgs-stream layout the encoders consume."""
    torch.manual_seed(seed)
    cams = [f"cam_{i}" for i in range(n_cams)]
    intrinsics = torch.eye(3).repeat(b, t, 1, 1)
    intrinsics[..., 0, 0] = 50.0
    intrinsics[..., 1, 1] = 50.0
    intrinsics[..., 0, 2] = w / 2
    intrinsics[..., 1, 2] = h / 2
    T_cam_ego = torch.eye(4).repeat(b, t, 1, 1)
    T_cam_ego[..., :3, 3] = torch.randn(b, t, 3) * 0.1
    # Timestep 0 is "now", history timesteps are in the past (seconds, negative).
    ts = -0.5 * torch.arange(t, dtype=torch.float32).expand(b, t)
    return dict(
        images={c: torch.rand(b, t, 3, h, w) for c in cams},
        intrinsics={c: intrinsics.clone() for c in cams},
        intrinsics_modified={c: intrinsics.clone() for c in cams},
        extrinsics={c: T_cam_ego.clone() for c in cams},
        distortion={c: torch.zeros(b, t, 6) for c in cams},
        imgs_h={c: torch.full((b,), h, dtype=torch.long) for c in cams},
        imgs_w={c: torch.full((b,), w, dtype=torch.long) for c in cams},
        T_cam_ego={c: T_cam_ego.clone() for c in cams},
        timestamps={c: ts.clone() for c in cams},
        batch_size=b,
        num_cams=n_cams,
        num_timesteps=t,
    )


class TestCervezaForward:
    @pytest.mark.parametrize("neck", ["simpleup", "simplefpn"])
    def test_single_timestep_shapes(self, neck):
        encoder = make_cerveza(make_cerveza_config(image_encoder_neck_type=neck)).eval()
        batch = make_camera_batch(b=2, t=1)
        with torch.no_grad():
            bev, info = encoder(**batch)
        # 16x16 lifter grid, then init_bev_processing's stride-2 conv halves it.
        assert bev.shape == (2, FEATURE_DIM, 8, 8)
        assert torch.isfinite(bev).all()
        # supervise_depth=True (default) exposes the per-camera depth logits.
        assert info["img_depths"].shape == (2, 1, 2, 8, 8, NUM_DEPTH_BINS)

    def test_temporal_detach_history_path(self):
        """T=2 exercises the now/history split (detach_previous_inputs=True default)."""
        encoder = make_cerveza().eval()
        batch = make_camera_batch(b=1, t=2)
        with torch.no_grad():
            bev, info = encoder(**batch)
        assert bev.shape == (1, FEATURE_DIM, 8, 8)
        assert torch.isfinite(bev).all()
        assert info["img_depths"].shape == (1, 2, 2, 8, 8, NUM_DEPTH_BINS)

    def test_temporal_no_detach_path(self):
        encoder = make_cerveza(make_cerveza_config(image_encoder_detach_previous_inputs=False)).eval()
        batch = make_camera_batch(b=1, t=2)
        with torch.no_grad():
            bev, _ = encoder(**batch)
        assert bev.shape == (1, FEATURE_DIM, 8, 8)
        assert torch.isfinite(bev).all()

    def test_gradients_flow_to_backbone_and_lifter(self):
        encoder = make_cerveza().train()
        batch = make_camera_batch(b=1, t=1)
        bev, _ = encoder(**batch)
        bev.sum().backward()
        for probe in ("image_encoder", "lifter", "init_bev_processing"):
            module = getattr(encoder, probe)
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), probe


class TestLSSProStandalone:
    B, N, HF, WF, C = 2, 2, 6, 6, 16
    BINS = 5

    def make_lifter(self, **config_overrides):
        config = LSSProConfig(depth_num_bins=self.BINS, max_depth=20.0, use_point_contraction=False, **config_overrides)
        return config.setup(
            in_channels=self.C,
            out_channels=self.C,
            point_cloud_range=POINT_CLOUD_RANGE,
            voxel_size=VOXEL_SIZE,
            initial_features="zeros",
            points_positional_encoding="sinusoidal",
            point_contraction=None,
        )

    def make_inputs(self, seed=0):
        torch.manual_seed(seed)
        intr = torch.eye(3).repeat(self.B, self.N, 1, 1)
        intr[..., 0, 0] = intr[..., 1, 1] = 50.0
        intr[..., 0, 2] = intr[..., 1, 2] = 24.0
        tce = torch.eye(4).repeat(self.B, self.N, 1, 1)
        return dict(
            features=torch.randn(self.B, self.N, self.C, self.HF, self.WF),
            H_in=48,
            W_in=48,
            intrinsics=intr,
            extrinsics=tce.clone(),
            T_cam_ego=tce.clone(),
            distortion=torch.zeros(self.B, self.N, 6),
            num_cams=self.N,
            timestamps=torch.zeros(self.B, self.N),
        )

    def test_forward_shapes(self):
        lifter = self.make_lifter().eval()
        with torch.no_grad():
            bev, depth, bev_occupancy, target_infos = lifter(**self.make_inputs())
        assert bev.shape == (self.B, self.C, 16, 16)
        assert torch.isfinite(bev).all()
        assert depth.shape == (self.B, self.N, self.HF, self.WF, self.BINS)
        assert bev_occupancy is None
        assert target_infos == {}

    def test_lidar_target_generation(self):
        """The no_grad target/occupancy block (extracted in Phase 3) must keep its outputs."""
        lifter = self.make_lifter().eval()
        inputs = self.make_inputs()
        depth_maps = torch.rand(self.B, self.N, self.HF, self.WF) * 15.0 + 2.0
        depth_maps_valid = torch.ones_like(depth_maps, dtype=torch.bool)
        with torch.no_grad():
            bev, _, bev_occupancy, target_infos = lifter(
                **inputs,
                depth_maps=depth_maps,
                depth_maps_valid=depth_maps_valid,
                time_indices=torch.zeros(self.B, dtype=torch.long),
            )
        assert bev_occupancy.shape == (self.B, 16, 16)
        assert torch.isfinite(bev_occupancy).all()
        assert set(target_infos) == {"target_bev_occupancy", "target_bev_visibility", "target_bev_valids"}
        for key, value in target_infos.items():
            assert value.shape == (self.B, 16, 16), key
            assert torch.isfinite(value).all(), key
        # Valid depths within range must mark at least some cells occupied/valid.
        assert target_infos["target_bev_valids"].sum() > 0
        assert target_infos["target_bev_occupancy"].sum() > 0

    def test_gradients_reach_depth_network(self):
        lifter = self.make_lifter().train()
        bev, _, _, _ = lifter(**self.make_inputs())
        bev.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lifter.depth_network.parameters())


class TestResnetInstanceNormSwap:
    """The BN->IN swap reassigned a loop variable and never touched the model (fixed in Phase 1)."""

    def test_swap_replaces_all_batch_norms(self):
        from queryocc.model.components.image_encoders.backbones.resnet import Resnet18

        model = Resnet18(swap_batch_norm_to_instance_norm=True)
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
        assert any(isinstance(m, torch.nn.InstanceNorm2d) for m in model.modules())
        outs = model(torch.rand(1, 3, 64, 64))
        assert all(torch.isfinite(o).all() for o in outs)

    def test_default_keeps_batch_norm(self):
        from queryocc.model.components.image_encoders.backbones.resnet import Resnet18

        model = Resnet18()
        assert any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())


def _state_dict_summary(module: torch.nn.Module) -> dict:
    return {key: list(value.shape) for key, value in sorted(module.state_dict().items())}


def build_snapshot_families():
    families = {
        "cerveza_resnet18_simpleup_lss": make_cerveza,
        "cerveza_resnet18_simplefpn_lss": lambda: make_cerveza(
            make_cerveza_config(image_encoder_neck_type="simplefpn")
        ),
    }
    return families


@pytest.mark.parametrize("family", sorted(build_snapshot_families()))
def test_state_dict_snapshot(family):
    """Checkpoint-compat guard: state_dict keys (attribute paths) and shapes must not change.

    On first run the snapshot is written and the test passes — commit the file. A later
    mismatch means existing checkpoints would fail to load: fix the code, not the snapshot
    (unless the break is intentional and remapped on load).
    """
    module = build_snapshot_families()[family]()
    summary = _state_dict_summary(module)

    snapshot = json.loads(SNAPSHOT_PATH.read_text()) if SNAPSHOT_PATH.exists() else {}
    if family not in snapshot:
        snapshot[family] = summary
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"snapshot for '{family}' created — commit {SNAPSHOT_PATH}")

    expected = snapshot[family]
    missing = sorted(set(expected) - set(summary))
    added = sorted(set(summary) - set(expected))
    changed = sorted(k for k in set(expected) & set(summary) if expected[k] != summary[k])
    assert not (missing or added or changed), (
        f"state_dict drift for '{family}' — old checkpoints would break.\n"
        f"missing: {missing[:10]}\nadded: {added[:10]}\nshape-changed: {changed[:10]}"
    )
