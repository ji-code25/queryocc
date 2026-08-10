"""End-to-end smoke test of the QueryOcc network on CPU.

Guards the whole assembled path — image encoder -> lift-contract-splat -> BEV encoder ->
unified query decoder — for every release preset, which the per-component tests do not
cover. Shrunken to a tiny backbone/BEV so it runs in seconds without a GPU.
"""

import copy

import pytest
import torch

from queryocc.config.method_configs import method_configs

PRESETS = ["queryocc-nusc", "queryocc-lidar-nusc", "queryocc-joint-nusc", "queryocc-nusc-av2"]
EXPECTED_SOURCES = {
    "queryocc-nusc": ["pseudo_occupancy", "pseudo_semantic_segmentation"],
    "queryocc-lidar-nusc": ["occupancy", "semantic_segmentation"],
    "queryocc-joint-nusc": [
        "occupancy",
        "semantic_segmentation",
        "pseudo_occupancy",
        "pseudo_semantic_segmentation",
    ],
    "queryocc-nusc-av2": ["occupancy", "semantic_segmentation"],
}

B, N_CAMS, H, W = 1, 2, 64, 176


def build_net(preset: str):
    """Instantiate a preset's network, shrunk to CPU-testable size.

    Deep-copied: method_configs holds shared preset instances, so mutating them here
    would leak into every other test in the session.
    """
    config = copy.deepcopy(method_configs[preset])
    model_config = config.model
    model_config.net.compile = False
    model_config.net.cerveza_encoder.image_encoder_backbone = "resnet18"
    model_config.net.hidden_dim = 32
    model_config.roi = "surround_30"
    model_config.configure_dataset(config.datamodule.dataset)
    return model_config.setup().eval().net


def make_camera_batch():
    cams = [f"CAM{i}" for i in range(N_CAMS)]
    per_cam = lambda tensor: {c: tensor.clone() for c in cams}  # noqa: E731

    intrinsics = torch.eye(3).expand(B, 1, 3, 3).clone()
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 100.0
    intrinsics[..., 0, 2], intrinsics[..., 1, 2] = W / 2, H / 2
    identity = torch.eye(4).expand(B, 1, 4, 4).clone()

    return dict(
        images=per_cam(torch.rand(B, 1, 3, H, W)),
        intrinsics=per_cam(intrinsics),
        intrinsics_modified=per_cam(intrinsics),
        extrinsics=per_cam(identity),
        T_cam_ego=per_cam(identity),
        distortion=per_cam(torch.zeros(B, 1, 4)),
        imgs_h={c: torch.full((B,), H, dtype=torch.long) for c in cams},
        imgs_w={c: torch.full((B,), W, dtype=torch.long) for c in cams},
        timestamps=per_cam(torch.zeros(B, 1)),
        batch_size=B,
        num_cams=N_CAMS,
        num_timesteps=1,
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_supervision_sources_match_preset(preset):
    net = build_net(preset)
    assert net.one_head_decoder.active_sources == EXPECTED_SOURCES[preset]


@pytest.mark.parametrize("preset", PRESETS)
def test_encode_decode_produces_finite_predictions(preset):
    net = build_net(preset)
    num_classes = len(net.config.semantic_segmentation_classes)

    with torch.no_grad():
        encoding, _ = net.encode({}, make_camera_batch())
        assert encoding.shape[0] == B and encoding.shape[1] == net.config.hidden_dim
        assert torch.isfinite(encoding).all()

        queries = torch.rand(B, 512, 4) * 20 - 10  # (x, y, z, t) inside the ROI
        out = net.decode(encoding, queries, to_decode=["occupancy", "semantic_segmentation"])

    assert out["occupancy"].shape == (B, 512, 1)
    # +2 output channels: the noise class and the free class.
    assert out["semantic_segmentation_logits"].shape == (B, 512, num_classes + 2)
    assert out["semantic_segmentation_preds"].shape == (B, 512, 1)
    for name, value in out.items():
        assert torch.isfinite(value).all(), name


def test_gradients_reach_the_image_backbone():
    net = build_net("queryocc-nusc")
    encoding, _ = net.encode({}, make_camera_batch())
    queries = torch.rand(B, 128, 4) * 20 - 10
    net.decode(encoding, queries, to_decode=["occupancy"])["occupancy"].sum().backward()

    backbone_grads = [p.grad for p in net.image_encoder.parameters() if p.requires_grad and p.grad is not None]
    assert backbone_grads, "no gradients reached the image encoder"
    assert all(torch.isfinite(g).all() for g in backbone_grads)
