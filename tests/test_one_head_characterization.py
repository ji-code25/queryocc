"""Forward characterization for OneHeadDecoder: the four QueryOcc supervision sources
(occupancy, pseudo_occupancy, semantic_segmentation, pseudo_semantic_segmentation) must
produce exactly the expected loss keys, all finite, with a finite total.

Run: pytest tests/test_one_head_characterization.py
"""

import torch

from queryocc.data.datasets.nuscenes_mappings import LabelMapper
from queryocc.model.components.decoders.one_head import OneHeadDecoder, OneHeadDecoderConfig

PCR = (-8.0, -8.0, -2.0, 8.0, 8.0, 2.0)
CLASSES = ("car", "pedestrian", "vegetation")
B, Q = 2, 48


def make_decoder():
    cfg = OneHeadDecoderConfig(
        occupancy=True,
        pseudo_occupancy=True,
        semantic_segmentation=True,
        pseudo_semantic_segmentation=True,
        hidden_dim=16,
        offset_dim=16,
        num_resblocks=1,
        num_offsets=1,
    )
    return OneHeadDecoder(
        cfg,
        input_dim=32,
        point_cloud_range=PCR,
        pc_dim=4,
        semantic_segmentation_classes=CLASSES,
    )


def make_inputs(seed=0):
    torch.manual_seed(seed)

    def query_group():
        xyz = torch.rand(B, Q, 3) * 12 - 6
        t = torch.rand(B, Q, 1) * 4 - 2
        return torch.cat([xyz, t], dim=-1)

    def semantic_labels():
        return {
            "semantic_segmentation_labels": torch.randint(0, len(CLASSES), (B, Q, 1)),
            "semantic_segmentation_valid": torch.ones(B, Q, 1),
        }

    return {
        "occupancy": {
            "occupancy_queries": query_group(),
            "occupancy_label": (torch.rand(B, Q, 1) > 0.5).float(),
        },
        "pseudo_occupancy": {
            "occupancy_queries": query_group(),
            "occupancy_label": (torch.rand(B, Q, 1) > 0.5).float(),
        },
        # semantic sources share their occupancy source's query group (row-aligned labels).
        "semantic_segmentation": semantic_labels(),
        "pseudo_semantic_segmentation": semantic_labels(),
    }


EXPECTED_LOSS_KEYS = {
    "occupancy_loss",
    "pseudo_occupancy_loss",
    "semantic_segmentation_loss",
    "pseudo_semantic_segmentation_loss",
}


def test_forward_loss_keys_and_finiteness():
    decoder = make_decoder()
    bev = torch.randn(B, 32, 16, 16, requires_grad=True)
    out = decoder.forward(bev, make_inputs(), label_mapper=LabelMapper(CLASSES))

    assert set(out["one_head_losses"]) == EXPECTED_LOSS_KEYS
    for key, loss in out["one_head_losses"].items():
        assert torch.isfinite(loss), key
    assert torch.isfinite(out["one_head_total_loss"])

    out["one_head_total_loss"].backward()
    assert bev.grad is not None and torch.isfinite(bev.grad).all()


def test_empty_pseudo_queries_use_grad_anchor():
    decoder = make_decoder()
    bev = torch.randn(B, 32, 16, 16, requires_grad=True)
    inputs = make_inputs()
    inputs["pseudo_occupancy"]["occupancy_queries"] = torch.zeros(B, 0, 4)
    inputs["pseudo_occupancy"]["occupancy_label"] = torch.zeros(B, 0, 1)
    inputs["pseudo_semantic_segmentation"] = {
        "semantic_segmentation_labels": torch.zeros(B, 0, 1, dtype=torch.long),
        "semantic_segmentation_valid": torch.zeros(B, 0, 1),
    }
    out = decoder.forward(bev, inputs, label_mapper=LabelMapper(CLASSES))
    assert set(out["one_head_losses"]) == EXPECTED_LOSS_KEYS
    assert torch.isfinite(out["one_head_total_loss"])


def test_forward_is_deterministic_in_eval():
    decoder = make_decoder()
    decoder.eval()
    bev = torch.randn(B, 32, 16, 16)
    inputs = make_inputs(seed=1)
    with torch.no_grad():
        a = decoder.forward(bev, inputs, label_mapper=LabelMapper(CLASSES))["one_head_total_loss"]
        b = decoder.forward(bev, inputs, label_mapper=LabelMapper(CLASSES))["one_head_total_loss"]
    torch.testing.assert_close(a, b)


def test_mixins_are_stateless_method_holders():
    """The one_head package rule: mixins hold methods only. A mixin that defines __init__ or
    registers parameters/buffers would silently change checkpoint keys — forbidden."""
    import inspect

    from queryocc.model.components.decoders.one_head import losses
    from queryocc.model.components.decoders.one_head.losses import QueryLossesMixin

    for mixin in (QueryLossesMixin,):
        assert "__init__" not in vars(mixin), f"{mixin.__name__} must not define __init__"
        assert not issubclass(mixin, torch.nn.Module) or mixin is torch.nn.Module.__mro__[0], (
            f"{mixin.__name__} must be a plain method holder, not an nn.Module"
        )
    for module in (losses,):
        src = inspect.getsource(module)
        for forbidden in ("nn.Parameter(", "register_buffer(", "register_parameter("):
            assert forbidden not in src, f"{module.__name__} must not create state: {forbidden}"

    mro_names = [cls.__name__ for cls in OneHeadDecoder.__mro__]
    assert "QueryLossesMixin" in mro_names, mro_names
