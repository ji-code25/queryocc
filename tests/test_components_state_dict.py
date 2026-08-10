"""State-dict snapshot guard for queryocc/model/components (refactor safety net).

Same mechanism as tests/test_image_encoders_characterization.py: checkpoint keys come from
module attribute paths, so these snapshots must stay identical when the attention/decoder
modules move between files.
A diff here means old checkpoints would no longer load — fix the code, not the snapshot.

The snapshot file is auto-created on first run (commit it); afterwards any mismatch fails.

Run: pytest tests/test_components_state_dict.py
"""

import json
from pathlib import Path

import pytest
import torch

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "model_components_state_dict.json"

PCR = (-8.0, -8.0, -2.0, 8.0, 8.0, 2.0)
VOXEL_SIZE = (1.0, 1.0, 1.0)


def make_uno_bev_encoder_py():
    from queryocc.model.components.uno_encoder import UnoBEVEncoder

    # deformable_py exercises the pure-torch deformable stack (moving in Phase B).
    return UnoBEVEncoder(feature_dim=32, attention="deformable_py", attention_layers=2)


def make_uno_decoder():
    from queryocc.model.components.decoders.uno_decoder import UnoDecoder

    return UnoDecoder(point_cloud_range=PCR, in_feature_dim=32, hidden_dim=16, n_occ_resblocks=2)


def make_one_head_decoder():
    from queryocc.model.components.decoders.one_head import OneHeadDecoder, OneHeadDecoderConfig

    cfg = OneHeadDecoderConfig(
        occupancy=True,
        pseudo_occupancy=True,
        semantic_segmentation=True,
        pseudo_semantic_segmentation=True,
        num_offsets=1,
        hidden_dim=16,
        offset_dim=16,
        num_resblocks=1,
    )
    return OneHeadDecoder(cfg, input_dim=32, point_cloud_range=PCR, pc_dim=4)


FAMILIES = {
    "uno_bev_encoder_deformable_py": make_uno_bev_encoder_py,
    "uno_decoder": make_uno_decoder,
    "one_head_decoder_tiny": make_one_head_decoder,
}


def _state_dict_summary(module: torch.nn.Module) -> dict:
    return {key: list(value.shape) for key, value in sorted(module.state_dict().items())}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_state_dict_snapshot(family):
    module = FAMILIES[family]()
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
