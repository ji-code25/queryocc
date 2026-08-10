"""The release-checkpoint contract: what we publish, and what train.py accepts.

These guard the properties that make a released file safe to hand out — it must not need
unpickling to read, it must announce which preset it belongs to, and a DDP-prefixed or
pre-rename state dict must normalize to the keys the loader expects.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

from queryocc.config.method_configs import method_configs
from queryocc.utils.release import (
    RELEASE_FORMAT_VERSION,
    RELEASE_KEY,
    clean_state_dict,
    read_release_header,
    resolve_checkpoint_spec,
)


def test_every_preset_is_stamped_with_its_name():
    # Released weights record the preset they were trained with, and train.py compares it
    # against this field; an unstamped preset would silently skip that check.
    for name, config in method_configs.items():
        assert config.preset_name == name


def test_clean_state_dict_normalizes_ddp_and_legacy_prefixes():
    raw = {
        "module.net.encoder.weight": torch.zeros(2),  # DDP-wrapped save
        "dos.decoder.bias": torch.zeros(2),  # pre-rename checkpoint
        "net.head.weight": torch.zeros(2),
        "metrics.confusion": torch.zeros(2),  # not part of the network
    }
    assert set(clean_state_dict(raw)) == {"net.encoder.weight", "net.decoder.bias", "net.head.weight"}


def test_release_header_reads_without_unpickling(tmp_path: Path):
    payload = {
        RELEASE_KEY: RELEASE_FORMAT_VERSION,
        "preset": "queryocc-nusc",
        "metrics": {"semantic_rayiou": 23.6},
        "state_dict": {"net.head.weight": torch.zeros(4)},
    }
    path = tmp_path / "release.pth"
    torch.save(payload, path)

    header = read_release_header(path)
    assert header["preset"] == "queryocc-nusc"
    assert header["metrics"]["semantic_rayiou"] == 23.6
    assert "state_dict" not in header  # header reads must not pull in the weights

    # The property that makes the file safe to publish: no arbitrary objects to unpickle.
    torch.load(path, map_location="cpu", weights_only=True)


def test_training_checkpoint_is_not_mistaken_for_a_release(tmp_path: Path):
    path = tmp_path / "train.ckpt"
    # A training checkpoint pickles the config tree — Paths and dataclasses — which the
    # weights_only unpickler rejects, so the header read bails out instead of misreading it.
    torch.save(
        {"state_dict": {"net.head.weight": torch.zeros(4)}, "hyper_parameters": {"out_dir": Path("/outputs")}},
        path,
    )
    assert read_release_header(path) is None

    plain = tmp_path / "plain.ckpt"
    torch.save({"state_dict": {"net.head.weight": torch.zeros(4)}}, plain)
    assert read_release_header(plain) is None  # loadable, but carries no release header


def test_local_paths_pass_through():
    assert resolve_checkpoint_spec("/data/run/last.ckpt") == Path("/data/run/last.ckpt")
    assert resolve_checkpoint_spec(Path("relative.pth")) == Path("relative.pth")


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("hf://owner/repo/weights.pth", ("owner/repo", "weights.pth")),
        ("hf://owner/repo/nested/weights.pth", ("owner/repo", "nested/weights.pth")),
        # tyro parses the CLI string into a Path, which collapses the double slash.
        ("hf:/owner/repo/weights.pth", ("owner/repo", "weights.pth")),
        # No filename: derived from the preset.
        ("hf://owner/repo", ("owner/repo", "queryocc-nusc.pth")),
    ],
)
def test_hub_specs_resolve_to_a_download(monkeypatch, spec, expected):
    calls = {}

    stub = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id, filename):
        calls["args"] = (repo_id, filename)
        return f"/cache/{repo_id}/{filename}"

    stub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    resolved = resolve_checkpoint_spec(Path(spec), preset="queryocc-nusc")
    assert calls["args"] == expected
    assert resolved == Path(f"/cache/{expected[0]}/{expected[1]}")


def test_malformed_hub_spec_is_rejected():
    with pytest.raises(ValueError, match="Malformed Hub spec"):
        resolve_checkpoint_spec("hf://owner")
