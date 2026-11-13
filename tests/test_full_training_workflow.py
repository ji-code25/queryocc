import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from generate_pseudo_labels.constants import CAMERA_ORDER
from tools.prepare_queryocc_full_training import prepare
from tools.select_best_queryocc_checkpoint import SEMANTIC_RAYIOU_KEY, select_best


def _record(scene, token):
    return {"scene_name": scene, "token": token}


def _write_converted(root, scene, token, valid):
    shape = (len(CAMERA_ORDER), 2, 3)
    labels = np.zeros(shape, dtype=np.int8)
    depth = np.ones(shape, dtype=np.float16)
    if valid:
        labels[0, 0, 0] = 1
    scores = np.ones(shape, dtype=np.float16)
    semantics_path = root / "semantics" / scene / f"{token}.npz"
    depth_path = root / "depth" / scene / f"{token}.npz"
    semantics_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(semantics_path, labels=labels, scores=scores, cam_order=np.asarray(CAMERA_ORDER))
    np.savez(depth_path, depth=depth, cam_order=np.asarray(CAMERA_ORDER))


def test_full_preflight_writes_blacklist_and_filtered_infos(tmp_path):
    train_info = tmp_path / "train.pkl"
    val_info = tmp_path / "val.pkl"
    records = [_record("scene-1", "valid"), _record("scene-1", "empty")]
    with train_info.open("wb") as handle:
        pickle.dump({"infos": records, "metadata": {"version": "test"}}, handle)
    with val_info.open("wb") as handle:
        pickle.dump({"infos": [_record("scene-2", "val")]}, handle)
    converted = tmp_path / "converted"
    _write_converted(converted, "scene-1", "valid", valid=True)
    _write_converted(converted, "scene-1", "empty", valid=False)

    summary = prepare(
        SimpleNamespace(
            train_info=train_info,
            val_info=val_info,
            semantics_root=converted / "semantics",
            depth_root=converted / "depth",
            infos_output_dir=tmp_path / "infos",
            artifact_dir=converted,
            infos_tag="queryocc-refined",
            workers=1,
        )
    )

    assert summary["records_checked"] == 2
    assert summary["blacklisted_records"] == 1
    with (tmp_path / "infos/queryocc-refined_infos_train.pkl").open("rb") as handle:
        filtered = pickle.load(handle)
    assert [record["token"] for record in filtered["infos"]] == ["valid"]
    blacklist = json.loads((converted / "full_training_blacklist.json").read_text())
    assert blacklist["records"][0]["token"] == "empty"


def test_checkpoint_manifest_finds_nested_recovery_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from tools.build_queryocc_checkpoint_manifest import build_manifest

    run_root = tmp_path / "queryocc-nusc-full16-job"
    timestamp_root = run_root / "2026-11-12_172358"
    scheduled_root = timestamp_root / "checkpoints"
    recovery_root = timestamp_root / "queryocc" / "run-id" / "checkpoints"
    scheduled_root.mkdir(parents=True)
    recovery_root.mkdir(parents=True)
    (timestamp_root / "config.yml").write_text("trainer:\n  max_epochs: 16\n")

    for epoch, global_step in ((10, 10490), (11, 11539)):
        torch.save(
            {"epoch": epoch - 1, "global_step": global_step},
            scheduled_root / f"epoch-{epoch:02d}.ckpt",
        )
    torch.save(
        {
            "epoch": 10,
            "global_step": 11539,
            "optimizer_states": [{}],
            "lr_schedulers": [{}],
            "loops": {},
        },
        recovery_root / "last.ckpt",
    )

    output = run_root / "checkpoint_manifest.json"
    manifest = build_manifest(run_root, (10, 11), output, "job")

    assert output.is_file()
    assert manifest["checkpoints"]["10"]["path"].endswith("epoch-10.ckpt")
    assert manifest["recovery_checkpoint"]["path"].endswith(
        "queryocc/run-id/checkpoints/last.ckpt"
    )


def test_semantic_rayiou_selector_requires_all_results_and_links_best(tmp_path):
    results_dir = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoints.mkdir()
    for epoch, score in ((10, 0.25), (11, 0.31), (12, 0.29)):
        checkpoint = checkpoints / f"epoch-{epoch:02d}.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        (results_dir / f"epoch-{epoch:02d}.json").write_text(
            json.dumps(
                {
                    "checkpoint_epoch": epoch,
                    "checkpoint_path": str(checkpoint),
                    "metrics": {SEMANTIC_RAYIOU_KEY: score, "test/other": epoch},
                }
            )
        )

    summary = select_best(
        results_dir,
        (10, 11, 12),
        SEMANTIC_RAYIOU_KEY,
        tmp_path / "evaluation_summary.json",
        tmp_path / "best_checkpoint.txt",
        tmp_path / "best.ckpt",
    )

    assert summary["best_epoch"] == 11
    assert summary["best_score"] == 0.31
    assert (tmp_path / "best.ckpt").resolve() == (checkpoints / "epoch-11.ckpt").resolve()
