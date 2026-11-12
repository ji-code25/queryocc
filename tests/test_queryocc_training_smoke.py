import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_smoke = load_module("prepare_queryocc_smoke", "tools/prepare_queryocc_smoke.py")
verify_smoke = load_module("verify_queryocc_smoke", "tools/verify_queryocc_smoke.py")
collect_smoke = load_module("collect_queryocc_smoke", "tools/collect_queryocc_smoke.py")
convert_legacy = load_module("convert_legacy_queryocc_pseudo_labels", "tools/convert_legacy_queryocc_pseudo_labels.py")


def dump_pickle(path: Path, payload) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_pseudo_pair(semantics_root: Path, depth_root: Path, scene: str, token: str) -> None:
    semantics_dir = semantics_root / scene
    depth_dir = depth_root / scene
    semantics_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((6, 2, 3), dtype=np.int8)
    depth = np.ones((6, 2, 3), dtype=np.float16)
    np.savez(semantics_dir / f"{token}.npz", labels=labels, cam_order=prepare_smoke.CAMERA_NAMES)
    # A different order is legal: each stream resolves its own embedded cam_order.
    np.savez(depth_dir / f"{token}.npz", depth=depth, cam_order=prepare_smoke.CAMERA_NAMES[::-1])


def test_prepare_and_verify_frozen_queryocc_inputs(tmp_path: Path):
    semantics_root = tmp_path / "semantics"
    depth_root = tmp_path / "depth"
    gt_root = tmp_path / "gts"
    output_dir = tmp_path / "smoke_inputs"

    train_records = []
    for index in range(3):
        token = f"train-{index}"
        record = {
            "token": token,
            "scene_token": "train-scene-token",
            "scene_name": "scene-train",
            "timestamp": index,
        }
        train_records.append(record)
        write_pseudo_pair(semantics_root, depth_root, "scene-train", token)

    val_records = [
        {"token": "val-0", "scene_token": "val-scene-token", "scene_name": "scene-val"},
        {"token": "val-1", "scene_token": "val-scene-token", "scene_name": "scene-val"},
    ]
    for token, classes in (("val-0", [2, 3, 4, 5]), ("val-1", [6, 7, 9, 10])):
        gt_dir = gt_root / "scene-val" / token
        gt_dir.mkdir(parents=True)
        np.savez(
            gt_dir / "labels.npz",
            semantics=np.asarray(classes, dtype=np.uint8),
            mask_camera=np.ones(len(classes), dtype=bool),
        )

    train_info = tmp_path / "train.pkl"
    val_info = tmp_path / "val.pkl"
    dump_pickle(train_info, {"infos": train_records, "metadata": {}})
    dump_pickle(val_info, {"infos": val_records, "metadata": {}})

    manifest = prepare_smoke.prepare(
        argparse.Namespace(
            train_info=train_info,
            val_info=val_info,
            data_root=tmp_path / "unused-nuscenes",
            semantics_root=semantics_root,
            depth_root=depth_root,
            gt_root=gt_root,
            output_dir=output_dir,
            infos_tag="queryocc-smoke",
            version="v1.0-trainval",
            train_count=2,
            train_sample_budget=4,
            val_count=2,
            world_size=1,
            samples_per_gpu=1,
            accumulate_grad_batches=1,
            seed="test",
            check_archives=True,
        )
    )

    assert manifest["expected_optimizer_steps"] == 4
    assert manifest["train_sample_budget"] == 4
    assert manifest["checked_archive_contents"]
    assert set(manifest["dynamic_coverage"]) == {str(value) for value in prepare_smoke.DYNAMIC_CLASS_IDS}
    result = verify_smoke.verify(output_dir, 2, 2, 4, check_archives=True)
    assert result == {
        "ok": True,
        "train_count": 2,
        "train_sample_budget": 4,
        "val_count": 2,
        "optimizer_steps": 4,
        "pseudo_label_depth_pairs": 2,
        "dynamic_class_ids": list(prepare_smoke.DYNAMIC_CLASS_IDS),
        "checked_archive_contents": True,
    }


def test_convert_legacy_layout_remaps_sky_and_camera_order(tmp_path: Path):
    semantics_root = tmp_path / "legacy_semantics"
    depth_root = tmp_path / "legacy_depth"
    output_root = tmp_path / "converted"
    semantics_root.mkdir()

    token = "sample-token"
    scene = "scene-0001"
    cameras = {}
    for camera_index, camera_name in enumerate(convert_legacy.CAMERA_ORDER):
        stem = f"image-{camera_name}"
        cameras[camera_name] = {"data_path": f"samples/{camera_name}/{stem}.jpg"}
        labels = np.full((2, 3), camera_index, dtype=np.int8)
        labels[0, 0] = convert_legacy.LEGACY_SKY_LABEL
        scores = np.full((2, 3), camera_index / 10, dtype=np.float16)
        np.savez(semantics_root / f"{stem}_labels.npz", labels=labels)
        np.savez(semantics_root / f"{stem}_scores.npz", scores=scores)

    legacy_depth_dir = depth_root / scene
    legacy_depth_dir.mkdir(parents=True)
    legacy_depth = np.stack([np.full((2, 3), index + 1, dtype=np.float16) for index in range(6)])
    np.save(legacy_depth_dir / f"{token}.npy", legacy_depth)
    record = {"token": token, "scene_name": scene, "scene_token": "scene-token", "cams": cameras}
    infos_path = tmp_path / "infos.pkl"
    dump_pickle(infos_path, {"infos": [record]})

    conversion_args = {
        "infos": [infos_path],
        "semantics_root": semantics_root,
        "depth_root": depth_root,
        "output_root": output_root,
        "require_scores": True,
        "overwrite": False,
        "max_samples": None,
        "min_complete": None,
    }
    dry_summary = convert_legacy.convert(argparse.Namespace(**conversion_args, dry_run=True))
    assert dry_summary["legacy_depth_layout"] == "packed"
    assert dry_summary["complete_records"] == 1
    assert dry_summary["selected_records"] == 1
    assert dry_summary["converted_records"] == 0
    assert not output_root.exists()

    with pytest.raises(RuntimeError, match="only 1 records.*need 2"):
        convert_legacy.convert(argparse.Namespace(**{**conversion_args, "min_complete": 2}, dry_run=True))

    summary = convert_legacy.convert(argparse.Namespace(**conversion_args, workers=2, dry_run=False))
    assert summary["converted_records"] == 1
    assert summary["conversion_workers"] == 2

    with np.load(output_root / "semantics" / scene / f"{token}.npz") as archive:
        assert tuple(archive["cam_order"]) == convert_legacy.CAMERA_ORDER
        assert archive["labels"].shape == (6, 2, 3)
        assert np.all(archive["labels"][:, 0, 0] == convert_legacy.DEFAULT_SKY_LABEL)
        assert "scores" in archive.files
    with np.load(output_root / "depth" / scene / f"{token}.npz") as archive:
        assert tuple(archive["cam_order"]) == convert_legacy.CAMERA_ORDER
        for output_index, camera_name in enumerate(convert_legacy.CAMERA_ORDER):
            legacy_index = convert_legacy.LEGACY_DEPTH_ORDER.index(camera_name)
            assert np.all(archive["depth"][output_index] == legacy_index + 1)


def test_convert_wazabi_per_camera_depth_layout(tmp_path: Path):
    semantics_root = tmp_path / "legacy_semantics"
    depth_root = tmp_path / "wazabi_metric3d"
    output_root = tmp_path / "converted"
    semantics_root.mkdir()
    depth_root.mkdir()

    token = "sample-token"
    scene = "scene-0001"
    cameras = {}
    for camera_index, camera_name in enumerate(convert_legacy.CAMERA_ORDER):
        stem = f"image-{camera_name}"
        cameras[camera_name] = {"data_path": f"samples/{camera_name}/{stem}.jpg"}
        np.savez(
            semantics_root / f"{stem}_labels.npz",
            labels=np.full((2, 3), camera_index, dtype=np.int8),
        )
        np.savez(
            semantics_root / f"{stem}_scores.npz",
            scores=np.full((2, 3), 0.5, dtype=np.float16),
        )
        np.save(
            depth_root / f"{stem}.npy",
            np.full((2, 3), camera_index + 1, dtype=np.float32),
        )

    infos_path = tmp_path / "infos.pkl"
    dump_pickle(
        infos_path,
        {"infos": [{"token": token, "scene_name": scene, "cams": cameras}]},
    )
    summary = convert_legacy.convert(
        argparse.Namespace(
            infos=[infos_path],
            semantics_root=semantics_root,
            depth_root=depth_root,
            depth_layout="auto",
            output_root=output_root,
            require_scores=True,
            overwrite=False,
            max_samples=None,
            min_complete=1,
            dry_run=False,
        )
    )
    assert summary["legacy_depth_layout"] == "per-camera"
    with np.load(output_root / "depth" / scene / f"{token}.npz") as archive:
        assert tuple(archive["cam_order"]) == convert_legacy.CAMERA_ORDER
        for camera_index in range(len(convert_legacy.CAMERA_ORDER)):
            assert np.all(archive["depth"][camera_index] == camera_index + 1)


def test_smoke_launcher_keeps_production_model_and_runs_100_steps():
    launcher = (ROOT / "scripts/train_queryocc_smoke_berzelius.sh").read_text()
    prepare_launcher = (ROOT / "scripts/prepare_queryocc_smoke_berzelius.sh").read_text()
    full_converter = (ROOT / "scripts/convert_queryocc_pseudo_labels_berzelius.sh").read_text()

    assert "#SBATCH --gpus=2" in launcher
    assert "#SBATCH --time=01:00:00" in launcher
    assert "--trainer.max-steps=200" in launcher
    assert "--trainer.accumulate-grad-batches=1" in launcher
    assert "--trainer.check-val-every-n-epoch=1" in launcher
    assert launcher.count("--datamodule.batch-size=1") == 3
    assert "--expected-steps 100" in launcher
    assert '"${NAME}-best-full-val"' in launcher
    assert '"${NAME}-final-full-val"' in launcher
    assert "--best-checkpoint" in launcher
    assert "--trainer.limit-test-batches=1.0" in launcher
    assert "--model.occupancy-sampler.num-query-points" not in launcher
    assert "--model.net.no-compile" not in launcher
    assert "--no-test" in launcher
    assert "--train-count 100" in prepare_launcher
    assert "--train-sample-budget 200" in prepare_launcher
    assert "--min-complete 100" in prepare_launcher
    assert "--datamodule.train-dataset-usage-num=100" in launcher
    assert "--val-count 32" in prepare_launcher
    assert "--check-archives" in prepare_launcher
    assert "SLURM_SUBMIT_DIR" in prepare_launcher
    assert "SLURM_SUBMIT_DIR" in launcher
    assert 'set -a\nsource "$SCRIPT_DIR/env.local.sh"\nset +a' in launcher
    assert 'CONVERTED_ROOT="$QUERYOCC_CONVERTED_ROOT"' in prepare_launcher
    assert '--min-complete "$FULL_CONVERSION_EXPECTED_TRAIN"' in full_converter
    assert "FULL_CONVERSION_EXPECTED_TRAIN:-28130" in full_converter
    assert "#SBATCH --cpus-per-task=16" in full_converter
    assert "#SBATCH --mem=64G" in full_converter
    assert "#SBATCH --time=06:00:00" in full_converter
    assert '--workers "$CONVERSION_WORKERS"' in full_converter
    assert "OMP_NUM_THREADS=1" in full_converter


def test_queryocc_full_training_validation_schedule():
    preset = (ROOT / "queryocc/config/methods/queryocc.py").read_text()
    trainer = (ROOT / "queryocc/config/trainer.py").read_text()
    train = (ROOT / "queryocc/train.py").read_text()

    assert "max_steps=-1" in preset
    assert "max_epochs=16" in preset
    assert "checkpoint_epochs=(10, 11, 12, 13, 14, 15, 16)" in preset
    assert "validation_epochs=()" in preset
    assert "limit_val_batches=0" in preset
    assert "shuffle_val=False" in preset
    assert "occ3d_only_eval=True" in preset
    assert "test_best_and_last_after_fit=False" in preset
    assert "validation_epochs: Tuple[int, ...] = ()" in trainer
    assert '("best", "last")' in train
    assert "EpochRecoveryCheckpoint" in train
    assert "ScheduledEpochCheckpoint" in train


def test_queryocc_full_berzelius_resource_and_dependency_workflow():
    training = (ROOT / "scripts/train_queryocc_full_berzelius.sh").read_text()
    evaluation = (ROOT / "scripts/evaluate_queryocc_full_berzelius.sh").read_text()
    submission = (ROOT / "scripts/submit_queryocc_full_berzelius.sh").read_text()
    selector = (ROOT / "scripts/select_queryocc_best_berzelius.sh").read_text()

    assert "#SBATCH --gpus-per-node=A100-SXM4-80GB:4" in training
    assert "#SBATCH --constraint=fat" in training
    assert "#SBATCH --time=24:00:00" in training
    assert "#SBATCH --account=berzelius-2026-281" in training
    assert "--trainer.limit-val-batches=0" in training

    assert "#SBATCH --gpus-per-node=A100-SXM4-40GB:1" in evaluation
    assert "#SBATCH --constraint=thin" in evaluation
    assert "--trainer.limit-test-batches=1.0" in evaluation
    assert "--evaluation-results-path=" in evaluation
    assert "--array=10-16" in submission
    assert "afterok:$prepare_job" in submission
    assert "afterok:$smoke_job" in submission
    assert "afterok:$train_job" in submission
    assert "afterok:$evaluation_job" in submission
    assert "select_best_queryocc_checkpoint.py" in selector


def test_checkpoint_summary_requires_exact_global_step(tmp_path: Path):
    torch = pytest.importorskip("torch")
    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "last.ckpt"
    output_path = tmp_path / "smoke_summary.json"
    manifest_path.write_text(
        """{
  "expected_optimizer_steps": 100,
  "train_count": 100,
  "train_sample_budget": 200,
  "val_count": 32,
  "world_size": 2,
  "samples_per_gpu": 1
}
"""
    )
    torch.save({"global_step": 100, "state_dict": {}}, checkpoint_path)

    summary = collect_smoke.collect(manifest_path, checkpoint_path, output_path, expected_steps=100)
    assert summary["ok"]
    assert summary["train_steps"] == 100
    assert output_path.is_file()


def test_checkpoint_summary_records_best_and_final_validations(tmp_path: Path):
    torch = pytest.importorskip("torch")
    manifest_path = tmp_path / "manifest.json"
    best_checkpoint_path = tmp_path / "best.ckpt"
    final_checkpoint_path = tmp_path / "last.ckpt"
    best_log = tmp_path / "best.log"
    final_log = tmp_path / "final.log"
    output_path = tmp_path / "smoke_summary.json"
    manifest_path.write_text(
        """{
  "expected_optimizer_steps": 100,
  "train_count": 100,
  "train_sample_budget": 200,
  "val_count": 32,
  "world_size": 2,
  "samples_per_gpu": 1
}
"""
    )
    torch.save({"global_step": 87, "state_dict": {}}, best_checkpoint_path)
    torch.save({"global_step": 100, "state_dict": {}}, final_checkpoint_path)
    best_log.write_text("best full smoke validation passed\n")
    final_log.write_text("final full smoke validation passed\n")

    summary = collect_smoke.collect(
        manifest_path,
        final_checkpoint_path,
        output_path,
        expected_steps=100,
        best_checkpoint_path=best_checkpoint_path,
        best_validation_log=best_log,
        final_validation_log=final_log,
        effective_val_samples=28,
    )
    assert summary["checkpoints"]["best"]["global_step"] == 87
    assert summary["checkpoints"]["final"]["global_step"] == 100
    assert summary["final_validation"]["selected_record_count"] == 32
    assert summary["final_validation"]["effective_sample_count"] == 28


def test_separate_semantics_and_depth_binds_are_supported():
    runner = (ROOT / "scripts/_run.sh").read_text()
    assert "PSEUDO_SEMANTICS_ROOT" in runner
    assert "PSEUDO_DEPTH_ROOT" in runner
    assert "$PSEUDO_SEMANTICS_ROOT:/datasets/pseudo_labels/semantics" in runner
    assert "$PSEUDO_DEPTH_ROOT:/datasets/pseudo_labels/depth" in runner
    assert "$NUSC_DATASET_ROOT/maps/:/datasets/nuscenes/maps" in runner
