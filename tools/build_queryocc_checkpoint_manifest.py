#!/usr/bin/env python3
"""Verify scheduled QueryOcc checkpoints and write a run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _only(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {description}, found {len(paths)}: {paths}")
    return paths[0]


def build_manifest(run_root: Path, epochs: tuple[int, ...], output: Path, training_job_id: str) -> dict:
    checkpoints = {}
    for epoch in epochs:
        path = _only(list(run_root.glob(f"*/checkpoints/epoch-{epoch:02d}.ckpt")), f"epoch {epoch} checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        stored_epoch = int(checkpoint["epoch"])
        if stored_epoch != epoch - 1:
            raise RuntimeError(f"{path}: stored epoch {stored_epoch}, expected {epoch - 1}")
        checkpoints[str(epoch)] = {
            "path": str(path.resolve()),
            "global_step": int(checkpoint["global_step"]),
            "bytes": path.stat().st_size,
        }

    recovery = _only(list(run_root.glob("*/checkpoints/last.ckpt")), "recovery checkpoint")
    recovery_payload = torch.load(recovery, map_location="cpu", weights_only=False, mmap=True)
    if int(recovery_payload["epoch"]) != max(epochs) - 1:
        raise RuntimeError(f"{recovery}: recovery checkpoint is not from the final epoch")
    config_path = _only(list(run_root.glob("*/config.yml")), "training config")
    payload = {
        "schema_version": 1,
        "training_job_id": training_job_id,
        "run_root": str(run_root.resolve()),
        "config": str(config_path.resolve()),
        "checkpoint_epochs": list(epochs),
        "checkpoints": checkpoints,
        "recovery_checkpoint": {
            "path": str(recovery.resolve()),
            "global_step": int(recovery_payload["global_step"]),
            "bytes": recovery.stat().st_size,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-job-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_manifest(args.run_root, tuple(args.epochs), args.output, args.training_job_id)
    print(json.dumps(result, indent=2, sort_keys=True))
