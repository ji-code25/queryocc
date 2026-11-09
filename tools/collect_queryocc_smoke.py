#!/usr/bin/env python3
"""Validate a QueryOcc smoke checkpoint and write a compact run summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _checkpoint_step(torch, checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    return int(checkpoint.get("global_step", -1))


def collect(
    manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    expected_steps: int,
    *,
    best_checkpoint_path: Path | None = None,
    best_validation_log: Path | None = None,
    final_validation_log: Path | None = None,
    effective_val_samples: int | None = None,
) -> dict:
    import torch

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    global_step = _checkpoint_step(torch, checkpoint_path)
    if global_step != expected_steps:
        raise ValueError(f"checkpoint global_step is {global_step}, expected {expected_steps}")
    if int(manifest["expected_optimizer_steps"]) != expected_steps:
        raise ValueError("manifest optimizer-step count differs from expectation")

    if best_checkpoint_path is not None:
        best_global_step = _checkpoint_step(torch, best_checkpoint_path)
        if not 0 < best_global_step <= expected_steps:
            raise ValueError(
                f"best checkpoint global_step is {best_global_step}, expected a step in [1, {expected_steps}]"
            )
    else:
        best_global_step = None

    selected_val_records = int(manifest["val_count"])
    if effective_val_samples is None:
        effective_val_samples = selected_val_records
    if not 0 < effective_val_samples <= selected_val_records:
        raise ValueError(
            f"effective validation samples is {effective_val_samples}, expected a value in "
            f"[1, {selected_val_records}]"
        )

    validation_logs = (best_validation_log, final_validation_log)
    if any(path is not None for path in validation_logs):
        if not all(path is not None for path in validation_logs):
            raise ValueError("both best and final validation logs are required")
        for path in validation_logs:
            assert path is not None
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"validation log is missing or empty: {path}")

    summary = {
        "schema_version": 1,
        "ok": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_sample_count": int(manifest["train_count"]),
        "train_sample_budget": int(manifest.get("train_sample_budget", manifest["train_count"])),
        "train_steps": global_step,
        # Kept for schema-v1 compatibility; this is the number of frozen info records.
        "val_sample_count": selected_val_records,
        "world_size": int(manifest["world_size"]),
        "samples_per_gpu": int(manifest["samples_per_gpu"]),
        "checkpoint": str(checkpoint_path),
        "checkpoints": {
            "best": {
                "path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                "global_step": best_global_step,
            },
            "final": {"path": str(checkpoint_path), "global_step": global_step},
        },
        "final_validation": {
            "scope": "all valid temporal anchors from the frozen smoke validation records",
            "selected_record_count": selected_val_records,
            "effective_sample_count": effective_val_samples,
            "best_log": str(best_validation_log) if best_validation_log is not None else None,
            "final_log": str(final_validation_log) if final_validation_log is not None else None,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument("--best-validation-log", type=Path)
    parser.add_argument("--final-validation-log", type=Path)
    parser.add_argument("--effective-val-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            collect(
                args.manifest,
                args.checkpoint,
                args.output,
                args.expected_steps,
                best_checkpoint_path=args.best_checkpoint,
                best_validation_log=args.best_validation_log,
                final_validation_log=args.final_validation_log,
                effective_val_samples=args.effective_val_samples,
            ),
            indent=2,
        )
    )
