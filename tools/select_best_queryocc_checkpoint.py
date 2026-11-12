#!/usr/bin/env python3
"""Select the best full-validation checkpoint by Semantic RayIoU."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

SEMANTIC_RAYIOU_KEY = "test/occ3d/metrics/occ3d_sem_rayiou/semantic_rayiou/overall_mean"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def select_best(
    results_dir: Path,
    epochs: tuple[int, ...],
    metric_key: str,
    output: Path,
    best_path_output: Path,
    best_symlink: Path,
) -> dict:
    records = []
    for epoch in epochs:
        result_path = results_dir / f"epoch-{epoch:02d}.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("checkpoint_epoch") != epoch:
            raise RuntimeError(f"{result_path}: checkpoint_epoch does not match {epoch}")
        checkpoint_path = Path(payload["checkpoint_path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"{result_path}: checkpoint does not exist: {checkpoint_path}")
        if metric_key not in payload.get("metrics", {}):
            raise KeyError(f"{result_path}: missing selection metric {metric_key!r}")
        score = float(payload["metrics"][metric_key])
        if not math.isfinite(score):
            raise ValueError(f"{result_path}: selection metric is not finite: {score}")
        records.append(
            {
                "epoch": epoch,
                "score": score,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "result_path": str(result_path.resolve()),
                "metrics": payload["metrics"],
            }
        )

    best = max(records, key=lambda record: (record["score"], record["epoch"]))
    summary = {
        "schema_version": 1,
        "selection_metric": metric_key,
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "best_checkpoint": best["checkpoint_path"],
        "evaluations": records,
    }
    _atomic_text(output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_text(best_path_output, best["checkpoint_path"] + "\n")

    best_symlink.parent.mkdir(parents=True, exist_ok=True)
    temporary_symlink = best_symlink.with_name(best_symlink.name + ".tmp")
    if temporary_symlink.is_symlink() or temporary_symlink.exists():
        temporary_symlink.unlink()
    relative_target = os.path.relpath(best["checkpoint_path"], best_symlink.parent)
    temporary_symlink.symlink_to(relative_target)
    temporary_symlink.replace(best_symlink)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--metric-key", default=SEMANTIC_RAYIOU_KEY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-path-output", type=Path, required=True)
    parser.add_argument("--best-symlink", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = select_best(
        args.results_dir,
        tuple(args.epochs),
        args.metric_key,
        args.output,
        args.best_path_output,
        args.best_symlink,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
