#!/usr/bin/env python3
"""Verify the frozen inputs used by the QueryOcc smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_queryocc_smoke import DYNAMIC_CLASS_IDS, load_records, pseudo_paths, record_token, validate_pseudo_pair


def unique_tokens(records: list[dict], name: str) -> list[str]:
    tokens = [record_token(record) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{name} contains duplicate tokens")
    return tokens


def verify(
    input_dir: Path,
    expected_train: int,
    expected_val: int,
    expected_steps: int,
    check_archives: bool,
) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    tag = manifest["infos_tag"]
    _, _, train_records = load_records(input_dir / f"{tag}_infos_train.pkl")
    _, _, val_records = load_records(input_dir / f"{tag}_infos_val.pkl")
    train_tokens = unique_tokens(train_records, "train subset")
    val_tokens = unique_tokens(val_records, "validation subset")

    if len(train_tokens) != expected_train:
        raise ValueError(f"expected {expected_train} train records, found {len(train_tokens)}")
    if len(val_tokens) != expected_val:
        raise ValueError(f"expected {expected_val} validation records, found {len(val_tokens)}")
    if train_tokens != manifest["train_tokens"]:
        raise ValueError("train tokens differ from the manifest")
    if val_tokens != manifest["val_tokens"]:
        raise ValueError("validation tokens differ from the manifest")
    if tuple(manifest["dynamic_class_ids"]) != DYNAMIC_CLASS_IDS:
        raise ValueError("manifest uses an unexpected dynamic-class contract")
    for class_id in DYNAMIC_CLASS_IDS:
        if not manifest["dynamic_coverage"].get(str(class_id)):
            raise ValueError(f"validation subset does not cover dynamic class {class_id}")

    denominator = (
        int(manifest["world_size"])
        * int(manifest["samples_per_gpu"])
        * int(manifest["accumulate_grad_batches"])
    )
    train_sample_budget = int(manifest.get("train_sample_budget", len(train_tokens)))
    if train_sample_budget < len(train_tokens):
        raise ValueError("train sample budget is smaller than the frozen train subset")
    steps, remainder = divmod(train_sample_budget, denominator)
    if remainder or steps != expected_steps:
        raise ValueError(
            f"train sample budget implies {steps} optimizer steps with remainder {remainder}, "
            f"expected exactly {expected_steps}"
        )
    if int(manifest["expected_optimizer_steps"]) != expected_steps:
        raise ValueError("manifest optimizer-step count differs from expectation")

    semantics_root = Path(manifest["semantics_root"])
    depth_root = Path(manifest["depth_root"])
    scene_names = {str(key): str(value) for key, value in manifest.get("scene_names", {}).items()}
    pseudo_pairs = 0
    for record in train_records:
        semantics, depth = pseudo_paths(record, semantics_root, depth_root, scene_names)
        if not semantics.is_file() or not depth.is_file():
            raise FileNotFoundError(f"{record_token(record)}: missing {semantics} or {depth}")
        if check_archives:
            validate_pseudo_pair(semantics, depth)
        pseudo_pairs += 1

    return {
        "ok": True,
        "train_count": len(train_tokens),
        "train_sample_budget": train_sample_budget,
        "val_count": len(val_tokens),
        "optimizer_steps": steps,
        "pseudo_label_depth_pairs": pseudo_pairs,
        "dynamic_class_ids": list(DYNAMIC_CLASS_IDS),
        "checked_archive_contents": check_archives,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=200)
    parser.add_argument("--expected-val", type=int, default=32)
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--check-archives", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            verify(
                args.input_dir,
                args.expected_train,
                args.expected_val,
                args.expected_steps,
                args.check_archives,
            ),
            indent=2,
            sort_keys=True,
        )
    )
