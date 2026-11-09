#!/usr/bin/env python3
"""Freeze deterministic QueryOcc train/validation subsets for a smoke run.

The subsets keep the production nuScenes records intact. QueryOcc still walks the
nuScenes devkit for the temporal lidar/camera streams; the smaller info files only
restrict which keyframes become dataset samples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from generate_pseudo_labels.constants import CAMERA_ORDER

CAMERA_NAMES = CAMERA_ORDER
DYNAMIC_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)
MAX_PSEUDO_LABEL = 18  # 18 is the explicit sky label used by generate_pseudo_labels.


def load_records(path: Path) -> tuple[dict[str, Any], str, list[dict]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a dict, got {type(payload).__name__}")
    for key in ("infos", "data_list"):
        records = payload.get(key)
        if isinstance(records, list):
            return payload, key, records
    raise KeyError(f"{path}: expected an 'infos' or 'data_list' list")


def record_token(record: dict) -> str:
    for key in ("token", "sample_idx"):
        value = record.get(key)
        if value is not None:
            return str(value)
    raise KeyError("annotation record has no token/sample_idx")


def record_scene_token(record: dict) -> str:
    value = record.get("scene_token")
    if value is None:
        raise KeyError(f"{record_token(record)}: annotation has no scene_token")
    return str(value)


def scene_name_for_record(record: dict, scene_names: dict[str, str]) -> str:
    value = record.get("scene_name") or record.get("scene_idx")
    if value is not None:
        return str(value)
    scene_token = record_scene_token(record)
    if scene_token not in scene_names:
        raise KeyError(f"{record_token(record)}: no nuScenes name for scene token {scene_token}")
    return scene_names[scene_token]


def load_scene_names(data_root: Path, version: str, records: list[dict]) -> dict[str, str]:
    missing = {
        record_scene_token(record)
        for record in records
        if record.get("scene_name") is None and record.get("scene_idx") is None
    }
    if not missing:
        return {}

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    scene_names = {str(scene["token"]): str(scene["name"]) for scene in nusc.scene}
    unknown = missing.difference(scene_names)
    if unknown:
        raise KeyError(f"unknown nuScenes scene tokens: {sorted(unknown)}")
    return scene_names


def pseudo_paths(
    record: dict,
    semantics_root: Path,
    depth_root: Path,
    scene_names: dict[str, str],
) -> tuple[Path, Path]:
    scene_name = scene_name_for_record(record, scene_names)
    token = record_token(record)
    return semantics_root / scene_name / f"{token}.npz", depth_root / scene_name / f"{token}.npz"


def _validate_cam_order(path: Path, archive: np.lib.npyio.NpzFile, n_planes: int) -> tuple[str, ...]:
    if "cam_order" not in archive.files:
        # The reader deliberately supports old files without cam_order, using the
        # generator's canonical order. Keep that compatibility in the smoke check.
        order = CAMERA_NAMES
    else:
        order = tuple(str(value) for value in np.asarray(archive["cam_order"]).tolist())
    if len(order) != n_planes or len(set(order)) != n_planes:
        raise ValueError(f"{path}: invalid cam_order {order} for {n_planes} planes")
    if set(order) != set(CAMERA_NAMES):
        raise ValueError(f"{path}: cam_order does not contain the six nuScenes cameras: {order}")
    return order


def validate_pseudo_pair(semantics_path: Path, depth_path: Path) -> tuple[tuple[int, int], tuple[str, ...]]:
    with np.load(semantics_path, allow_pickle=False) as archive:
        if "labels" not in archive.files:
            raise KeyError(f"{semantics_path}: missing array 'labels'")
        labels = np.asarray(archive["labels"])
        if labels.ndim != 3 or labels.shape[0] != len(CAMERA_NAMES):
            raise ValueError(f"{semantics_path}: labels must have shape [6, H, W], got {labels.shape}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(f"{semantics_path}: labels must be integer, got {labels.dtype}")
        invalid = (labels < 0) | (labels > MAX_PSEUDO_LABEL)
        if np.any(invalid):
            raise ValueError(
                f"{semantics_path}: labels outside [0, {MAX_PSEUDO_LABEL}]: "
                f"{np.unique(labels[invalid]).tolist()}"
            )
        semantics_order = _validate_cam_order(semantics_path, archive, labels.shape[0])
        spatial_shape = tuple(int(value) for value in labels.shape[1:])

    with np.load(depth_path, allow_pickle=False) as archive:
        if "depth" not in archive.files:
            raise KeyError(f"{depth_path}: missing array 'depth'")
        depth = np.asarray(archive["depth"])
        if depth.ndim != 3 or depth.shape[0] != len(CAMERA_NAMES):
            raise ValueError(f"{depth_path}: depth must have shape [6, H, W], got {depth.shape}")
        if tuple(depth.shape[1:]) != spatial_shape:
            raise ValueError(
                f"{depth_path}: depth shape {tuple(depth.shape[1:])} differs from labels {spatial_shape}"
            )
        if not np.issubdtype(depth.dtype, np.number):
            raise TypeError(f"{depth_path}: depth must be numeric, got {depth.dtype}")
        if not np.all(np.isfinite(depth)) or np.any(depth < 0):
            raise ValueError(f"{depth_path}: depth must be finite and non-negative")
        if not np.any(depth > 0):
            raise ValueError(f"{depth_path}: depth contains no positive pixels")
        depth_order = _validate_cam_order(depth_path, archive, depth.shape[0])

    # Different orders are valid because each stream embeds and resolves its own order.
    return spatial_shape, semantics_order if semantics_order == depth_order else semantics_order


def stable_rank(token: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{token}".encode()).hexdigest()


def select_train_records(
    records: list[dict],
    semantics_root: Path,
    depth_root: Path,
    scene_names: dict[str, str],
    count: int,
    seed: str,
    check_archives: bool,
) -> tuple[list[dict], list[str]]:
    if count <= 0:
        raise ValueError("train count must be positive")
    complete = []
    seen: set[str] = set()
    for record in records:
        token = record_token(record)
        if token in seen:
            raise ValueError(f"duplicate train token: {token}")
        seen.add(token)
        semantics, depth = pseudo_paths(record, semantics_root, depth_root, scene_names)
        if semantics.is_file() and depth.is_file():
            complete.append((token, record, semantics, depth))
    complete.sort(key=lambda row: stable_rank(row[0], seed))
    if len(complete) < count:
        raise RuntimeError(
            f"only {len(complete)} train records have both semantics and depth; need {count}"
        )
    selected = complete[:count]
    if check_archives:
        for _, _, semantics, depth in selected:
            validate_pseudo_pair(semantics, depth)
    return [row[1] for row in selected], [row[0] for row in selected]


def dynamic_classes_for_record(record: dict, gt_root: Path, scene_names: dict[str, str]) -> set[int]:
    scene_name = scene_name_for_record(record, scene_names)
    token = record_token(record)
    gt_path = gt_root / scene_name / token / "labels.npz"
    with np.load(gt_path, allow_pickle=False) as archive:
        if "semantics" not in archive.files:
            raise KeyError(f"{gt_path}: missing array 'semantics'")
        semantics = np.asarray(archive["semantics"])
        if "mask_camera" in archive.files:
            semantics = semantics[np.asarray(archive["mask_camera"], dtype=bool)]
    return set(map(int, np.unique(semantics))).intersection(DYNAMIC_CLASS_IDS)


def select_val_records(
    records: list[dict],
    gt_root: Path,
    scene_names: dict[str, str],
    count: int,
    seed: str,
) -> tuple[list[dict], list[str], dict[str, list[str]]]:
    if count <= 0:
        raise ValueError("validation count must be positive")
    by_token = {record_token(record): record for record in records}
    if len(by_token) != len(records):
        raise ValueError("validation annotations contain duplicate tokens")

    candidates = []
    for token, record in by_token.items():
        scene_name = scene_name_for_record(record, scene_names)
        gt_path = gt_root / scene_name / token / "labels.npz"
        if gt_path.is_file():
            candidates.append(token)
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} validation records have Occ3D ground truth; need {count}")

    class_sets = {
        token: dynamic_classes_for_record(by_token[token], gt_root, scene_names)
        for token in candidates
    }
    uncovered = set(DYNAMIC_CLASS_IDS)
    selected: list[str] = []
    remaining = set(candidates)
    while uncovered:
        ranked = sorted(
            remaining,
            key=lambda token: (-len(class_sets[token].intersection(uncovered)), stable_rank(token, seed)),
        )
        if not ranked:
            break
        best = ranked[0]
        newly_covered = class_sets[best].intersection(uncovered)
        if not newly_covered:
            break
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(newly_covered)
    if uncovered:
        raise RuntimeError(f"validation split does not cover dynamic IDs {sorted(uncovered)}")
    if len(selected) > count:
        raise RuntimeError(f"dynamic-class coverage needs {len(selected)} records, but count is {count}")

    fill = sorted(remaining, key=lambda token: stable_rank(token, seed))
    selected.extend(fill[: count - len(selected)])
    coverage = {
        str(class_id): sorted(token for token in selected if class_id in class_sets[token])
        for class_id in DYNAMIC_CLASS_IDS
    }
    return [by_token[token] for token in selected], selected, coverage


def subset_payload(payload: dict[str, Any], key: str, records: list[dict]) -> dict[str, Any]:
    result = copy.copy(payload)
    result[key] = records
    return result


def dump_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    train_sample_budget = args.train_sample_budget or args.train_count
    if train_sample_budget < args.train_count:
        raise ValueError("train sample budget must be at least the number of unique train records")
    denominator = args.world_size * args.samples_per_gpu * args.accumulate_grad_batches
    expected_steps, remainder = divmod(train_sample_budget, denominator)
    if remainder:
        raise ValueError(
            "train sample budget must be divisible by world size times samples per GPU times accumulation"
        )

    train_payload, train_key, train_records = load_records(args.train_info)
    val_payload, val_key, val_records = load_records(args.val_info)
    all_records = train_records + val_records
    scene_names = load_scene_names(args.data_root, args.version, all_records)

    selected_train, train_tokens = select_train_records(
        train_records,
        args.semantics_root,
        args.depth_root,
        scene_names,
        args.train_count,
        args.seed,
        args.check_archives,
    )
    selected_val, val_tokens, coverage = select_val_records(
        val_records,
        args.gt_root,
        scene_names,
        args.val_count,
        args.seed,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / f"{args.infos_tag}_infos_train.pkl"
    val_out = output_dir / f"{args.infos_tag}_infos_val.pkl"
    dump_pickle(train_out, subset_payload(train_payload, train_key, selected_train))
    dump_pickle(val_out, subset_payload(val_payload, val_key, selected_val))

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "infos_tag": args.infos_tag,
        "semantics_root": str(args.semantics_root.resolve()),
        "depth_root": str(args.depth_root.resolve()),
        "gt_root": str(args.gt_root.resolve()),
        "train_count": len(train_tokens),
        "train_sample_budget": train_sample_budget,
        "val_count": len(val_tokens),
        "world_size": args.world_size,
        "samples_per_gpu": args.samples_per_gpu,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "expected_optimizer_steps": expected_steps,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "dynamic_class_ids": list(DYNAMIC_CLASS_IDS),
        "dynamic_coverage": coverage,
        "scene_names": scene_names,
        "train_info": str(train_out),
        "val_info": str(val_out),
        "checked_archive_contents": args.check_archives,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-info", type=Path, required=True)
    parser.add_argument("--val-info", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantics-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--infos-tag", default="queryocc-smoke")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument(
        "--train-sample-budget",
        type=int,
        help="Total sample reads, including repeats; defaults to --train-count",
    )
    parser.add_argument("--val-count", type=int, default=32)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--samples-per-gpu", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--seed", default="queryocc-smoke-v1")
    parser.add_argument("--check-archives", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))
