#!/usr/bin/env python3
"""Validate converted pseudo labels and build filtered full-training infos."""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from generate_pseudo_labels.constants import CAMERA_ORDER, DEFAULT_SKY_LABEL
from queryocc.data.vfm.pseudo_label_npz import read_camera_array

NOISE_LABEL = 0


def _records_key(payload: dict[str, Any], path: Path) -> str:
    for key in ("infos", "data_list"):
        if isinstance(payload.get(key), list):
            return key
    raise KeyError(f"{path}: expected an 'infos' or 'data_list' list")


def _load_payload(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a dict, got {type(payload).__name__}")
    return payload, _records_key(payload, path)


def _record_token(record: dict[str, Any]) -> str:
    value = record.get("token") or record.get("sample_idx")
    if value is None:
        raise KeyError("record has no token/sample_idx")
    return str(value)


def _record_scene(record: dict[str, Any]) -> str:
    value = record.get("scene_name") or record.get("scene_idx")
    if value is None:
        raise KeyError(f"{_record_token(record)}: record has no scene_name/scene_idx")
    return str(value)


def _member_header(path: Path, member_name: str) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(path) as archive:
        if member_name not in archive.namelist():
            raise KeyError(f"{path}: missing {member_name}")
        with archive.open(member_name) as member:
            version = np.lib.format.read_magic(member)
            shape, fortran_order, dtype = np.lib.format._read_array_header(member, version)
    if fortran_order or dtype.hasobject:
        raise ValueError(f"{path}:{member_name}: unsupported numpy layout")
    return tuple(shape), dtype


def _inspect_record(record: dict[str, Any], semantics_root: Path, depth_root: Path) -> dict[str, Any]:
    token = _record_token(record)
    scene = _record_scene(record)
    semantics_path = semantics_root / scene / f"{token}.npz"
    depth_path = depth_root / scene / f"{token}.npz"

    label_shape, label_dtype = _member_header(semantics_path, "labels.npy")
    score_shape, score_dtype = _member_header(semantics_path, "scores.npy")
    depth_shape, depth_dtype = _member_header(depth_path, "depth.npy")
    expected_planes = len(CAMERA_ORDER)
    if label_shape != score_shape or label_shape != depth_shape:
        raise ValueError(f"{token}: labels/scores/depth shapes differ: {label_shape}, {score_shape}, {depth_shape}")
    if len(label_shape) != 3 or label_shape[0] != expected_planes:
        raise ValueError(f"{token}: expected [{expected_planes}, H, W], got {label_shape}")
    if not np.issubdtype(label_dtype, np.integer):
        raise TypeError(f"{token}: labels must be integer, got {label_dtype}")
    if not np.issubdtype(score_dtype, np.floating) or not np.issubdtype(depth_dtype, np.floating):
        raise TypeError(f"{token}: scores/depth must be floating point")

    has_positive_candidate = False
    for camera in CAMERA_ORDER:
        labels = read_camera_array(semantics_root, scene, token, camera, "labels")
        depth = read_camera_array(depth_root, scene, token, camera, "depth")
        if labels is None or depth is None or labels.shape != depth.shape:
            raise ValueError(f"{token}/{camera}: missing or shape-mismatched labels/depth")
        valid = (
            (labels != NOISE_LABEL) & (labels != DEFAULT_SKY_LABEL) & np.isfinite(depth) & (depth > 0) & (depth < 160)
        )
        if np.any(valid):
            has_positive_candidate = True
            break

    return {"token": token, "scene": scene, "blacklisted": not has_positive_candidate}


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    train_payload, train_key = _load_payload(args.train_info)
    val_payload, _ = _load_payload(args.val_info)
    train_records = train_payload[train_key]

    inspect_one = partial(
        _inspect_record,
        semantics_root=args.semantics_root,
        depth_root=args.depth_root,
    )
    results = []
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    mapped = executor.map(inspect_one, train_records, chunksize=8) if executor else map(inspect_one, train_records)
    try:
        for index, result in enumerate(mapped, start=1):
            results.append(result)
            if index % 500 == 0 or index == len(train_records):
                print(f"Preflight {index}/{len(train_records)} records", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    blacklist = [result for result in results if result["blacklisted"]]
    blacklisted_tokens = {item["token"] for item in blacklist}
    filtered_payload = dict(train_payload)
    filtered_payload[train_key] = [
        record for record in train_records if _record_token(record) not in blacklisted_tokens
    ]

    args.infos_output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    filtered_train_path = args.infos_output_dir / f"{args.infos_tag}_infos_train.pkl"
    filtered_val_path = args.infos_output_dir / f"{args.infos_tag}_infos_val.pkl"
    blacklist_path = args.artifact_dir / "full_training_blacklist.json"
    preflight_path = args.artifact_dir / "full_training_preflight.json"
    _atomic_pickle(filtered_train_path, filtered_payload)
    _atomic_pickle(filtered_val_path, val_payload)
    _atomic_json(blacklist_path, {"schema_version": 1, "records": blacklist})

    summary = {
        "schema_version": 1,
        "train_info": str(args.train_info.resolve()),
        "val_info": str(args.val_info.resolve()),
        "semantics_root": str(args.semantics_root.resolve()),
        "depth_root": str(args.depth_root.resolve()),
        "records_checked": len(train_records),
        "blacklisted_records": len(blacklist),
        "filtered_train_records": len(filtered_payload[train_key]),
        "filtered_train_info": str(filtered_train_path.resolve()),
        "filtered_val_info": str(filtered_val_path.resolve()),
        "blacklist": str(blacklist_path.resolve()),
        "infos_tag": args.infos_tag,
        "workers": args.workers,
        "ready": True,
    }
    _atomic_json(preflight_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-info", type=Path, required=True)
    parser.add_argument("--val-info", type=Path, required=True)
    parser.add_argument("--semantics-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--infos-output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--infos-tag", default="queryocc-refined")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))
