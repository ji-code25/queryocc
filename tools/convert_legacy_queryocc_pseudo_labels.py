#!/usr/bin/env python3
"""Convert DragonBall/GaussianFlowOcc pseudo labels to QueryOcc's NPZ layout.

Legacy inputs:
  semantics: <flat-root>/<image-stem>_{labels,scores}.npz (one camera/file)
  depth:     either Wazabi's <depth-root>/<image-stem>.npy (one camera/file)
             or GaussianFlowOcc's <depth-root>/<scene>/<sample-token>.npy
             (six cameras/file)

QueryOcc output:
  <output-root>/{semantics,depth}/<scene>/<sample-token>.npz

The conversion also fixes the legacy vocabulary collision where label 17 means
sky. QueryOcc reserves 17 for free space and uses 18 for sky.
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from generate_pseudo_labels.constants import CAMERA_ORDER, DEFAULT_SKY_LABEL, LEGACY_DEPTH_ORDER

LEGACY_SKY_LABEL = 17
DEPTH_LAYOUTS = ("auto", "per-camera", "packed")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a dict, got {type(payload).__name__}")
    for key in ("infos", "data_list"):
        records = payload.get(key)
        if isinstance(records, list):
            return records
    raise KeyError(f"{path}: expected an 'infos' or 'data_list' list")


def record_token(record: dict) -> str:
    for key in ("token", "sample_idx"):
        value = record.get(key)
        if value is not None:
            return str(value)
    raise KeyError("annotation record has no token/sample_idx")


def record_scene_name(record: dict) -> str:
    value = record.get("scene_name") or record.get("scene_idx")
    if value is None:
        raise KeyError(f"{record_token(record)}: annotation has no scene_name/scene_idx")
    return str(value)


def camera_stem(record: dict, camera_name: str) -> str:
    cameras = record.get("cams") or record.get("images")
    if not isinstance(cameras, dict) or camera_name not in cameras:
        raise KeyError(f"{record_token(record)}: missing camera {camera_name}")
    camera = cameras[camera_name]
    image_path = camera.get("data_path") or camera.get("img_path")
    if image_path is None:
        raise KeyError(f"{record_token(record)}: {camera_name} has no image path")
    return Path(image_path).stem


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"{path}: missing array {key!r}; available keys: {archive.files}")
        return np.asarray(archive[key]).copy()


def source_paths(
    record: dict,
    semantics_root: Path,
    depth_root: Path,
    depth_layout: str,
) -> tuple[list[Path], list[Path], list[Path]]:
    stems = [camera_stem(record, camera) for camera in CAMERA_ORDER]
    labels = [semantics_root / f"{stem}_labels.npz" for stem in stems]
    scores = [semantics_root / f"{stem}_scores.npz" for stem in stems]
    if depth_layout == "per-camera":
        depth = [depth_root / f"{stem}.npy" for stem in stems]
    elif depth_layout == "packed":
        depth = [depth_root / record_scene_name(record) / f"{record_token(record)}.npy"]
    else:
        raise ValueError(f"unsupported resolved depth layout: {depth_layout}")
    return labels, scores, depth


def resolve_depth_layout(depth_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if next(depth_root.glob("*.npy"), None) is not None:
        return "per-camera"
    if next(depth_root.glob("*/*.npy"), None) is not None:
        return "packed"
    raise FileNotFoundError(
        f"{depth_root}: could not detect per-camera '<image-stem>.npy' or packed "
        "'<scene>/<token>.npy' depth layout"
    )


def record_is_complete(
    record: dict,
    semantics_root: Path,
    depth_root: Path,
    require_scores: bool,
    depth_layout: str,
    label_names: set[str] | None = None,
    score_names: set[str] | None = None,
    depth_names: set[str] | None = None,
) -> bool:
    labels, scores, depth = source_paths(record, semantics_root, depth_root, depth_layout)
    labels_exist = (
        all(path.name in label_names for path in labels)
        if label_names is not None
        else all(path.is_file() for path in labels)
    )
    scores_exist = (
        all(path.name in score_names for path in scores)
        if score_names is not None
        else all(path.is_file() for path in scores)
    )
    depth_exists = (
        all(path.name in depth_names for path in depth)
        if depth_names is not None
        else all(path.is_file() for path in depth)
    )
    return labels_exist and depth_exists and (not require_scores or scores_exist)


def _validate_camera_planes(arrays: list[np.ndarray], name: str, token: str) -> np.ndarray:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or len(next(iter(shapes), ())) != 2:
        raise ValueError(
            f"{token}: {name} camera arrays must share one 2-D shape, got {sorted(shapes)}"
        )
    return np.stack(arrays)


def convert_record(
    record: dict,
    semantics_root: Path,
    depth_root: Path,
    output_root: Path,
    require_scores: bool,
    overwrite: bool,
    depth_layout: str,
) -> tuple[Path, Path]:
    token = record_token(record)
    scene_name = record_scene_name(record)
    semantics_out = output_root / "semantics" / scene_name / f"{token}.npz"
    depth_out = output_root / "depth" / scene_name / f"{token}.npz"
    if semantics_out.is_file() and depth_out.is_file() and not overwrite:
        return semantics_out, depth_out

    label_paths, score_paths, depth_paths = source_paths(
        record, semantics_root, depth_root, depth_layout
    )
    missing = [path for path in (*label_paths, *depth_paths) if not path.is_file()]
    if require_scores:
        missing.extend(path for path in score_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"{token}: incomplete legacy inputs: {missing}")

    labels = _validate_camera_planes(
        [load_npz_array(path, "labels") for path in label_paths], "labels", token
    )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{token}: labels must be integer, got {labels.dtype}")
    invalid = (labels < 0) | (labels > LEGACY_SKY_LABEL)
    if np.any(invalid):
        raise ValueError(
            f"{token}: labels outside [0, {LEGACY_SKY_LABEL}]: "
            f"{np.unique(labels[invalid]).tolist()}"
        )
    labels = labels.astype(np.int8, copy=False)
    labels[labels == LEGACY_SKY_LABEL] = DEFAULT_SKY_LABEL

    semantics_payload: dict[str, np.ndarray] = {
        "labels": labels,
        "cam_order": np.asarray(CAMERA_ORDER),
    }
    if all(path.is_file() for path in score_paths):
        scores = _validate_camera_planes(
            [load_npz_array(path, "scores") for path in score_paths], "scores", token
        )
        if scores.shape != labels.shape:
            raise ValueError(
                f"{token}: scores shape {scores.shape} differs from labels {labels.shape}"
            )
        semantics_payload["scores"] = scores.astype(np.float16, copy=False)
    elif require_scores:
        raise FileNotFoundError(f"{token}: one or more score archives are missing")

    if depth_layout == "per-camera":
        depth = _validate_camera_planes(
            [np.load(path, allow_pickle=False) for path in depth_paths], "depth", token
        )
    else:
        depth = np.load(depth_paths[0], allow_pickle=False)
        if depth.ndim != 3 or depth.shape[0] != len(LEGACY_DEPTH_ORDER):
            raise ValueError(
                f"{depth_paths[0]}: depth must have shape [6, H, W], got {depth.shape}"
            )
        reorder = [LEGACY_DEPTH_ORDER.index(camera) for camera in CAMERA_ORDER]
        depth = depth[reorder]
    if tuple(depth.shape[1:]) != tuple(labels.shape[1:]):
        raise ValueError(
            f"{token}: depth shape {depth.shape[1:]} differs from labels {labels.shape[1:]}"
        )
    if not np.all(np.isfinite(depth)) or np.any(depth < 0):
        raise ValueError(f"{token}: depth must be finite and non-negative")
    depth = np.asarray(depth, dtype=np.float16)

    semantics_out.parent.mkdir(parents=True, exist_ok=True)
    depth_out.parent.mkdir(parents=True, exist_ok=True)
    semantics_tmp = semantics_out.with_suffix(".tmp.npz")
    depth_tmp = depth_out.with_suffix(".tmp.npz")
    np.savez(semantics_tmp, **semantics_payload)
    np.savez(depth_tmp, depth=depth, cam_order=np.asarray(CAMERA_ORDER))
    semantics_tmp.replace(semantics_out)
    depth_tmp.replace(depth_out)
    return semantics_out, depth_out


def convert(args: argparse.Namespace) -> dict[str, Any]:
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    depth_layout = resolve_depth_layout(args.depth_root, getattr(args, "depth_layout", "auto"))
    records = []
    seen: set[str] = set()
    for infos_path in args.infos:
        for record in load_records(infos_path):
            token = record_token(record)
            if token not in seen:
                records.append(record)
                seen.add(token)

    # One directory scan avoids six existence syscalls for every one of the ~34k
    # nuScenes records (a costly metadata storm on the shared filesystem).
    label_names = {path.name for path in args.semantics_root.glob("*_labels.npz")}
    score_names = {path.name for path in args.semantics_root.glob("*_scores.npz")}
    depth_names = (
        {path.name for path in args.depth_root.glob("*.npy")}
        if depth_layout == "per-camera"
        else None
    )
    complete = [
        record
        for record in records
        if record_is_complete(
            record,
            args.semantics_root,
            args.depth_root,
            args.require_scores,
            depth_layout,
            label_names,
            score_names,
            depth_names,
        )
    ]
    complete.sort(key=lambda record: (record_scene_name(record), record_token(record)))
    available_complete_records = len(complete)
    if args.min_complete is not None and available_complete_records < args.min_complete:
        raise RuntimeError(
            f"only {available_complete_records} records have complete legacy semantics, "
            f"scores and depth; need {args.min_complete}"
        )
    if args.max_samples is not None:
        complete = complete[: args.max_samples]

    converted = 0
    if not args.dry_run:
        convert_one = partial(
            convert_record,
            semantics_root=args.semantics_root,
            depth_root=args.depth_root,
            output_root=args.output_root,
            require_scores=args.require_scores,
            overwrite=args.overwrite,
            depth_layout=depth_layout,
        )
        if workers == 1:
            results = map(convert_one, complete)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(convert_one, complete, chunksize=1)

        progress_interval = max(1, min(500, len(complete) // 20))
        try:
            for _ in results:
                converted += 1
                if converted % progress_interval == 0 or converted == len(complete):
                    print(
                        f"Converted {converted}/{len(complete)} records "
                        f"with {workers} worker(s)",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    summary = {
        "schema_version": 1,
        "legacy_semantics_root": str(args.semantics_root.resolve()),
        "legacy_depth_root": str(args.depth_root.resolve()),
        "legacy_depth_layout": depth_layout,
        "output_root": str(args.output_root.resolve()),
        "records_scanned": len(records),
        "complete_records": available_complete_records,
        "selected_records": len(complete),
        "converted_records": converted,
        "conversion_workers": workers,
        "dry_run": args.dry_run,
        "camera_order": list(CAMERA_ORDER),
        "legacy_depth_order": list(
            CAMERA_ORDER if depth_layout == "per-camera" else LEGACY_DEPTH_ORDER
        ),
        "legacy_sky_label": LEGACY_SKY_LABEL,
        "queryocc_sky_label": DEFAULT_SKY_LABEL,
    }
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "conversion_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infos", type=Path, nargs="+", required=True)
    parser.add_argument("--semantics-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--depth-layout", choices=DEPTH_LAYOUTS, default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-scores", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--min-complete", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(convert(parse_args()), indent=2, sort_keys=True))
