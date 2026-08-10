"""Verify generated pseudo-labels against the data contract, and optionally
against legacy GaussianFlowOcc label directories.

Contract check (always): samples N random tokens from the infos pkl, loads
the corresponding output files, and asserts shapes / dtypes / camera order /
value ranges.

Legacy cross-check (optional): compares against an existing GaussianFlowOcc
label dir, reordering cameras appropriately. Semantics are compared as
per-pixel agreement (expect ~1.0 only when generated with
`--vocabulary gaussianflowocc --sky-label 0`; DINO/SAM are deterministic in
eval mode but driver/library versions can shift a few boundary pixels).
Depth is compared as mean/max absolute difference (expect a systematic gap
unless generated with `--color-order bgr`).

Example:
    python -m generate_pseudo_labels.verify_outputs \
        --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
        --semantics-root data/pseudo_labels/semantics \
        --depth-root data/pseudo_labels/depth \
        --legacy-mask-root data/grounded_sam_nusc \
        --legacy-depth-root data/metric_3d_nusc
"""

import argparse
import random
from pathlib import Path

import numpy as np

try:
    from .common import load_infos
    from . import constants as C
except ImportError:
    from common import load_infos
    import constants as C


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--infos-pkl", required=True, type=Path)
    parser.add_argument("--semantics-root", type=Path, default=None)
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument("--legacy-mask-root", type=Path, default=None,
                        help="GaussianFlowOcc grounded_sam dir "
                             "(<scene>/<token>.npy, LEGACY_MASK_ORDER)")
    parser.add_argument("--legacy-depth-root", type=Path, default=None,
                        help="GaussianFlowOcc metric_3d dir "
                             "(<scene>/<token>.npy, LEGACY_DEPTH_ORDER)")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def reorder(stack, from_order, to_order):
    return stack[[list(from_order).index(c) for c in to_order]]


def check_semantics(root, info):
    path = root / info["scene_name"] / f"{info['token']}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    labels = data["labels"]
    cam_order = tuple(data["cam_order"])
    assert labels.dtype == np.int8, f"{path}: labels dtype {labels.dtype}"
    assert labels.ndim == 3 and labels.shape[0] == len(C.CAMERA_ORDER), \
        f"{path}: labels shape {labels.shape}"
    assert cam_order == C.CAMERA_ORDER, \
        f"{path}: cam_order {cam_order} != contract {C.CAMERA_ORDER}"
    valid = set(C.CLASS_NAMES) | {0}
    found = set(np.unique(labels).tolist())
    assert found <= valid, f"{path}: unexpected labels {found - valid}"
    return labels, cam_order


def check_depth(root, info):
    path = root / info["scene_name"] / f"{info['token']}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    depth = data["depth"]
    cam_order = tuple(data["cam_order"])
    assert depth.dtype == np.float16, f"{path}: depth dtype {depth.dtype}"
    assert depth.ndim == 3 and depth.shape[0] == len(C.CAMERA_ORDER), \
        f"{path}: depth shape {depth.shape}"
    assert cam_order == C.CAMERA_ORDER, \
        f"{path}: cam_order {cam_order} != contract {C.CAMERA_ORDER}"
    lo, hi = C.METRIC3D_DEPTH_CLAMP
    assert float(depth.min()) >= lo and float(depth.max()) <= hi, \
        f"{path}: depth outside [{lo}, {hi}]"
    return depth, cam_order


def main():
    args = parse_args()
    infos, _ = load_infos(args.infos_pkl)
    rng = random.Random(args.seed)
    picks = rng.sample(infos, min(args.num_samples, len(infos)))

    checked = {"semantics": 0, "depth": 0}
    agreements, depth_diffs = [], []

    for info in picks:
        scene, token = info["scene_name"], info["token"]

        if args.semantics_root is not None:
            result = check_semantics(args.semantics_root, info)
            if result is not None:
                checked["semantics"] += 1
                labels, cam_order = result
                if args.legacy_mask_root is not None:
                    legacy_path = (args.legacy_mask_root / scene /
                                   f"{token}.npy")
                    if legacy_path.exists():
                        legacy = np.load(legacy_path)
                        legacy = reorder(legacy, C.LEGACY_MASK_ORDER,
                                         cam_order)
                        if legacy.shape == labels.shape:
                            agreements.append(
                                float((legacy == labels).mean()))
                        else:
                            print(f"[warn] {token}: legacy mask shape "
                                  f"{legacy.shape} != {labels.shape} "
                                  "(different --scale?), skipping compare")

        if args.depth_root is not None:
            result = check_depth(args.depth_root, info)
            if result is not None:
                checked["depth"] += 1
                depth, cam_order = result
                if args.legacy_depth_root is not None:
                    legacy_path = (args.legacy_depth_root / scene /
                                   f"{token}.npy")
                    if legacy_path.exists():
                        legacy = np.load(legacy_path)
                        legacy = reorder(legacy, C.LEGACY_DEPTH_ORDER,
                                         cam_order)
                        if legacy.shape == depth.shape:
                            diff = np.abs(legacy.astype(np.float32)
                                          - depth.astype(np.float32))
                            depth_diffs.append(
                                (float(diff.mean()), float(diff.max())))
                        else:
                            print(f"[warn] {token}: legacy depth shape "
                                  f"{legacy.shape} != {depth.shape} "
                                  "(different --scale?), skipping compare")

    print(f"\ncontract: {checked['semantics']} semantics / "
          f"{checked['depth']} depth files checked "
          f"(of {len(picks)} sampled) — all assertions passed")
    if agreements:
        print(f"legacy mask agreement: mean {np.mean(agreements):.4f}, "
              f"min {np.min(agreements):.4f} over {len(agreements)} samples")
    if depth_diffs:
        means, maxes = zip(*depth_diffs)
        print(f"legacy depth |diff| (m): mean {np.mean(means):.3f}, "
              f"worst-pixel max {np.max(maxes):.2f} over "
              f"{len(depth_diffs)} samples")
    if checked["semantics"] == 0 and checked["depth"] == 0:
        raise SystemExit("No output files found for any sampled token — "
                         "wrong --save-path / not generated yet?")


if __name__ == "__main__":
    main()
