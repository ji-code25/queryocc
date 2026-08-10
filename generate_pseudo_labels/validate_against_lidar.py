"""Validate generated pseudo-labels against lidar ground truth.

`verify_outputs.py` checks the data contract (shapes, dtypes, camera order).
This script checks whether the labels are *right*: it projects the lidar
point cloud of each sample into every camera and compares

  * depth      — predicted depth at the projected pixel vs the lidar range
                 (abs-rel error, RMSE, delta<1.25 — the standard monocular
                 depth metrics), and
  * semantics  — the predicted class at the projected pixel vs the nuScenes
                 lidarseg class of the point (overall + per-class accuracy).

This catches the failure modes a contract check cannot: wrong camera order,
transposed extrinsics, a wrong metric-depth scale factor, a shifted class-id
space. Agreement is expected to be good but not perfect — the pseudo-labels
are open-vocabulary predictions, the projection ignores occlusion (a lidar
point can be hidden behind a closer surface), and lidar returns are sparse.
What matters is that the numbers are in a sane regime and stable between
runs; a broken pipeline is off by an order of magnitude, not a few percent.

Needs the raw lidar (`lidar_path` in the infos) and, for the semantic part,
nuScenes-lidarseg at <data-root>/lidarseg/<version>/<token>_lidarseg.bin.

Example:
    python -m generate_pseudo_labels.validate_against_lidar \
        --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
        --data-root data/nuscenes \
        --semantics-root data/pseudo_labels/semantics \
        --depth-root data/pseudo_labels/depth
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

try:
    from . import constants as C
    from .common import load_infos, resolve_path
except ImportError:  # run as a plain file, not a package
    import constants as C
    from common import load_infos, resolve_path


# nuScenes-lidarseg raw class index (0..31) -> the 16-class id space used by
# the pseudo-labels (constants.CLASS_NAMES). This is the standard nuScenes
# challenge mapping; everything not in the 16 classes becomes 0 = ignore.
LIDARSEG_TO_CLASS = np.array([
    0,   # 0  noise
    0,   # 1  animal
    7,   # 2  human.pedestrian.adult
    7,   # 3  human.pedestrian.child
    7,   # 4  human.pedestrian.construction_worker
    0,   # 5  human.pedestrian.personal_mobility
    7,   # 6  human.pedestrian.police_officer
    0,   # 7  human.pedestrian.stroller
    0,   # 8  human.pedestrian.wheelchair
    1,   # 9  movable_object.barrier
    0,   # 10 movable_object.debris
    0,   # 11 movable_object.pushable_pullable
    8,   # 12 movable_object.trafficcone
    0,   # 13 static_object.bicycle_rack
    2,   # 14 vehicle.bicycle
    3,   # 15 vehicle.bus.bendy
    3,   # 16 vehicle.bus.rigid
    4,   # 17 vehicle.car
    5,   # 18 vehicle.construction
    0,   # 19 vehicle.emergency.ambulance
    0,   # 20 vehicle.emergency.police
    6,   # 21 vehicle.motorcycle
    9,   # 22 vehicle.trailer
    10,  # 23 vehicle.truck
    11,  # 24 flat.driveable_surface
    12,  # 25 flat.other
    13,  # 26 flat.sidewalk
    14,  # 27 flat.terrain
    15,  # 28 static.manmade
    0,   # 29 static.other
    16,  # 30 static.vegetation
    0,   # 31 vehicle.ego
], dtype=np.int16)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--infos-pkl", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--semantics-root", type=Path, default=None)
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument("--lidarseg-version", default=None,
                        help="defaults to the infos metadata version")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth", type=float, default=80.0,
                        help="ignore lidar points beyond this range")
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--check-camera-order", action="store_true",
                        help="score every stored depth plane against every "
                             "camera's lidar projection; the diagonal must "
                             "win. The decisive test for a mixed-up "
                             "cam_order (see constants.LEGACY_*_ORDER)")
    return parser.parse_args()


def transform_matrix(translation, rotation, inverse=False):
    """4x4 from a nuScenes (translation, quaternion [w,x,y,z]) pair."""
    matrix = np.eye(4)
    rot = Quaternion(rotation).rotation_matrix
    if inverse:
        matrix[:3, :3] = rot.T
        matrix[:3, 3] = -rot.T @ np.asarray(translation)
    else:
        matrix[:3, :3] = rot
        matrix[:3, 3] = np.asarray(translation)
    return matrix


def lidar_to_camera(info, cam):
    """4x4 taking lidar-frame points to `cam`'s frame, via the two ego poses
    (the lidar and the camera are captured at different timestamps)."""
    cam_info = info["cams"][cam]
    lidar2ego = transform_matrix(info["lidar2ego_translation"],
                                 info["lidar2ego_rotation"])
    ego2global = transform_matrix(info["ego2global_translation"],
                                  info["ego2global_rotation"])
    global2ego_cam = transform_matrix(cam_info["ego2global_translation"],
                                      cam_info["ego2global_rotation"],
                                      inverse=True)
    ego2cam = transform_matrix(cam_info["sensor2ego_translation"],
                               cam_info["sensor2ego_rotation"], inverse=True)
    return ego2cam @ global2ego_cam @ ego2global @ lidar2ego


def load_lidar(path):
    points = np.fromfile(str(path), dtype=np.float32)
    return points.reshape(-1, 5)[:, :3]


def load_lidarseg(data_root, version, lidar_token):
    path = (Path(data_root) / "lidarseg" / version
            / f"{lidar_token}_lidarseg.bin")
    if not path.exists():
        return None
    return np.fromfile(str(path), dtype=np.uint8)


def label_scale(root, meta_file):
    """The --scale the labels in `root` were written with (see write_meta)."""
    if root is None:
        return 1.0
    meta_path = Path(root) / meta_file
    if not meta_path.exists():
        return 1.0
    return float(json.loads(meta_path.read_text()).get("scale", 1.0))


def project(points, info, cam, image_hw, intrinsic_scale=1.0):
    """Lidar points -> (u, v, depth) inside `cam`, plus the keep mask.

    `intrinsic_scale` mirrors the generators' --scale: labels stored
    downscaled need fx, fy, cx, cy scaled by the same factor.
    """
    hom = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    in_cam = (lidar_to_camera(info, cam) @ hom.T).T[:, :3]
    depth = in_cam[:, 2]
    valid = depth > 0.1
    intrinsic = np.asarray(info["cams"][cam]["cam_intrinsic"], dtype=float)
    if intrinsic_scale != 1.0:
        intrinsic = intrinsic.copy()
        intrinsic[:2] *= intrinsic_scale
    uv = (intrinsic @ in_cam[valid].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    h, w = image_hw
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w)
              & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    keep = np.flatnonzero(valid)[inside]
    return uv[inside], depth[keep], keep


def depth_metrics(pred, gt):
    """Standard monocular depth metrics on matched pixels."""
    abs_rel = float(np.mean(np.abs(pred - gt) / gt))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "delta<1.25": float(np.mean(ratio < 1.25)),
        "delta<1.25^2": float(np.mean(ratio < 1.25 ** 2)),
        # A systematic metric-scale error shows up here and nowhere else:
        # 1.0 = unbiased, 0.5 / 2.0 = the scale factor is wrong.
        "median_pred/gt": float(np.median(pred / gt)),
        "points": float(len(pred)),
    }


def camera_order_matrix(infos, picks, depth_root, data_root, scale,
                        min_depth, max_depth):
    """abs_rel of each stored depth plane against each camera's lidar
    projection. Correct ordering => the diagonal wins every row."""
    error = counts = cams = None
    for i in picks:
        info = infos[i]
        path = depth_root / info["scene_name"] / f"{info['token']}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        cams = [str(c) for c in data["cam_order"]]
        planes = data["depth"]
        if error is None:
            error = np.zeros((len(cams), len(cams)))
            counts = np.zeros((len(cams), len(cams)))
        points = load_lidar(resolve_path(info["lidar_path"], data_root))
        h, w = planes[0].shape
        for j, cam in enumerate(cams):
            uv, gt, _ = project(points, info, cam, (h, w), scale)
            inside = (gt >= min_depth) & (gt <= max_depth)
            uv, gt = uv[inside], gt[inside]
            if not len(gt):
                continue
            rows = uv[:, 1].astype(int)
            cols = uv[:, 0].astype(int)
            for plane_idx in range(len(cams)):
                pred = planes[plane_idx][rows, cols].astype(np.float32)
                ok = pred > 0
                error[plane_idx, j] += float(
                    np.sum(np.abs(pred[ok] - gt[ok]) / gt[ok]))
                counts[plane_idx, j] += int(ok.sum())
    if error is None:
        return None, None
    return error / np.maximum(counts, 1), cams


def main():
    args = parse_args()
    random.seed(args.seed)
    infos, metadata = load_infos(args.infos_pkl)
    version = args.lidarseg_version or metadata.get("version", "v1.0-trainval")
    picks = random.sample(range(len(infos)),
                          min(args.num_samples, len(infos)))
    scales = {
        "labels": label_scale(args.semantics_root, C.SEMANTICS_META_FILE),
        "depth": label_scale(args.depth_root, C.DEPTH_META_FILE),
    }

    depth_pred, depth_gt = [], []
    depth_per_cam = defaultdict(lambda: ([], []))
    sem_hits = defaultdict(int)
    sem_total = defaultdict(int)
    sem_confusion = defaultdict(int)
    missing = []

    for i in picks:
        info = infos[i]
        scene, token = info["scene_name"], info["token"]
        points = load_lidar(resolve_path(info["lidar_path"], args.data_root))
        seg = load_lidarseg(args.data_root, version, info["lidar_idx"])

        sem_file = (args.semantics_root / scene / f"{token}.npz"
                    if args.semantics_root else None)
        depth_file = (args.depth_root / scene / f"{token}.npz"
                      if args.depth_root else None)
        sem = np.load(sem_file) if sem_file and sem_file.exists() else None
        dep = (np.load(depth_file)
               if depth_file and depth_file.exists() else None)
        if args.semantics_root and sem is None:
            missing.append(str(sem_file))
        if args.depth_root and dep is None:
            missing.append(str(depth_file))

        for source, key in ((sem, "labels"), (dep, "depth")):
            if source is None:
                continue
            cam_order = [str(c) for c in source["cam_order"]]
            array = source[key]
            # labels/depth may be stored downscaled (--scale)
            for cam_idx, cam in enumerate(cam_order):
                plane = array[cam_idx]
                h, w = plane.shape
                uv, gt_depth, keep = project(points, info, cam, (h, w),
                                             scales[key])
                if not len(keep):
                    continue
                inside = ((gt_depth >= args.min_depth)
                          & (gt_depth <= args.max_depth))
                uv, gt_depth, keep = uv[inside], gt_depth[inside], keep[inside]
                if not len(keep):
                    continue
                # Keep only the NEAREST lidar return per pixel. Without this
                # every point hidden behind a closer surface is compared
                # against that surface's prediction, which biases depth low
                # and scrambles the semantic comparison.
                flat = (uv[:, 1].astype(int) * w + uv[:, 0].astype(int))
                far_to_near = np.argsort(-gt_depth)
                nearest_depth = np.full(h * w, np.nan, dtype=np.float32)
                nearest_depth[flat[far_to_near]] = gt_depth[far_to_near]
                pixels = np.flatnonzero(~np.isnan(nearest_depth))
                gt_at_pixel = nearest_depth[pixels]
                flat_plane = plane.reshape(-1)

                if key == "depth":
                    pred = flat_plane[pixels].astype(np.float32)
                    ok = pred > 0
                    depth_pred.append(pred[ok])
                    depth_gt.append(gt_at_pixel[ok])
                    depth_per_cam[cam][0].append(pred[ok])
                    depth_per_cam[cam][1].append(gt_at_pixel[ok])
                elif seg is not None:
                    nearest_class = np.zeros(h * w, dtype=np.int16)
                    nearest_class[flat[far_to_near]] = \
                        LIDARSEG_TO_CLASS[seg[keep]][far_to_near]
                    gt_class = nearest_class[pixels]
                    pred_class = flat_plane[pixels].astype(np.int16)
                    labelled = gt_class > 0        # ignore lidarseg 'noise'
                    for g, p in zip(gt_class[labelled], pred_class[labelled]):
                        sem_total[int(g)] += 1
                        sem_hits[int(g)] += int(g == p)
                        if g != p:
                            sem_confusion[(int(g), int(p))] += 1

    print(f"{len(picks)} samples, lidarseg version {version}, "
          f"label scales {scales}")
    if missing:
        print(f"WARN {len(missing)} missing output files, e.g. {missing[0]}")

    if depth_pred:
        pred = np.concatenate(depth_pred)
        gt = np.concatenate(depth_gt)
        print(f"\n[depth] {len(pred)} matched lidar points "
              f"({args.min_depth}-{args.max_depth} m)")
        for name, value in depth_metrics(pred, gt).items():
            print(f"  {name:>15}: {value:.4f}")
        # Per camera: a mismatched camera order or a bad extrinsic shows up
        # as one camera being far worse than the rest.
        per_cam = {
            cam: depth_metrics(np.concatenate(pr), np.concatenate(gt_))
            for cam, (pr, gt_) in sorted(depth_per_cam.items())
        }
        print(f"  {'per camera':>15}: "
              + ", ".join(f"{cam.replace('CAM_', '')} {m['abs_rel']:.3f}"
                          for cam, m in per_cam.items()))
    elif args.depth_root:
        print("\n[depth] no matched points")

    if args.check_camera_order and args.depth_root:
        matrix, cams = camera_order_matrix(
            infos, picks, args.depth_root, args.data_root, scales["depth"],
            args.min_depth, args.max_depth)
        print("\n[camera order] abs_rel of stored plane (row) vs the camera "
              "the lidar was projected into (column)")
        header = "plane / camera"
        print(f"  {header:>16}"
              + "".join(f"{c.replace('CAM_', ''):>13}" for c in cams))
        wrong = []
        for i, cam in enumerate(cams):
            print(f"  {cam.replace('CAM_', ''):>16}"
                  + "".join(f"{v:>13.3f}" for v in matrix[i]))
            if int(matrix[i].argmin()) != i:
                wrong.append(f"{cam} matches {cams[int(matrix[i].argmin())]}")
        if wrong:
            print("  FAILED: " + "; ".join(wrong))
            raise SystemExit(1)
        print("  OK: every plane matches its own camera")

    if sem_total:
        hits = sum(sem_hits.values())
        total = sum(sem_total.values())
        print(f"\n[semantics] {total} matched lidar points, "
              f"overall accuracy {hits / total:.3f}")
        print(f"  {'class':>22} {'points':>9} {'acc':>7}   most confused with")
        for class_id in sorted(sem_total, key=lambda c: -sem_total[c]):
            worst = max(
                ((p, n) for (g, p), n in sem_confusion.items()
                 if g == class_id),
                key=lambda item: item[1], default=(None, 0))
            confused = ("-" if worst[0] is None else
                        f"{C.CLASS_NAMES.get(worst[0], worst[0])} "
                        f"({worst[1] / sem_total[class_id]:.0%})")
            print(f"  {C.CLASS_NAMES.get(class_id, class_id):>22} "
                  f"{sem_total[class_id]:>9} "
                  f"{sem_hits[class_id] / sem_total[class_id]:>7.3f}   "
                  f"{confused}")
    elif args.semantics_root:
        print("\n[semantics] no lidarseg labels found — skipped")


if __name__ == "__main__":
    main()
