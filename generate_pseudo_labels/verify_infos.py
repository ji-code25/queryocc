"""Verify an infos pkl produced by create_infos.py.

Checks the schema and the internal consistency of every sample (cheap, all
samples), and that the referenced files actually exist on disk (expensive,
on a random subset unless --check-files-all).

Optionally cross-checks against a reference pkl (e.g. the original bevdetv2
pkls shipped with GaussianFlowOcc / BEVDet): every token present in both is
compared on poses, intrinsics and box counts, so a regenerated index can be
proven equivalent to the one the published results used.

Example:
    python -m generate_pseudo_labels.verify_infos \
        --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
        --data-root data/nuscenes \
        --reference-pkl data/orig/bevdetv2-nuscenes_infos_train.pkl
"""

import argparse
import random
from pathlib import Path

import numpy as np

try:
    from .common import load_infos, resolve_path
    from . import constants as C
except ImportError:  # run as a plain file, not a package
    from common import load_infos, resolve_path
    import constants as C


REQUIRED_KEYS = (
    "token", "scene_token", "scene_name", "timestamp", "lidar_idx",
    "lidar_path", "occ_path", "cams",
    "lidar2ego_translation", "lidar2ego_rotation",
    "ego2global_translation", "ego2global_rotation",
)
ANNOTATION_KEYS = (
    "gt_boxes", "gt_names", "gt_velocity", "num_lidar_pts", "num_radar_pts",
    "valid_flag", "ann_infos",
)
CAM_KEYS = (
    "data_path", "cam_intrinsic", "sample_data_token",
    "sensor2ego_translation", "sensor2ego_rotation",
    "ego2global_translation", "ego2global_rotation", "timestamp",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--infos-pkl", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="root the stored data paths are relative to")
    parser.add_argument("--occ-root", type=Path, default=None,
                        help="root for occ_path (default: --data-root); "
                             "checks <root>/<occ_path>/labels.npz")
    parser.add_argument("--reference-pkl", type=Path, default=None,
                        help="compare shared tokens against this pkl")
    parser.add_argument("--num-file-checks", type=int, default=20,
                        help="samples to check on disk (0 disables)")
    parser.add_argument("--check-files-all", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def check(self, ok, msg):
        if not ok:
            self.error(msg)
        return ok


def is_unit_quaternion(q):
    q = np.asarray(q, dtype=float)
    return q.shape == (4,) and abs(np.linalg.norm(q) - 1.0) < 1e-3


def check_schema(infos, rep):
    """Per-sample field presence, shapes and internal consistency."""
    tokens, has_ann = set(), 0
    for i, info in enumerate(infos):
        where = f"info[{i}]"
        for key in REQUIRED_KEYS:
            rep.check(key in info, f"{where}: missing '{key}'")
        if "token" in info:
            rep.check(info["token"] not in tokens,
                      f"{where}: duplicate token {info.get('token')}")
            tokens.add(info["token"])
        for key in ("lidar2ego_rotation", "ego2global_rotation"):
            if key in info:
                rep.check(is_unit_quaternion(info[key]),
                          f"{where}: {key} is not a unit quaternion")

        cams = info.get("cams", {})
        rep.check(set(cams) == set(C.CAMERA_ORDER),
                  f"{where}: cams {sorted(cams)} != {sorted(C.CAMERA_ORDER)}")
        for cam, cam_info in cams.items():
            at = f"{where}.cams[{cam}]"
            for key in CAM_KEYS:
                rep.check(key in cam_info, f"{at}: missing '{key}'")
            k = np.asarray(cam_info.get("cam_intrinsic"))
            rep.check(k.shape == (3, 3),
                      f"{at}: intrinsic {k.shape} != (3, 3)")
            if k.shape == (3, 3):
                rep.check(k[0, 0] > 0 and k[1, 1] > 0 and k[2, 2] == 1,
                          f"{at}: implausible intrinsic {k.tolist()}")
            rep.check(
                is_unit_quaternion(cam_info.get("sensor2ego_rotation", [])),
                f"{at}: sensor2ego_rotation is not a unit quaternion")

        if "gt_boxes" not in info:
            continue  # test split
        has_ann += 1
        for key in ANNOTATION_KEYS:
            rep.check(key in info, f"{where}: missing '{key}'")
        n = len(info["gt_boxes"])
        for key in ("gt_names", "gt_velocity", "num_lidar_pts",
                    "num_radar_pts", "valid_flag"):
            rep.check(len(info.get(key, [])) == n,
                      f"{where}: len({key})={len(info.get(key, []))} != "
                      f"len(gt_boxes)={n}")
        if n:  # an empty gt_boxes has no inferable column count
            boxes = np.asarray(info["gt_boxes"], dtype=float).reshape(n, -1)
            rep.check(boxes.shape[1] == 7,
                      f"{where}: gt_boxes has {boxes.shape[1]} columns, "
                      "expected 7")
            rep.check(bool(np.all(np.isfinite(boxes))),
                      f"{where}: non-finite gt_boxes")
            rep.check(bool(np.all(boxes[:, 3:6] > 0)),
                      f"{where}: non-positive box dimensions")
        ann_boxes, ann_labels = info["ann_infos"]
        rep.check(len(ann_boxes) == len(ann_labels),
                  f"{where}: ann_infos boxes/labels length mismatch")
        for b in ann_boxes:
            rep.check(len(b) == 9,
                      f"{where}: ann_infos box has {len(b)} values, "
                      "expected 9")
        # ann_infos keeps only lidar/radar-visible boxes of detection classes.
        rep.check(len(ann_boxes) <= n,
                  f"{where}: more ann_infos boxes than gt_boxes")
    return has_ann


def check_order(infos, rep):
    keys = [(i["scene_name"], i["timestamp"]) for i in infos]
    rep.check(keys == sorted(keys),
              "infos are not sorted by (scene_name, timestamp) — index +/- t "
              "adjacency will not work")
    # Each scene must be one contiguous block, and one scene_token per name.
    seen, prev = set(), None
    name_to_token = {}
    for info in infos:
        name = info["scene_name"]
        if name != prev:
            rep.check(name not in seen,
                      f"scene {name} appears in several blocks")
            seen.add(name)
            prev = name
        token = name_to_token.setdefault(name, info["scene_token"])
        rep.check(token == info["scene_token"],
                  f"scene {name} maps to several scene_tokens")


def check_files(infos, data_root, occ_root, rep, indices):
    for i in indices:
        info = infos[i]
        where = f"info[{i}]"
        try:
            resolve_path(info["lidar_path"], data_root)
        except FileNotFoundError as e:
            rep.error(f"{where}: {e}")
        for cam, cam_info in info["cams"].items():
            try:
                resolve_path(cam_info["data_path"], data_root)
            except FileNotFoundError as e:
                rep.error(f"{where}.cams[{cam}]: {e}")
        if info.get("panoptic_path"):
            try:
                resolve_path(info["panoptic_path"], data_root)
            except FileNotFoundError as e:
                rep.warn(f"{where}: panoptic {e}")
        if occ_root is not None:
            occ = Path(occ_root) / str(info["occ_path"]).lstrip("./")
            if not (occ / "labels.npz").exists():
                rep.warn(f"{where}: missing occ GT {occ / 'labels.npz'}")


def compare_reference(infos, ref_infos, rep):
    """Compare every token present in both files."""
    ref = {i["token"]: i for i in ref_infos if "token" in i}
    shared = [i for i in infos if i["token"] in ref]
    print(f"[reference] {len(shared)} shared tokens "
          f"({len(infos)} generated, {len(ref_infos)} reference)")
    if not shared:
        rep.warn("no shared tokens with the reference pkl")
        return
    worst = {}
    for info in shared:
        r = ref[info["token"]]
        where = f"token {info['token']}"
        for key in ("lidar2ego_translation", "lidar2ego_rotation",
                    "ego2global_translation", "ego2global_rotation"):
            if key not in r:
                continue
            d = float(np.max(np.abs(np.asarray(info[key], dtype=float)
                                    - np.asarray(r[key], dtype=float))))
            worst[key] = max(worst.get(key, 0.0), d)
            rep.check(d < 1e-6, f"{where}: {key} differs by {d:g}")
        for cam in info["cams"]:
            if cam not in r.get("cams", {}):
                continue
            d = float(np.max(np.abs(
                np.asarray(info["cams"][cam]["cam_intrinsic"], dtype=float)
                - np.asarray(r["cams"][cam]["cam_intrinsic"], dtype=float))))
            worst["cam_intrinsic"] = max(worst.get("cam_intrinsic", 0.0), d)
            rep.check(d < 1e-6,
                      f"{where}.cams[{cam}]: intrinsics differ by {d:g}")
            gen_name = Path(str(info["cams"][cam]["data_path"])).name
            ref_name = Path(str(r["cams"][cam]["data_path"])).name
            rep.check(gen_name == ref_name,
                      f"{where}.cams[{cam}]: image {gen_name} != {ref_name}")
        if "gt_boxes" in info and "gt_boxes" in r:
            rep.check(len(info["gt_boxes"]) == len(r["gt_boxes"]),
                      f"{where}: {len(info['gt_boxes'])} gt_boxes vs "
                      f"{len(r['gt_boxes'])} in the reference")
            gen_boxes = np.asarray(info["gt_boxes"], dtype=float)
            ref_boxes = np.asarray(r["gt_boxes"], dtype=float)
            if len(gen_boxes) and gen_boxes.shape == ref_boxes.shape:
                d = float(np.max(np.abs(gen_boxes - ref_boxes)))
                worst["gt_boxes"] = max(worst.get("gt_boxes", 0.0), d)
                rep.check(d < 1e-4, f"{where}: gt_boxes differ by {d:g}")
    print("[reference] max abs difference: "
          + ", ".join(f"{k}={v:g}" for k, v in sorted(worst.items())))


def main():
    args = parse_args()
    random.seed(args.seed)

    infos, metadata = load_infos(args.infos_pkl)
    print(f"{args.infos_pkl}: {len(infos)} samples, metadata={metadata}")
    scenes = {i["scene_name"] for i in infos}
    print(f"{len(scenes)} scenes, "
          f"{sum('gt_boxes' in i for i in infos)} with annotations")

    rep = Report()
    rep.check(len(infos) > 0, "empty infos list")
    rep.check("version" in metadata, "metadata has no 'version'")
    check_schema(infos, rep)
    check_order(infos, rep)

    if args.num_file_checks or args.check_files_all:
        indices = (range(len(infos)) if args.check_files_all
                   else random.sample(range(len(infos)),
                                      min(args.num_file_checks, len(infos))))
        indices = list(indices)
        occ_root = args.occ_root or args.data_root
        print(f"[files] checking {len(indices)} samples "
              f"(data_root={args.data_root}, occ_root={occ_root})")
        check_files(infos, args.data_root, occ_root, rep, indices)

    if args.reference_pkl:
        ref_infos, _ = load_infos(args.reference_pkl)
        compare_reference(infos, ref_infos, rep)

    for msg in rep.warnings[:20]:
        print(f"WARN  {msg}")
    if len(rep.warnings) > 20:
        print(f"... and {len(rep.warnings) - 20} more warnings")
    for msg in rep.errors[:20]:
        print(f"ERROR {msg}")
    if len(rep.errors) > 20:
        print(f"... and {len(rep.errors) - 20} more errors")

    if rep.errors:
        print(f"FAILED: {len(rep.errors)} errors, {len(rep.warnings)} warnings")
        raise SystemExit(1)
    print(f"OK: no errors, {len(rep.warnings)} warnings")


if __name__ == "__main__":
    main()
