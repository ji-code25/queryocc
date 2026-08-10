"""Create bevdetv2-compatible infos pkl files directly from nuscenes-devkit.

Standalone replacement for GaussianFlowOcc's `tools/create_data_bevdet.py`
(which routes through mmdet3d's nuscenes_converter). Requires only:
    nuscenes-devkit, pyquaternion, numpy

The point: run this ONCE inside the pseudo-label container, then your
training environment only ever reads plain pickles — no nuscenes-devkit, no
mmcv/mmdet3d, no locked-in dependencies.

Output: <out-dir>/<tag>_infos_{train,val}.pkl (or _test.pkl), a dict
    {"infos": [sample, ...], "metadata": {"version": ...}}
with per-sample fields identical to the bevdetv2 pkls this repo already
uses (see README for the full schema). Extra convenience: infos are sorted
by (scene_name, timestamp), so index +/- t adjacency within a scene works
directly (guard with scene_token as usual).

Example:
    python -m generate_pseudo_labels.create_infos \
        --root-path data/nuscenes --version v1.0-mini --out-dir data
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


# mmdet3d's NuScenesDataset.NameMapping (for gt_names).
NAME_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

# BEVDet's category -> detection-class mapping (for ann_infos); categories
# mapped to 'ignore' are dropped.
GENERAL_TO_DETECTION = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}
DETECTION_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root-path", type=Path,
                        default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval",
                        choices=["v1.0-trainval", "v1.0-test", "v1.0-mini"])
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--tag", default=None,
                        help="filename prefix; defaults to bevdetv2-nuscenes"
                             " (or bevdetv2-nuscenes-mini for v1.0-mini)")
    parser.add_argument("--occ-path-prefix", default="./data/gts",
                        help="prefix for the occ3d GT path stored per sample")
    parser.add_argument("--relative-to", type=Path, default=None,
                        help="if set, store data paths relative to this "
                             "directory (default: keep devkit paths, made "
                             "relative to the current working directory "
                             "when possible)")
    return parser.parse_args()


def make_relative(path, relative_to):
    path = str(path)
    base = str(relative_to) if relative_to is not None else os.getcwd()
    if path.startswith(base + os.sep):
        return os.path.relpath(path, base)
    return path


def panoptic_file(nusc, lidar_token, relative_to):
    """Path to the panoptic .npz of a lidar sample_data token, or None.

    Prefers the devkit's `panoptic` table, and falls back to the documented
    layout (panoptic/<version>/<token>_panoptic.npz) — dataset copies often
    ship the payloads without the index json, in which case the table lookup
    raises. Returns None when the file is genuinely absent.
    """
    try:
        path = os.path.join(nusc.dataroot,
                            nusc.get("panoptic", lidar_token)["filename"])
    except Exception:
        path = os.path.join(nusc.dataroot, "panoptic", nusc.version,
                            f"{lidar_token}_panoptic.npz")
    return make_relative(path, relative_to) if os.path.exists(path) else None


def sensor_info(nusc, sensor_token, sensor_type, relative_to):
    """Per-camera dict, same fields as mmdet3d's obtain_sensor2top minus the
    sensor2lidar_* derivatives (recompute from the poses if you need them)."""
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor",
                         sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    return {
        "data_path": make_relative(
            nusc.get_sample_data_path(sd_rec["token"]), relative_to),
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }


def boxes_in_lidar_frame(nusc, sample):
    """Detection-style annotation fields in the lidar frame, byte-compatible
    with mmdet3d's `_fill_trainval_infos`: gt_boxes [N,7] =
    (x, y, z, w, l, h, -yaw - pi/2), gt_names, gt_velocity (lidar frame),
    num_lidar_pts, num_radar_pts, valid_flag.

    NOTE the two conventions in one pkl, both inherited from upstream:
    `gt_boxes` here keeps nuScenes' wlh order and mmdet3d's rotated yaw,
    while `ann_infos` (bevdet_ann_infos) stores lwh and the raw yaw in the
    ego frame. Keeping each bit-identical to its upstream producer is what
    makes these pkls drop-in replacements.
    """
    lidar_token = sample["data"]["LIDAR_TOP"]
    _, boxes, _ = nusc.get_sample_data(lidar_token)
    annotations = [nusc.get("sample_annotation", t) for t in sample["anns"]]

    sd_rec = nusc.get("sample_data", lidar_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    l2e_r_mat = Quaternion(cs_record["rotation"]).rotation_matrix
    e2g_r_mat = Quaternion(pose_record["rotation"]).rotation_matrix

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                     for b in boxes]).reshape(-1, 1)
    velocity = np.array(
        [nusc.box_velocity(t)[:2] for t in sample["anns"]]).reshape(-1, 2)
    # global -> lidar frame velocity
    for i in range(len(boxes)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velocity[i] = velo[:2]

    names = np.array([NAME_MAPPING.get(b.name, b.name) for b in boxes])
    valid_flag = np.array(
        [(a["num_lidar_pts"] + a["num_radar_pts"]) > 0 for a in annotations],
        dtype=bool).reshape(-1)
    # mmdet3d convention: nuScenes wlh, and yaw mapped into the LiDAR box
    # frame (-yaw - pi/2). Verified against the released nuScenes infos pkls
    # (verify_infos.py --reference-pkl): exact match on all shared tokens.
    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
    return {
        "gt_boxes": gt_boxes,
        "gt_names": names,
        "gt_velocity": velocity,
        "num_lidar_pts": np.array([a["num_lidar_pts"] for a in annotations]),
        "num_radar_pts": np.array([a["num_radar_pts"] for a in annotations]),
        "valid_flag": valid_flag,
    }


def bevdet_ann_infos(nusc, sample, info):
    """BEVDet-style (gt_boxes, gt_labels) in the CAM_FRONT ego frame at image
    time — 9-dim boxes [xyz, dxdydz, yaw, vx, vy]. Mirrors
    create_data_bevdet.get_gt()."""
    from nuscenes.utils.data_classes import Box

    ego2global_rotation = info["cams"]["CAM_FRONT"]["ego2global_rotation"]
    ego2global_translation = \
        info["cams"]["CAM_FRONT"]["ego2global_translation"]
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse

    gt_boxes, gt_labels = [], []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        detection_name = GENERAL_TO_DETECTION[ann["category_name"]]
        if (detection_name not in DETECTION_CLASSES
                or ann["num_lidar_pts"] + ann["num_radar_pts"] <= 0):
            continue
        velocity = nusc.box_velocity(ann["token"])
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)
        box = Box(ann["translation"], ann["size"],
                  Quaternion(ann["rotation"]), velocity=velocity)
        box.translate(trans)
        box.rotate(rot)
        gt_boxes.append(np.concatenate([
            np.array(box.center),
            np.array(box.wlh)[[1, 0, 2]],
            np.array([box.orientation.yaw_pitch_roll[0]]),
            np.array(box.velocity[:2]),
        ]))
        gt_labels.append(DETECTION_CLASSES.index(detection_name))
    return gt_boxes, gt_labels


def fill_infos(nusc, scene_tokens, test, occ_path_prefix, relative_to):
    infos = []
    for sample in nusc.sample:
        if sample["scene_token"] not in scene_tokens:
            continue
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_record = nusc.get("calibrated_sensor",
                             sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        scene = nusc.get("scene", sample["scene_token"])

        panoptic_path = panoptic_file(nusc, lidar_token, relative_to)

        info = {
            "lidar_path": make_relative(
                nusc.get_sample_data_path(lidar_token), relative_to),
            "token": sample["token"],
            "sweeps": [],
            "cams": {},
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            "scene_name": scene["name"],
            "scene_token": sample["scene_token"],
            "panoptic_path": panoptic_path,
            "lidar_idx": lidar_token,
            "occ_path": f"{occ_path_prefix}/{scene['name']}/{sample['token']}",
        }
        for cam in CAMERA_TYPES:
            cam_token = sample["data"][cam]
            cam_info = sensor_info(nusc, cam_token, cam, relative_to)
            cs = nusc.get("calibrated_sensor",
                          nusc.get("sample_data",
                                   cam_token)["calibrated_sensor_token"])
            cam_info["cam_intrinsic"] = np.array(cs["camera_intrinsic"])
            info["cams"][cam] = cam_info

        if not test:
            info.update(boxes_in_lidar_frame(nusc, sample))
            info["ann_infos"] = bevdet_ann_infos(nusc, sample, info)
        infos.append(info)

    infos.sort(key=lambda i: (i["scene_name"], i["timestamp"]))
    return infos


def main():
    args = parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    tag = args.tag or ("bevdetv2-nuscenes-mini"
                       if args.version == "v1.0-mini"
                       else "bevdetv2-nuscenes")
    nusc = NuScenes(version=args.version, dataroot=str(args.root_path),
                    verbose=True)

    if args.version == "v1.0-trainval":
        split_names = {"train": splits.train, "val": splits.val}
    elif args.version == "v1.0-mini":
        split_names = {"train": splits.mini_train, "val": splits.mini_val}
    else:
        split_names = {"test": splits.test}

    name_to_token = {s["name"]: s["token"] for s in nusc.scene}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"version": args.version}

    for split, scene_names in split_names.items():
        scene_tokens = {name_to_token[n] for n in scene_names
                        if n in name_to_token}
        print(f"[{split}] {len(scene_tokens)} scenes")
        infos = fill_infos(nusc, scene_tokens, test=(split == "test"),
                           occ_path_prefix=args.occ_path_prefix,
                           relative_to=args.relative_to)
        out_file = args.out_dir / f"{tag}_infos_{split}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump({"infos": infos, "metadata": metadata}, f)
        print(f"[{split}] {len(infos)} samples -> {out_file}")


if __name__ == "__main__":
    main()
