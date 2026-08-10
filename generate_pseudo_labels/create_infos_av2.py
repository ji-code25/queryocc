"""Create infos pkl files for the Argoverse 2 sensor dataset.

Same on-disk schema as ``create_infos.py`` (nuScenes), so
``generate_semantics.py`` / ``generate_depth.py`` and ``tools/lidar_pseudoseg``
consume it unchanged. Requires only the ``av2`` devkit, pandas and numpy — no
nuScenes devkit, no mmcv/mmdet3d.

One sample per **lidar sweep** (AV2 has no keyframe concept and no sample
tokens): ``token`` is the lidar timestamp stem, ``scene_name`` is the log id.
Each camera is matched to the sweep by nearest timestamp (cameras run at 20 Hz,
the lidar at 10 Hz, and the two are not synchronised), and every camera carries
the ego pose interpolated at *its own* capture time.

Fields per sample::

    token, scene_token, scene_name, timestamp, lidar_idx, lidar_path
    lidar2ego_{translation,rotation}     # up_lidar sensor pose, quaternion [w,x,y,z]
    ego2global_{translation,rotation}    # ego pose at the lidar timestamp
    cams: {<ring_camera>: data_path, cam_intrinsic [3,3], sensor2ego_{translation,
           rotation}, ego2global_{translation,rotation}, timestamp, image_size,
           type, sample_data_token}

The nuScenes-only fields are deliberately **absent**: ``occ_path``,
``panoptic_path``, ``sweeps``, ``gt_boxes``, ``gt_names``, ``gt_velocity``,
``num_lidar_pts``, ``num_radar_pts``, ``valid_flag``, ``ann_infos``. AV2 ships
no Occ3D ground truth and this index is only used to *generate* pseudo labels,
so nothing downstream of it reads those.

CAUTION — two AV2 specifics that differ from nuScenes:

* **Lidar sweeps are stored in the ego frame already.** ``lidar2ego_*`` is
  recorded for completeness (it is the ``up_lidar`` calibration) but must NOT be
  applied to the points read from ``lidar_path``; ``tools/lidar_pseudoseg``
  treats AV2 points as ego-frame.
* **Mixed camera resolutions.** ``ring_front_center`` is portrait (1550x2048),
  the other six rings are landscape (2048x1550). Both generators batch all
  cameras of a sample into one forward pass, so one index must cover only
  cameras of the same resolution. Emit one index per resolution group with
  ``--cameras`` (+ a distinct ``--tag``) and run the generators once per group
  into separate ``--save-path`` roots. See the README.

Example (landscape group of the val split)::

    python -m generate_pseudo_labels.create_infos_av2 \
        --root-path /datasets/argoverse2 --split val --out-dir data/infos \
        --relative-to /datasets/argoverse2 \
        --cameras ring_front_left ring_front_right ring_side_left \
                  ring_side_right ring_rear_left ring_rear_right \
        --tag av2-landscape
"""

import argparse
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.geometry import mat_to_quat, quat_to_mat
from av2.geometry.interpolate import interpolate_pose
from av2.geometry.se3 import SE3
from tqdm import tqdm

# The seven ring cameras (av2.datasets.sensor.constants.RingCameras), grouped by
# sensor resolution. ring_front_center is mounted portrait.
PORTRAIT_CAMERAS = ("ring_front_center",)
LANDSCAPE_CAMERAS = (
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)
RING_CAMERAS = PORTRAIT_CAMERAS + LANDSCAPE_CAMERAS

# The lidar the sweeps are timestamped and calibrated against.
LIDAR_SENSOR = "up_lidar"

# Default tolerance for lidar <-> camera timestamp matching (AV2 cameras run at
# 20 Hz, so a correctly recorded log matches within ~25 ms).
DEFAULT_MAX_CAM_OFFSET_NS = 50_000_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("data/argoverse2"),
        help="AV2 root. Both layouts are accepted: <root>/<split>/<log> and <root>/sensor/<split>/<log>.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--tag", default="av2", help="filename prefix: <tag>_infos_<split>.pkl")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(RING_CAMERAS),
        help="cameras to index (default: all seven rings). AV2 mixes resolutions, and both generators batch "
        "all cameras of a sample together, so emit one index per resolution group: "
        f"'--cameras {' '.join(LANDSCAPE_CAMERAS)}' and '--cameras {' '.join(PORTRAIT_CAMERAS)}'.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="if set, store data paths relative to this directory (recommended: the same directory the "
        "training/generation side passes as --data-root). Default: absolute paths.",
    )
    parser.add_argument(
        "--max-cam-offset-ms",
        type=float,
        default=DEFAULT_MAX_CAM_OFFSET_NS / 1e6,
        help="drop a lidar sweep when any camera's nearest frame is further away than this",
    )
    parser.add_argument("--logs", nargs="+", default=None, help="only these log ids")
    parser.add_argument("--max-logs", type=int, default=None, help="stop after this many logs (smoke tests)")
    parser.add_argument("--num-workers", type=int, default=8, help="threads used to walk the logs")
    return parser.parse_args()


def make_relative(path, relative_to):
    """Same convention as create_infos.py: relative to --relative-to when below it, else unchanged."""
    path = str(path)
    base = str(relative_to) if relative_to is not None else os.getcwd()
    if path.startswith(base + os.sep):
        return os.path.relpath(path, base)
    return path


def find_split_dir(root_path: Path, split: str) -> Path:
    """Locate the directory holding the logs of `split`.

    Handles both layouts in the wild: the released tarballs extract to
    ``<root>/<split>/<log>``, while the AV2 devkit's SensorDataloader (and the
    queryocc AV2Dataset, via ``subset``) expects ``<root>/sensor/<split>/<log>``.
    """
    candidates = [root_path / split, root_path / "sensor" / split]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No {split} split under {root_path} (tried: {', '.join(str(c) for c in candidates)})")


class EgoPoses:
    """``city_SE3_egovehicle`` of one log, queryable at arbitrary timestamps.

    The pose stream is logged at ~150 Hz, well above both sensor rates, but the
    camera timestamps never coincide with it exactly, so queries are slerp +
    linear interpolated between the two bracketing entries (clamped at the ends).
    """

    def __init__(self, log_dir: Path):
        frame = pd.read_feather(log_dir / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
        self.timestamps = frame["timestamp_ns"].to_numpy().astype(np.int64)
        self.translations = frame[["tx_m", "ty_m", "tz_m"]].to_numpy().astype(np.float64)
        self.rotations = quat_to_mat(frame[["qw", "qx", "qy", "qz"]].to_numpy().astype(np.float64))

    def _se3(self, index: int) -> SE3:
        return SE3(rotation=self.rotations[index].copy(), translation=self.translations[index].copy())

    def at(self, timestamp_ns: int):
        """(translation [3], rotation quaternion [w,x,y,z]) of the ego vehicle in the city frame."""
        index = int(np.searchsorted(self.timestamps, timestamp_ns))
        if index <= 0:
            pose = self._se3(0)
        elif index >= len(self.timestamps):
            pose = self._se3(len(self.timestamps) - 1)
        else:
            pose = interpolate_pose(
                key_timestamps=(int(self.timestamps[index - 1]), int(self.timestamps[index])),
                key_poses=(self._se3(index - 1), self._se3(index)),
                query_timestamp=int(timestamp_ns),
            )
        return pose.translation.tolist(), mat_to_quat(pose.rotation).tolist()


def lidar2ego(log_dir: Path):
    """(translation [3], rotation quaternion [w,x,y,z]) of the up lidar in the ego frame."""
    frame = pd.read_feather(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    row = frame[frame["sensor_name"] == LIDAR_SENSOR]
    if row.empty:
        raise KeyError(f"{LIDAR_SENSOR} missing from {log_dir}/calibration/egovehicle_SE3_sensor.feather")
    row = row.iloc[0]
    return (
        [float(row["tx_m"]), float(row["ty_m"]), float(row["tz_m"])],
        [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
    )


def camera_calibration(log_dir: Path, cameras):
    """Static per-camera calibration: intrinsics, sensor->ego pose, image size."""
    calibration = {}
    for camera_name in cameras:
        camera = PinholeCamera.from_feather(log_dir=log_dir, cam_name=camera_name)
        ego_se3_cam = camera.ego_SE3_cam
        calibration[camera_name] = {
            "cam_intrinsic": camera.intrinsics.K.copy(),
            "sensor2ego_translation": ego_se3_cam.translation.tolist(),
            "sensor2ego_rotation": mat_to_quat(ego_se3_cam.rotation).tolist(),
            "image_size": [int(camera.intrinsics.width_px), int(camera.intrinsics.height_px)],
        }
    return calibration


def camera_frames(log_dir: Path, cameras):
    """Sorted (timestamps_ns, paths) of the jpgs of every requested camera."""
    frames = {}
    for camera_name in cameras:
        camera_dir = log_dir / "sensors" / "cameras" / camera_name
        paths = sorted(camera_dir.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No images for camera {camera_name} in {log_dir}")
        frames[camera_name] = (np.array([int(p.stem) for p in paths], dtype=np.int64), paths)
    return frames


def nearest_index(sorted_timestamps: np.ndarray, query: int) -> int:
    """Index of the entry closest to `query` in a sorted timestamp array."""
    index = int(np.searchsorted(sorted_timestamps, query))
    if index == 0:
        return 0
    if index >= len(sorted_timestamps):
        return len(sorted_timestamps) - 1
    before, after = sorted_timestamps[index - 1], sorted_timestamps[index]
    return index - 1 if (query - before) <= (after - query) else index


def log_infos(log_dir: Path, cameras, relative_to, max_cam_offset_ns: int):
    """Infos of one AV2 log, one entry per lidar sweep (unsorted)."""
    lidar_files = sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
    if not lidar_files:
        return [], 0

    poses = EgoPoses(log_dir)
    calibration = camera_calibration(log_dir, cameras)
    frames = camera_frames(log_dir, cameras)
    lidar_translation, lidar_rotation = lidar2ego(log_dir)

    infos = []
    for lidar_file in lidar_files:
        lidar_timestamp = int(lidar_file.stem)
        ego_translation, ego_rotation = poses.at(lidar_timestamp)

        cams = {}
        for camera_name in cameras:
            timestamps, paths = frames[camera_name]
            index = nearest_index(timestamps, lidar_timestamp)
            camera_timestamp = int(timestamps[index])
            if abs(camera_timestamp - lidar_timestamp) > max_cam_offset_ns:
                cams = None
                break
            cam_translation, cam_rotation = poses.at(camera_timestamp)
            cams[camera_name] = {
                **calibration[camera_name],
                "data_path": make_relative(paths[index], relative_to),
                "type": camera_name,
                # AV2 has no sample_data tokens; the image timestamp is the unique id.
                "sample_data_token": paths[index].stem,
                "ego2global_translation": cam_translation,
                "ego2global_rotation": cam_rotation,
                "timestamp": camera_timestamp,
            }
        if cams is None:  # a camera dropped out around this sweep
            continue

        infos.append(
            {
                "token": lidar_file.stem,
                "scene_name": log_dir.name,
                "scene_token": log_dir.name,
                "lidar_idx": lidar_file.stem,
                "lidar_path": make_relative(lidar_file, relative_to),
                # NOTE nanoseconds (nuScenes uses microseconds); only used for ordering.
                "timestamp": lidar_timestamp,
                "lidar2ego_translation": lidar_translation,
                "lidar2ego_rotation": lidar_rotation,
                "ego2global_translation": ego_translation,
                "ego2global_rotation": ego_rotation,
                "cams": cams,
            }
        )
    return infos, len(lidar_files)


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)  # pyarrow/av2 feather chatter

    unknown = [c for c in args.cameras if c not in RING_CAMERAS]
    if unknown:
        raise SystemExit(f"Unknown camera(s) {unknown}; choose from {list(RING_CAMERAS)}")
    if set(args.cameras) & set(PORTRAIT_CAMERAS) and set(args.cameras) & set(LANDSCAPE_CAMERAS):
        print(
            "WARNING: mixing portrait (ring_front_center, 1550x2048) and landscape (2048x1550) cameras in one "
            "index. generate_semantics/generate_depth batch all cameras of a sample together and reject such a "
            "sample; emit one index per resolution group (see --cameras). This index is still usable for "
            "tools/lidar_pseudoseg, which reads the cameras one at a time."
        )

    split_dir = find_split_dir(args.root_path, args.split)
    logs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if args.logs:
        wanted = set(args.logs)
        logs = [d for d in logs if d.name in wanted]
        missing = wanted - {d.name for d in logs}
        if missing:
            raise SystemExit(f"Log(s) not found under {split_dir}: {sorted(missing)}")
    if args.max_logs is not None:
        logs = logs[: args.max_logs]
    print(f"[{args.split}] {len(logs)} logs under {split_dir}")

    max_cam_offset_ns = int(args.max_cam_offset_ms * 1e6)

    failed = []

    def process(log_dir):
        # A log missing a camera (or with an unreadable calibration) must not kill a 700-log
        # run, but it must not vanish either: skip it loudly and list it in the summary.
        try:
            return log_infos(log_dir, args.cameras, args.relative_to, max_cam_offset_ns)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"WARNING: skipping log {log_dir.name}: {type(exc).__name__}: {exc}")
            failed.append(log_dir.name)
            return [], 0

    infos, num_sweeps = [], 0
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        for log_infos_, log_sweeps in tqdm(executor.map(process, logs), total=len(logs), desc=args.split):
            infos.extend(log_infos_)
            num_sweeps += log_sweeps

    # Same ordering contract as create_infos.py: index +/- t adjacency works within a scene.
    infos.sort(key=lambda info: (info["scene_name"], info["timestamp"]))

    metadata = {
        "version": f"av2-{args.split}",
        "dataset": "argoverse2",
        "split": args.split,
        "cameras": list(args.cameras),
        "root_path": str(args.root_path),
        "split_dir": str(split_dir),
        "max_cam_offset_ns": max_cam_offset_ns,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"{args.tag}_infos_{args.split}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({"infos": infos, "metadata": metadata}, f)
    dropped = num_sweeps - len(infos)
    print(f"[{args.split}] {len(infos)} samples ({dropped} sweeps dropped, no camera match) -> {out_file}")
    if failed:
        print(f"[{args.split}] {len(failed)} log(s) skipped entirely: {sorted(failed)}")


if __name__ == "__main__":
    main()
