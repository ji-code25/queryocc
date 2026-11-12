"""Build the derived nuScenes index ("queryocc infos") consumed by NuScenesDataset.

Deliberately free of torch and every other heavy training dependency: only numpy,
pyquaternion and nuscenes-devkit. That lets the index be precomputed on a CPU node in a
small environment (see tools/create_queryocc_infos.py) while the dataset imports the very
same builder, so the two can never drift.

Inputs are the nuScenes database plus the split definition written by
`generate_pseudo_labels.create_infos`. Note that those index pkls carry one entry per
keyframe with an empty ``sweeps`` list and dataset-relative paths, so the lidar frame pool
is walked from the devkit instead (see `collect_lidar_frames`).
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pyquaternion import Quaternion

from queryocc.common import paths

# Bumped whenever the cached structure changes, so stale caches are rebuilt rather than
# silently served. v3: the lidar pool is the full 20 Hz stream again — v2 wrongly applied
# only_tokens_for_offsets (a camera-only setting) to it, cutting the pool ~10x.
CACHE_SCHEMA_VERSION = "devkit_lidar_frames_v4"

# The nuScenes camera rig, in the canonical order used everywhere.
NUSCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)


def index_tag(config) -> str:
    """Filename tag of the generate_pseudo_labels index pkls for this config."""
    if config.infos_tag:
        return config.infos_tag
    return "bevdetv2-nuscenes-mini" if config.version == "mini" else "bevdetv2-nuscenes"


def cache_path(config, train_or_val: str, split: str) -> Path:
    """Where the derived index is cached, hashed over everything it depends on."""
    values = [
        train_or_val,
        config.version,
        split,
        index_tag(config),
        list(config.nuscenes_cameras_to_load),
        config.only_tokens_for_offsets,
        CACHE_SCHEMA_VERSION,
    ]
    values_hash = hashlib.md5(json.dumps(values).encode()).hexdigest()
    return paths.INFOS_DIR / f"queryocc_infos_{values_hash}.pkl"


def collect_lidar_frames(nusc, scene_token: str, config) -> List[Dict[str, Any]]:
    """Every LIDAR_TOP frame of a scene, oldest first, as sweep records.

    Walks the sample_data `next` chain from the scene's first keyframe, so keyframes
    (samples/) *and* the intermediate 20 Hz sweeps (sweeps/) are included.

    The pool is deliberately NOT filtered by ``only_tokens_for_offsets``: that setting
    applies to the camera pool only (see build_infos below), matching the original
    pipeline, where this list came from the index pkl's 20 Hz ``sweeps`` entries. Two
    things depend on the full stream: the offset-frame sampler draws its +/-3 s temporal
    window from it, and the scene-length check in NuScenesDataset.__init__ subtracts a
    buffer computed as seconds x 20 Hz (120 frames for the released presets), which a
    ~40-frame keyframe-only pool cannot satisfy.
    """
    del config  # kept for signature stability; the lidar pool is never subsampled here
    first_sample = nusc.get("sample", nusc.get("scene", scene_token)["first_sample_token"])
    sample_data = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])

    frames = []
    while True:
        calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        frames.append(
            {
                "data_path": nusc.get_sample_data_path(sample_data["token"]),
                "type": "lidar",
                "sample_data_token": sample_data["token"],
                "sensor2ego_translation": calib["translation"],
                "sensor2ego_rotation": calib["rotation"],
                "ego2global_translation": ego_pose["translation"],
                "ego2global_rotation": ego_pose["rotation"],
                "timestamp": sample_data["timestamp"],
                "sample_token": sample_data["sample_token"],
            }
        )
        if sample_data["next"] == "":
            break
        sample_data = nusc.get("sample_data", sample_data["next"])

    return sorted(frames, key=lambda f: f["timestamp"])


def build_infos(config, train_or_val: str, split: str, version: str) -> Dict[str, Any]:
    """Walk the devkit + split definition and return the derived index."""
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(config.data), verbose=False)
    if config.version == "mini":
        mini_scene_tokens = [scene["token"] for scene in nusc.scene]
        if len(config.ids_to_load) == 0:
            config.ids_to_load = mini_scene_tokens
        else:
            config.ids_to_load = [i for i in config.ids_to_load if i in mini_scene_tokens]

    index_path = paths.INFOS_DIR / f"{index_tag(config)}_infos_{train_or_val}.pkl"
    with open(index_path, "rb") as f:
        nusc_infos = pickle.load(f)

    scenes = list({x["scene_token"] for x in nusc_infos["infos"]})
    if len(config.ids_to_load) > 0:
        scenes = [scene for scene in scenes if scene in config.ids_to_load]
    scenes.sort()

    infos_per_scene = {scene: [] for scene in scenes}
    for info in nusc_infos["infos"]:
        if info["scene_token"] in scenes:
            infos_per_scene[info["scene_token"]].append(info)
    for scene, infos in infos_per_scene.items():
        infos_per_scene[scene] = sorted(infos, key=lambda x: x["timestamp"])

    sample_data_token_to_sample_token = {}
    sweeps_per_scene = {}
    scene_token_to_real_scene_name = {}
    for scene, infos in infos_per_scene.items():
        scene_token_to_real_scene_name[scene] = nusc.get("scene", scene)["name"]
        for info in infos:
            lidar_sample_data_token = nusc.get("sample", info["token"])["data"]["LIDAR_TOP"]
            sample_data_token_to_sample_token[lidar_sample_data_token] = info["token"]
        sweeps_per_scene[scene] = collect_lidar_frames(nusc, scene, config)

    scene_to_location = {}
    cameras = config.nuscenes_cameras_to_load
    cameras_per_scene = {scene: {cam: [] for cam in cameras} for scene in sweeps_per_scene}
    camera_timestamps_per_scene = {scene: {cam: [] for cam in cameras} for scene in sweeps_per_scene}
    # Sample token per camera frame, used to locate that frame's pseudo-label npz
    # (<root>/<scene>/<token>.npz). Sweep frames belong to a sample but have no pseudo
    # labels of their own, so only keyframes get a token; the rest stay None.
    camera_tokens_per_scene = {scene: {cam: [] for cam in cameras} for scene in sweeps_per_scene}
    camera_calib_per_scene = {scene: {cam: {} for cam in cameras} for scene in sweeps_per_scene}

    for scene in sweeps_per_scene:
        scene_record = nusc.get("scene", scene)
        sample = nusc.get("sample", scene_record["first_sample_token"])["data"]
        scene_to_location[scene] = nusc.get("log", scene_record["log_token"])["location"]

        first_timestamp = sweeps_per_scene[scene][0]["timestamp"]
        last_timestamp = sweeps_per_scene[scene][-1]["timestamp"]

        for cam in cameras:
            sample_data = nusc.get("sample_data", sample[cam])
            calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = Quaternion(calib["rotation"]).rotation_matrix
            extrinsics[:3, 3] = calib["translation"]
            camera_calib_per_scene[scene][cam] = {
                "extrinsics": extrinsics,
                "intrinsics": calib["camera_intrinsic"],
            }
            while sample_data["next"] != "":
                timestamp = sample_data["timestamp"]
                keep = first_timestamp <= timestamp <= last_timestamp and not (
                    config.only_tokens_for_offsets and "sweeps" in sample_data["filename"]
                )
                if keep:
                    cameras_per_scene[scene][cam].append(sample_data["filename"])
                    camera_timestamps_per_scene[scene][cam].append(timestamp)
                    camera_tokens_per_scene[scene][cam].append(
                        sample_data["sample_token"] if sample_data["is_key_frame"] else None
                    )
                sample_data = nusc.get("sample_data", sample_data["next"])

    return {
        "sweeps_per_scene": sweeps_per_scene,
        "sample_data_token_to_sample_token": sample_data_token_to_sample_token,
        "scene_token_to_real_scene_name": scene_token_to_real_scene_name,
        "scene_to_location": scene_to_location,
        "cameras_per_scene": cameras_per_scene,
        "camera_timestamps_per_scene": camera_timestamps_per_scene,
        "camera_tokens_per_scene": camera_tokens_per_scene,
        "camera_calib_per_scene": camera_calib_per_scene,
    }


def load_or_build_infos(config, train_or_val: str, split: str, version: str) -> Dict[str, Any]:
    """Return the derived index, building and caching it when it is missing."""
    path = cache_path(config, train_or_val, split)

    if config.load_queryocc_infos_if_exists and path.exists():
        print(f"[{train_or_val}:{split}] Loading queryocc infos from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"[{train_or_val}:{split}] Creating queryocc infos")
    infos = build_infos(config, train_or_val, split, version)
    if config.save_queryocc_infos:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(infos, f)
        print(f"[{train_or_val}:{split}] Saved queryocc infos to {path}")
    return infos
