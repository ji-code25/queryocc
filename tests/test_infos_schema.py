"""The dataset index must be built from the generate_pseudo_labels pkl schema.

Two properties of that schema differ from the legacy bevdetv2 pkls the code was
originally written against, and both fail silently rather than loudly:

* `sweeps` is empty — every sample carries only its keyframe, so a builder that
  harvested non-keyframe lidar frames from it would quietly drop the 20 Hz stream
  the offset-frame sampler needs.
* paths are stored relative to the dataset root, so any absolute-path surgery on
  them breaks.

These tests build the cache from a synthetic devkit + index pkl, which keeps them
runnable without the real nuScenes download.
"""

import pickle
from types import SimpleNamespace

import pytest

from queryocc.data.datasets.nuscenes_infos import collect_lidar_frames

SCENE_TOKEN = "scene0"
N_KEYFRAMES = 3
SWEEPS_PER_KEYFRAME = 2  # non-keyframe lidar frames following each keyframe


class FakeNuScenes:
    """Minimal stand-in for the devkit surface the cache builder touches."""

    def __init__(self):
        self.tables = {"scene": {}, "sample": {}, "sample_data": {}, "calibrated_sensor": {}, "ego_pose": {}, "log": {}}
        self.tables["log"]["log0"] = {"location": "singapore-onenorth"}
        self.tables["calibrated_sensor"]["calib0"] = {
            "translation": [0.0, 0.0, 1.8],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]],
        }

        frames = []  # (token, is_key_frame, filename)
        timestamp = 1_000_000
        for k in range(N_KEYFRAMES):
            frames.append((f"lidar_kf{k}", True, f"samples/LIDAR_TOP/kf{k}.pcd.bin"))
            for s in range(SWEEPS_PER_KEYFRAME):
                frames.append((f"lidar_sw{k}_{s}", False, f"sweeps/LIDAR_TOP/kf{k}_{s}.pcd.bin"))

        for i, (token, is_key, filename) in enumerate(frames):
            self.tables["ego_pose"][f"pose_{token}"] = {
                "translation": [float(i), 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
            self.tables["sample_data"][token] = {
                "token": token,
                "is_key_frame": is_key,
                "filename": filename,
                "timestamp": timestamp + i * 50_000,
                "sample_token": f"sample{i // (SWEEPS_PER_KEYFRAME + 1)}",
                "calibrated_sensor_token": "calib0",
                "ego_pose_token": f"pose_{token}",
                "next": frames[i + 1][0] if i + 1 < len(frames) else "",
            }

        for k in range(N_KEYFRAMES):
            self.tables["sample"][f"sample{k}"] = {"token": f"sample{k}", "data": {"LIDAR_TOP": f"lidar_kf{k}"}}
        self.tables["scene"][SCENE_TOKEN] = {
            "token": SCENE_TOKEN,
            "name": "scene-0001",
            "first_sample_token": "sample0",
            "log_token": "log0",
        }

    def get(self, table, token):
        return self.tables[table][token]

    def get_sample_data_path(self, token):
        # The devkit returns absolute paths; the index pkl stores relative ones.
        return f"/datasets/nuscenes/{self.tables['sample_data'][token]['filename']}"


def collect(only_tokens_for_offsets: bool):
    config = SimpleNamespace(only_tokens_for_offsets=only_tokens_for_offsets)
    return collect_lidar_frames(FakeNuScenes(), SCENE_TOKEN, config)


def test_full_stream_includes_non_keyframe_sweeps():
    """The 20 Hz pool must come from the devkit, not from the index pkl's empty `sweeps`."""
    frames = collect(only_tokens_for_offsets=False)
    assert len(frames) == N_KEYFRAMES * (1 + SWEEPS_PER_KEYFRAME)
    assert sum("samples/" in f["data_path"] for f in frames) == N_KEYFRAMES
    assert sum("sweeps/" in f["data_path"] for f in frames) == N_KEYFRAMES * SWEEPS_PER_KEYFRAME


def test_lidar_pool_ignores_only_tokens_for_offsets():
    """only_tokens_for_offsets is a CAMERA setting. Applying it to the lidar pool cuts it
    ~10x, which silently shrinks the offset-frame window and makes the scene-length check
    in NuScenesDataset.__init__ (buffer = seconds x 20 Hz) unsatisfiable."""
    assert collect(only_tokens_for_offsets=True) == collect(only_tokens_for_offsets=False)


def test_frames_are_time_ordered_and_complete():
    frames = collect(only_tokens_for_offsets=False)
    required = {
        "data_path",
        "sample_data_token",
        "timestamp",
        "sensor2ego_translation",
        "sensor2ego_rotation",
        "ego2global_translation",
        "ego2global_rotation",
        "sample_token",
    }
    for frame in frames:
        assert required <= set(frame), required - set(frame)
    assert [f["timestamp"] for f in frames] == sorted(f["timestamp"] for f in frames)


def test_data_paths_are_absolute():
    """Downstream loading opens these directly, so the relative paths in the index pkl
    must not leak through."""
    for frame in collect(only_tokens_for_offsets=False):
        assert frame["data_path"].startswith("/")


@pytest.mark.parametrize("field", ["sweeps", "lidar_path"])
def test_index_pkl_expectations_are_documented(field, tmp_path):
    """Pin the two schema facts this builder is designed around."""
    sample = {"lidar_path": "samples/LIDAR_TOP/x.pcd.bin", "sweeps": [], "token": "t", "scene_token": SCENE_TOKEN}
    path = tmp_path / "index.pkl"
    path.write_bytes(pickle.dumps({"infos": [sample]}))
    loaded = pickle.loads(path.read_bytes())["infos"][0]

    if field == "sweeps":
        assert loaded["sweeps"] == [], "create_infos writes an empty sweeps list"
    else:
        assert not loaded["lidar_path"].startswith("/"), "create_infos stores paths relative to the dataset root"
