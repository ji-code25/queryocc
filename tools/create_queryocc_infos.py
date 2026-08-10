"""Precompute the QueryOcc dataset cache (CPU only, no GPU needed).

The nuScenes dataset builds a derived index — lidar frame pool, camera frame lists with
their sample tokens, per-scene calibration — from the nuscenes-devkit database plus the
split definition in the `generate_pseudo_labels` index pkls. Training does this lazily on
first construction, which means every rank of a GPU job spends several minutes walking the
devkit before the first step (and re-does it whenever the cache schema changes).

This script performs exactly that step ahead of time and writes the same
`queryocc_infos_<hash>.pkl` files the dataset would have written, so a training job just
loads them. It needs no GPU and no pseudo labels — only the nuScenes database and the
index pkls.

Usage (from the repo root):

    python -m tools.create_queryocc_infos --version full
    python -m tools.create_queryocc_infos --version mini --splits train

The hash covers the settings the cache depends on (version, split, cameras,
only_tokens_for_offsets), so pass the same values the training preset uses — the defaults
match the released presets.
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):  # allow `python tools/create_queryocc_infos.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from queryocc.common import paths  # noqa: E402
from queryocc.data.datasets.nuscenes_infos import NUSCENES_CAMERAS, load_or_build_infos  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version", choices=["full", "mini"], default="full", help="nuScenes version to index (default: full)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="which splits to build (default: both)",
    )
    parser.add_argument(
        "--data", type=Path, default=None, help=f"nuScenes root (default: {paths.NUSC_ROOT}, i.e. $NUSC_ROOT)"
    )
    parser.add_argument(
        "--infos-dir",
        type=Path,
        default=None,
        help=f"where the index pkls live and the cache is written (default: {paths.INFOS_DIR})",
    )
    parser.add_argument(
        "--infos-tag",
        default="",
        help="index pkl filename tag; empty picks bevdetv2-nuscenes[-mini] from --version",
    )
    parser.add_argument(
        "--all-lidar-frames",
        action="store_true",
        help="also index non-keyframe CAMERA frames (sets only_tokens_for_offsets=False; "
        "must match the training config). The lidar pool is always the full 20 Hz stream.",
    )
    parser.add_argument("--overwrite", action="store_true", help="rebuild even if the cache file already exists")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.infos_dir is not None:
        # The dataset resolves both the index pkls and the cache through this module-level
        # default, so redirecting it here is what makes --infos-dir take effect.
        paths.INFOS_DIR = args.infos_dir

    # A plain settings holder rather than the dataset config: this script must stay
    # importable without torch so it can run in a small CPU environment.
    config = SimpleNamespace(
        version=args.version,
        infos_tag=args.infos_tag,
        only_tokens_for_offsets=not args.all_lidar_frames,
        nuscenes_cameras_to_load=NUSCENES_CAMERAS,
        data=args.data if args.data is not None else paths.NUSC_ROOT,
        ids_to_load=[],
        load_queryocc_infos_if_exists=not args.overwrite,
        save_queryocc_infos=True,
    )

    print(f"nuScenes root : {config.data}")
    print(f"index pkls    : {paths.INFOS_DIR}")
    print(f"cache output  : {paths.INFOS_DIR}")
    print("lidar frames  : full 20 Hz stream (always)")
    print(f"camera frames : {'keyframes only' if config.only_tokens_for_offsets else 'full rate'}")

    for split in args.splits:
        print(f"\n=== {split} ===", flush=True)
        started = time.time()
        version = "v1.0-mini" if config.version == "mini" else "v1.0-trainval"
        infos = load_or_build_infos(config, split, "original", version)

        scenes = len(infos["sweeps_per_scene"])
        frames = sum(len(v) for v in infos["sweeps_per_scene"].values())
        cameras = sum(len(cams) for scene in infos["cameras_per_scene"].values() for cams in scene.values())
        keyframes = sum(
            1 for scene in infos["camera_tokens_per_scene"].values() for cams in scene.values() for t in cams if t
        )
        print(
            f"{split}: {scenes} scenes, {frames} lidar frames, {cameras} camera frames "
            f"({keyframes} with pseudo-label tokens) in {time.time() - started:.0f}s"
        )

    print("\nDone. Training jobs will now load these instead of rebuilding them.")


if __name__ == "__main__":
    main()
