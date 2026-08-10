"""Shared helpers: infos loading, sharding, path resolution, atomic saves."""

import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_infos(infos_pkl):
    """Load a bevdetv2-style infos pkl -> (list of sample dicts, metadata)."""
    with open(infos_pkl, "rb") as f:
        data = pickle.load(f)
    return data["infos"], data.get("metadata", {})


def shard_bounds(num_items, shard, num_shards):
    """Contiguous [start, end) slice for `shard` of `num_shards`."""
    if num_shards <= 1:
        return 0, num_items
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} out of range [0, {num_shards})")
    start = shard * num_items // num_shards
    end = (shard + 1) * num_items // num_shards
    return start, end


def select_samples(infos, start=None, end=None, shard=None, num_shards=None,
                   scene_prefixes=None):
    """Apply --start/--end, then --shard/--num-shards, then scene filtering."""
    infos = infos[start:end]
    if num_shards is not None and num_shards > 1:
        lo, hi = shard_bounds(len(infos), shard or 0, num_shards)
        infos = infos[lo:hi]
    if scene_prefixes:
        infos = [
            info for info in infos
            if any(info["scene_name"].startswith(p) for p in scene_prefixes)
        ]
    return infos


def resolve_path(path, data_root=None):
    """Resolve an image path from an infos pkl.

    Infos store paths as written at creation time (often relative, e.g.
    './data/nuscenes/samples/...'). If the path does not exist as-is and
    `data_root` is given, retry relative to `data_root`.
    """
    p = Path(path)
    if p.exists() or data_root is None:
        return p
    stripped = str(path)
    while stripped.startswith("./"):
        stripped = stripped[2:]
    candidate = Path(data_root) / stripped
    if candidate.exists():
        return candidate
    # Last resort: strip a leading 'data/' component, common when the pkl
    # was created from a repo root and data_root already points at 'data'.
    parts = Path(stripped).parts
    if parts and parts[0] == "data":
        candidate = Path(data_root).joinpath(*parts[1:])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve '{path}' (data_root={data_root}). "
        "Pass --data-root pointing at the directory the pkl paths are "
        "relative to.")


def atomic_save_npz(path, compress=False, **arrays):
    """Write an .npz atomically (tmp file + rename) so that job-array
    restarts never leave truncated files behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        if compress:
            np.savez_compressed(f, **arrays)
        else:
            np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def write_meta(save_path, filename, payload):
    """Write provenance metadata next to the generated labels. Adds the
    command line, timestamp and package version automatically."""
    try:
        from . import __version__
    except ImportError:  # run as a plain file, not a package
        __version__ = "unknown"
    payload = dict(payload)
    payload.setdefault("command", " ".join(sys.argv))
    payload.setdefault("created_utc",
                       datetime.now(timezone.utc).isoformat(timespec="seconds"))
    payload.setdefault("generate_pseudo_labels_version", __version__)
    # Normalize through JSON (int keys -> str, Paths -> str) so that the
    # rerun comparison below compares like with like.
    payload = json.loads(json.dumps(payload, default=str))
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    meta_path = save_path / filename
    # Never silently change the contract of an existing dataset: if a meta
    # file already exists with different settings, refuse.
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        keys = [k for k in payload
                if k not in ("command", "created_utc")]
        diffs = {
            k: (existing.get(k), payload[k])
            for k in keys if existing.get(k) != payload[k]
        }
        if diffs:
            raise RuntimeError(
                f"{meta_path} exists with different settings: {diffs}. "
                "Use a fresh --save-path (or delete the old dataset) instead "
                "of mixing generation settings in one directory.")
        return
    meta_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def install_mmcv_shim():
    """Make Metric3D's torch.hub code importable without mmcv.

    `mono/utils/comm.py` does an unguarded `from mmcv.utils import
    collect_env`, and a few model files do `from mmcv.utils import Config`.
    mmcv is a heavy, CUDA-compiled dependency this package deliberately keeps
    out (the rest of Metric3D only needs mmengine, which ships the very same
    objects — hubconf.py itself already falls back to it). So when mmcv is
    genuinely absent, register a minimal stand-in backed by mmengine.

    No-op when a real mmcv is installed.
    """
    import importlib.util
    import types

    try:
        if importlib.util.find_spec("mmcv") is not None:
            return
    except (ImportError, ValueError):
        pass

    from mmengine import Config, DictAction
    from mmengine.utils import get_git_hash
    from mmengine.utils.dl_utils import collect_env

    utils = types.ModuleType("mmcv.utils")
    utils.Config = Config
    utils.DictAction = DictAction
    utils.collect_env = collect_env
    utils.get_git_hash = get_git_hash

    mmcv = types.ModuleType("mmcv")
    mmcv.__version__ = "0.0.0+mmengine-shim"
    mmcv.utils = utils

    sys.modules.setdefault("mmcv", mmcv)
    sys.modules.setdefault("mmcv.utils", utils)


def add_common_args(parser):
    """Arguments shared by both generators."""
    parser.add_argument("--infos-pkl", required=True, type=Path,
                        help="bevdetv2-style infos pkl (see create_infos.py)")
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="root to resolve relative image paths against")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="downscale factor applied to the saved labels "
                             "(1.0 = native camera resolution)")
    parser.add_argument("--layout", choices=["npz", "legacy"], default="npz",
                        help="'npz': self-describing per-sample .npz in the "
                             "canonical camera order. 'legacy': plain .npy "
                             "stacks bit-compatible with the GaussianFlowOcc "
                             "trainer (mind: legacy mask and depth camera "
                             "orders differ!)")
    parser.add_argument("--compress", action="store_true",
                        help="use compressed npz (slower load, smaller disk)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="camera keys to process, in the order they are "
                             "stored in the output (default: the nuScenes ring, "
                             "constants.CAMERA_ORDER). Needed for datasets with "
                             "other camera names, and to process ONE RESOLUTION "
                             "GROUP at a time: both generators batch all cameras "
                             "of a sample into a single forward pass, so their "
                             "images must share one resolution (e.g. AV2's "
                             "portrait ring_front_center vs. the six landscape "
                             "rings). Camera groups need separate --save-path "
                             "roots — they would otherwise overwrite each "
                             "other's per-sample npz.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0,
                        help="shard index in [0, num-shards)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--scene-prefixes", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser
