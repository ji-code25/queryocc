"""Shared access to the pseudo-label npz layout written by ``generate_pseudo_labels``.

One file per sample, per stream:

    <root>/<scene_name>/<token>.npz
        <array>    [6, H, W]   the stream's payload, stored in `cam_order`
        cam_order  <U18 [6]    the camera order the arrays are stored in

Files are opened lazily and the per-camera slice is taken from the embedded
`cam_order`, so a change of camera ordering upstream cannot silently transpose
the data (the original pipeline used two different orders for masks and depth,
which this layout deliberately removes).

Reading strategy: the zip layout of each (file, member) — data offset, plane size,
dtype, `cam_order` — is resolved once per process and cached, after which reading a
camera plane is a single ``open`` + one exact-offset ``read``. The obvious
alternatives are both dataloader killers at scale: ``np.load(...)[name]``
materialises all six planes (6x the bytes), and seeking inside ``zipfile``'s
``ZipExtFile`` reads and discards every byte it skips, so even a "seek to the
plane" implementation re-reads half the file on average. Missing files (sweep
frames have no pseudo labels) are cached as missing, so the per-epoch stat storm
on the shared filesystem is paid once per worker instead of once per access.
"""

import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from generate_pseudo_labels.constants import CAMERA_ORDER

# Native image sizes, used to shape the placeholder returned for frames without labels.
DEFAULT_SHAPES = {
    "nuscenes": (900, 1600),
    "argoverse2": (1550, 2048),
    "av2": (1550, 2048),
}

# _member_meta sentinels: the file or member does not exist (a legitimate absence —
# callers get None), vs. the file exists but the direct-read layout does not apply
# (compressed member, Fortran order, ...) and the caller must take the slow full read.
_MISSING = "missing"
_FALLBACK = "fallback"


def npz_path(root: Path, scene_name: str, token: str) -> Path:
    return Path(root) / scene_name / f"{token}.npz"


@lru_cache(maxsize=16384)
def _member_meta(path_str: str, array_name: str):
    """Resolve where `array_name`'s planes live inside the npz, once per process.

    Returns ``(data_offset, n_planes, plane_bytes, plane_shape, dtype_str, cam_order)``
    when one plane can be read straight out of the file, `_MISSING` when the file or
    member does not exist, `_FALLBACK` when the layout rules this out. Pseudo-label
    files are immutable once generated, so absence is cached like presence.
    """
    try:
        with zipfile.ZipFile(path_str) as zf:
            names = set(zf.namelist())
            member_name = f"{array_name}.npy"
            if member_name not in names:
                return _MISSING
            if "cam_order.npy" in names:
                cam_order = tuple(str(c) for c in np.load(io.BytesIO(zf.read("cam_order.npy"))))
            else:
                cam_order = tuple(CAMERA_ORDER)

            info = zf.getinfo(member_name)
            if info.compress_type != zipfile.ZIP_STORED:
                return _FALLBACK
            with zf.open(member_name) as member:
                version = np.lib.format.read_magic(member)
                shape, fortran_order, dtype = np.lib.format._read_array_header(member, version)
                npy_header_size = member.tell()
            if fortran_order or dtype.hasobject or len(shape) < 2:
                return _FALLBACK

        # The local file header's name/extra lengths can differ from the central
        # directory's, so the data offset must come from the local header itself.
        with open(path_str, "rb") as f:
            f.seek(info.header_offset)
            local_header = f.read(30)
            if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
                return _FALLBACK
            name_len = int.from_bytes(local_header[26:28], "little")
            extra_len = int.from_bytes(local_header[28:30], "little")

        data_offset = info.header_offset + 30 + name_len + extra_len + npy_header_size
        plane_shape = shape[1:]
        plane_bytes = int(np.prod(plane_shape)) * dtype.itemsize
        return (data_offset, shape[0], plane_bytes, plane_shape, dtype.str, cam_order)
    except FileNotFoundError:
        return _MISSING
    except Exception:  # noqa: BLE001 — any surprise falls back to the correct slow path
        return _FALLBACK


def read_camera_array(
    root: Path, scene_name: str, token: Optional[str], camera_name: str, array_name: str
) -> Optional[np.ndarray]:
    """Return `array_name`'s slice for `camera_name`, or None when it is not available.

    None covers every legitimate absence: pseudo labels exist for keyframes only (so sweep
    frames have no file), and the optional streams (`scores`, `confidence`) are only present
    when the generator was asked for them. Callers substitute their own "no supervision here"
    placeholder rather than having to distinguish these cases.

    The returned array may be a read-only view of an internal buffer; callers copy
    (``.astype``) before mutating, as all current callers do.
    """
    if token is None:
        return None
    path = npz_path(root, scene_name, token)
    meta = _member_meta(str(path), array_name)
    if meta is _MISSING:
        return None
    if meta is _FALLBACK:
        return _read_full(path, camera_name, array_name)

    data_offset, n_planes, plane_bytes, plane_shape, dtype_str, cam_order = meta
    if camera_name not in cam_order:
        return None
    index = cam_order.index(camera_name)
    if not 0 <= index < n_planes:
        return None
    with open(path, "rb") as f:
        f.seek(data_offset + index * plane_bytes)
        buffer = f.read(plane_bytes)
    if len(buffer) != plane_bytes:
        return _read_full(path, camera_name, array_name)
    return np.frombuffer(buffer, dtype=np.dtype(dtype_str)).reshape(plane_shape)


def _read_full(path: Path, camera_name: str, array_name: str) -> Optional[np.ndarray]:
    """Slow path: materialise the whole member and slice. Correct for any npz."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if array_name not in data:
            return None
        cam_order = [str(c) for c in data["cam_order"]] if "cam_order" in data else list(CAMERA_ORDER)
        if camera_name not in cam_order:
            return None
        return data[array_name][cam_order.index(camera_name)]
