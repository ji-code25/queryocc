"""Metric depth reader for the pseudo-label npz layout.

Reads the depth stream written by ``generate_pseudo_labels.generate_depth``:

    <depth_root>/<scene_name>/<token>.npz
        depth      float16 [6, H, W]   metres, clamped to [0, 300], CAMERA_ORDER
        cam_order  <U18    [6]         the camera order the arrays are stored in

``cam_order`` is embedded in every file, so the per-camera slice is resolved
from the file itself rather than from a hard-coded ordering.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np

from queryocc.common import paths
from queryocc.config.base_config import InstantiateConfig
from queryocc.data.vfm.pseudo_label_npz import read_camera_array


@dataclass
class Metric3DReaderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Metric3DReader)

    depth_root: Path = paths.PSEUDO_DEPTH_ROOT
    """Root of the generate_pseudo_labels depth stream (<root>/<scene>/<token>.npz)."""


class Metric3DReader:
    def __init__(self, config: Metric3DReaderConfig, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name

    def read_camera(self, scene_name: str, token: Optional[str], camera_name: str) -> Optional[np.ndarray]:
        """Depth map [H, W] in metres for one camera of one sample, or None when unavailable."""
        depth = read_camera_array(self.config.depth_root, scene_name, token, camera_name, "depth")
        return None if depth is None else depth.astype(np.float32)
