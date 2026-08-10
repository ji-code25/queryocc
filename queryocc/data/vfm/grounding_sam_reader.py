"""Open-vocabulary semantic pseudo-label reader for the npz layout.

Reads the semantics stream written by ``generate_pseudo_labels.generate_semantics``:

    <semantics_root>/<scene_name>/<token>.npz
        labels     int8    [6, H, W]   nuScenes-lidarseg class ids, CAMERA_ORDER
        scores     float16 [6, H, W]   optional (--save-scores)
        cam_order  <U18    [6]

Class ids are the nuScenes-lidarseg space (0 = noise/ignore, 1-16 = classes,
18 = sky). Sky is not a learning class, so ``nusc_to_learning`` would collapse
it to noise and the downstream noise filter would drop those pixels; with
``sky_to_free`` it becomes the *free* learning class instead, which turns sky
pixels into free-space rays (the "w-sky" label set used for the paper).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np

from generate_pseudo_labels.constants import DEFAULT_SKY_LABEL
from queryocc.common import paths
from queryocc.config.base_config import InstantiateConfig
from queryocc.data.datasets.nuscenes_mappings import LabelMapper
from queryocc.data.vfm.pseudo_label_npz import read_camera_array


@dataclass
class GroundingSAMReaderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: GroundingSAMReader)

    semantics_root: Path = paths.PSEUDO_SEMANTICS_ROOT
    """Root of the generate_pseudo_labels semantics stream (<root>/<scene>/<token>.npz)."""


class GroundingSAMReader:
    def __init__(self, config: GroundingSAMReaderConfig, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name

    def read_camera(
        self,
        scene_name: str,
        token: Optional[str],
        camera_name: str,
        label_mapper: Optional[LabelMapper] = None,
        sky_to_free: bool = False,
    ) -> Optional[np.ndarray]:
        """Semantic labels [H, W] for one camera of one sample, or None when unavailable."""
        mask = read_camera_array(self.config.semantics_root, scene_name, token, camera_name, "labels")
        if mask is None:
            return None
        mask = mask.astype(np.int64)
        if label_mapper is not None:
            sky_pixels = mask == DEFAULT_SKY_LABEL if sky_to_free else None
            # Sky is not a learning class, so nusc_to_learning collapses it to noise (idx 0).
            mask = label_mapper.nusc_to_learning(mask)
            if sky_pixels is not None:
                # Keep sky as the "free" learning class instead (sky == empty space along the ray)
                # so the downstream noise filter does not drop it and it becomes free-space rays.
                mask[sky_pixels] = label_mapper.free_idx
        return mask

    def read_camera_scores(self, scene_name: str, token: Optional[str], camera_name: str) -> Optional[np.ndarray]:
        """Per-pixel mask confidence [H, W], or None when scores were not generated."""
        scores = read_camera_array(self.config.semantics_root, scene_name, token, camera_name, "scores")
        return None if scores is None else scores.astype(np.float32)
