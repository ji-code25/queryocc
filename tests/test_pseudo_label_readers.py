"""The pseudo-label readers must honour the on-disk contract of generate_pseudo_labels.

Covers the two properties that would silently corrupt supervision if broken:
the per-camera slice is taken by the file's own `cam_order` (not by position),
and sky is remapped to the free class rather than being dropped as noise.
"""

import numpy as np
import pytest

from generate_pseudo_labels.constants import CAMERA_ORDER, DEFAULT_SKY_LABEL
from queryocc.data.datasets.nuscenes_mappings import DEFAULT_NUSCENES_CLASSES, LabelMapper
from queryocc.data.vfm.grounding_sam_reader import GroundingSAMReader, GroundingSAMReaderConfig
from queryocc.data.vfm.metric3d_reader import Metric3DReader, Metric3DReaderConfig

SCENE, TOKEN = "scene-0061", "cafed58c98cd45e6b0e0e8de6bcf4b8e"
H, W = 4, 6
CAR_LABEL = 4  # nuScenes-lidarseg id, see generate_pseudo_labels.constants.CLASS_NAMES


@pytest.fixture
def label_roots(tmp_path):
    """Write one semantics + one depth npz in the documented layout.

    Camera c of CAMERA_ORDER gets the constant depth c and label c+1, so a
    mis-ordered read is visible as an off-by-index value.
    """
    semantics_root, depth_root = tmp_path / "semantics", tmp_path / "depth"
    for root in (semantics_root, depth_root):
        (root / SCENE).mkdir(parents=True)

    labels = np.stack([np.full((H, W), i + 1, dtype=np.int8) for i in range(len(CAMERA_ORDER))])
    labels[CAMERA_ORDER.index("CAM_BACK")][0, 0] = DEFAULT_SKY_LABEL
    np.savez(
        semantics_root / SCENE / f"{TOKEN}.npz",
        labels=labels,
        scores=np.full((len(CAMERA_ORDER), H, W), 0.75, dtype=np.float16),
        cam_order=np.array(CAMERA_ORDER),
    )
    np.savez(
        depth_root / SCENE / f"{TOKEN}.npz",
        depth=np.stack([np.full((H, W), i, dtype=np.float16) for i in range(len(CAMERA_ORDER))]),
        cam_order=np.array(CAMERA_ORDER),
    )
    return semantics_root, depth_root


def make_readers(label_roots):
    semantics_root, depth_root = label_roots
    semantics = GroundingSAMReader(GroundingSAMReaderConfig(semantics_root=semantics_root), "nuscenes")
    depth = Metric3DReader(Metric3DReaderConfig(depth_root=depth_root), "nuscenes")
    return semantics, depth


@pytest.mark.parametrize("camera", CAMERA_ORDER)
def test_camera_slice_follows_embedded_cam_order(label_roots, camera):
    semantics, depth = make_readers(label_roots)
    expected = CAMERA_ORDER.index(camera)

    depth_map = depth.read_camera(SCENE, TOKEN, camera)
    assert depth_map.shape == (H, W)
    assert np.all(depth_map == expected)

    labels = semantics.read_camera(SCENE, TOKEN, camera)  # raw ids without a label mapper
    assert np.all(labels[labels != DEFAULT_SKY_LABEL] == expected + 1)


def test_reversed_cam_order_is_honoured(tmp_path):
    """A file written in a different camera order must still read out correctly."""
    root = tmp_path / "semantics"
    (root / SCENE).mkdir(parents=True)
    reversed_order = list(reversed(CAMERA_ORDER))
    np.savez(
        root / SCENE / f"{TOKEN}.npz",
        labels=np.stack([np.full((H, W), i + 1, dtype=np.int8) for i in range(len(reversed_order))]),
        cam_order=np.array(reversed_order),
    )
    reader = GroundingSAMReader(GroundingSAMReaderConfig(semantics_root=root), "nuscenes")
    for i, camera in enumerate(reversed_order):
        assert np.all(reader.read_camera(SCENE, TOKEN, camera) == i + 1)


def test_sky_becomes_free_class_when_requested(label_roots):
    semantics, _ = make_readers(label_roots)
    mapper = LabelMapper(DEFAULT_NUSCENES_CLASSES)

    free = semantics.read_camera(SCENE, TOKEN, "CAM_BACK", mapper, sky_to_free=True)
    assert free[0, 0] == mapper.free_idx

    # Without sky_to_free, sky is not a learning class and collapses to noise.
    noise = semantics.read_camera(SCENE, TOKEN, "CAM_BACK", mapper, sky_to_free=False)
    assert noise[0, 0] == mapper.noise_idx


def test_missing_sample_reads_as_none(label_roots):
    """Sweep frames have no pseudo labels; that must be reported, not raised."""
    semantics, depth = make_readers(label_roots)
    assert semantics.read_camera(SCENE, "not-a-token", "CAM_FRONT") is None
    assert depth.read_camera(SCENE, "not-a-token", "CAM_FRONT") is None
    assert semantics.read_camera(SCENE, None, "CAM_FRONT") is None
    assert depth.read_camera("no-such-scene", TOKEN, "CAM_FRONT") is None


def test_scores_are_optional(tmp_path, label_roots):
    semantics, _ = make_readers(label_roots)
    assert np.allclose(semantics.read_camera_scores(SCENE, TOKEN, "CAM_FRONT"), 0.75)

    root = tmp_path / "no_scores"
    (root / SCENE).mkdir(parents=True)
    np.savez(
        root / SCENE / f"{TOKEN}.npz",
        labels=np.zeros((len(CAMERA_ORDER), H, W), dtype=np.int8),
        cam_order=np.array(CAMERA_ORDER),
    )
    reader = GroundingSAMReader(GroundingSAMReaderConfig(semantics_root=root), "nuscenes")
    assert reader.read_camera_scores(SCENE, TOKEN, "CAM_FRONT") is None
