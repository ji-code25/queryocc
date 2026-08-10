"""Site-specific default paths, overridable via environment variables.

Every dataset/asset location the training code needs is resolved here, from an
environment variable with a container-layout default (the defaults match the
bind mounts used by the SLURM launchers in scripts/). Put your site values in
the git-excluded ``scripts/env.local.sh`` (same mechanism as
``generate_pseudo_labels/scripts/env.local.sh``) — the launchers source it and
export these variables before python starts. All of these are also plain
config fields, so any of them can still be overridden per-run on the CLI.
"""

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


# Dataset roots.
NUSC_ROOT = _env_path("NUSC_ROOT", "/datasets/nuscenes")
AV2_ROOT = _env_path("AV2_ROOT", "/datasets/argoverse2")

# Dataset index pkls produced by generate_pseudo_labels.create_infos.
INFOS_DIR = _env_path("QUERYOCC_INFOS_DIR", str(NUSC_ROOT / "infos"))

# Pseudo-label npz streams produced by generate_pseudo_labels (semantics + depth).
PSEUDO_SEMANTICS_ROOT = _env_path("QUERYOCC_PSEUDO_SEMANTICS_ROOT", "/datasets/pseudo_labels/semantics")
PSEUDO_DEPTH_ROOT = _env_path("QUERYOCC_PSEUDO_DEPTH_ROOT", "/datasets/pseudo_labels/depth")

# Occ3D-nuScenes ground truth (evaluation only).
OCC3D_ROOT = _env_path("QUERYOCC_OCC3D_ROOT", "/datasets/occ3d-nuscenes")

# Precomputed free-ray ("missing points") caches.
NUSC_MISSING_POINTS_ROOT = _env_path("QUERYOCC_NUSC_MISSING_POINTS_ROOT", "/datasets/nuscenes_missing_points")
AV2_MISSING_POINTS_ROOT = _env_path("QUERYOCC_AV2_MISSING_POINTS_ROOT", "/datasets/av2_missing_points")

# Per-point lidar pseudo-segmentation labels (tools/lidar_pseudoseg) for the
# lidar-supervised arms.
NUSC_PSEUDOSEG_ROOT = _env_path("QUERYOCC_NUSC_PSEUDOSEG_ROOT", "/outputs/pseudo_labeler/pseudoseg")
AV2_PSEUDOSEG_ROOT = _env_path("QUERYOCC_AV2_PSEUDOSEG_ROOT", "/outputs/pseudo_labeler/pseudoseg/av2_delta_fused")

# DINOv3-pretrained ConvNeXt backbone checkpoints (see backbones/convnext.py for
# the download pointer).
DINOV3_CKPT_DIR = _env_path("QUERYOCC_DINOV3_CKPT_DIR", "/outputs/ckpts")
