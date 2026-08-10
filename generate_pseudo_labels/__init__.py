"""Standalone pseudo-label generation for camera-only occupancy training.

Self-contained extraction of the GaussianFlowOcc pseudo-label pipelines:
  * semantic masks   : GroundingDINO (SwinB) + SAM (ViT-H), open-vocabulary
  * metric depth     : Metric3D (ViT-L) via torch.hub
  * dataset infos    : bevdetv2-style .pkl files built directly from
                       nuscenes-devkit, so the *training* environment never
                       needs the nuScenes devkit or mm* packages.

No dependency on mmdet3d / mmcv / the GaussianFlowOcc repo. Copy this
directory into any repo and run the scripts with `python -m
generate_pseudo_labels.<script>` (or directly as files).
"""

__version__ = "0.1.0"
