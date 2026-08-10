# Third-party code, datasets and model weights

QueryOcc itself is released under the Apache License, Version 2.0 (see `LICENSE`).
This file records everything in or around the repository that carries *different* terms,
so it is clear what applies to what.

## Code vendored into this repository

Each file keeps its original copyright header. The upstream licenses below (Apache-2.0 and
MIT) all permit inclusion in an Apache-2.0 work provided their notices are retained, which
`NOTICE` does.

| Path | Upstream | License |
|---|---|---|
| `queryocc/model/ops/functions/` | [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) (SenseTime) | Apache-2.0 |
| `queryocc/utils/geometry/poses.py`, `rotations.py` | [NeuRAD](https://github.com/georghess/neurad-studio) / [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | Apache-2.0 |
| `queryocc/model/components/image_encoders/backbones/convnext.py` | [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) (Meta) | MIT |
| `queryocc/model/ops/dvr/` | [4d-occ-forecasting](https://github.com/tarashakhurana/4d-occ-forecasting) | MIT |
| `queryocc/model/components/dynamic_conv.py` | [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch) | MIT |
| `queryocc/optimizer/cAdam.py` | [C-Optim](https://github.com/kyleliang919/C-Optim) | MIT |

Some files note a SciPy-derived implementation in comments; those follow their surrounding
file's Apache-2.0 terms as inherited from NeuRAD/Nerfstudio.

## Datasets

**Not distributed with this repository.** You download them yourself and are bound by the
terms you accept at the source. Both nuScenes and Argoverse 2 require accepting terms of
use before download, and both are widely used under research/non-commercial conditions —
check the current terms yourself before any commercial use.

| Dataset | Used for | Terms |
|---|---|---|
| [nuScenes](https://www.nuscenes.org/nuscenes) | training and evaluation | accept the nuScenes Terms of Use at download |
| [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) | evaluation labels only | per Occ3D; derived from nuScenes, so nuScenes terms also apply |
| [Argoverse 2](https://www.argoverse.org/av2.html) | optional multi-dataset arm | per the Argoverse 2 terms |

## Pretrained model weights

Also not distributed here. Fetched by you, under their own licenses:

| Weights | Used by | Notes |
|---|---|---|
| [DINOv3](https://github.com/facebookresearch/dinov3) ConvNeXt-Base | image backbone initialization | **Gated** — request access from Meta and accept the DINOv3 license. Only *training* needs these; evaluating a released QueryOcc checkpoint does not, because the checkpoint overwrites every backbone weight. |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) SwinB + [SAM](https://github.com/facebookresearch/segment-anything) ViT-H | `generate_pseudo_labels/generate_semantics.py` | per upstream |
| [Metric3D](https://github.com/YvanYin/Metric3D) ViT-L | `generate_pseudo_labels/generate_depth.py` | per upstream |

## Checkpoints released by this project

QueryOcc weights published by the authors are covered by this repository's Apache-2.0
license as to the code, but they are derived from the datasets above. Anyone using them
should honour the terms of the data they were trained on.
