---
license: apache-2.0
tags:
  - 3d-occupancy-prediction
  - autonomous-driving
  - self-supervised
  - nuscenes
library_name: pytorch
---

<!-- Upload this file as README.md in the Hub repo. Fill in every {PLACEHOLDER}. -->

# QueryOcc: Query-based Self-Supervision for 3D Semantic Occupancy

Weights for [QueryOcc](https://arxiv.org/abs/2511.17221) (Lilja, Lan, Fu, Hammarstrand), CVPR 2026.

[Project page](https://research.zenseact.com/publications/queryocc/) ·
[Paper (arXiv)](https://arxiv.org/abs/2511.17221) ·
[Code]({REPO_URL})

QueryOcc learns continuous 3D semantic occupancy from multi-view camera images by supervising
directly in 4D space-time: positive and negative queries are sampled along rays from point
clouds observed in adjacent frames — no rendering losses, no voxelized lidar aggregation. The
supervision comes either from a vision foundation model (metric depth + open-vocabulary
semantics) or from real lidar. No human annotation is used at any stage.

## Files

| File | Preset | Supervision | Sem. RayIoU | Sem. IoU | Occ. IoU | sha256 |
|---|---|---|---|---|---|---|
| `queryocc-nusc.pth` | `queryocc-nusc` | pseudo point clouds (Metric3D + Grounded-SAM) | {SEM_RAYIOU} | {SEM_IOU} | {OCC_IOU} | `{SHA256}` |

Metrics are on the Occ3D-nuScenes validation split under the self-supervised protocol,
112 M parameters, ~450 MB per file.

## Usage

```bash
git clone {REPO_URL} && cd queryocc
./setup_uv_env.sh && source .venv/bin/activate

python queryocc/train.py queryocc-nusc --no-train --test \
    --load_ckpt_from=hf://{HF_REPO}
```

The checkpoint is downloaded and cached automatically, and records the preset it belongs to —
loading it into a different preset is an error rather than a silent partial load. To use a
manually downloaded file, pass its path instead.

Evaluation additionally requires nuScenes, the Occ3D-nuScenes labels and a CUDA toolkit (the
RayIoU raycaster is JIT-compiled on first use); the repository README covers the setup. Nothing
else is needed — in particular, evaluating these weights does **not** require the gated DINOv3
backbone checkpoint that training uses.

Add `--allow_visualization --visualizer.f_viz_occ_3d_bev --visualizer.f_viz_bev_features` to log
BEV predictions and a PCA projection of the BEV features to Weights & Biases.

## Loading the weights directly

The file is a plain `torch.save` payload containing only tensors and primitives, so it needs no
trust in the publisher:

```python
import torch

payload = torch.load("queryocc-nusc.pth", map_location="cpu", weights_only=True)
payload["preset"]  # 'queryocc-nusc'
payload["metrics"]  # evaluation metrics recorded at export
payload["state_dict"]  # 'net.'-prefixed network weights
```

## Training data and intended use

Trained on nuScenes (camera images and, for the lidar arms, lidar). Research artifact: a
demonstration of self-supervised occupancy learning, not a validated driving component. Its
outputs reflect the geometry, sensor suite and geographies of nuScenes (Boston and Singapore)
and should not be relied on for safety-critical decisions. Use is subject to the
[nuScenes terms](https://www.nuscenes.org/terms-of-use) as well as the license below.

## License

Apache 2.0, matching the code. {ADD_ANY_DATASET_DERIVED_RESTRICTIONS}

## Citation

```bibtex
@inproceedings{lilja2026queryocc,
  title={QueryOcc: Query-based Self-Supervision for 3D Semantic Occupancy},
  author={Lilja, Adam and Lan, Ji and Fu, Junsheng and Hammarstrand, Lars},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={21399--21408},
  year={2026}
}
```
