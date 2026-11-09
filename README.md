# QueryOcc: Query-based Self-Supervision for 3D Semantic Occupancy

Official implementation of
[**QueryOcc: Query-based Self-Supervision for 3D Semantic Occupancy**](https://arxiv.org/abs/2511.17221)
(Adam Lilja, Ji Lan, Junsheng Fu, Lars Hammarstrand), **CVPR 2026**.

[Project page](https://research.zenseact.com/publications/queryocc/) ·
[Paper (arXiv)](https://arxiv.org/abs/2511.17221) ·
[Weights](https://huggingface.co/QueryOcc/queryocc)

QueryOcc learns continuous 3D semantic occupancy from multi-view camera images by
supervising directly in 4D space-time: positive and negative queries are sampled along
rays from point clouds observed in adjacent frames, with no rendering losses and no
voxelized lidar aggregation. The point clouds come either from a vision foundation model
(metric depth + open-vocabulary semantics) or from real lidar.

On the self-supervised Occ3D-nuScenes benchmark QueryOcc reaches **23.6 semantic RayIoU**
at 11.6 FPS, a 26% relative improvement over the previous best camera-only method.

## Repository structure

| Path | Contents |
|---|---|
| `queryocc/` | The training/evaluation package (model, data, metrics, visualization). |
| `generate_pseudo_labels/` | Standalone pseudo-label generation: dataset index pkls, Grounded-SAM semantics, Metric3D depth. See [its README](generate_pseudo_labels/README.md). |
| `tools/lidar_pseudoseg/` | Offline per-point lidar semantic labels, needed by the lidar-supervised arms. |
| `tools/missing_points/` | Offline free-ray ("missing points") generation — free-space supervision for the lidar-supervised arms. |
| `tools/release/` | How the released weights are produced from a training checkpoint, and how to publish new ones. See [its README](tools/release/README.md). |
| `scripts/` | SLURM launchers for a generic cluster, and the release run matrix. |
| `containers/` | Docker / Apptainer build files. |

## Installation

```bash
./setup_uv_env.sh          # creates .venv with the full stack
source .venv/bin/activate
```

There is no compile step at install time. The only native code is the `dvr` raycaster used
by the RayIoU metrics, which PyTorch JIT-compiles on first use — that needs a CUDA toolkit
and a GPU, i.e. inside the container or on a cluster node.

For clusters, build the container and convert it to an Apptainer image:

```bash
containers/build_docker.sh                  # optionally --build-arg CUDA_ARCHITECTURES="80"
containers/convert_docker_to_apptainer.sh   # -> queryocc_YYYYMMDD.sif
```

Copy the resulting `.sif` to the cluster and point `CONTAINER` at it in
`scripts/env.local.sh` (the launchers default to `$PROJECT_ROOT/containers/queryocc.sif`).
`CUDA_ARCHITECTURES` only sets `TORCH_CUDA_ARCH_LIST` for the JIT-compiled `dvr` raycaster,
so building for just your target GPU (80 = A100) is enough.

## Data preparation

### 1. nuScenes and the Occ3D ground truth

Download [nuScenes](https://www.nuscenes.org/nuscenes) (samples, sweeps, `v1.0-*`, plus
`lidarseg` and `panoptic` for the lidar arms) and the
[Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) labels, which are used for
evaluation only and are expected as `<occ3d_root>/<scene>/<token>/labels.npz`.

### 2. Dataset index and pseudo labels

Both come from `generate_pseudo_labels/` (see its README for cluster job arrays, sharding
and verification):

```bash
python -m generate_pseudo_labels.create_infos \
    --root-path data/nuscenes --version v1.0-trainval --out-dir data/infos \
    --relative-to data/nuscenes --occ-path-prefix occ3d-nuscenes
python -m generate_pseudo_labels.generate_semantics \
    --infos-pkl data/infos/bevdetv2-nuscenes_infos_train.pkl \
    --save-path data/pseudo_labels/semantics --data-root $PWD
python -m generate_pseudo_labels.generate_depth \
    --infos-pkl data/infos/bevdetv2-nuscenes_infos_train.pkl \
    --save-path data/pseudo_labels/depth --data-root $PWD
```

This writes `<root>/<scene_name>/<token>.npz` for both streams, which is exactly what the
training-time readers consume. Full nuScenes trainval costs roughly 90 GPU-hours and
~850 GB at native resolution.

### 3. Dataset cache (optional but recommended)

Training derives a per-split index (lidar frame pool, camera frames with their sample
tokens, calibration) from the nuScenes database plus the split definition above. It is
built lazily on first use, which makes every rank of a GPU job walk the database before
the first step. Precompute it once on CPU instead:

```bash
python -m tools.create_queryocc_infos --version full                    # locally
scripts/create_infos.sh --version full                            # cluster login node
sbatch --gpus-per-node=T4:1 scripts/create_infos.sh --version full # or as a batch job
```

This needs only the nuScenes database and the index pkls — no GPU, no pseudo labels, and
no training container: the builder (`queryocc/data/datasets/nuscenes_infos.py`) imports
neither torch nor any other heavy dependency, just numpy, pyquaternion and
nuscenes-devkit, so it runs in a small CPU environment. The dataset imports the same
builder, so the precomputed cache can never drift from what training expects.

The cache filename hashes the settings it depends on, so pass the same `--version` /
`--all-lidar-frames` the training preset uses (the defaults match the released presets).

The camera-only arm (`queryocc-nusc`) needs nothing else.

### 3a. Lidar supervision: per-point labels and free rays

`queryocc-lidar-nusc`, `queryocc-joint-nusc` and `queryocc-nusc-av2` supervise from real
lidar, which needs two offline products beyond the dataset itself:

| Product | Tool | Supervises |
|---|---|---|
| per-point semantic labels | [`tools/lidar_pseudoseg/`](tools/lidar_pseudoseg/README.md) | the semantic head (lidar has no ground-truth labels here) |
| free rays / "missing points" | [`tools/missing_points/`](tools/missing_points/README.md) | free space — the negative queries |

Free rays are synthetic points placed along lidar rays that returned *nothing*: an
unreturned ray certifies empty space out to the sensor's range, which is exactly the
negative evidence the query sampler needs. Every preset sets `add_missing_points=True`, so
without this cache the lidar arms train on materially weaker free-space supervision — the
dataloader warns once when it finds none.

```bash
python -m tools.missing_points.engine --dataset nuscenes   --version v1.0-trainval
python -m tools.missing_points.engine --dataset argoverse2 --split train
```

Both are CPU-only (no CUDA, no compiled extensions) and resumable, so they can run on a
login node or a cheap CPU allocation; see the tool README for cost and the exact on-disk
contract. Training finds them via `QUERYOCC_NUSC_MISSING_POINTS_ROOT` /
`QUERYOCC_AV2_MISSING_POINTS_ROOT` (bound by the SLURM launchers from
`NUSC_MISSING_POINTS_DIR` / `AV2_MISSING_POINTS_DIR` in `scripts/env.local.sh`).

### 3b. Argoverse 2 (only for `queryocc-nusc-av2`)

The cross-dataset arm additionally needs the **AV2 sensor dataset** (`AV2_ROOT`, either
layout: `<root>/<split>/<log>` or `<root>/sensor/<split>/<log>`) and its own pseudo labels
— the whole chain is in this repo and none of it is shared with nuScenes:

```bash
AV2=/datasets/argoverse2
LANDSCAPE="ring_front_left ring_front_right ring_side_left ring_side_right ring_rear_left ring_rear_right"

# index: one over all seven cameras (for the projection) + one per resolution group
python -m generate_pseudo_labels.create_infos_av2 --root-path $AV2 --split train \
    --out-dir $AV2 --relative-to $AV2 --tag av2
python -m generate_pseudo_labels.create_infos_av2 --root-path $AV2 --split train \
    --out-dir data/infos --relative-to $AV2 --cameras $LANDSCAPE --tag av2-landscape
python -m generate_pseudo_labels.create_infos_av2 --root-path $AV2 --split train \
    --out-dir data/infos --relative-to $AV2 --cameras ring_front_center --tag av2-front-center

# 2D semantics: AV2's ring_front_center is portrait (1550x2048) and the six other rings
# are landscape (2048x1550); the generator batches all cameras of a sample, so one run
# and one --save-path per group
python -m generate_pseudo_labels.generate_semantics --data-root $AV2 --cameras $LANDSCAPE \
    --infos-pkl data/infos/av2-landscape_infos_train.pkl \
    --save-path data/pseudo_labels/av2/semantics_landscape
python -m generate_pseudo_labels.generate_semantics --data-root $AV2 --cameras ring_front_center \
    --infos-pkl data/infos/av2-front-center_infos_train.pkl \
    --save-path data/pseudo_labels/av2/semantics_front_center

# per-point lidar pseudo-seg (what queryocc-nusc-av2 actually reads, via AV2_PSEUDOSEG_ROOT)
python -m tools.lidar_pseudoseg.engine --dataset argoverse2 --split train \
    --infos_pkl $AV2/av2_infos_train.pkl --argoverse_data $AV2 \
    --semantics_root data/pseudo_labels/av2/semantics_landscape \
                     data/pseudo_labels/av2/semantics_front_center \
    --pseudo_label_name av2_delta --scenewise_fusion true
```

Repeat with `--split val`. Note what is and is not needed: on AV2 the training-time VFM
reader consumes **only** the per-point lidar pseudo-seg (`queryocc/data/vfm/vfm_reader.py`
gates the 2D semantic/depth streams on nuScenes), so the 2D semantics above are an
intermediate product and `generate_depth` is not required at all for this arm. The AV2
index carries no occupancy ground truth or boxes (AV2 has no Occ3D labels); details and
caveats are in `generate_pseudo_labels/README.md` ("Argoverse 2") and
`tools/lidar_pseudoseg/README.md`.

### 4. Site configuration

Every dataset location is an environment variable with a container-layout default, resolved
in `queryocc/common/paths.py`: `NUSC_ROOT`, `AV2_ROOT`, `QUERYOCC_INFOS_DIR`,
`QUERYOCC_PSEUDO_SEMANTICS_ROOT`, `QUERYOCC_PSEUDO_DEPTH_ROOT`, `QUERYOCC_OCC3D_ROOT`,
`QUERYOCC_NUSC_MISSING_POINTS_ROOT`, `QUERYOCC_{NUSC,AV2}_PSEUDOSEG_ROOT` and
`QUERYOCC_DINOV3_CKPT_DIR`. Every one is also a config field, so it can be overridden
per-run on the command line.

On a cluster, put your paths in `scripts/env.local.sh` (git-excluded); the launchers source
it and bind the host directories to the container layout. Locally, `cp .env.example .env`
and fill it in for `containers/start_docker.sh`.

The ConvNeXt backbone is initialized from DINOv3 weights — download
`dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` from
[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) (gated) into the
directory `QUERYOCC_DINOV3_CKPT_DIR` points at. The SLURM launchers bind `DINOV3_CKPT_DIR`
(host) to that location inside the container, so only the host path needs setting in
`scripts/env.local.sh`.

## Training

```bash
python queryocc/train.py queryocc-nusc                     # single GPU
python queryocc/train.py queryocc-nusc --trainer.devices=4 # multi-GPU
```

Any config field can be overridden from the CLI, e.g.
`--model.occupancy_sampler.num_query_points 400000` for the paper's exact query budget
(the preset halves it for speed) or `--datamodule.batch_size 2` if memory is tight.

On SLURM:

```bash
sbatch scripts/train_multi_gpu.sh <run-name> queryocc-nusc
sbatch --array=1-4 scripts/arrays/array_train.sh release   # the full release matrix
```

### Berzelius smoke training

Before a full `queryocc-nusc` run, the smoke workflow exercises the production model and
the real Grounded-SAM + Metric3D input path with a bounded, reproducible workload:

- 100 deterministic training records, repeated to a 200-sample budget on 2 GPUs with
  batch size 1/GPU and no accumulation;
- QueryOcc's normal sample-budget scaling turns `max_steps=200` into exactly 100
  optimizer steps while the datamodule repeats the small frozen subset;
- 32 frozen validation records, selected to cover every dynamic Occ3D class and
  yielding 28 valid temporal anchors after scene-boundary filtering;
- the production ConvNeXt/DINOv3 backbone, compile path and 200k-query budget (none are
  reduced for the smoke run);
- preflight validation of every selected semantics/depth archive, followed by a
  `last.ckpt` check whose `global_step` must be exactly 100;
- final evaluation of both the validation-best and final checkpoints on all 28
  valid smoke anchors, mirroring the full-run best/final protocol.

Copy `scripts/env.local.example` to the git-ignored `scripts/env.local.sh` and fill in
the Berzelius paths. Grounded-SAM and Metric3D can be stored separately via
`PSEUDO_SEMANTICS_ROOT` and `PSEUDO_DEPTH_ROOT`; the launcher binds them into QueryOcc's
fixed container layout. The existing DragonBall/GaussianFlowOcc format is also supported:
set `LEGACY_PSEUDO_SEMANTICS_ROOT` and `LEGACY_PSEUDO_DEPTH_ROOT` instead. Preparation
packs the six per-camera masks into one file, fixes the legacy sky-label collision
(`17 -> 18`), reorders Metric3D to the embedded `cam_order`, and only exposes complete
label/depth records to QueryOcc. Then submit preparation and training as one dependency chain:

```bash
scripts/submit_queryocc_smoke_berzelius.sh
```

Set `QUERYOCC_CONVERTED_ROOT` to the canonical materialized dataset location. Smoke
preparation writes its first complete records there, and full training reuses them. Once
all 28,130 training records have refined six-camera labels/scores, run the resumable
one-time full conversion on the CPU partition:

```bash
sbatch scripts/convert_queryocc_pseudo_labels_berzelius.sh
```

The full job gates on all expected records before writing, skips already converted
samples, and atomically renames each output pair, so it is safe to resubmit after a
timeout without duplicating data. Its default request is 16 CPU cores, 64 GB RAM and a
six-hour limit; all 16 workers convert independent records in parallel while numerical
libraries are restricted to one thread per worker.

The preparation job writes frozen info files plus `manifest.json` under
`$QUERYOCC_SMOKE_ROOT/smoke_inputs`. The GPU job writes
`$OUTPUT_DIR/smoke_runs/queryocc-nusc-smoke100-<job-id>/smoke_summary.json` only after
the input preflight, 100 training steps, validation and checkpoint checks all pass. The
GPU request has no `fat`/80 GB constraint: the production camera-only preset is expected
to fit an ordinary 40 GB Berzelius A100.

### Release matrix

| Preset | Supervision | Datasets |
|---|---|---|
| `queryocc-nusc` | pseudo point clouds (Metric3D + Grounded-SAM) | nuScenes |
| `queryocc-lidar-nusc` | real lidar | nuScenes |
| `queryocc-joint-nusc` | pseudo point clouds **and** real lidar | nuScenes |
| `queryocc-nusc-av2` | real lidar | nuScenes + Argoverse 2 |

Training runs a full evaluation on the test split after fitting, using the best checkpoint.

## Evaluation

```bash
python queryocc/train.py queryocc-nusc --no-train --test --load_ckpt_from=<checkpoint>
```

`<checkpoint>` is a `.ckpt` file, a run directory (add `--load_ckpt_from_best`), or an
`hf://<owner>/<repo>` released checkpoint (see [Model zoo](#model-zoo)). Weights load strictly:
missing or unexpected keys fail the run rather than leaving part of the network at its random
initialization. `--allow_partial_load` downgrades that to a warning, for the cases where the
mismatch is intended.

Metrics are logged to Weights & Biases under the project `queryocc`. Runs go to your own
default entity (resolved from your `~/.netrc` wandb credentials); set `WANDB_ENTITY` in the
git-excluded `scripts/env.local.sh` only if you want them in a shared team instead. The keys
behind the paper's tables:

| Paper metric | Logged key |
|---|---|
| Semantic RayIoU | `test/occ3d/metrics/occ3d_sem_rayiou/semantic_rayiou/overall_mean` |
| Dynamic RayIoU | `…/semantic_rayiou/overall_mean_dynamic` |
| Occupancy RayIoU | `test/occ3d/metrics/occ3d_binary_rayiou/max_binary_rayiou` |
| Semantic IoU | `test/occ3d/metrics/occ3d_multiclass_iou/camera/iou/mean` |
| Occupancy IoU | `test/occ3d/metrics/occ3d_binary_iou/camera/max_iou` |

Two visualizations are available (`--allow_visualization` plus
`--visualizer.f_viz_bev_features` and `--visualizer.f_viz_occ_3d_bev`): a PCA projection of
the BEV feature map, and Occ3D predictions against ground truth in BEV.

## Model zoo

| Preset | Sem. RayIoU | Sem. IoU | Occ. IoU | Checkpoint |
|---|---|---|---|---|
| `queryocc-nusc` | 24.25 | 21.51 | 55.06 | [QueryOcc/queryocc](https://huggingface.co/QueryOcc/queryocc) |
| `queryocc-lidar-nusc` | 23.2 | 20.4 | 56.9 | [QueryOcc/queryocc](https://huggingface.co/QueryOcc/queryocc) |

The `queryocc-nusc` row is above the paper's 23.6 / 21.3 / 55.0 because the preset's peak LR was
tuned after publication (5e-5 → 7.5e-5); the method, architecture and input resolution are
unchanged, and `--model.max_lr=5e-5` reproduces the paper configuration. Three seeds of the
released configuration scored 23.91 / 24.24 / 24.25 semantic RayIoU (mean 24.13) — the number
above is the released checkpoint's own score, and 24.13 is what to expect from a retrain. As
Occ3D-nuScenes has no held-out test split, these are validation-split numbers, which is both
what the benchmark reports and what the LR sweep selected on.

Released checkpoints are single ~450 MB files on the Hugging Face Hub, downloaded and cached by
`--load_ckpt_from`:

```bash
python queryocc/train.py queryocc-nusc --no-train --test --load_ckpt_from=hf://<owner>/queryocc
```

The spec is `hf://<owner>/<repo>[/<file>]`, defaulting to `<preset>.pth` — so the same repo serves
every preset, and selecting a preset it does not publish says so rather than 404ing. A local path
works the same way. Each file records the preset it was trained with, and loading it into a different one
is an error rather than a partially initialized model. Evaluating a released checkpoint needs
nuScenes, the Occ3D labels and a CUDA toolkit for the `dvr` raycaster — but **not** the gated
DINOv3 weights, which only training uses (the checkpoint overwrites them). Producing and
publishing these files is [`tools/release/`](tools/release/README.md).

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

## License and terms of use

The QueryOcc code is released under the [Apache License, Version 2.0](LICENSE). Portions
vendored from other projects keep their own licenses (Apache-2.0 and MIT) and are listed in
[`NOTICE`](NOTICE).

Two things are **not** covered by that license, because this repository does not distribute
them — you obtain them yourself and are bound by the terms you accept at the source:

- **Datasets.** nuScenes, the Occ3D-nuScenes evaluation labels and Argoverse 2 each require
  accepting their own terms before download, and are commonly available on
  research/non-commercial conditions. Check the current terms before any commercial use.
- **Pretrained weights.** The DINOv3 backbone is gated by Meta and carries the DINOv3
  license; GroundingDINO, SAM and Metric3D carry theirs. Note that only *training* needs the
  DINOv3 weights — evaluating a released QueryOcc checkpoint does not, since the checkpoint
  overwrites the backbone.

[`THIRD_PARTY.md`](THIRD_PARTY.md) has the full per-file and per-artifact breakdown.

## Acknowledgements

The pseudo-label pipeline follows GaussianFlowOcc; the dataset index format follows BEVDet;
the query-based occupancy formulation builds on UnO and GASP. This code was developed in an
internal research platform and lifted here as the official implementation.
