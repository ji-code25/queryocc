# generate_pseudo_labels

Standalone pseudo-label generation for camera-only 3D occupancy training.
Extracted from GaussianFlowOcc, with **zero dependency** on mmdet3d / mmcv /
the GaussianFlowOcc repo. Copy this directory into any repo and go.

Three tools, one container:

| tool | what | models | deps |
|---|---|---|---|
| `create_infos.py` | bevdetv2-compatible dataset index pkls (nuScenes) | — | nuscenes-devkit (container only) |
| `create_infos_av2.py` | the same index for Argoverse 2 sensor logs | — | av2 devkit, pandas |
| `generate_semantics.py` | open-vocab semantic masks | GroundingDINO SwinB + SAM ViT-H | upstream pip packages |
| `generate_depth.py` | metric depth | Metric3D ViT-L (torch.hub) | mmengine, timm |

plus a preflight check and three verifiers:

| checker | what it proves |
|---|---|
| `preflight.py` | the array you are about to submit will *start*: run it before `sbatch` |
| `verify_infos.py` | the index is complete and self-consistent, every file it points at exists, and (with `--reference-pkl`) it matches an existing bevdetv2/mmdet3d pkl token for token |
| `verify_outputs.py` | the generated labels satisfy the on-disk contract (shapes, dtypes, camera order, ranges) |
| `validate_against_lidar.py` | the labels are *geometrically right*: lidar projected into every camera vs the predicted depth (abs-rel, δ<1.25, scale bias) and vs the predicted class (lidarseg agreement), plus `--check-camera-order` |

The design goal: your **training environment reads plain `.pkl` / `.npz`
files and nothing else**. All heavyweight, version-fragile dependencies
(nuscenes-devkit, GroundingDINO's CUDA extension, Metric3D's stack) live in
the generation container and never leak into the training repo.

## Quick start

```bash
# 0. build the container (from this directory)
containers/build.sh docker           # or: sif / sif-from-docker

# 1. dataset index (once per nuScenes version)
python -m generate_pseudo_labels.create_infos \
    --root-path data/nuscenes --version v1.0-mini --out-dir data \
    --relative-to data/nuscenes --occ-path-prefix occ3d-nuscenes
python -m generate_pseudo_labels.verify_infos \
    --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
    --data-root data/nuscenes --check-files-all

# 2. pre-download model weights (once, on a node WITH internet)
export TORCH_HOME=/path/to/model_cache/torch
export HF_HOME=/path/to/model_cache/huggingface
python -m generate_pseudo_labels.prefetch_weights --sam-dir ckpts

# 3. generate
python -m generate_pseudo_labels.generate_semantics \
    --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
    --save-path data/pseudo_labels/semantics --data-root $PWD
python -m generate_pseudo_labels.generate_depth \
    --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
    --save-path data/pseudo_labels/depth --data-root $PWD

# 4. verify: contract, then geometry against lidar
python -m generate_pseudo_labels.verify_outputs \
    --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
    --semantics-root data/pseudo_labels/semantics \
    --depth-root data/pseudo_labels/depth
python -m generate_pseudo_labels.validate_against_lidar \
    --infos-pkl data/bevdetv2-nuscenes-mini_infos_train.pkl \
    --data-root data/nuscenes \
    --semantics-root data/pseudo_labels/semantics \
    --depth-root data/pseudo_labels/depth --check-camera-order
```

For **Argoverse 2** the same three steps run against `create_infos_av2.py`
plus a `--cameras` group — see [Argoverse 2](#argoverse-2) below.

On the cluster, use `scripts/slurm_*.sh` (SLURM job arrays + Apptainer;
sharding is derived from the array index, resumable, no split files).

### Preflight the array first

Every way a task can die at *startup* is per-shard — each one loads its own
first sample and runs its own DINO cache self-check — so a shard can fail on
sample 0 while its siblings run for hours. `preflight.py` reproduces those
startup steps for all shards in one short job (~3 s per shard) and exits
non-zero if any of them would fail, so it can gate the submission:

```bash
NUM_SHARDS=16 sbatch --gpus-per-node=A100:1 scripts/slurm_preflight.sh
# or directly, gating an sbatch:
python -m generate_pseudo_labels.preflight \
    --infos-pkl data/bevdetv2-nuscenes_infos_train.pkl \
    --data-root data/nuscenes --num-shards 16 --tf32 \
    --sam-checkpoint ckpts/sam_vit_h_4b8939.pth \
&& sbatch --array=0-15%8 scripts/slurm_generate_semantics.sh
```

It checks the pkl and every shard's slice, the SAM checkpoint, the save
path (writable, free space, no conflicting `_meta_*.json`), that each
shard's first sample decodes, and the DINO cache self-check on each. Pass
the same `--tf32` / thresholds the real run will use, or the check proves
nothing about it. `--skip-dino` keeps only the cheap checks and needs
neither a GPU nor torch; `--check-depth` additionally loads Metric3D, which
is the quickest way to confirm the torch.hub cache works on an offline
compute node.

## Site configuration

The scripts contain no cluster-specific paths. Everything is an environment
variable with a repo-relative default (`CONTAINER`, `NUSC_ROOT`, `OUT_DIR`,
`PSEUDO_LABEL_ROOT`, `DATA_ROOT`, `MODEL_CACHE`, `SAM_CKPT`, `BIND_PATHS`,
`VERSION`, `INFOS_PKL`, `EXTRA_ARGS`, …), read by `scripts/_common.sh`, which
also sources `scripts/env.local.sh` when that file exists. Put your cluster
paths, Apptainer binds, SLURM account and submit helpers in that one file and
keep it out of version control — nothing else needs to change:

```bash
# scripts/env.local.sh
NUSC_ROOT=/data/nuscenes
CONTAINER=/path/to/containers/pseudo_labels.sif
BIND_PATHS="/proj:/proj"
export SBATCH_ACCOUNT=<your-account>
```

`CONTAINER=` (empty) runs the commands in the currently active Python
environment instead of Apptainer — handy for the CPU-only `create_infos`
step, which needs nothing but nuscenes-devkit.

The `#SBATCH` blocks deliberately carry no account or partition: pass those
at submit time (`sbatch -A <account> --gpus=<type>:1 …`), or export
`SBATCH_ACCOUNT`, which sbatch picks up from the environment.

## Output contract

Everything is pinned in [`constants.py`](constants.py) and echoed into a
`_meta_*.json` provenance file at each output root (vocabulary, thresholds,
model ids, camera order, scale, exact command line). A meta mismatch on a
rerun into the same directory is a hard error — no silently mixed datasets.

Default (`--layout npz`), one file per sample:

```
<save-path>/<scene_name>/<token>.npz
  semantics: labels     int8    [6, H, W]     # class ids below
             scores     float16 [6, H, W]     # optional (--save-scores)
             cam_order  str     [6]           # always CAMERA_ORDER
  depth:     depth      float16 [6, H, W]     # meters, clamped [0, 300]
             confidence float16 [6, H, W]     # optional (--save-confidence)
             cam_order  str     [6]
```

* **One camera order for both streams** (`constants.CAMERA_ORDER`), embedded
  in every file. (The original pipeline used *different* orders for masks
  and depth — a silent-corruption hazard this layout removes.)
* Class ids are nuScenes-lidarseg v1 ids (`constants.CLASS_NAMES`):
  0 = ignore, 1–16 = classes, 18 = sky (default `--sky-label`; the original
  pipeline used 0 = ignore for sky, or 17 which collides with "free").
* `--scale` bakes downscaling into the files (masks: nearest, depth:
  bilinear). At the GaussianFlowOcc render resolution (`--scale 0.44`) this
  cuts disk ~5x and removes the per-epoch resize; remember to scale your
  intrinsics by the same factor at load time (`fx,fy,cx,cy *= scale`).

`--layout legacy` instead reproduces the exact GaussianFlowOcc trainer
format: plain `.npy` stacks at native resolution, masks in
`LEGACY_MASK_ORDER`, depth in `LEGACY_DEPTH_ORDER` — useful for A/B against
the original training code.

## What the pipelines do (and where they differ from the original)

**Semantics** — per image: GroundingDINO (SwinB, HF checkpoint) is prompted
once per vocabulary entry (42 prompts, thresholds 0.20/0.20, input resized
to short-side 800 / long-side ≤ 1333 → 750×1333 for nuScenes); detections
whose decoded phrase ≠ prompt are dropped; SAM ViT-H segments each box at
full resolution; masks are fused by per-pixel argmax over
confidence-weighted masks; the winner maps to a class id. Reproduces the
published GaussianFlowOcc labels with `--vocabulary gaussianflowocc
--sky-label 0`. `--save-phrase-scores` additionally stores per-prompt score
maps (large; for offline fusion experiments on subsets — the fusion rule is
the pipeline's weakest link and the first thing worth improving).

**Depth** — per sample: all 6 images resized/padded to Metric3D's 616×1064
canvas, one batched forward, un-pad, upsample to native resolution, scale
canonical depth to metric via `fx·scale/1000`, clamp to [0, 300] m.
Metric3D's torch.hub code does an unguarded `from mmcv.utils import
collect_env` (env logging only) and a few `from mmcv.utils import Config`;
`common.install_mmcv_shim()` registers a minimal mmengine-backed stand-in so
that mmcv — a heavy CUDA-compiled dependency — stays out of the container.
It is a no-op if a real mmcv is installed.

Deliberate changes vs. the original scripts (each has a compat flag):

| change | rationale | reproduce original |
|---|---|---|
| RGB input to Metric3D | original fed cv2 BGR into RGB normalization (likely a bug) | `--color-order bgr` |
| sky → 18 | 0 loses information, 17 collides with "free" | `--sky-label 0` |
| one camera order + embedded `cam_order` | two silent different orders originally | `--layout legacy` |
| Metric3D confidence saveable | original discarded it; free depth-loss weight | (off by default) |
| DDP removed | was vestigial (world_size 1 + `.module`); sharding does the parallelism | — |
| zeros for box-less images | original wrote −1, which poisons the semantic loss | — |
| atomic writes + per-root meta.json | resumable job arrays, no truncated files, full provenance | — |

Bit-exactness caveat: even with all compat flags, expect a few boundary
pixels to differ from labels generated years ago — different
torch/CUDA/library versions shift DINO/SAM outputs slightly.
`verify_outputs.py --legacy-mask-root/--legacy-depth-root` quantifies this.

## infos pkl schema (what your dataloader needs)

`create_infos.py` writes `{"infos": [...], "metadata": {"version": ...}}`,
sorted by `(scene_name, timestamp)` so `index ± t` adjacency works within a
scene (guard on `scene_token`). Per sample:

```
token, scene_token, scene_name, timestamp, lidar_idx
lidar_path, panoptic_path, occ_path            # occ3d GT for evaluation
lidar2ego_{translation,rotation}               # quaternion [w,x,y,z]
ego2global_{translation,rotation}              # lidar-time ego pose
cams: {CAM_*: data_path, cam_intrinsic [3,3], sample_data_token,
       sensor2ego_{translation,rotation},      # camera extrinsics
       ego2global_{translation,rotation},      # image-time ego pose
       timestamp}
# train/val only:
gt_boxes [N,7], gt_names, gt_velocity [N,2],   # lidar frame, mmdet3d
                                               #   convention: xyz, w l h,
                                               #   -yaw - pi/2
num_lidar_pts, num_radar_pts, valid_flag
ann_infos: (boxes [N,9], labels)               # BEVDet ego-frame boxes:
                                               #   xyz, l w h, yaw, vx, vy
```

Same fields as the bevdetv2 pkls GaussianFlowOcc uses (minus the unused
`sweeps` content and `sensor2lidar_*`, which are derivable from the poses),
so they also work as drop-in replacements there.

`create_infos_av2.py` writes the same structure for Argoverse 2, minus the
nuScenes-only fields (occ/panoptic paths, boxes, `ann_infos`) and plus
`cams.*.image_size`; see [Argoverse 2](#argoverse-2).

## Containers

`containers/Dockerfile` and `containers/pseudo_labels.def` are equivalent;
`containers/build.sh {docker|sif|sif-from-docker}` builds either. Notes:

* Needs a CUDA **devel** base image: GroundingDINO compiles a CUDA extension
  (`groundingdino._C`, deformable attention) at install time. The build sets
  `TORCH_CUDA_ARCH_LIST` so it compiles on GPU-less build machines, and
  fails loudly if the extension is missing.
* Compute nodes without internet: run `prefetch_weights.py` once on a login
  node with `TORCH_HOME`/`HF_HOME` on persistent storage, then bind those
  into the job (the SLURM scripts already do). Alternatively bake weights
  into the image with `--build-arg PREFETCH_WEIGHTS=1` (~5 GB extra).
* The package is baked into the image at `/opt/generate_pseudo_labels`; for
  development, bind-mount your checkout and prepend it to `PYTHONPATH`.

## Performance

The semantics stream is the expensive one. The original scripts ran **252
batch-1 DINO forwards per sample** (42 prompts × 6 cameras, sequentially).
This port restructures that with identical outputs:

* **Camera batching** (always on): all 6 cameras go through each prompt's
  pass as one batch → 42 batch-6 passes per sample. GroundingDINO contains
  only batch-independent layers (LayerNorm / FrozenBatchNorm), and batched
  captions are its native training path, so results are unchanged up to
  float nondeterminism.
* **Backbone once per image + text once per run** (default, `dino_cached.py`):
  the Swin backbone is prompt-independent and now runs once per camera
  batch; the BERT encoding of all prompts is computed once at startup. Only
  the text-conditioned transformer + heads loop over prompts. This
  replicates `GroundingDINO.forward()` internals, so it is guarded: on the
  first sample the cached path is compared against the stock forward and
  the run aborts on divergence (`--skip-cache-check` to skip,
  `--no-dino-cache` to fall back to the public forward entirely).
* **`--prompt-chunk K`**: additionally tiles K prompts per transformer pass
  (batch = 6K) — same outputs, more memory. Try 2–4 on a 40 GB card.
* **Prefetch**: the next sample's JPEG decode + resize runs on a background
  thread while the GPU works.
* **`--tf32`**: TF32 matmuls (Ampere+), near-exact, essentially free.
* **`--amp bf16`**: autocast SAM, roughly halves its compute — *not*
  bit-exact (argmax labels virtually identical; boundary pixels may flip).
  Validate on a subset with `verify_outputs.py` before a full run.
  GroundingDINO is *not* autocast by this flag and has its own
  (default-off) `--amp-dino`: its CUDA deformable-attention kernel is built
  with `AT_DISPATCH_FLOATING_TYPES`, so a half-precision input raises
  `ms_deform_attn_forward_cuda not implemented for BFloat16`. `--tf32` is
  the speed lever for DINO.

Combined, expect roughly 5–10× lower walltime than the original pipeline.
The SLURM script defaults to `--tf32 --amp bf16 --prompt-chunk 2` (drop
`--amp bf16` from `EXTRA_ARGS` if you need exact fp32 labels). After the
DINO speedups, the 6 SAM ViT-H encodes per sample become a comparable
cost — `--amp` is the main lever there.

## Sizing (nuScenes trainval, 34k samples × 6 cams)

Measured on one A100 with the SLURM defaults (`--tf32 --amp bf16
--prompt-chunk 2`), nuScenes 900×1600:

| stream | per sample | 34k samples | bytes/sample (native) | 34k samples |
|---|---|---|---|---|
| semantics (42 batch-6 DINO forwards + 6 SAM ViT-H encodes) | 7.1 s | 68 GPU-h | 8.6 MB (labels only) | 295 GB |
| depth (one batched forward) | 2.1 s | 20 GPU-h | 17.3 MB | 590 GB |

`--save-scores` doubles the semantics footprint; `--save-confidence` doubles
the depth footprint. `--scale 0.44` cuts both by ~5x (~57 GB + ~115 GB).
Shard wide (`--array=0-15%8`); tasks are resumable, so lost tasks — or a run
that fills the disk — cost nothing but the rerun.

## Adapting to other datasets (e.g. AV2, ZOD)

Both generators only consume the infos pkl: `scene_name`, `token`, and per
camera `data_path` + `cam_intrinsic` (that is the *complete* list — verified
by grepping every `sample[...]` access in both scripts). Write a small
`create_infos`-style script for your dataset that emits the same schema and
both pipelines work unchanged. Two things to pass on the command line:

* `--cameras <name> ...` — the camera keys to read from `cams` and the order
  they are stored in (default: `CAMERA_ORDER`, the nuScenes ring). It is
  written into the npz as `cam_order` and into the `_meta_*.json`, so files
  stay self-describing.
* one `--save-path` (and one index) **per resolution group**: both generators
  batch all cameras of a sample into a single forward pass, so their images
  must share one resolution. Groups cannot share a `--save-path` — the
  per-sample npz is keyed by `<scene>/<token>` only, so the second run would
  overwrite the first (the `_meta_*.json` guard turns that into a hard error).

### Argoverse 2

`create_infos_av2.py` is the AV2 index generator: one sample per lidar sweep,
`token` = lidar timestamp stem, `scene_name` = log id, cameras matched to the
sweep by nearest timestamp and each carrying the ego pose interpolated at its
own capture time. Both AV2 root layouts are handled (`<root>/<split>/<log>` as
the tarballs extract, and `<root>/sensor/<split>/<log>` as the devkit and
`AV2Dataset` expect).

AV2 mixes camera resolutions — `ring_front_center` is portrait 1550x2048, the
six other rings are landscape 2048x1550 — so the full chain is run twice, once
per group:

```bash
AV2=/datasets/argoverse2
LANDSCAPE="ring_front_left ring_front_right ring_side_left ring_side_right ring_rear_left ring_rear_right"

# 1. index, one per resolution group (add --max-logs 1 for a smoke test)
python -m generate_pseudo_labels.create_infos_av2 \
    --root-path $AV2 --split train --out-dir data/infos --relative-to $AV2 \
    --cameras $LANDSCAPE --tag av2-landscape
python -m generate_pseudo_labels.create_infos_av2 \
    --root-path $AV2 --split train --out-dir data/infos --relative-to $AV2 \
    --cameras ring_front_center --tag av2-front-center

# 2. semantics, one run per group, separate save paths
python -m generate_pseudo_labels.generate_semantics \
    --infos-pkl data/infos/av2-landscape_infos_train.pkl --cameras $LANDSCAPE \
    --save-path data/pseudo_labels/av2/semantics_landscape --data-root $AV2
python -m generate_pseudo_labels.generate_semantics \
    --infos-pkl data/infos/av2-front-center_infos_train.pkl --cameras ring_front_center \
    --save-path data/pseudo_labels/av2/semantics_front_center --data-root $AV2

# 3. depth — optional: not read by tools/lidar_pseudoseg, and queryocc's AV2 arm
#    consumes only the lidar pseudo-seg (its 2D streams are nuScenes-gated)
python -m generate_pseudo_labels.generate_depth \
    --infos-pkl data/infos/av2-landscape_infos_train.pkl --cameras $LANDSCAPE \
    --save-path data/pseudo_labels/av2/depth_landscape --data-root $AV2
python -m generate_pseudo_labels.generate_depth \
    --infos-pkl data/infos/av2-front-center_infos_train.pkl --cameras ring_front_center \
    --save-path data/pseudo_labels/av2/depth_front_center --data-root $AV2
```

Consumers are given both roots (`tools/lidar_pseudoseg`: `--semantics_root
<landscape> <front_center>`, searched in order per camera). For the lidar
pseudo-seg chain, generate the semantics at native resolution (`--scale 1.0`,
the default): the projection takes the image size from the mask but the
intrinsics from the index.

AV2 differences from the nuScenes index, both documented in the module
docstring:

* **No occ/box fields.** `occ_path`, `panoptic_path`, `sweeps`, `gt_boxes`,
  `gt_names`, `gt_velocity`, `num_lidar_pts`, `num_radar_pts`, `valid_flag`
  and `ann_infos` are simply absent — AV2 ships no Occ3D ground truth, and
  nothing in this package or in `tools/lidar_pseudoseg` reads them.
  `verify_infos.py` / `validate_against_lidar.py` are nuScenes-specific and do
  not run on this index.
* **Sweeps are stored in the ego frame**, so `lidar2ego_*` (the `up_lidar`
  calibration, recorded for completeness) must NOT be applied to the points.
  `timestamp` is in nanoseconds (nuScenes uses microseconds); it is only used
  for ordering. A sweep whose nearest camera frame is more than
  `--max-cam-offset-ms` (default 50 ms) away is dropped.
