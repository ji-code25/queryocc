# tools/lidar_pseudoseg

Offline generation of **per-point lidar semantic pseudo-labels**: the camera semantic
masks from `generate_pseudo_labels` are projected onto each sample's lidar point cloud,
so every lidar point gets one nuScenes-lidarseg class id.

Why: the lidar-supervised arms (`queryocc-lidar-nusc`, `queryocc-joint-nusc`,
`queryocc-nusc-av2`) supervise occupancy with *labelled lidar points*. Without ground
truth lidarseg they read these files instead — see `_get_pseudo_panoptic_labels` in
`queryocc/data/datasets/nuscenes_dataset.py` and `queryocc/data/datasets/argoverse2.py`.
The camera-only arms do not need this tool.

## Output (do not change — training reads it)

One file per keyframe, raw `uint8`, one label per point, in point-cloud order:

```
nuScenes:     <pseudoseg_root>/<pseudo_label_name>[_fused]/{lidar_idx}_pseudoseg.bin
Argoverse 2:  <pseudoseg_root>/<pseudo_label_name>[_fused]/<scene_name>/{lidar_token}_pseudoseg.bin
```

Values are nuScenes-lidarseg class ids (`NUSCENES_IDX_TO_LABELS_MAP`, 0 = noise,
1–16 = classes, 17 = free); points no camera saw are written as `free` (17), which the
dataloader turns into "invalid" (unsupervised) points. With `--with_score true` an extra
`{lidar_idx}_pseudoseg_scores.bin` (`float16`, one score per point) is written next to it;
training does not read it.

`--scenewise_fusion true` additionally writes the `_fused` directory: point clouds of a
whole scene are accumulated in the middle frame, a voxel majority vote (1 m) settles
static-vs-dynamic and the static class, and the result is scattered back per sample.
**This is the variant training uses** (`panoptic_pseudo_label_name` defaults to
`delta_fused`).

## Inputs

| input | what | default |
|---|---|---|
| infos pkl | dataset index from `generate_pseudo_labels.create_infos` (nuScenes) or `.create_infos_av2` (AV2). Needs `token`, `scene_name`, `lidar_path`, `lidar2ego_*` (nuScenes only), `ego2global_*`, `cams.*.{data_path,cam_intrinsic,sensor2ego_*}`, plus `lidar_idx` on nuScenes | `paths.INFOS_DIR/bevdetv2-nuscenes{-mini}_infos_{split}.pkl` / `<argoverse_data>/av2_infos_{split}.pkl` |
| semantics npz | `<semantics_root>/<scene_name>/<token>.npz` with `labels` int8 `[n_cams,H,W]` (+ optional `scores`) and `cam_order` | `paths.PSEUDO_SEMANTICS_ROOT` (`QUERYOCC_PSEUDO_SEMANTICS_ROOT`) |
| dataset root | lidar point clouds (also resolves relative paths from the pkl) | `paths.NUSC_ROOT` / `paths.AV2_ROOT` |

Generate the semantics at **native resolution** (`generate_semantics --scale 1.0`): the
projection takes the image size from the mask but uses the un-scaled intrinsics from the
infos pkl, so a downscaled mask would misalign.

## Run

nuScenes (from the repo root, with the venv active):

```bash
python -m tools.lidar_pseudoseg.engine \
    --dataset nuscenes \
    --version v1.0-trainval \
    --split train \
    --semantics_root /datasets/pseudo_labels/semantics \
    --nuscenes_data /datasets/nuscenes \
    --pseudo_label_name delta \
    --scenewise_fusion true
```

Argoverse 2 — the index comes from `create_infos_av2`, and because AV2's ring cameras do
not share a resolution its semantics are generated one resolution group at a time, so
`--semantics_root` takes both roots (searched in order, per camera):

```bash
LANDSCAPE="ring_front_left ring_front_right ring_side_left ring_side_right ring_rear_left ring_rear_right"

python -m generate_pseudo_labels.create_infos_av2 \
    --root-path /datasets/argoverse2 --split train \
    --out-dir /datasets/argoverse2 --relative-to /datasets/argoverse2 --tag av2
# (+ the two per-group indices and the two generate_semantics runs:
#  see generate_pseudo_labels/README.md, "Argoverse 2")

python -m tools.lidar_pseudoseg.engine \
    --dataset argoverse2 \
    --split train \
    --infos_pkl /datasets/argoverse2/av2_infos_train.pkl \
    --semantics_root /datasets/pseudo_labels/av2/semantics_landscape \
                     /datasets/pseudo_labels/av2/semantics_front_center \
    --argoverse_data /datasets/argoverse2 \
    --pseudo_label_name av2_delta \
    --scenewise_fusion true
```

The index passed to the engine must contain **all** cameras to project (`--tag av2` above,
i.e. the default all-seven index), while the per-group indices exist only to drive the
semantics generator. A sample is skipped when any of its cameras has no mask in any root.

Other flags: `--out_dir` (pseudoseg root, see below), `--infos_pkl`, `--override true`
(recompute existing files), `--with_score true`, `--num_workers N` (thread pool, default
16). The run is resumable: existing `*_pseudoseg.bin` files are skipped, and a scene is
skipped in the fusion pass when all of its fused files already exist.

AV2 specifics: the output stem is the lidar timestamp (the index's `token`); sweeps are
read from `lidar_path` and are already in the **ego** frame, so `lidar2ego_*` is not
applied; camera extrinsics come from `sensor2ego_*` (inverted to project ego → camera) and
the pinhole model is used instead of Kannala-Brandt. The index must follow the
`create_infos_av2` schema (a bare-list container is tolerated, but the per-sample fields
must match).

## Where the output lands / how training finds it

`--out_dir` (config `pseudoseg_root`) is the directory that *contains* the
`<pseudo_label_name>` and `<pseudo_label_name>_fused` directories. Defaults come from
`queryocc/common/paths.py`:

| dataset | default `--out_dir` | env var | read by training as |
|---|---|---|---|
| nuScenes | `paths.NUSC_PSEUDOSEG_ROOT` (`/outputs/pseudo_labeler/pseudoseg`) | `QUERYOCC_NUSC_PSEUDOSEG_ROOT` | `NUSC_PSEUDOSEG_ROOT/<panoptic_pseudo_label_name>/{sample_data_token}_pseudoseg.bin` |
| Argoverse 2 | `paths.AV2_PSEUDOSEG_ROOT.parent` (`/outputs/pseudo_labeler/pseudoseg`) | `QUERYOCC_AV2_PSEUDOSEG_ROOT` | `AV2_PSEUDOSEG_ROOT/<scene>/{frame}_pseudoseg.bin` (the env var points at the `*_fused` directory itself) |

So with the defaults, `--pseudo_label_name delta --scenewise_fusion true` produces exactly
what the dataset config's default `panoptic_pseudo_label_name = "delta_fused"` expects, and
`--pseudo_label_name av2_delta --scenewise_fusion true` produces the default
`AV2_PSEUDOSEG_ROOT = .../pseudoseg/av2_delta_fused`. Set the env vars (e.g. in the
git-excluded `scripts/env.local.sh`) to move them elsewhere; both the generator and the
training code read the same variables.
