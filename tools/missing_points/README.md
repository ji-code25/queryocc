# tools/missing_points

> **Verified against the published cache.** All 3,935 nuScenes sweeps regenerated from
> the mini split are byte-identical to the cache the paper's results were trained on. AV2
> matches on intensity/time/channel exactly and on xyz for 92.8% of rows, the rest within
> 2.5e-4 m (float32 pose interpolation in the AV2 devkit; nuScenes interpolates in float64
> and is exact).

Generates the **free-ray ("missing points") cache** — synthetic points placed along lidar
rays that returned *nothing*.

A ray that comes back empty is evidence of free space out to the sensor's range, so these
points are exactly the negative evidence QueryOcc's query sampler needs. They are stored
per sweep and read at training time as extra "missing rays" alongside the out-of-range
points.

## Why

Every preset sets `add_missing_points=True` (`queryocc/config/methods/queryocc.py`), and
the lidar-supervised arms — `queryocc-lidar-nusc`, `queryocc-joint-nusc`,
`queryocc-nusc-av2` — depend on it for their free-space supervision. Without the cache the
dataloader silently falls back to only the out-of-range points: training still runs, the
loss still looks reasonable, and free-space supervision is quietly much weaker. Both
readers (`_load_missing_points` in `queryocc/data/datasets/nuscenes_dataset.py` and
`argoverse2.py`) warn once when they find nothing.

The camera-only arm (`queryocc-nusc`) does not need this tool.

## Output (do not change — training reads it)

One `.npy` per lidar sweep, `float32 [N, 6]`:

```
columns:  [x, y, z, intensity, time, channel]
          time    = per-point offset in seconds relative to the sweep timestamp
          channel = laser index

nuScenes:    <out_dir>/<lidar_stem>_missing.npy            # flat, no subdirectories
             (from "<lidar>.pcd.bin" -> "<lidar>_missing.npy")
Argoverse 2: <out_dir>/<split>/<log_id>/<timestamp_ns>_missing.npy
```

The loaders read columns `0:3` and `4:5` only, but the array must be at least 5 wide.
Frames differ by dataset and are **not** interchangeable:

| Dataset | Frame | Dummy range | Channels |
|---|---|---|---|
| nuScenes | LIDAR_TOP **sensor** frame | 150 m | 2–31 (0, 1 skipped) |
| Argoverse 2 | **ego** frame | 500 m | 0–63 (up + down lidar, down offset by 32) |

## Inputs

| Input | What | Default |
|---|---|---|
| dataset root | raw lidar sweeps + calibration | `paths.NUSC_ROOT` / `paths.AV2_ROOT` (`$NUSC_ROOT` / `$AV2_ROOT`) |
| out dir | where the cache is written | `paths.NUSC_MISSING_POINTS_ROOT` / `paths.AV2_MISSING_POINTS_ROOT` |

No pseudo labels, no index pkl, no infos cache — only the raw dataset. The generators are
**CPU-only** (numpy, torch CPU ops, tqdm, pyquaternion, the two devkits): no CUDA, no
compiled extensions, so they run in a small CPU environment rather than the training
container.

## Run

```bash
# nuScenes (all LIDAR_TOP sample_data: keyframes AND 20 Hz sweeps)
python -m tools.missing_points.engine --dataset nuscenes --version v1.0-trainval \
    --num_workers 8

# Argoverse 2 (per split)
python -m tools.missing_points.engine --dataset argoverse2 --split train --num_workers 8
python -m tools.missing_points.engine --dataset argoverse2 --split val   --num_workers 8
```

On a cluster, `scripts/create_missing_points.sh` wraps this with the site paths from
`scripts/env.local.sh` and runs in a venv (or a container if `CONTAINER` is set):

```bash
scripts/create_missing_points.sh --dataset nuscenes --version v1.0-trainval
sbatch --gpus-per-node=T4:1 scripts/create_missing_points.sh --dataset argoverse2 --split train
```

Other flags: `--out_dir`, `--nuscenes_data` / `--argoverse_data`, `--override true|false`,
`--num_workers`. Both AV2 root layouts are accepted (`<root>/<split>/<log>` and
`<root>/sensor/<split>/<log>`).

**Resumable**: a sweep whose output already exists is skipped, so an interrupted or
timed-out run costs only the unfinished tail. Pass `--override true` to force regeneration.
Work is parallelised one scene (nuScenes) / log (AV2) per worker thread — threads rather
than processes because the nuScenes devkit object is not cheaply picklable.

## Cost

Measured (single thread, spinning disk — cluster storage will be faster; both are I/O-bound):

| Dataset | Rate | Bytes/sweep | Full split |
|---|---|---|---|
| nuScenes | 0.78 s/sweep | 115 kB | ~331.9k sweeps -> ~72 h single-thread (~5 h at 16 workers), ~38 GB |
| Argoverse 2 | 32-54 s/log (157 sweeps) | 0.6-0.9 MB | train (700 logs) -> ~6-11 h single-thread, ~66-99 GB |

The nuScenes cache covers *every* LIDAR_TOP sample_data, not just keyframes, because the
offset-frame sampler draws from the full 20 Hz stream.

**Note:** AV2 output is `float64` (nuScenes is `float32`). That matches the published cache
and `argoverse2.py::_load_missing_points` casts downstream, so it is kept as-is.

## Where the output lands / how training finds it

| Dataset | Default out dir | Env var | Read by training as |
|---|---|---|---|
| nuScenes | `/datasets/nuscenes_missing_points` | `QUERYOCC_NUSC_MISSING_POINTS_ROOT` | `nusc_cached_missing_points_dir` |
| Argoverse 2 | `/datasets/av2_missing_points` | `QUERYOCC_AV2_MISSING_POINTS_ROOT` | `av2_cached_missing_points_dir` |

The SLURM launchers bind the host directories (`NUSC_MISSING_POINTS_DIR` /
`AV2_MISSING_POINTS_DIR` in `scripts/env.local.sh`) onto those container defaults, so the
generator and the trainer cannot drift apart.
