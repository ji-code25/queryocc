# QueryOcc nuScenes full-16 training and evaluation

This record summarizes the 16-epoch QueryOcc training run and the full-validation sweep over
epochs 10 through 16. All reported metric values are multiplied by 100, matching the paper-table
convention.

## Run summary

| Item | Value |
|---|---|
| Training job | `18225549` (`queryocc_full16`) |
| Training account | `berzelius-2026-289` |
| Training hardware | 4 x NVIDIA A100-SXM4 80 GB (`fat`) |
| Training schedule | 16 epochs; intermediate validation disabled |
| Training time | 2026-11-12 18:23:36 to 2026-11-13 04:38:09 (10:14:33) |
| Evaluated checkpoints | epochs 10, 11, 12, 13, 14, 15, and 16 |
| Evaluation job | `18241523_[10-16]` (`queryocc_fullval`) |
| Evaluation account | `berzelius-2026-281` |
| Evaluation hardware | 1 x NVIDIA A100-SXM4 40 GB (`thin`) per checkpoint |
| Evaluation duration | 16:58 to 17:38 per checkpoint |
| Selection job | `18241524` (`queryocc_select`) |
| Selection metric | Semantic RayIoU |
| Selected checkpoint | **epoch 16** |
| Selected score | **28.9754 Semantic RayIoU** |

The training process reached `max_epochs=16` and wrote all requested checkpoints. Slurm marked
job `18225549` as `FAILED` only because the post-training manifest builder searched the wrong DDP
output root after `Trainer.fit` completed. The manifest path handling was repaired separately, and
all seven full-validation tasks and the selection job completed successfully.

## Full-validation results

| Epoch | Semantic RayIoU | Dynamic RayIoU | Binary Occupancy RayIoU | Semantic IoU | Dynamic IoU | Binary Occupancy IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 27.9812 | 28.8835 | 44.8682 | 24.9914 | 18.6889 | 55.3436 |
| 11 | 27.6062 | 28.0978 | 45.7613 | 25.0373 | 18.7820 | **56.2513** |
| 12 | 28.2241 | 28.4606 | **46.3006** | 24.8454 | 18.6254 | 53.9148 |
| 13 | 28.6231 | 29.5728 | 45.6819 | 25.0270 | 18.8784 | 55.7279 |
| 14 | 28.8937 | **29.9443** | 46.0143 | 25.1676 | 19.1055 | 55.2978 |
| 15 | 28.8803 | 29.8687 | 45.9604 | 25.2675 | 19.1703 | 55.2253 |
| 16 | **28.9754** | 29.8738 | 46.1014 | **25.3516** | **19.2306** | 55.3181 |

The best value in each column is bold. Epoch 16 is selected because Semantic RayIoU is the
checkpoint-selection metric; individual secondary metrics peak at different epochs.

## Metric definitions

| Reported metric | Logged key |
|---|---|
| Semantic RayIoU | `test/occ3d/metrics/occ3d_sem_rayiou/semantic_rayiou/overall_mean` |
| Dynamic RayIoU | `test/occ3d/metrics/occ3d_sem_rayiou/semantic_rayiou/overall_mean_dynamic` |
| Binary Occupancy RayIoU | `test/occ3d/metrics/occ3d_binary_rayiou/max_binary_rayiou` |
| Semantic IoU | `test/occ3d/metrics/occ3d_multiclass_iou/camera/iou/mean` |
| Dynamic IoU | `test/occ3d/metrics/occ3d_multiclass_iou/camera/iou/mean_dynamic` |
| Binary Occupancy IoU | `test/occ3d/metrics/occ3d_binary_iou/camera/max_iou` |

`Binary Occupancy RayIoU` and `Binary Occupancy IoU` are distinct metrics. The former evaluates
binary occupancy along rays; the latter evaluates binary voxel overlap inside the camera mask.

## Cluster artifacts

- Run root: `/proj/agp/projects/queryocc/output/experiments/queryocc-nusc-full16-18225549`
- Checkpoints: `/proj/agp/projects/queryocc/output/experiments/queryocc-nusc-full16-18225549/2026-11-12_172358/checkpoints`
- Per-epoch results: `/proj/agp/projects/queryocc/output/experiments/queryocc-nusc-full16-18225549/full_validation/epoch-{10..16}.json`
- Evaluation summary: `/proj/agp/projects/queryocc/output/experiments/queryocc-nusc-full16-18225549/evaluation_summary.json`
- Training log: `/proj/agp/users/x_jilan/logs/18225549.out`
- Evaluation logs: `/proj/agp/users/x_jilan/logs/18241523_<epoch>.out`
