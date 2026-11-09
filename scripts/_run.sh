#!/bin/bash
# Apptainer runner, called by the train*.sh SBATCH wrappers and the array
# launchers. Site-specific host paths live in the git-excluded
# scripts/env.local.sh (same mechanism as generate_pseudo_labels): every
# variable below has a default and can be overridden there.
#
# Inside the container the data always lands at the fixed /datasets layout that
# queryocc/common/paths.py defaults to, so no path config is needed on the CLI.
name=${1:?"No name given"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${QUERYOCC_SKIP_ENV_LOCAL:-}" ] && [ -f "$SCRIPT_DIR/env.local.sh" ]; then
    source "$SCRIPT_DIR/env.local.sh"
fi

# W&B: leave WANDB_ENTITY unset to log to your own default entity (your personal
# account, resolved from ~/.netrc). Set it in scripts/env.local.sh only to log into a
# shared team instead. It is intentionally never committed.
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY

# Host-side locations (override in scripts/env.local.sh).
PROJECT_ROOT=${PROJECT_ROOT:?"Set PROJECT_ROOT in scripts/env.local.sh"}
CONTAINER=${CONTAINER:-"$PROJECT_ROOT/containers/queryocc.sif"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT"}
# DINOv3-pretrained ConvNeXt backbone weights (see README "Site configuration").
DINOV3_CKPT_DIR=${DINOV3_CKPT_DIR:-"$PROJECT_ROOT/ckpts"}
# Per-point lidar pseudo-segmentation labels (tools/lidar_pseudoseg), needed by the
# lidar-supervised arms.
PSEUDOSEG_ROOT=${PSEUDOSEG_ROOT:-"$PROJECT_ROOT/pseudo_labeler/pseudoseg"}
NUSC_DATASET_ROOT=${NUSC_DATASET_ROOT:-"$PROJECT_ROOT/nuscenes"}
# panoptic/ and lidarseg/ are separate nuScenes add-on downloads, so they get their own
# variables: point these elsewhere if your shared dataset copy does not include them.
NUSC_PANOPTIC=${NUSC_PANOPTIC:-"$NUSC_DATASET_ROOT/panoptic"}
NUSC_LIDARSEG=${NUSC_LIDARSEG:-"$NUSC_DATASET_ROOT/lidarseg"}
NUSC_OCC3D_DATASET_ROOT=${NUSC_OCC3D_DATASET_ROOT:-"$PROJECT_ROOT/occ3d-nuscenes"}
NUSC_INFOS_ROOT=${NUSC_INFOS_ROOT:-"$PROJECT_ROOT/infos"}
PSEUDO_LABEL_ROOT=${PSEUDO_LABEL_ROOT:-"$PROJECT_ROOT/pseudo_labels/bevdetv2-nuscenes"}
# The two streams may be produced/stored independently. PSEUDO_LABEL_ROOT remains
# the backwards-compatible common parent, while these two variables let a site
# bind the exact Grounded-SAM and Metric3D directories it already has.
PSEUDO_SEMANTICS_ROOT=${PSEUDO_SEMANTICS_ROOT:-"$PSEUDO_LABEL_ROOT/semantics"}
PSEUDO_DEPTH_ROOT=${PSEUDO_DEPTH_ROOT:-"$PSEUDO_LABEL_ROOT/depth"}
NUSC_MISSING_POINTS_DIR=${NUSC_MISSING_POINTS_DIR:-"$PROJECT_ROOT/nuscenes_missing_points_cache"}
AV2_DATASET_ROOT=${AV2_DATASET_ROOT:-"$PROJECT_ROOT/av2"}
AV2_MISSING_POINTS_DIR=${AV2_MISSING_POINTS_DIR:-"$PROJECT_ROOT/av2_missing_points_cache"}

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/nccl_dumps"

# Ask the OS for a free port; $RANDOM alone can roll a privileged port (<1024)
# -> EACCES on the DDP socket bind.
find_free_port() {
  python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
export MASTER_PORT=$(find_free_port)

# Full name including slurm job id
full_name=$name-$SLURM_JOB_ID

# Normally the container runs the trainer. Set ENTRYPOINT to run something else under the
# exact same image and bind mounts -- diagnostics that must see the data exactly the way
# training does, e.g.
#   ENTRYPOINT="python -u -m tools.bench_dataloader --preset queryocc-nusc" \
#       sbatch scripts/train.sh bench
# The trainer's own arguments are not passed in that case; ENTRYPOINT carries its own.
if [ -n "${ENTRYPOINT:-}" ]; then
    read -r -a COMMAND <<< "$ENTRYPOINT"
else
    COMMAND=(python -u queryocc/train.py ${2} --exp-name="$full_name" \
        --trainer.accumulate_grad_batches=2 "${@:3}")
fi

apptainer exec --nv \
    --env PYTHONNOUSERSITE=1 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env NUMEXPR_NUM_THREADS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
    --env TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
    --env TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/outputs/nccl_dumps/${full_name}_ \
    --env QUERYOCC_DINOV3_CKPT_DIR=/dinov3_ckpts \
    --env QUERYOCC_NUSC_PSEUDOSEG_ROOT=/pseudoseg \
    --bind $PWD:/queryocc \
    --bind $OUTPUT_DIR:/outputs \
    --bind $DINOV3_CKPT_DIR:/dinov3_ckpts \
    --bind $PSEUDOSEG_ROOT:/pseudoseg \
    --bind $NUSC_DATASET_ROOT/samples/:/datasets/nuscenes/samples \
    --bind $NUSC_DATASET_ROOT/sweeps/:/datasets/nuscenes/sweeps \
    --bind $NUSC_DATASET_ROOT/v1.0-mini/:/datasets/nuscenes/v1.0-mini \
    --bind $NUSC_DATASET_ROOT/v1.0-test/:/datasets/nuscenes/v1.0-test \
    --bind $NUSC_DATASET_ROOT/v1.0-trainval/:/datasets/nuscenes/v1.0-trainval \
    --bind $NUSC_DATASET_ROOT/maps/:/datasets/nuscenes/maps \
    --bind $NUSC_PANOPTIC:/datasets/nuscenes/panoptic \
    --bind $NUSC_LIDARSEG:/datasets/nuscenes/lidarseg \
    --bind $NUSC_OCC3D_DATASET_ROOT:/datasets/occ3d-nuscenes/ \
    --bind $NUSC_INFOS_ROOT:/datasets/nuscenes/infos/ \
    --bind $PSEUDO_SEMANTICS_ROOT:/datasets/pseudo_labels/semantics \
    --bind $PSEUDO_DEPTH_ROOT:/datasets/pseudo_labels/depth \
    --bind $NUSC_MISSING_POINTS_DIR:/datasets/nuscenes_missing_points \
    --bind $AV2_DATASET_ROOT:/datasets/argoverse2/ \
    --bind $AV2_MISSING_POINTS_DIR:/datasets/av2_missing_points \
    --pwd /queryocc \
    $CONTAINER \
    "${COMMAND[@]}"

# Interactive mode:
# srun -A <account> -n 1 -t 01:30:00 --gpus-per-node=1 --pty bash
# apptainer shell --nv --bind $PWD:/queryocc --pwd /queryocc $CONTAINER
