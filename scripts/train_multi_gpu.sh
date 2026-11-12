#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4
#SBATCH --time 1-12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --job-name=queryocc
#
# Multi-GPU (single node, DDP) training. If your cluster requires a type-qualified
# GPU request, pass it at submit time (it overrides the directive above):
#   sbatch --gpus-per-node=A100:4 scripts/train_multi_gpu.sh <run-name> <preset>


# Dataset version: override with DATASET_VERSION=mini for smoke tests.
DATASET_VERSION=${DATASET_VERSION:-full}

NUM_GPUS=${NUM_GPUS:-4}
# datamodule.num-workers is PER DDP RANK: each rank spawns its own worker pool, so do
# not multiply by the GPU count — oversubscribing the node's cores collapses throughput
# even while GPU utilization still reads 100%.
WORKERS_PER_GPU=${WORKERS_PER_GPU:-5}

scripts/_run.sh $@ --trainer.devices=$NUM_GPUS --datamodule.num-workers=$WORKERS_PER_GPU --datamodule.version=$DATASET_VERSION


#
#EOF
