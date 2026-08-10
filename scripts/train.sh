#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --time 4-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --job-name=queryocc
#
# Single-GPU training. If your cluster requires a type-qualified GPU request,
# pass it at submit time (it overrides the directive above):
#   sbatch --gpus-per-node=A100:1 scripts/train.sh <run-name> <preset>


# Dataset version: override with DATASET_VERSION=mini for smoke tests.
DATASET_VERSION=${DATASET_VERSION:-full}

# Dataloader workers; tune to the CPU cores of your allocation.
NUM_WORKERS=${NUM_WORKERS:-16}

scripts/_run.sh $@ --datamodule.num-workers=$NUM_WORKERS --datamodule.version=$DATASET_VERSION


#
#EOF
