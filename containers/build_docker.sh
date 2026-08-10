#!/bin/bash
# Build the queryocc image. Run from the repo root.
#
# Useful build args (see containers/Dockerfile):
#   --build-arg CUDA_ARCHITECTURES="80"   architectures the dvr raycaster JIT-compiles for
#                                         (80 = A100; fewer = smaller TORCH_CUDA_ARCH_LIST)

DOCKER_BUILDKIT=1 docker build -t queryocc:latest -f containers/Dockerfile "$@" .
