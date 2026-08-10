#!/bin/bash
# Start the queryocc container in the background. Run from the repo root.
# Secrets and host data paths are read from .env (cp .env.example .env and fill it in).
# Then use containers/into_docker.sh to get a shell and containers/stop_docker.sh to stop.

set -e

if [ ! -f .env ]; then
    echo "No .env file found. Create one with: cp .env.example .env (then fill in your values)"
    exit 1
fi
source .env

if docker ps --format '{{.Names}}' | grep -qx queryocc; then
    echo "Container 'queryocc' is already running. Use containers/into_docker.sh to enter it."
    exit 0
fi

# Only mount paths that exist on this machine (external drives may be unplugged).
MOUNTS=(-v "$(pwd)":/queryocc)
add_mount() {
    if [ -n "$1" ] && [ -d "$1" ]; then
        MOUNTS+=(-v "$1":"$2")
    else
        echo "Skipping mount $2 ($1 not found)"
    fi
}
add_mount "$DATASETS_DIR" /datasets
add_mount "$OUTPUTS_DIR" /outputs
add_mount "$PSEUDO_SEMANTICS_DIR" /datasets/pseudo_labels/semantics
add_mount "$PSEUDO_DEPTH_DIR" /datasets/pseudo_labels/depth

# Persist the huggingface cache across container restarts.
mkdir -p "$HOME/.cache/huggingface"

# NVIDIA_DISABLE_REQUIRE: the runtime's strict check rejects CUDA 12.6 images on
# driver 535, but cu126 userspace officially supports drivers >= 525 (CUDA minor
# version compatibility), so the check is safe to skip.
docker run -d --rm \
    --name queryocc \
    --gpus all \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    --shm-size=64g \
    --ipc=host \
    --env-file .env \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    "${MOUNTS[@]}" \
    -p 8081:8081 \
    queryocc:latest \
    sleep infinity

echo "Container 'queryocc' started. Enter it with: ./containers/into_docker.sh"
