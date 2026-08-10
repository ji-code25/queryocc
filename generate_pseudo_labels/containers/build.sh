#!/usr/bin/env bash
# Build the pseudo-label container.
#
# Usage (run from the generate_pseudo_labels package root):
#   containers/build.sh docker            # Docker image
#   containers/build.sh sif               # .sif directly via Apptainer
#   containers/build.sh sif-from-docker   # Docker image -> .sif
#
# Env overrides:
#   IMAGE_TAG   (default generate-pseudo-labels:latest)
#   SIF_PATH    (default pseudo_labels.sif)
#   PREFETCH_WEIGHTS=1  bake all model weights into the Docker image (~5 GB)
set -euo pipefail

MODE="${1:-docker}"
IMAGE_TAG="${IMAGE_TAG:-generate-pseudo-labels:latest}"
SIF_PATH="${SIF_PATH:-pseudo_labels.sif}"
PREFETCH_WEIGHTS="${PREFETCH_WEIGHTS:-0}"

if [ ! -f "containers/Dockerfile" ]; then
    echo "error: run this from the generate_pseudo_labels package root" >&2
    exit 1
fi

case "$MODE" in
docker)
    docker build \
        -f containers/Dockerfile \
        --build-arg PREFETCH_WEIGHTS="$PREFETCH_WEIGHTS" \
        -t "$IMAGE_TAG" \
        .
    echo "built docker image: $IMAGE_TAG"
    ;;
sif)
    # Direct Apptainer build (no Docker daemon needed, works on clusters).
    apptainer build --fakeroot "$SIF_PATH" containers/pseudo_labels.def
    echo "built: $SIF_PATH"
    ;;
sif-from-docker)
    docker build \
        -f containers/Dockerfile \
        --build-arg PREFETCH_WEIGHTS="$PREFETCH_WEIGHTS" \
        -t "$IMAGE_TAG" \
        .
    apptainer build "$SIF_PATH" "docker-daemon://${IMAGE_TAG}"
    echo "built: $SIF_PATH (from docker image $IMAGE_TAG)"
    ;;
*)
    echo "usage: containers/build.sh {docker|sif|sif-from-docker}" >&2
    exit 1
    ;;
esac
