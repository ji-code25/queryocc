# Container for BOTH pseudo-label pipelines (semantics + depth) and infos
# creation. Build from the package root:
#
#   cd generate_pseudo_labels
#   docker build -f containers/Dockerfile -t generate-pseudo-labels:latest .
#
# A CUDA *devel* base is required: GroundingDINO compiles a CUDA extension
# (multi-scale deformable attention) at install time. TORCH_CUDA_ARCH_LIST
# makes that build work on machines without a GPU (e.g. CI / login nodes).
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    # cover V100 / A100 / RTX30xx / RTX40xx / H100
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    # Metric3D's DINOv2 backbone falls back to plain attention without
    # xformers; we skip xformers to avoid pinning it to the torch build.
    XFORMERS_DISABLED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# GroundingDINO (compiles the _C CUDA extension) + SAM, both from upstream.
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/IDEA-Research/GroundingDINO.git" \
    && pip install --no-cache-dir \
        "git+https://github.com/facebookresearch/segment-anything.git"

# Fail the build loudly if the CUDA extension did not compile — without it
# GroundingDINO silently degrades to CPU-only. (torch must be imported first:
# it is what pulls libc10.so into the process.)
RUN python -c "import torch; from groundingdino import _C; print('groundingdino CUDA ext: OK')"

# Bake the package into the image; bind-mount your checkout over it for
# development (see scripts/slurm_*.sh).
COPY . /opt/generate_pseudo_labels/generate_pseudo_labels
ENV PYTHONPATH=/opt/generate_pseudo_labels

# Optional: pre-download all model weights into the image (~5 GB). Enable
# with --build-arg PREFETCH_WEIGHTS=1 if your compute nodes have no internet
# and you prefer a fat image over a bind-mounted cache.
ARG PREFETCH_WEIGHTS=0
RUN if [ "$PREFETCH_WEIGHTS" = "1" ]; then \
        python -m generate_pseudo_labels.prefetch_weights \
            --sam-dir /opt/weights/sam; \
    fi

WORKDIR /workspace
