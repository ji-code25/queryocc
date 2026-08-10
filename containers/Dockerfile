ARG CUDA_VERSION=12.6.3
ARG OS_VERSION=22.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION

# metainformation
LABEL org.opencontainers.image.title="queryocc"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

# Variables used at build time.
## CUDA architectures for the JIT-compiled dvr raycaster (RayIoU metrics).
## 61: Pascal (local dev, e.g. TITAN Xp) | 75: Turing | 80/86: Ampere (A100/A40) | 89: Ada | 90: Hopper
ARG CUDA_ARCHITECTURES=90;89;86;80;75;61

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libeigen3-dev \
    openssh-client \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    vim-tiny \
    wget \
    less \
    && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# Upgrade pip and install build tooling.
# setuptools<80: kept for compatibility with older build tooling.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip "setuptools<80" pybind11 ninja

# Install python dependencies before copying the source code, so that source
# changes do not invalidate these (slow, large) layers.
COPY requirements/ /queryocc/requirements/
## Install torch first in its own layer (largest download, changes least often).
RUN python3.10 -m pip install --no-cache-dir \
    $(grep -E '^(torch|torchvision)==' /queryocc/requirements/requirements.txt) \
    --extra-index-url https://download.pytorch.org/whl/cu126
RUN python3.10 -m pip install --no-cache-dir -r /queryocc/requirements/requirements.txt

# Setup
RUN mkdir -p ~/.ssh && ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# Add source code to container
COPY . /queryocc

# Install queryocc itself.
WORKDIR /queryocc
RUN python3.10 -m pip install --no-cache-dir -e .

# Remove the CUDA forward-compatibility driver libs: the nvidia container runtime
# injects them on hosts with older drivers, but they only work on data-center GPUs
# (error 804 on GeForce/Titan). We rely on CUDA minor version compatibility instead
# (cu126 userspace runs on any >= 525 driver).
RUN rm -rf /usr/local/cuda/compat

# Bash as default entrypoint.
CMD /bin/bash -l
