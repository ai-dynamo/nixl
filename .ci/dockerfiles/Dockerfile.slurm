# NIXL GPU Test Environment Dockerfile
#
# This Dockerfile creates a GPU-enabled test environment for NIXL (NVIDIA I/O eXchange Layer)
# development and testing. It provides a containerized environment with:
#
# - NVIDIA cuda-dl-base image with CUDA support
# - Non-root user setup for security
# - Sudo access for package installation and system configuration
# - Optimized for CI/CD pipeline testing
#
# Usage:
#   docker build -f .ci/dockerfiles/Dockerfile.gpu_test -t nixl-gpu-test .
#   docker run --gpus all --privileged -it nixl-gpu-test
#
# Build arguments:
#   BASE_IMAGE: Base NVIDIA cuda-dl-base image (default: nvcr.io/nvidia/cuda-dl-base:25.10-cuda13.0-devel-ubuntu24.04)
#   _UID: User ID for the non-root user (default: 148069)
#   _GID: Group ID for the user (default: 30)
#   _LOGIN: Username (default: svc-nixl)
#   _GROUP: Group name (default: hardware)
#   _HOME: Home directory path (default: /home/svc-nixl)
#   WORKSPACE: Workspace directory path
#

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.10-cuda13.0-devel-ubuntu24.04

FROM ${BASE_IMAGE}

# Build arguments
ARG _UID=148069
ARG _GID=30
ARG NIXL_INSTALL_DIR
ARG NIXL_BUILD_DIR
ARG PRE_INSTALLED_ENV
ARG PRE_INSTALLED_NIXL_ENV
ARG PRE_INSTALLED_UCX_ENV
ARG UCX_VERSION=v1.20.x
ARG HAS_GPU=true

COPY --chown="${_UID}":"${_GID}" . .

RUN .gitlab/build.sh ${NIXL_INSTALL_DIR}
RUN sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]

# Default command
CMD ["/bin/bash"]
