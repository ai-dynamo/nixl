#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This script is used to test the GPU support of the NIXL library.
# Running on slurm GPU cluster

set -ex

DOCKER_IMAGE=${TEST_DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:25.02-py3"}
GIT_REF=${GIT_REF:-"main"}

# temporary fix for installing nvidia-container-toolkit on rock cluster -
# TODO: remove this once this ticket is closed: https://jirasw.nvidia.com/browse/SWXI-331
if ! command -v nvidia-ctk &> /dev/null; then
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
        sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    sudo yum install -y nvidia-container-toolkit nvidia-container-runtime libnvidia-container1 libnvidia-container-tools
else
    echo "nvidia-container-toolkit is already installed"
fi
if ! sudo nvidia-ctk runtime configure --runtime=docker --status &>/dev/null; then
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Enhanced error handling
handle_error() {
    local exit_code=$?
    log "Error occurred in ${BASH_SOURCE[1]}:${BASH_LINENO[0]}"
    log "Exit code: ${exit_code}"
    log "Command: ${BASH_COMMAND}"

    # Collect debug information
    log "Collecting debug information..."
    nvidia-smi || true
    docker info || true
    ibv_devinfo || true

    # Cleanup
    log "Cleaning up..."
    sudo docker rm -f "${CONTAINER_ID}" 2>/dev/null || true

    exit ${exit_code}
}

# Set up error handling
trap handle_error ERR
trap 'sudo docker rm -f "${CONTAINER_ID}" 2>/dev/null || true' EXIT

# Enhanced container creation with health check
log "Starting container with image: ${DOCKER_IMAGE}"
CONTAINER_ID=$(sudo docker run -dt \
    --ulimit memlock=-1:-1 \
    --privileged \
    --net=host \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --gpus all \
    --device=/dev/infiniband \
    --device=/dev/gdrdrv \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -w /tmp \
    "${DOCKER_IMAGE}")

# TODO: find a better way to wait for the container to be ready - there is no nvidia-smi in the current container
# Wait for container to be healthy
# log "Waiting for container to be ready..."
# timeout 60s bash -c "while ! sudo docker exec ${CONTAINER_ID} nvidia-smi >/dev/null 2>&1; do sleep 1; done" || {
#     log "Container failed to become healthy"
#     exit 1
# }

# Execute commands with enhanced logging
log "Cloning repository..."
sudo docker exec "${CONTAINER_ID}" /bin/bash -c "git clone https://github.com/ai-dynamo/nixl && git -C nixl checkout ${GIT_REF}"

log "Building NIXL..."
sudo docker exec -w /tmp/nixl "${CONTAINER_ID}" /bin/bash -c ".gitlab/build.sh /opt/nixl /usr/local"

log "Running tests..."
sudo docker exec -w /tmp/nixl "${CONTAINER_ID}" /bin/bash -c ".gitlab/test_cpp.sh /opt/nixl"
sudo docker exec -w /tmp/nixl "${CONTAINER_ID}" /bin/bash -c ".gitlab/test_python.sh /opt/nixl"

log "Tests completed successfully"
