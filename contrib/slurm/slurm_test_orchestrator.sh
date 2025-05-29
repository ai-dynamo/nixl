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

set -eE -o pipefail

# Errors trap
export PS4='+ ${BASH_SOURCE}:${LINENO}: '
exit_code=0
trap 'exit_code=$?; echo "ERROR: command \"${BASH_COMMAND}\" exited with status ${exit_code} @ ${BASH_SOURCE}:${LINENO}" >&2; exit ${exit_code}' ERR

usage() {
    echo "Usage: $0 <test_cmd> [-p <partition>] [-i <docker_image>] [-t <timeout>]"
    echo "Example: $0 \".gitlab/test_cpp.sh /opt/nixl\" -p rock -t 02:00:00"
    echo "Options:"
    echo "  -p <partition>    Slurm partition (default: rock)"
    echo "  -i <docker_image> Docker image to use for tests"
    echo "  -t <timeout>      Job timeout in HH:MM:SS format (default: 01:00:00)"
    exit 1
}

# Validate required parameter
if [ -z "$1" ]; then
    echo "Error: Test command is required"
    usage
fi

TEST_CMD="$1"
shift

PARTITION="${SLURM_PARTITION:-rock}"
TIMEOUT="${SLURM_TIMEOUT:-01:00:00}"
DOCKER_IMAGE="harbor.mellanox.com/ucx/x86_64/pytorch:25.02-py3"
SLURM_JOB_NAME="NIXL-${JOB_NAME:-local}-${BUILD_ID:-$$}"
SLURM_CLIENT_IMAGE="harbor.mellanox.com/ucx/x86_64/slurm-client:v1.0"
SLURM_USER="svc-nixl"
SLURM_HOME="/labhome/svc-nixl"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

# CI environment variables
NIXL_INSTALL_DIR="${NIXL_INSTALL_DIR:-/opt/nixl}"
UCX_INSTALL_DIR="${UCX_INSTALL_DIR:-/usr/local}"

# Parse optional parameters
while getopts ":p:i:t:" opt; do
    case "$opt" in
    p) PARTITION="$OPTARG" ;;
    i) DOCKER_IMAGE="$OPTARG" ;;
    t) TIMEOUT="$OPTARG" ;;
    *) usage ;;
    esac
done

# TMP - sha1 should work
if [ -n "${sha1:-}" ]; then
    GIT_REF="$sha1"
elif [ -n "$GIT_COMMIT" ]; then
    GIT_REF="$GIT_COMMIT"
elif [ -n "$GIT_BRANCH" ]; then
    GIT_REF="$GIT_BRANCH"
elif [ -n "$CHANGE_BRANCH" ]; then
    GIT_REF="$CHANGE_BRANCH"
else
    GIT_REF="main"
fi

BUILD_AND_TEST_CMD="pip install --upgrade meson && \
    git clone https://github.com/ai-dynamo/nixl && \
    cd nixl && \
    git checkout ${GIT_REF} && \
    .gitlab/build.sh ${NIXL_INSTALL_DIR} ${UCX_INSTALL_DIR} && \
    ${TEST_CMD}"

DOCKER_RUN_CMD="sudo docker run --rm \
    --ulimit memlock=-1:-1 \
    --net=host \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --gpus all \
    --device=/dev/gdrdrv \
    -e LDFLAGS='-lpthread -ldl' \
    -e NIXL_PLUGIN_DIR=${NIXL_INSTALL_DIR}/lib/x86_64-linux-gnu/plugins \
    -e UCX_INSTALL_DIR=${UCX_INSTALL_DIR} \
    -e NIXL_INSTALL_DIR=${NIXL_INSTALL_DIR} \
    -e DEBIAN_FRONTEND=noninteractive \
    -w /tmp \
    ${DOCKER_IMAGE} \
    /bin/bash -c '${BUILD_AND_TEST_CMD}'"

# NVIDIA Container Toolkit installation for RHEL
NVIDIA_TOOLKIT_INSTALL="curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo && \
    sudo yum install -y nvidia-container-toolkit nvidia-container-runtime libnvidia-container1 libnvidia-container-tools && \
    sudo nvidia-ctk runtime configure --runtime=docker"

# Command for sbatch wrap
WRAP_CMD="set -ex; ${NVIDIA_TOOLKIT_INSTALL} && sudo systemctl restart docker && ${DOCKER_RUN_CMD}"

# Cleanup function
# shellcheck disable=SC2317  # Function is called by EXIT trap
cleanup() {
    if [ -n "${SLURM_CONFIG_DIR:-}" ]; then
        rm -rf "$SLURM_CONFIG_DIR"
    fi
}
trap cleanup EXIT

# Copy Slurm config
SLURM_CONFIG_DIR=$(mktemp -d)
echo "Copying Slurm configuration..."
scp -rq "${SSH_OPTS[@]}" "${SLURM_USER}@hpchead:/etc/slurm/" "$SLURM_CONFIG_DIR/" || {
    echo "Error: Failed to copy Slurm configuration"
    exit 1
}

# Disable SPANK plugins
find "$SLURM_CONFIG_DIR/slurm" -name "plugstack.conf*" -type f -print0 | \
    xargs -0 -r sed -i 's/^\(include\|required\|optional\)/#\1/' 2>/dev/null || true
find "$SLURM_CONFIG_DIR/slurm" -path "*/plugstack.conf.d/*.conf" -type f -print0 | \
    xargs -0 -r sed -i 's/^\(include\|required\|optional\)/#\1/' 2>/dev/null || true

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export variables for container
export SLURM_JOB_NAME
export PARTITION
export TIMEOUT
export WRAP_CMD

# Submit job using Slurm client container
echo "Starting Slurm job..."
echo "  Job Name: $SLURM_JOB_NAME"
echo "  Partition: $PARTITION"
echo "  Timeout: $TIMEOUT"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Git Reference: $GIT_REF"
echo "  NIXL Install Dir: $NIXL_INSTALL_DIR"
echo "  UCX Install Dir: $UCX_INSTALL_DIR"

DOCKER_CONTAINER_ID=$(docker run -d \
    -v "$SLURM_CONFIG_DIR/slurm:/etc/slurm:ro" \
    -v "$SCRIPT_DIR/slurm_client_runner.sh:/slurm_client_runner.sh:ro" \
    -v "$SLURM_HOME:$SLURM_HOME" \
    -e SLURM_JOB_NAME \
    -e PARTITION \
    -e TIMEOUT \
    -e WRAP_CMD \
    -e DEBUG="${DEBUG:-0}" \
    "$SLURM_CLIENT_IMAGE" /slurm_client_runner.sh)

if [ -z "$DOCKER_CONTAINER_ID" ]; then
    echo "Error: Failed to start Slurm client container"
    exit 1
fi

# Follow logs and get exit code
docker logs -f "$DOCKER_CONTAINER_ID"
EXIT_CODE=$(docker wait "$DOCKER_CONTAINER_ID")

# Set exit code
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Job completed successfully"
    exit 0
fi
echo "Job failed with exit code: $EXIT_CODE"
exit "$EXIT_CODE"
