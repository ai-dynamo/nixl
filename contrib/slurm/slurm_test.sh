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
trap 'exit_code=$?; echo "ERROR: command \"${BASH_COMMAND}\" exited with status ${exit_code} @ ${BASH_SOURCE}:${LINENO}" >&2; exit ${exit_code}' ERR

usage() {
    echo "Usage: $0 <test_cmd> [-p <partition>] [-i <docker_image>]"
    echo "Example: $0 \".gitlab/test_cpp.sh /opt/nixl\" -p rock"
    exit 1
}

# Validate required parameter
if [ -z "$1" ]; then
    echo "Error: Test command is required"
    usage
fi

TEST_CMD="$1"
shift

PARTITION="rock"
TIMEOUT="01:00:00" # Slurm job runtime
DOCKER_IMAGE="harbor.mellanox.com/ucx/x86_64/pytorch:25.02-py3"
SLURM_JOB_NAME="NIXL-${JOB_NAME:-local}-${BUILD_ID:-$$}"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR svc-nixl@hpchead"

# Parse optional parameters
while getopts ":p:i:" opt; do
    case "$opt" in
    p) PARTITION="$OPTARG" ;;
    i) DOCKER_IMAGE="$OPTARG" ;;
    *) usage ;;
    esac
done

# Temp - check with Jenkins
if [ -n "$GIT_COMMIT" ]; then
    GIT_REF="$GIT_COMMIT"
elif [ -n "$GIT_BRANCH" ]; then
    GIT_REF="$GIT_BRANCH"
elif [ -n "$CHANGE_BRANCH" ]; then
    GIT_REF="$CHANGE_BRANCH"
else
    GIT_REF="main"
fi

BUILD_AND_TEST_CMD="pip install --upgrade meson && \
    git clone --branch ${GIT_REF} https://github.com/ai-dynamo/nixl && \
    cd nixl && \
    .gitlab/build.sh /opt/nixl /usr/local && \
    ${TEST_CMD}"

DOCKER_RUN_CMD="sudo docker run --rm --quiet \
    --ulimit memlock=-1:-1 \
    --net=host \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --gpus all \
    --device=/dev/gdrdrv \
    -e LDFLAGS='-lpthread -ldl' \
    -e NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    -e DEBIAN_FRONTEND=noninteractive \
    -w /tmp \
    ${DOCKER_IMAGE} \
    /bin/bash -c '${BUILD_AND_TEST_CMD}'"

# NVIDIA Container Toolkit installation for RHEL 8.6 nodes
NVIDIA_TOOLKIT_INSTALL="curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo && \
    sudo yum install -y nvidia-container-toolkit nvidia-container-runtime libnvidia-container1 libnvidia-container-tools && \
    sudo nvidia-ctk runtime configure --runtime=docker"

# command for sbatch wrap - install Container Toolkit, configure, then restart docker
WRAP_CMD="set -ex; ${NVIDIA_TOOLKIT_INSTALL} && sudo systemctl restart docker && ${DOCKER_RUN_CMD}"

# Submit job
JOB_ID=$($SSH_CMD "sbatch --parsable \
    -J \"$SLURM_JOB_NAME\" \
    -N 1 \
    -p \"$PARTITION\" \
    -t \"$TIMEOUT\" \
    --wrap \"${WRAP_CMD}\"")

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    exit 1
fi
echo "Job $JOB_ID submitted"

# Wait for job completion
TERMINAL_STATES="COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL"
MONITOR_TIMEOUT=7200 # Queue pending + job runtime
SECONDS=0

while [ $SECONDS -lt $MONITOR_TIMEOUT ]; do
    STATUS=$($SSH_CMD "scontrol show job $JOB_ID -o | grep -oP 'JobState=\K\S+'")
    if echo "$STATUS" | grep -qE "$TERMINAL_STATES"; then
        break
    fi
    sleep 60
done

# Handle timeout
if [ $SECONDS -ge $MONITOR_TIMEOUT ]; then
    echo "Error: Monitoring timeout reached"
    $SSH_CMD "scancel $JOB_ID"
    exit 1
fi

# Show logs and cleanup
$SSH_CMD "cat slurm-${JOB_ID}.out; rm slurm-${JOB_ID}.out"

# Show node info
echo "Ran on node: $($SSH_CMD "scontrol show job $JOB_ID | grep -oP 'NodeList=\K\S+' | tail -1")"

# Set exit code
if [[ "$STATUS" == "COMPLETED" ]]; then
    echo "Job completed successfully"
    exit 0
fi
echo "Job failed with status: $STATUS"
exit 1
