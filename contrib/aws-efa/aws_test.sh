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

set -exE -o pipefail

# Get test type from argument, default to cpp
TEST_TYPE=${1:-"cpp"}

# Set AWS container image (can be overridden via environment)
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"nvcr.io/nvidia/pytorch:25.02-py3"}

# Set Git checkout command based on GITHUB_REF
if [ -z "$GITHUB_REF" ]; then   # manual run
    echo "Error: GITHUB_REF environment variable must be set"
    echo "For a branch, use: export GITHUB_REF=\"main\""
    echo "For a PR, use: export GITHUB_REF=\"refs/pull/187/head\""
    exit 1
fi

case "$GITHUB_REF" in
    refs/pull/*)
        export GIT_CHECKOUT_CMD="git fetch origin ${GITHUB_REF} && git checkout FETCH_HEAD"
        ;;
    *)
        export GIT_CHECKOUT_CMD="git checkout ${GITHUB_REF}"
        ;;
esac

# Construct command to run in AWS
setup_cmd="set -x && \
    git clone ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY} && \
    cd nixl && \
    ${GIT_CHECKOUT_CMD}"
build_cmd=".gitlab/build.sh \${NIXL_INSTALL_DIR} \${UCX_INSTALL_DIR}"
test_script="test_${TEST_TYPE}"
export AWS_CMD="${setup_cmd} && ${build_cmd} && .gitlab/${test_script}.sh \${NIXL_INSTALL_DIR}"

# Generate properties json from template
envsubst < aws_vars.template > aws_vars.json
jq . aws_vars.json >/dev/null

# Submit AWS job
aws eks update-kubeconfig --name ucx-ci
JOB_NAME="NIXL_${TEST_TYPE}_${GITHUB_RUN_NUMBER:-$RANDOM}"
JOB_ID=$(aws batch submit-job \
    --job-name "$JOB_NAME" \
    --job-definition "NIXL-Ubuntu-JD" \
    --job-queue ucx-ci-JQ \
    --eks-properties-override file://./aws_vars.json \
    --query 'jobId' --output text)

# Function to wait for a specific job status
wait_for_status() {
    local target_status="$1"
    local timeout=600
    local interval=60
    local status=""
    SECONDS=0

    while [ $SECONDS -lt $timeout ]; do
        status=$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].status' --output text)
        echo "Current status: $status (${SECONDS}s elapsed)"
        if echo "$status" | grep -qE "$target_status"; then
            echo "Reached status $status (completed in ${SECONDS}s)"
            return 0
        fi
        sleep $interval
    done

    echo "Timeout waiting for status $target_status after ${SECONDS}s. Final status: $status"
    return 1
}

# Wait for the job to start running
echo "Waiting for job to start running..."
if ! wait_for_status "RUNNING"; then
    echo "Job failed to start"
    exit 1
fi

# Stream logs from the pod
POD=$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].eksProperties.podProperties.podName' --output text)
echo "Streaming logs from pod: $POD"
kubectl -n ucx-ci-batch-nodes logs -f "$POD"

# Check final job status
echo "Waiting for job completion..."
exit_status=$(wait_for_status "SUCCEEDED|FAILED")
if [[ "$exit_status" =~ FAILED ]]; then
    echo "Failure running NIXL tests"
    exit 1
fi

echo "NIXL tests completed successfully"
