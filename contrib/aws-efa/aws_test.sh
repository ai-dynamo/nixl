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

usage() {
    echo ""
    echo "Description: Run NIXL tests on AWS infrastructure"
    echo ""
    echo "Usage: $0 <test script> <test script args>"
    echo ""
    echo "Example: $0 .gitlab/test_cpp.sh \$NIXL_INSTALL_DIR && .test_script123.sh param123"
    echo ""
    echo "Required environment variables:"
    echo "  GITHUB_REF        - Git reference to checkout (e.g., main, branch-xyz, commit SHA)"
    echo "  GITHUB_SERVER_URL - GitHub server URL (e.g., \"https://github.com\")"
    echo "  GITHUB_REPOSITORY - GitHub repository (e.g., \"ai-dynamo/nixl\")"
    echo ""
    echo "Optional environment variables:"
    echo "  CONTAINER_IMAGE   - Container image to use (default: nvcr.io/nvidia/pytorch:25.02-py3)"
    exit 1
}

# Validate required parameters and environment variables
if [ -z "$1" ]; then
    echo "Error: Test command string argument is required"
    usage
fi

if [ -z "$GITHUB_REF" ] || [ -z "$GITHUB_SERVER_URL" ] || [ -z "$GITHUB_REPOSITORY" ]; then
    echo "Error: Missing required environment variables"
    usage
fi

test_cmd="$1"
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"nvcr.io/nvidia/pytorch:25.02-py3"}

# Set Git checkout command based on GITHUB_REF
case "$GITHUB_REF" in
    refs/heads/*)
        export GIT_CHECKOUT_CMD="git checkout ${GITHUB_REF#refs/heads/}"
        ;;
    *)
        export GIT_CHECKOUT_CMD="git checkout $GITHUB_REF"
        ;;
esac

# Construct command sequence to run within the AWS container
setup_cmd="set -x && \
    git clone ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY} && \
    cd nixl && \
    ${GIT_CHECKOUT_CMD}"
build_cmd=".gitlab/build.sh \${NIXL_INSTALL_DIR} \${UCX_INSTALL_DIR}"
export AWS_CMD="${setup_cmd} && ${build_cmd} && ${test_cmd}"

# Generate AWS job properties json from template
envsubst < aws_vars.template > aws_vars.json
jq . aws_vars.json >/dev/null

# Submit AWS job
aws eks update-kubeconfig --name ucx-ci
JOB_NAME="NIXL_${GITHUB_RUN_NUMBER:-$RANDOM}"
JOB_ID=$(aws batch submit-job \
    --job-name "$JOB_NAME" \
    --job-definition "NIXL-Ubuntu-JD" \
    --job-queue ucx-nxil-jq \
    --eks-properties-override file://./aws_vars.json \
    --retry-strategy '{"attempts":3}' \
    --query 'jobId' --output text)

# Function to wait for a specific job status
wait_for_status() {
    local target_status="$1"
    local timeout="$2"
    local interval="$3"
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
echo "Waiting for job to start running (timeout: 30m)..."
if ! wait_for_status "RUNNING" 1800 10; then
    echo "Job failed to start"
    exit 1
fi

# Stream logs from the pod
POD=$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].eksProperties.podProperties.podName' --output text)
echo "Streaming logs from pod: $POD"
kubectl -n ucx-ci-batch-nodes logs -f "$POD" || kubectl -n ucx-ci-batch-nodes logs "$POD" --previous || true

# Check final job status
echo "Waiting for job completion (timeout: 10m)..."
exit_status=$(wait_for_status "SUCCEEDED|FAILED" 600 10)
if [[ "$exit_status" =~ FAILED ]]; then
    echo "Failure running NIXL tests"
    exit 1
fi

echo "NIXL tests completed successfully"
