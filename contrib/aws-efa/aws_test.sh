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
    echo "  CONTAINER_IMAGE   - Container image to use (default: nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04)"
    echo "  TEST_TIMEOUT      - Timeout for test execution in minutes"
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
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04"}

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
efa_validation_cmd="fi_info -p efa"
build_cmd=".gitlab/build.sh \${NIXL_INSTALL_DIR} \${UCX_INSTALL_DIR}"

# Add timeout only if TEST_TIMEOUT is set (expects minutes)
if [ -n "$TEST_TIMEOUT" ]; then
    test_cmd="timeout ${TEST_TIMEOUT}m ${test_cmd}"
fi

export AWS_CMD="${setup_cmd} && ${build_cmd} && ${efa_validation_cmd} && ${test_cmd}"

# Generate AWS job properties json from template
envsubst < aws_vars.template > aws_vars.json
jq . aws_vars.json >/dev/null

# Build tags JSON for AWS
TAGS_JSON=$(jq -n '{
    github_ref: env.GITHUB_REF,
    github_repository: env.GITHUB_REPOSITORY,
    github_run_number: env.GITHUB_RUN_NUMBER,
    github_run_id: env.GITHUB_RUN_ID,
    github_job_url: "\(env.GITHUB_SERVER_URL)/\(env.GITHUB_REPOSITORY)/actions/runs/\(env.GITHUB_RUN_ID)",
    container_image: env.CONTAINER_IMAGE
}')

# Submit AWS job
aws eks update-kubeconfig --name ucx-ci
JOB_NAME="NIXL_${GITHUB_RUN_NUMBER:-$RANDOM}"
JOB_ID=$(aws batch submit-job \
    --job-name "$JOB_NAME" \
    --job-definition "NIXL-Ubuntu-JD" \
    --job-queue ucx-nxil-jq \
    --eks-properties-override file://./aws_vars.json \
    --tags "$TAGS_JSON" \
    --query 'jobId' --output text)

# Function to wait for a specific job status
wait_for_status() {
    set +x
    local target_status="$1"
    local timeout="$2"
    local interval="$3"
    local status=""
    SECONDS=0

    while [ $SECONDS -lt $timeout ]; do
        status=$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].status' --output text)
        if echo "$status" | grep -qE "$target_status"; then
            echo -e "\nReached status $status (completed in ${SECONDS}s)"
            set -x
            return 0
        fi
        printf "."
        sleep $interval
    done

    echo -e "\nTimeout waiting for status $target_status after ${SECONDS}s. Final status: $status"
    set -x
    return 1
}

# Wait for the job to start running
echo "Waiting for job to start running (timeout: 30m)..."
if ! wait_for_status "RUNNING" 1800 10; then
    echo "Job failed to start"
    aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].[status,statusReason]' --output table
    exit 1
fi

# Stream logs from the pod
POD=$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].eksProperties.podProperties.podName' --output text)
echo "Streaming logs from pod: $POD"
kubectl -n ucx-ci-batch-nodes logs -f "$POD" || kubectl -n ucx-ci-batch-nodes logs "$POD" --previous || true

# Check final job status
echo "Waiting for job completion (timeout: 10m)..."
if ! wait_for_status "SUCCEEDED" 600 10; then
    echo "Failure running NIXL tests"
    exit 1
fi

echo "NIXL tests completed successfully"
