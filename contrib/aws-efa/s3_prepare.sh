#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Prepare S3 state and AWS pod env for S3 plugin tests.

if [ -z "$NIXL_AWS_ACCESS_KEY_ID" ] || [ -z "$NIXL_AWS_SECRET_ACCESS_KEY" ]; then
    echo "Missing NIXL S3 credentials"
    exit 1
fi

if [ -z "$GITHUB_RUN_ID" ]; then
    echo "Missing required GitHub metadata: GITHUB_RUN_ID"
    exit 1
fi

export AWS_ACCESS_KEY_ID="$NIXL_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$NIXL_AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-eu-central-1}"
export AWS_DEFAULT_BUCKET="nixl-ci-test-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT:-1}"

set -exE -o pipefail

# Validate credentials before creating resources.
aws sts get-caller-identity >/dev/null
aws s3 mb "s3://${AWS_DEFAULT_BUCKET}"

max_attempts=5
attempt=1
while [ $attempt -le $max_attempts ]; do
    if aws s3 ls "s3://${AWS_DEFAULT_BUCKET}" >/dev/null 2>&1; then
        echo "S3 bucket exists successfully on attempt $attempt"
        break
    fi
    echo "S3 bucket check failed on attempt $attempt/$max_attempts"
    if [ $attempt -eq $max_attempts ]; then
        echo "S3 bucket creation failed after $max_attempts attempts"
        exit 1
    fi
    sleep 2
    attempt=$((attempt + 1))
done

echo test > testfile.txt
aws s3 cp testfile.txt "s3://${AWS_DEFAULT_BUCKET}/testfile.txt"

jq '.podProperties.containers[0].env += [{"name":"AWS_ACCESS_KEY_ID","value":env.AWS_ACCESS_KEY_ID},{"name":"AWS_SECRET_ACCESS_KEY","value":env.AWS_SECRET_ACCESS_KEY},{"name":"AWS_DEFAULT_BUCKET","value":env.AWS_DEFAULT_BUCKET},{"name":"AWS_DEFAULT_REGION","value":env.AWS_DEFAULT_REGION}]' \
    aws_vars.template > aws_vars.tmp.template
mv aws_vars.tmp.template aws_vars.template

