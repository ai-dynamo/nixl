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
if [ -z "$NIXL_AWS_ACCESS_KEY_ID" ] || [ -z "$NIXL_AWS_SECRET_ACCESS_KEY" ]; then
    echo "Missing NIXL S3 credentials"
    exit 1
fi

export AWS_ACCESS_KEY_ID="$NIXL_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$NIXL_AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_BUCKET="nixl-ci-test-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT:-1}"

set -exE -o pipefail

aws s3 rb "s3://${AWS_DEFAULT_BUCKET}" --force || true
