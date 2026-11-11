#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Builds and pushes the per-PR vLLM/SGLang sanity image for build-wheel-matrix.yaml:
# layers the extracted wheels onto the ci-demo-built framework base
# (contrib/Dockerfile.<fw> force-reinstalls them), then copies the workspace in so
# the sanity script and its helpers are present.
#
# Usage: build-sanity-image.sh <vllm|sglang> <wheels_dir>
# Environment: arch, PR_REGISTRY, CI_IMAGE_TAG, BUILD_NUMBER

set -eux

fw="$1"
wheels_dir="$2"

BASE="${PR_REGISTRY}/${arch}/${fw}-nixl-base:${CI_IMAGE_TAG}"
IMG="${PR_REGISTRY}/pr/${arch}/${fw}-nixl:${BUILD_NUMBER}"
# 1. Layer the freshly built wheels onto the base (wheels-only context).
docker build --platform "linux/${arch}" \
  --build-arg BASE_IMAGE="${BASE}" \
  -f "contrib/Dockerfile.${fw}" \
  -t "${IMG}" \
  "${wheels_dir}/"
# 2. Copy the whole workspace in (test script + helpers), like the other pipelines.
printf 'FROM %s\nCOPY . /workspace/nixl\nWORKDIR /workspace/nixl\n' "${IMG}" \
  | docker build --platform "linux/${arch}" -t "${IMG}" -f - .
for i in {1..6}; do
    docker push "${IMG}" && break
    [ "$i" = 6 ] && exit 1
    sleep $((2 ** (i - 1)))
done
