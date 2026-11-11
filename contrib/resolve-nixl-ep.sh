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

set -e

CUDA_MAJOR="${CUDA_VERSION%%.*}"
EP="${BUILD_NIXL_EP:-}"

if [ -z "$EP" ]; then
    if [ "$CUDA_MAJOR" = "13" ]; then EP="true"; else EP="false"; fi
fi

case "$EP" in
    true|false) ;;
    *) echo "ERROR: BUILD_NIXL_EP must be 'true', 'false' or empty, got '$EP'" >&2; exit 1 ;;
esac

if [ "$EP" = "true" ] && [ "$CUDA_MAJOR" != "13" ]; then
    echo "ERROR: BUILD_NIXL_EP=true requires CUDA 13 (NIXL EP needs torch >= 2.14, only on the cu13 index); got CUDA_VERSION=$CUDA_VERSION" >&2
    exit 1
fi

echo "$EP"
