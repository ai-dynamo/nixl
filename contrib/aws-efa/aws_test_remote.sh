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

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <nixl_install_dir>"
    exit 1
fi

NIXL_INSTALL_DIR="$1"
UCX_INSTALL_DIR="${UCX_INSTALL_DIR:-$NIXL_INSTALL_DIR}"

run_group() {
    local group_name="$1"
    shift
    echo "::group::${group_name}"
    local rc
    if "$@"; then
        rc=0
    else
        rc=$?
    fi
    echo "::endgroup::"
    return "$rc"
}

run_group "Build" .gitlab/build.sh "${NIXL_INSTALL_DIR}" "${UCX_INSTALL_DIR}"
run_group "EFA C++ Tests" .gitlab/test_cpp.sh "${NIXL_INSTALL_DIR}"
run_group "EFA Python Tests" .gitlab/test_python.sh "${NIXL_INSTALL_DIR}"
run_group "EFA Rust Tests" .gitlab/test_rust.sh "${NIXL_INSTALL_DIR}"
run_group "S3 Tests" .gitlab/test_plugins.sh "${NIXL_INSTALL_DIR}"
