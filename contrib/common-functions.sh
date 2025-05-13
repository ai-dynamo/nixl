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


_missing_requirement() {
    _error "ERROR: ${1:-} requires an argument."
}

# Error handling functions
_error() {
    printf '%s %s\n' "${1:-}" "${2:-}" >&2
    return 1
}

_setup_environment() {
    local install_dir=${1:-$_NIXL_INSTALL_DIR}

    export LIBRARY_PATH=/usr/local/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}

    export LD_LIBRARY_PATH=${install_dir}/lib:${install_dir}/lib/${NIXL_ARCH}-linux-gnu:${install_dir}/lib/${NIXL_ARCH}-linux-gnu/plugins:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

    export CPATH=${install_dir}/include${CPATH:+:$CPATH}
    export PATH=${install_dir}/bin${PATH:+:$PATH}
    export PKG_CONFIG_PATH=${install_dir}/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
    export NIXL_PLUGIN_DIR=${install_dir}/lib/${NIXL_ARCH}-linux-gnu/plugins
}

_is_valid_os_type() {
    if ! grep -q "Ubuntu" /etc/os-release; then
        echo "This script only supports Ubuntu for now"
        return 1
    fi
    return 0
}

# Check if script is being sourced
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    # Script is being executed directly
    _error "This script is not meant to be executed directly"
    exit 1
fi
