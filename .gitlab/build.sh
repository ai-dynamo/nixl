#!/bin/sh
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

set -e
set -x

# Parse commandline arguments with first argument being the install directory
# and second argument being the UCX installation directory.
INSTALL_DIR=$1
UCX_INSTALL_DIR=$2

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir> <ucx_install_dir>"
    exit 1
fi

if [ -z "$UCX_INSTALL_DIR" ]; then
    UCX_INSTALL_DIR=$INSTALL_DIR
fi

# shellcheck disable=SC1091
. "$(dirname "$0")/../contrib/build.env"
nixl_install_build_dependencies
nixl_build_ucx
nixl_build --doxygen
