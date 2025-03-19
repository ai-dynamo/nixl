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

# Parse commandline arguments with first argument being the build directory
# and second argument being the install directory. A third argument is optional
# and specifies the UCX installation directory. If not provided, the UCX
# installation directory is assumed to be the same as the install directory.
BUILD_DIR=$1
INSTALL_DIR=$2
UCX_INSTALL_DIR=$3

if [ -z "$BUILD_DIR" ]; then
    echo "Usage: $0 <install_dir> <build_dir>"
    exit 1
fi

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir> <build_dir>"
    exit 1
fi

if [ -z "$UCX_INSTALL_DIR" ]; then
    UCX_INSTALL_DIR=$INSTALL_DIR
fi

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:$LD_LIBRARY_PATH
export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH

meson setup ${BUILD_DIR} --prefix=${INSTALL_DIR} -Ducx_path=${UCX_INSTALL_DIR}
cd ${BUILD_DIR} && ninja && ninja install

# TODO(kapila): Copy the nixl.pc file to the install directory if needed.
# cp ${BUILD_DIR}/nixl.pc ${INSTALL_DIR}/lib/pkgconfig/nixl.pc
