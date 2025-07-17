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

#
# Common functions for CI scripts
#

#
# Check test script usage
# Parse commandline arguments with first argument being the install directory.
#
check_install_dir() {
    INSTALL_DIR=$1
    if [ -z "$INSTALL_DIR" ]; then
        echo "Usage: $0 <install_dir>"
        exit 1
    fi
}

#
# Set environment variables for the build
#
set_env() {
    INSTALL_DIR=$1

    ARCH=$(uname -m)
    [ "$ARCH" = "arm64" ] && ARCH="aarch64"

    export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
    export CPATH=${INSTALL_DIR}/include:$CPATH
    export PATH=${INSTALL_DIR}/bin:$PATH
    export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
    export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins
}


#
# Set initial port number for client/server applications to be updated with
# function below
#
server_port_range=1000
min_port_number=10500
max_port_number=65535

# GITLAB CI
if [ -n "$CI_CONCURRENT_ID" ]; then
    EXECUTOR_NUMBER=$CI_CONCURRENT_ID
# Jenkins CI
elif [ -z "$EXECUTOR_NUMBER" ]; then
    # Fallback to random number if both CI_CONCURRENT_ID and EXECUTOR_NUMBER are not set
    EXECUTOR_NUMBER=$(($RANDOM % $(((max_port_number - min_port_number) / server_port_range))))
fi

echo EXECUTOR_NUMBER=$EXECUTOR_NUMBER

server_port_min=$((min_port_number + EXECUTOR_NUMBER * server_port_range))
server_port_max=$((server_port_min + server_port_range))
server_port=${server_port_min}

get_next_server_port() {
    # Cycle server_port between (server_port_min)..(server_port_max-1)
    server_port=$((server_port + 1))
    server_port=$((server_port >= server_port_max ? server_port_min : server_port))
    echo $server_port
}
