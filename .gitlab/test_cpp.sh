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
TEXT_YELLOW="\033[1;33m"
TEXT_CLEAR="\033[0m"

# Parse commandline arguments with first argument being the install directory.
INSTALL_DIR=$1

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir>"
    exit 1
fi

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/x86_64-linux-gnu:${INSTALL_DIR}/lib/x86_64-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/x86_64-linux-gnu/plugins

echo "==== Running C++ tests ===="
cd ${INSTALL_DIR}
./bin/desc_example
./bin/agent_example
./bin/ucx_backend_test
./bin/ucx_mo_backend_test
./bin/ucx_backend_multi
./bin/serdes_test

# Run NIXL client-server test
./bin/nixl_test initiator 127.0.0.1 1234&
sleep 1
./bin/nixl_test target 127.0.0.1 1234

echo "${TEXT_YELLOW}==== Disabled tests==="
echo "./bin/md_streamer disabled"
echo "./bin/nixl_test disabled"
echo "./bin/p2p_test disabled"
echo "./bin/ucx_worker_test disabled"
echo "${TEXT_CLEAR}"