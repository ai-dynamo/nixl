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

source $(dirname $0)/../.ci/scripts/common.sh

set -e
set -x
TEXT_YELLOW="\033[1;33m"
TEXT_CLEAR="\033[0m"

# For running as user - check if running as root, if not set sudo variable
if [ "$(id -u)" -ne 0 ]; then
    SUDO=sudo
else
    SUDO=""
fi

$SUDO apt-get update
$SUDO apt-get -qq install -y libaio-dev


# Parse commandline arguments with first argument being the install directory.
INSTALL_DIR=$1

check_install_dir $INSTALL_DIR

set_env $INSTALL_DIR

echo "==== Show system info ===="
env
nvidia-smi topo -m || true
ibv_devinfo || true
uname -a || true

echo "==== Running C++ tests ===="
cd ${INSTALL_DIR}
./bin/desc_example
./bin/agent_example
./bin/nixl_example
./bin/ucx_backend_test
./bin/ucx_mo_backend_test

# POSIX test disabled until we solve io_uring and Docker compatibility

./bin/nixl_posix_test -n 128 -s 1048576

./bin/ucx_backend_multi
./bin/serdes_test
./bin/gtest
./bin/test_plugin

# Run NIXL client-server test
nixl_test_port=$(get_next_server_port)

./bin/nixl_test target 127.0.0.1 $nixl_test_port&
sleep 1
./bin/nixl_test initiator 127.0.0.1 $nixl_test_port

echo "${TEXT_YELLOW}==== Disabled tests==="
echo "./bin/md_streamer disabled"
echo "./bin/p2p_test disabled"
echo "./bin/ucx_worker_test disabled"
echo "${TEXT_CLEAR}"
