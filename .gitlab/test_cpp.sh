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


# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

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

ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH

export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins

echo "==== Show system info ===="
env
nvidia-smi topo -m || true
ibv_devinfo || true
uname -a || true
cat /sys/devices/virtual/dmi/id/product_name || true

echo "==== Running ETCD server ===="
etcd_port=$(get_next_tcp_port)
etcd_peer_port=$(get_next_tcp_port)
export NIXL_ETCD_ENDPOINTS="http://127.0.0.1:${etcd_port}"
export NIXL_ETCD_PEER_URLS="http://127.0.0.1:${etcd_peer_port}"
etcd --listen-client-urls ${NIXL_ETCD_ENDPOINTS} --advertise-client-urls ${NIXL_ETCD_ENDPOINTS} \
     --listen-peer-urls ${NIXL_ETCD_PEER_URLS} --initial-advertise-peer-urls ${NIXL_ETCD_PEER_URLS} \
     --initial-cluster default=${NIXL_ETCD_PEER_URLS} &
sleep 5

echo "==== Running C++ tests ===="
cd ${INSTALL_DIR}
./bin/desc_example
./bin/agent_example
./bin/nixl_example
if $TEST_LIBFABRIC ; then
    ./bin/nixl_example LIBFABRIC
fi
./bin/nixl_etcd_example
./bin/ucx_backend_test
mkdir -p /tmp/telemetry_test
NIXL_TELEMETRY_ENABLE=y NIXL_TELEMETRY_DIR=/tmp/telemetry_test ./bin/agent_example &
sleep 1
./bin/telemetry_reader /tmp/telemetry_test/Agent001 &
telePID=$!
sleep 6
kill -s INT $telePID

# POSIX test disabled until we solve io_uring and Docker compatibility

./bin/nixl_posix_test -n 128 -s 1048576
./bin/nixl_gusli_test -n 4 -s 16
./bin/ucx_backend_multi
./bin/serdes_test

# shellcheck disable=SC2154
gtest-parallel --workers=1 --serialize_test_cases ./bin/gtest -- --min-tcp-port="$min_gtest_port" --max-tcp-port="$max_gtest_port"
./bin/test_plugin

# Run NIXL client-server test
nixl_test_port=$(get_next_tcp_port)

./bin/nixl_test target 127.0.0.1 "$nixl_test_port"&
sleep 1
./bin/nixl_test initiator 127.0.0.1 "$nixl_test_port"

echo "${TEXT_YELLOW}==== Disabled tests==="
echo "./bin/md_streamer disabled"
echo "./bin/p2p_test disabled"
echo "./bin/ucx_worker_test disabled"
echo "${TEXT_CLEAR}"

pkill etcd
