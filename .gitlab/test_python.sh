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

# Parse commandline arguments with first argument being the install directory.
INSTALL_DIR=$1

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir>"
    exit 1
fi

ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
export CPATH=${INSTALL_DIR}/include::$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins
export NIXL_PREFIX=${INSTALL_DIR}
# Raise exceptions for logging errors
export NIXL_DEBUG_LOGGING=yes

# Set the correct wheel name based on the CUDA version
cuda_major=$(nvcc --version | grep -oP 'release \K[0-9]+')
case "$cuda_major" in
    12|13) echo "CUDA $cuda_major detected" ;;
    *) echo "Error: Unsupported CUDA version $cuda_major"; exit 1 ;;
esac
pip3 install --break-system-packages tomlkit
./contrib/tomlutil.py --wheel-name "nixl-cu${cuda_major}" pyproject.toml
# Control ninja parallelism during pip build to prevent OOM (NPROC from common.sh)
pip3 install --break-system-packages --config-settings=compile-args="-j${NPROC}" .
pip3 install --break-system-packages dist/nixl-*none-any.whl
pip3 install --break-system-packages pytest
pip3 install --break-system-packages pytest-timeout
pip3 install --break-system-packages zmq

# Add user pip packages to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "==== Running ETCD server ===="
etcd_port=$(get_next_tcp_port)
etcd_peer_port=$(get_next_tcp_port)
export NIXL_ETCD_ENDPOINTS="http://127.0.0.1:${etcd_port}"
export NIXL_ETCD_PEER_URLS="http://127.0.0.1:${etcd_peer_port}"
export NIXL_ETCD_NAMESPACE="/nixl/python_ci/${etcd_port}"
etcd --listen-client-urls ${NIXL_ETCD_ENDPOINTS} --advertise-client-urls ${NIXL_ETCD_ENDPOINTS} \
     --listen-peer-urls ${NIXL_ETCD_PEER_URLS} --initial-advertise-peer-urls ${NIXL_ETCD_PEER_URLS} \
     --initial-cluster default=${NIXL_ETCD_PEER_URLS} &
sleep 5

echo "==== Running python tests ===="
pytest -s test/python

if $TEST_LIBFABRIC ; then
    cat /sys/devices/virtual/dmi/id/product_name
    echo "Product Name: $(cat /sys/devices/virtual/dmi/id/product_name)"
    echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
    # To collect libfabric debug logs, uncomment the following lines
    #export FI_LOG_LEVEL=debug
    #export FI_LOG_PROV=efa
    #export NIXL_LOG_LEVEL=TRACE
    pytest -s test/python --backend LIBFABRIC
fi

python3 test/python/prep_xfer_perf.py list
python3 test/python/prep_xfer_perf.py array

echo "==== Running python examples ===="
cd examples/python
python3 nixl_api_example.py
python3 partial_md_example.py --init-port "$(get_next_tcp_port)" --target-port "$(get_next_tcp_port)"
python3 partial_md_example.py --etcd
python3 query_mem_example.py

# Running telemetry for the last test
basic_two_peers_port=$(get_next_tcp_port)
mkdir -p /tmp/telemetry_test

python3 basic_two_peers.py --mode="target" --ip=127.0.0.1 --port="$basic_two_peers_port"&
sleep 5
NIXL_TELEMETRY_ENABLE=y NIXL_TELEMETRY_DIR=/tmp/telemetry_test \
python3 basic_two_peers.py --mode="initiator" --ip=127.0.0.1 --port="$basic_two_peers_port"

python3 telemetry_reader.py --telemetry_path /tmp/telemetry_test/initiator &
telePID=$!
sleep 6
kill -s INT $telePID

pkill etcd

