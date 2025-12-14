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

export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins

echo "==== Show system info ===="
env
nvidia-smi topo -m || true
ibv_devinfo || true
uname -a || true

echo "==== Running ETCD server ===="
etcd_port=$(get_next_tcp_port)
etcd_peer_port=$(get_next_tcp_port)
export NIXL_ETCD_ENDPOINTS="http://127.0.0.1:${etcd_port}"
export NIXL_ETCD_PEER_URLS="http://127.0.0.1:${etcd_peer_port}"
export NIXL_ETCD_NAMESPACE="/nixl/nixlbench_ci/${etcd_port}"
etcd --listen-client-urls ${NIXL_ETCD_ENDPOINTS} --advertise-client-urls ${NIXL_ETCD_ENDPOINTS} \
     --listen-peer-urls ${NIXL_ETCD_PEER_URLS} --initial-advertise-peer-urls ${NIXL_ETCD_PEER_URLS} \
     --initial-cluster default=${NIXL_ETCD_PEER_URLS} &
sleep 5

echo "==== Running Nixlbench tests ===="
cd ${INSTALL_DIR}

DEFAULT_NB_PARAMS="--filepath /tmp --total_buffer_size 80000000 --start_block_size 4096 --max_block_size 16384 --start_batch_size 1 --max_batch_size 4"

run_nixlbench() {
    args="$@"
    ./bin/nixlbench --etcd-endpoints ${NIXL_ETCD_ENDPOINTS} $DEFAULT_NB_PARAMS $args
}

run_nixlbench_noetcd() {
    args="$@"
    ./bin/nixlbench $DEFAULT_NB_PARAMS $args
}

run_nixlbench_one_worker() {
    args="$@"
    run_nixlbench_noetcd $args
}

run_nixlbench_two_workers() {
    benchmark_group=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
    args="$@"
    run_nixlbench --benchmark_group $benchmark_group $args &
    pid=$!
    sleep 1
    run_nixlbench --benchmark_group $benchmark_group $args
    wait $pid
}

if $HAS_GPU ; then
    seg_types="VRAM DRAM"
else
    seg_types="DRAM"
    echo "Worker without GPU, skipping VRAM tests"
fi

for op_type in READ WRITE; do
    for initiator in $seg_types; do
        for target in $seg_types; do
            run_nixlbench_two_workers --backend UCX --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target --check_consistency
        done
    done
done

for op_type in READ WRITE; do
    run_nixlbench_one_worker --backend POSIX --op_type $op_type --check_consistency
done

# UCCL has a bug for data validation
if $HAS_GPU ; then
    for op_type in READ WRITE; do
        for initiator in $seg_types; do
            for target in $seg_types; do
                UCCL_RCMODE=1 run_nixlbench_two_workers --backend UCCL --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target
            done
        done
    done
fi

pkill etcd
