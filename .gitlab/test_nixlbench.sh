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


# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

set -e
set -x
ulimit -c unlimited

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

echo "==== Running Nixlbench tests ===="
cd ${INSTALL_DIR}

DEFAULT_NB_PARAMS="--filepath /tmp --total_buffer_size 80000000 --start_block_size 16384 --max_block_size 16384 --start_batch_size 4 --max_batch_size 4"

run_nixlbench_one_worker() {
    args="$@"
    ./bin/nixlbench $DEFAULT_NB_PARAMS $args
}

for op_type in READ WRITE; do
    run_nixlbench_one_worker --backend POSIX --op_type $op_type --check_consistency
done

if $HAS_GPU ; then
    seg_types="VRAM DRAM"
else
    seg_types="DRAM"
    echo "Worker without GPU, skipping VRAM tests"
fi

get_random_tcp_port() {
    python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((\"\", 0))
print(s.getsockname()[1])
s.close()
"
}

run_nixlbench_two_workers_asio() {
    args="$@"
    asio_port=$(get_random_tcp_port)
    command_line="./bin/nixlbench --runtime_type=ASIO --asio_port=$asio_port $DEFAULT_NB_PARAMS $args"
    parallel --line-buffer --halt now,fail=1 ::: "$command_line" "sleep 4 ; $command_line"
}

for op_type in READ WRITE; do
    for initiator in $seg_types; do
        for target in $seg_types; do
            run_nixlbench_two_workers_asio --backend UCX --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target --check_consistency
        done
    done
done

# TODO: remove this once https://github.com/ai-dynamo/nixl/issues/1999 is fixed
# Skip UCCL nixlbench transfer tests to reduce CI flakiness
# if $HAS_GPU ; then
#     for op_type in READ WRITE; do
#         for initiator in $seg_types; do
#             for target in $seg_types; do
#                 run_nixlbench_two_workers_asio --backend UCCL --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target --check_consistency
#             done
#         done
#     done
# fi

run_nixlbench_two_workers_etcd() {
    args="$@"
    benchmark_group=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
    command_line="./bin/nixlbench --etcd_endpoints ${NIXL_ETCD_ENDPOINTS} $DEFAULT_NB_PARAMS --benchmark_group $benchmark_group $args"
    parallel --line-buffer --halt now,fail=1 ::: "$command_line" "sleep 4 ; $command_line"
}

start_etcd_server "/nixl/nixlbench_ci"

for op_type in READ WRITE; do
    for initiator in $seg_types; do
        for target in $seg_types; do
            run_nixlbench_two_workers_etcd --backend UCX --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target --check_consistency
        done
    done
done

# TODO: remove this once https://github.com/ai-dynamo/nixl/issues/1999 is fixed
# Skip UCCL nixlbench transfer tests to reduce CI flakiness
# if $HAS_GPU ; then
#     for op_type in READ WRITE; do
#         for initiator in $seg_types; do
#             for target in $seg_types; do
#                 run_nixlbench_two_workers_etcd --backend UCCL --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target --check_consistency
#             done
#         done
#     done
# fi

capture_gpunetio_hang() {
    local group=$1 child= pid
    # Keep the watcher and its timer/debugger from surviving a completed case.
    trap 'if [[ -n "${child:-}" ]]; then kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi' EXIT
    trap 'exit 0' TERM INT
    sleep "${2:-30}" & child=$!
    wait "$child" || exit 0
    child=
    echo "GPUNETIO diagnostic: workers still running after the capture delay"
    uname -m || true
    printf 'CUDA_DEVICE_MAX_CONNECTIONS=%s CUDA_LAUNCH_BLOCKING=%s CUDA_MODULE_LOADING=%s\n' \
        "${CUDA_DEVICE_MAX_CONNECTIONS:-unset}" "${CUDA_LAUNCH_BLOCKING:-unset}" "${CUDA_MODULE_LOADING:-unset}"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv || true
    sha256sum ./bin/nixlbench "${NIXL_PLUGIN_DIR:-.}/libplugin_GPUNETIO.so" || true
    # The generated group is unique to this case; never attach to other GPU users.
    for pid in $(pgrep -f -- "^./bin/nixlbench .*--benchmark_group $group( |$)" || true); do
        echo "GPUNETIO diagnostic: worker PID $pid"
        ps -L -p "$pid" -o pid,tid,stat,wchan:32 || true
        if command -v gdb >/dev/null 2>&1; then
            timeout --kill-after=2s 10s gdb -batch \
                -ex 'set pagination off' -ex 'set print frame-arguments none' \
                -ex 'info sharedlibrary' -ex 'thread apply all bt' -p "$pid" & child=$!
            wait "$child" || echo "GPUNETIO diagnostic: gdb capture failed for PID $pid"
            child=
        else
            echo "GPUNETIO diagnostic: gdb unavailable; CPU wait channels shown above"
        fi
    done
    exit 0
}

run_nixlbench_two_workers_gpunetio() {
    local benchmark_group port0 port1 command_line
    # Take the OOB ports from the reserved range: ephemeral ports (get_random_tcp_port) can be
    # grabbed by lingering sockets of earlier cases before the worker binds, failing with EADDRINUSE.
    port0=$(get_next_tcp_port)
    port1=$(get_next_tcp_port)
    benchmark_group=${port0}-$$
    command_line="timeout --signal=INT --kill-after=15s 360s ./bin/nixlbench --etcd_endpoints ${NIXL_ETCD_ENDPOINTS} $DEFAULT_NB_PARAMS --benchmark_group gpunetio-$benchmark_group $*"
    # Each worker keeps a GPUNetIO progress kernel running, and two processes on
    # one GPU get time-sliced: give each worker its own GPU when there are two.
    local gpus gpu0 gpu1
    IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
    gpu0=${gpus[0]:-0}
    gpu1=${gpus[1]:-$gpu0}
    # Each local worker owns its listener; never share an SO_REUSEPORT endpoint.
    # Bound a failed case even if its peer is still waiting for a notification.
    local diagnostic_pid status=0
    capture_gpunetio_hang "gpunetio-$benchmark_group" & diagnostic_pid=$!
    parallel --line-buffer --halt now,fail=1 ::: \
        "CUDA_VISIBLE_DEVICES=$gpu0 $command_line --gpunetio_oob_port=$port0" \
        "sleep 4 ; CUDA_VISIBLE_DEVICES=$gpu1 $command_line --gpunetio_oob_port=$port1" || status=$?
    kill "$diagnostic_pid" 2>/dev/null || true
    wait "$diagnostic_pid" 2>/dev/null || true
    return "$status"
}

if $HAS_GPU ; then
    # WRITE first: so far every hang was READ, and CI stops at the first failure
    for op_type in WRITE READ; do
        for initiator in $seg_types; do
            for target in $seg_types; do
                for qp_count in 1 2 4; do
                    run_nixlbench_two_workers_gpunetio --backend GPUNETIO --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target --check_consistency --gpunetio_data_qp_count=$qp_count
                done
            done
        done
    done
fi

kill -9 $ETCD_PID 2>/dev/null || true
