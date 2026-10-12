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
GPUNETIO_DIAGNOSTIC_SCRIPT=$(realpath "$(dirname "$0")/capture_gpunetio_hang.py")

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

run_nixlbench_two_workers_gpunetio() {
    local benchmark_group port0 port1 command_line
    benchmark_group=$(get_random_tcp_port)-$$
    port0=$(get_random_tcp_port)
    port1=$(get_random_tcp_port)
    while [ "$port0" = "$port1" ]; do
        port1=$(get_random_tcp_port)
    done
    # Log the GPU-side state of a transfer stuck for 30s, before the capture and the timeout
    local nb_prefix="NIXL_GPUNETIO_HANG_REPORT_SEC=30 timeout --signal=INT --kill-after=15s 180s python3 '$GPUNETIO_DIAGNOSTIC_SCRIPT' --"
    local nb_args="--etcd_endpoints ${NIXL_ETCD_ENDPOINTS} $DEFAULT_NB_PARAMS --benchmark_group gpunetio-$benchmark_group $*"
    command_line="$nb_prefix ./bin/nixlbench $nb_args"
    # TEMPORARY diagnostic: run only the first worker (the initiator that hit an illegal
    # memory access) under compute-sanitizer to get the faulting kernel and address.
    # kernel_progress is excluded: memcheck flags its start-up QP reads and kills it.
    # Opt-in: set GPUNETIO_SANITIZE=1 (it slows the run and can abort the progress kernel).
    local command_line0="$command_line"
    if [ "${GPUNETIO_SANITIZE:-0}" = 1 ] && command -v compute-sanitizer > /dev/null; then
        command_line0="$nb_prefix compute-sanitizer --tool memcheck --print-limit 50 --launch-timeout 0 --kernel-name-exclude kns=kernel_progress ./bin/nixlbench $nb_args"
    fi
    # TEMPORARY diagnostic, opt-in (GPUNETIO_COREDUMP=1): ask the driver for a GPU core dump
    # when the initiator takes a device exception, and print the faulting kernel and PC below.
    # Untested on x86; on GH200 it was seen to swallow the fault instead (no error, no core).
    local core_pattern=/tmp/gpunetio_core_%h_%p
    if [ "${GPUNETIO_COREDUMP:-0}" = 1 ]; then
        rm -f /tmp/gpunetio_core_*
        command_line0="CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory CUDA_COREDUMP_FILE=$core_pattern $command_line0"
    fi
    # Each worker keeps a GPUNetIO progress kernel running, and two processes on
    # one GPU get time-sliced: give each worker its own GPU when there are two.
    local gpus gpu0 gpu1
    IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
    gpu0=${gpus[0]:-0}
    gpu1=${gpus[1]:-$gpu0}
    # Each local worker owns its listener; never share an SO_REUSEPORT endpoint.
    # Bound a failed case even if its peer is still waiting for a notification.
    local rc=0
    parallel --line-buffer --halt now,fail=1 ::: \
        "CUDA_VISIBLE_DEVICES=$gpu0 $command_line0 --gpunetio_oob_port=$port0" \
        "sleep 4 ; CUDA_VISIBLE_DEVICES=$gpu1 $command_line --gpunetio_oob_port=$port1" || rc=$?
    if [ "${GPUNETIO_COREDUMP:-0}" = 1 ]; then
        print_gpunetio_coredumps
    fi
    return $rc
}

# Print the faulting kernel, thread and PC from any GPU core dump left by the initiator.
print_gpunetio_coredumps() {
    local core
    for core in /tmp/gpunetio_core_*; do
        [ -e "$core" ] || { echo "GPUNETIO_COREDUMP: no GPU core dump written"; return 0; }
        echo "GPUNETIO_COREDUMP: $core ($(stat -c %s "$core") bytes)"
        if command -v cuda-gdb > /dev/null; then
            timeout 120 cuda-gdb -batch -nx -ex "target cudacore $core" \
                -ex "info cuda kernels" -ex "info cuda threads" -ex "bt" || true
        else
            echo "GPUNETIO_COREDUMP: cuda-gdb not found, cannot decode $core"
        fi
    done
}

if $HAS_GPU ; then
    # WRITE first: so far every hang was READ, and CI stops at the first failure
    for op_type in WRITE READ; do
        for initiator in $seg_types; do
            for target in $seg_types; do
                run_nixlbench_two_workers_gpunetio --backend GPUNETIO --op_type $op_type --initiator_seg_type $initiator --target_seg_type $target --check_consistency
            done
        done
    done
fi

kill -9 $ETCD_PID 2>/dev/null || true
