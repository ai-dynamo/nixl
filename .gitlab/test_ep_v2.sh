#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NIXL EP V2 CI: run the runtime/ep correctness tests on one node with 4 GPUs.

# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

set -e
set -x

INSTALL_DIR=$1

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir>"
    exit 1
fi

# NIXL EP V2 is built against the vLLM venv's Python (NIXL_PYTHON in test-dl-ep-matrix.yaml).
EP_PYTHON="${VLLM_ELASTIC_TEST_DIR}/.venv/bin/python"

if [ ! -x "${EP_PYTHON}" ]; then
    echo "ERROR: Python environment is missing: ${EP_PYTHON}" >&2
    exit 1
fi

# UCX and NIXL libraries.
export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$(uname -m)-linux-gnu:$LD_LIBRARY_PATH

# Load the source dispatcher and the backend installed in the shared venv.
export PYTHONPATH="${PWD}/src/bindings/python/nixl-meta${PYTHONPATH:+:$PYTHONPATH}"

# The container limits UCX to one Ethernet port: let UCX select the RDMA devices.
unset UCX_NET_DEVICES
unset NIXL_ETCD_ENDPOINTS NIXL_ETCD_PEER_URLS NIXL_ETCD_NAMESPACE UCX_TLS

echo "==== Show system info ===="
env
nvidia-smi topo -m || true

# NIXL EP V2 runtime not merged yet (main has only its API shell): nothing to test.
if [ ! -d runtime/ep/csrc ] && ! "${EP_PYTHON}" -c "import nixl.runtime.ep._C" 2>/dev/null; then
    echo "SKIP: NIXL EP V2 runtime is not in this tree"
    exit 0
fi

echo "==== Running NIXL EP V2 tests ===="
cd runtime/ep

# run_v2_test [args...]: test_elastic_ep.py on 4 GPUs, each run on a free port.
run_v2_test() {
    local port
    port=$(get_next_tcp_port)
    WORLD_SIZE=1 RANK=0 SERVER_ADDR=127.0.0.1 SERVER_PORT=${port} \
    NIXL_TCPSTORE_ENDPOINT=127.0.0.1:${port} \
    timeout 600 "${EP_PYTHON}" tests/elastic/test_elastic_ep.py --num-processes 4 "$@"
}

# All tests run on 1x4 (one NVLink domain, direct mode).

# 1. Single-domain membership patterns: the default plan
#    [[0,1,2,3],[0,1],[0,1,2,3],[0,1,2],[0,1,2,3],[1,2,3],[0,1,2,3],[1,3],[0,1,2,3],[0],[0,1,2,3]]
run_v2_test

# 2. Consecutive expansion and contraction
run_v2_test --plan '[[0],[0,1,2],[0,1,2,3],[0,1,2],[0]]' --num-topk 16 --num-tokens 8192 --num-experts 512

# 3. Short direct NVFP4 expansion
run_v2_test --plan '[[0,1],[0,1,2,3]]' --dtype nvfp4

# 4. Short direct FP8 expansion
run_v2_test --plan '[[0,1],[0,1,2,3]]' --dtype fp8

# TODO: once the test can split the node into 2 domains of 2 GPUs (2x2, hybrid mode), keep tests 1-2 and:
#   3. Direct/hybrid topology coverage: 2x2, default plan
#      [[0,1,2,3],[0,1],[0,1,2,3],[0,2],[0,1,2,3],[1,3],[0,1,2,3],[2,3],[0,1,2,3],[0],[0,1,2,3]]
#   4. Late join domain, domain replacement, mode switch with the same number of GPUs and a 1-GPU phase:
#      2x2, [[0,1],[0,1,2,3],[0,1],[2,3],[0,2],[3],[0,1]] --reconnect-on-leave
#      --num-topk 6 --num-tokens 63 --num-experts 96
#   5. Short hybrid NVFP4 expansion: 2x2, [[0,2],[0,1,2,3]] --dtype nvfp4 (replaces test 3)
#   6. Short hybrid FP8 expansion: 2x2, [[0,2],[0,1,2,3]] --dtype fp8 (replaces test 4)

echo "==== NIXL EP V2 tests done ===="
