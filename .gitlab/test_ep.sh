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

# NIXL EP CI: run nixl_ep native elastic.py and vLLM+nixl_ep tests.

# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

set -e
set -x
set -o pipefail

INSTALL_DIR=$1

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir>"
    exit 1
fi

: "${VLLM_ELASTIC_TEST_DIR:?vLLM Elastic EP test environment is not installed}"
VLLM_PYTHON="${VLLM_ELASTIC_TEST_DIR}/.venv/bin/python"

if [ ! -x "${VLLM_PYTHON}" ]; then
    echo "ERROR: vLLM Python environment is missing: ${VLLM_PYTHON}" >&2
    exit 1
fi

ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins
export NIXL_PREFIX=${INSTALL_DIR}
export NIXL_DEBUG_LOGGING=yes

# Load the source dispatcher and the backend installed in the shared venv.
export PYTHONPATH="${PWD}/src/bindings/python/nixl-meta${PYTHONPATH:+:$PYTHONPATH}"

echo "==== Show system info ===="
env
nvidia-smi topo -m || true
ibv_devinfo || true
uname -a || true
cat /sys/devices/virtual/dmi/id/product_name || true

echo "==== Running elastic EP tests ===="
EP_SRC_DIR="examples/device/ep"
NIXL_BUILD_DIR=${NIXL_BUILD_DIR:-nixl_build}

run_elastic_test() {
    local plan_file=$1
    local extra_flags=${2:-}
    echo "---- elastic: plan=$(basename "$plan_file") flags=[$extra_flags] ----"
    (
        unset NIXL_ETCD_ENDPOINTS NIXL_ETCD_PEER_URLS NIXL_ETCD_NAMESPACE
        unset UCX_NET_DEVICES  # let UCX auto-select GPU-capable transport
        # Force NVLink-only transports.
        if [[ "$extra_flags" != *--disable-ll-nvlink* ]]; then
            export UCX_TLS=^rc_gda
        fi
        PYTHONPATH="${NIXL_BUILD_DIR}/${EP_SRC_DIR}:${EP_SRC_DIR}/tests:${EP_SRC_DIR}/tests/elastic${PYTHONPATH:+:$PYTHONPATH}" \
        timeout 300 "${VLLM_PYTHON}" ${EP_SRC_DIR}/tests/elastic/elastic.py \
            --plan "$plan_file" \
            --num-processes 4 \
            --num-experts-per-rank 32 \
            --num-topk 8 \
            --num-tokens 256 \
            --timeout-ms 10000 \
            --validate-phase-failures $extra_flags
    )
}

# NVLink (default)
run_elastic_test "${EP_SRC_DIR}/tests/elastic/no_expansion.json"
run_elastic_test "${EP_SRC_DIR}/tests/elastic/expansion_fault_contraction.json"

# Only run the --disable-ll-nvlink (RDMA) elastic tests when all four CX-7
# NICs (mlx5_0..mlx5_3) report PORT_ACTIVE.
all_rdma_nics_active() {
    local hca
    for hca in mlx5_0 mlx5_1 mlx5_2 mlx5_3; do
        if ! ibv_devinfo -d "$hca" 2>/dev/null | grep -q "state:.*PORT_ACTIVE"; then
            return 1
        fi
    done
    return 0
}

# RDMA (--disable-ll-nvlink)
if all_rdma_nics_active; then
    run_elastic_test "${EP_SRC_DIR}/tests/elastic/no_expansion.json" "--disable-ll-nvlink"
    run_elastic_test "${EP_SRC_DIR}/tests/elastic/expansion_fault_contraction.json" "--disable-ll-nvlink"
else
    echo "Skipping RDMA elastic tests: not all of mlx5_0..mlx5_3 are PORT_ACTIVE on $(hostname)"
fi

echo "==== nixl_ep elastic tests done ===="

echo "==== Running vLLM Elastic EP test ===="
(
    # Avoid SPCx loading HPC-X UCX 1.21; NIXL EP requires UCX >=1.22.
    unset NCCL_NET_PLUGIN
    unset UCX_NET_DEVICES
    # TODO: remove this override when vLLM updates FlashInfer with
    # https://github.com/flashinfer-ai/flashinfer/pull/4377.
    # FlashInfer 0.6.16.post3's default MLA backend fails to JIT on CUDA 13.3.
    export VLLM_ATTENTION_BACKEND=CUTLASS_MLA
    export PATH="${VLLM_ELASTIC_TEST_DIR}/.venv/bin:${PATH}"
    VLLM_LOG="${PWD}/elastic_ep_vllm_single_node.log"
    VLLM_REF="$(git -C "${VLLM_ELASTIC_TEST_DIR}" describe --tags --exact-match HEAD)"
    VLLM_COMMIT="$(git -C "${VLLM_ELASTIC_TEST_DIR}" rev-parse HEAD)"

    echo "vLLM source: VLLM_REF=${VLLM_REF} VLLM_COMMIT=${VLLM_COMMIT}"

    # Run vLLM's 2 -> 4 -> 2 Elastic EP scaling test with NIXL EP (Cover eager heavy traffic and CUDA graphs testing).
    (
        cd "${VLLM_ELASTIC_TEST_DIR}"
        VLLM_NIXL_EP_MAX_NUM_RANKS=4 \
        VLLM_TEST_ELASTIC_EP_ALL2ALL_BACKEND=nixl_ep \
        timeout 4500 "${VLLM_PYTHON}" -m pytest \
            "tests/distributed/test_elastic_ep.py::test_elastic_ep_scaling[enforce_eager_heavy]" \
            "tests/distributed/test_elastic_ep.py::test_elastic_ep_scaling[cuda_graphs_heavy]" \
            -v -s --tb=short 2>&1 | tee "${VLLM_LOG}"
    )

    if grep -Eiq '(^|[[:space:]])[0-9]+ skipped|SKIPPED' "${VLLM_LOG}"; then
        echo "ERROR: vLLM Elastic EP test was skipped" >&2
        exit 1
    fi

    echo "==== vLLM Elastic EP test done ===="
)
