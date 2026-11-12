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

NUM_RANKS=4
NIXL_EP_RDMA_HCAS=${NIXL_EP_RDMA_HCAS:-"mlx5_0 mlx5_1 mlx5_2 mlx5_3"}

# GPU-initiated RDMA needs every configured NIC to be active and UCX to offer
# rc_gda on all ranks. Print the discovered connectivity and the reason for a
# negative decision so new GPU partitions can be diagnosed from the CI log.
rdma_supported() {
    local hca info states link ucx_out ucx_rc=0 gda gda_devs gda_failed gda_gpus
    local why=""
    local ucx_info="${INSTALL_DIR}/bin/ucx_info"
    local -a hcas

    echo "==== RDMA check on $(hostname) ===="
    read -r -a hcas <<<"${NIXL_EP_RDMA_HCAS}"
    for hca in "${hcas[@]}"; do
        info=$(ibv_devinfo -d "$hca" 2>/dev/null) || true
        states=$(awk '$1 == "state:" { printf "%s%s", sep, $2; sep = "," }' <<<"$info")
        link=$(awk '$1 == "link_layer:" { print $2; exit }' <<<"$info")
        echo "${hca}: ${states:-not found}${link:+ (${link})}"
        [[ "$states" == *PORT_ACTIVE* ]] ||
            why+="${why:+; }${hca} is ${states:-not found}"
    done

    if [ ! -x "$ucx_info" ]; then
        why+="${why:+; }${ucx_info} not found; UCX rc_gda capability is unverified"
    else
        "$ucx_info" -v | head -n 4 || true
        # A fresh process has no active CUDA context, and UCX then hides rc_gda.
        # Debug logging explains why UCX leaves a NIC or GPU out.
        ucx_out=$(env -u UCX_TLS -u UCX_NET_DEVICES \
            UCX_IB_GDA_RETAIN_INACTIVE_CTX=yes UCX_LOG_LEVEL=debug \
            timeout 120 "$ucx_info" -d 2>&1) || ucx_rc=$?
        sed -nE 's/.*gdaki\.c:[0-9]+ +UCX +[A-Z]+ +/UCX: /p' <<<"$ucx_out" |
            sort -u | head -n 20 || true
        if [ "$ucx_rc" -ne 0 ]; then
            why+="${why:+; }ucx_info -d failed (exit ${ucx_rc}); UCX rc_gda capability is unverified"
        else
            # ucx_info lists a device before opening it and marks a device it
            # cannot open with "< failed to open interface >".
            gda=$(awk '
                function flush() { if (dev != "") print state, dev; dev = "" }
                /^# +Transport:/ { flush(); transport = $NF }
                /^# +Device:/ && transport == "rc_gda" { dev = $NF; state = "ok" }
                /^# +< failed to open interface >/ { state = "failed" }
                END { flush() }
            ' <<<"$ucx_out" | sort -u)
            gda_devs=$(awk '$1 == "ok" { print $2 }' <<<"$gda" | paste -sd' ')
            gda_failed=$(awk '$1 == "failed" { print $2 }' <<<"$gda" | paste -sd' ')
            gda_gpus=$(tr ' ' '\n' <<<"$gda_devs" |
                sed -n 's/-.*//p' | sort -u | wc -l)
            echo "UCX rc_gda devices (GPU-NIC): ${gda_devs:-none}"
            [ -z "$gda_failed" ] ||
                echo "UCX rc_gda devices that failed to open: ${gda_failed}"
            [ "$gda_gpus" -ge "$NUM_RANKS" ] ||
                why+="${why:+; }UCX offers rc_gda on ${gda_gpus} of ${NUM_RANKS} GPUs"
        fi
    fi

    if [ -n "$why" ]; then
        echo "RDMA check: not supported: ${why}"
        return 1
    fi
    echo "RDMA check: supported"
}

echo "==== Show system info ===="
env
nvidia-smi topo -m || true
ibv_devinfo || true
uname -a || true
cat /sys/devices/virtual/dmi/id/product_name || true
if (set +x; rdma_supported); then
    RDMA_SUPPORTED=true
else
    RDMA_SUPPORTED=false
fi

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

# RDMA (--disable-ll-nvlink), only when the capability check above passed.
if [ "$RDMA_SUPPORTED" = true ]; then
    run_elastic_test "${EP_SRC_DIR}/tests/elastic/no_expansion.json" "--disable-ll-nvlink"
    run_elastic_test "${EP_SRC_DIR}/tests/elastic/expansion_fault_contraction.json" "--disable-ll-nvlink"
else
    echo "Skipping RDMA elastic tests on $(hostname): see the RDMA check above"
fi

echo "==== nixl_ep elastic tests done ===="

echo "==== Running free-after-disconnect test ===="
(
    unset NIXL_ETCD_ENDPOINTS NIXL_ETCD_PEER_URLS NIXL_ETCD_NAMESPACE
    unset UCX_NET_DEVICES UCX_TLS
    PYTHONPATH="${NIXL_BUILD_DIR}/${EP_SRC_DIR}${PYTHONPATH:+:$PYTHONPATH}" \
    timeout 300 "${VLLM_PYTHON}" ${EP_SRC_DIR}/tests/free_after_disconnect.py
)
echo "==== free-after-disconnect test done ===="

echo "==== Running vLLM Elastic EP test ===="
(
    # Avoid SPCx loading HPC-X UCX 1.21; NIXL EP requires UCX >=1.22.
    unset NCCL_NET_PLUGIN
    unset UCX_NET_DEVICES
    export PATH="${VLLM_ELASTIC_TEST_DIR}/.venv/bin:${PATH}"
    VLLM_LOG="${PWD}/elastic_ep_vllm_single_node.log"
    VLLM_REF="$(git -C "${VLLM_ELASTIC_TEST_DIR}" describe --tags --exact-match HEAD)"
    VLLM_COMMIT="$(git -C "${VLLM_ELASTIC_TEST_DIR}" rev-parse HEAD)"

    echo "vLLM source: VLLM_REF=${VLLM_REF} VLLM_COMMIT=${VLLM_COMMIT}"

    (
        cd "${VLLM_ELASTIC_TEST_DIR}"
        # TODO: drop VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS once VLLM_REF includes
        # the upstream fix for FlashInfer autotuning on Elastic EP process groups.
        VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=trtllm_batch_decode_mla \
        timeout 4500 "${VLLM_PYTHON}" -m pytest \
            "tests/distributed/test_elastic_ep.py::test_elastic_ep_scaling[cuda_graphs_heavy_nixl_ep]" \
            -v -s --tb=short --color=no 2>&1 | tee "${VLLM_LOG}"
    )

    if ! grep -Eq '=+ 1 passed(, [0-9]+ warnings?)? in [0-9.]+s' "${VLLM_LOG}"; then
        echo "ERROR: vLLM Elastic EP test did not pass" >&2
        exit 1
    fi

    echo "==== vLLM Elastic EP test done ===="
)
