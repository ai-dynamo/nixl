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

# nixl_ep elastic CI: run only EP tests (invoked from nixl-ci-dl-gpu-ep flow).

# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

set -e
set -x

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
export NIXL_DEBUG_LOGGING=yes

# The PR image installs the CUDA-versioned bindings (nixl_ep_cu*) under
# ${INSTALL_DIR}/lib/python3/dist-packages via meson's --prefix, which is not
# on Python's default sys.path. Expose it so the nixl_ep dispatcher can load
# nixl_ep_cu13 at runtime.
export PYTHONPATH="${INSTALL_DIR}/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"

# Install the nixl meta wheel (provides the nixl_ep dispatcher package that
# re-exports from nixl_ep_cu13). The wheel is staged in dist/ by the PR image
# build; without this install `import nixl_ep` fails with ModuleNotFoundError.
if [ -n "$VIRTUAL_ENV" ] && grep -q '^uv =' "$VIRTUAL_ENV/pyvenv.cfg" 2>/dev/null; then
    pip3="uv pip"
else
    pip3="python3 -m pip"
fi
$pip3 install --break-system-packages dist/nixl-*none-any.whl

echo "==== Show system info ===="
env
nvidia-smi topo -m || true
ibv_devinfo || true
uname -a || true
cat /sys/devices/virtual/dmi/id/product_name || true

echo "==== NVIDIA Peermem check ===="
if ! lsmod | grep -q nvidia_peermem; then
    echo "nvidia_peermem module not loaded"
fi

if [ -f /sys/kernel/mm/memory_peers/nv_mem/version ]; then
    cat /sys/kernel/mm/memory_peers/nv_mem/version
else
    echo "/sys/kernel/mm/memory_peers/nv_mem/version not found "
fi

if [ -f /sys/module/nvidia_peermem/version ]; then
    cat /sys/module/nvidia_peermem/version
else
    echo "/sys/module/nvidia_peermem/version not found"
fi

if [ -f /sys/module/nv_peer_mem/version ]; then
    cat /sys/module/nv_peer_mem/version
else
    echo "/sys/module/nv_peer_mem/version not found"
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
            export UCX_TLS=cuda_copy,cuda_ipc,sm,self
        fi
        PYTHONPATH="${NIXL_BUILD_DIR}/${EP_SRC_DIR}:${EP_SRC_DIR}/tests:${EP_SRC_DIR}/tests/elastic${PYTHONPATH:+:$PYTHONPATH}" \
        timeout 300 python3 ${EP_SRC_DIR}/tests/elastic/elastic.py \
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
