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
SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)

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

echo "==== Show memory info ===="
cat /proc/meminfo

ldd ./bin/nixlbench || true

echo "==== Building DOCA KV mock ===="
# This CI node has no NVMe-KV device. Build the DOCA_MEMOS unit-test mocks as a
# shared library and load it in place of every libdoca_* the plugin needs. The
# mock keeps values in memory and completes tasks in doca_pe_progress().
PLUGIN="${NIXL_PLUGIN_DIR}/libplugin_DOCA_MEMOS.so"
[ -f "${PLUGIN}" ] || { echo "ERROR: ${PLUGIN} was not built" >&2; exit 1; }
MOCK_DIR="${SRC_DIR}/test/unit/plugins/doca_memos"
MOCK_STAGE=$(mktemp -d)
MOCK_LIB="${MOCK_STAGE}/libdoca_mock.so"
cat > "${MOCK_STAGE}/mock_init.cpp" <<'EOF'
#include "doca_memos_mocks.h"
#include <limits>
__attribute__((constructor)) static void
mock_init() {
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.max_value_len = std::numeric_limits<uint32_t>::max();
}
EOF
# Mock headers first: they shadow the real doca-common ones, as in the unit test build.
DOCA_KV_CFLAGS=$(PKG_CONFIG_PATH="/opt/doca_memos/doca_kv/lib/${ARCH}-linux-gnu/pkgconfig" pkg-config --cflags doca-kv)
g++ -std=c++20 -O2 -fPIC -shared -DDOCA_ALLOW_EXPERIMENTAL_API -I"${MOCK_DIR}" ${DOCA_KV_CFLAGS} \
    "${MOCK_DIR}/doca_memos_mocks.cpp" "${MOCK_STAGE}/mock_init.cpp" -o "${MOCK_LIB}"

DOCA_SONAMES=$(readelf -d "${PLUGIN}" | sed -n 's/.*NEEDED.*\[\(libdoca_[^]]*\)\].*/\1/p')
: "${DOCA_SONAMES:?could not read any libdoca_* soname from ${PLUGIN}}"
for soname in ${DOCA_SONAMES}; do
    ln -sf "${MOCK_LIB}" "${MOCK_STAGE}/${soname}"
done
export LD_LIBRARY_PATH="${MOCK_STAGE}:${LD_LIBRARY_PATH}"
echo "Using mock ${MOCK_LIB} as:" ${DOCA_SONAMES}
# nixlbench treats a plugin that fails to load as a network backend and fails on etcd instead.
if ldd -r "${PLUGIN}" 2>&1 | grep 'undefined symbol: doca_'; then
    echo "ERROR: the mock does not provide every DOCA symbol ${PLUGIN} needs" >&2
    exit 1
fi

echo "==== Running Nixlbench tests (DOCA_MEMOS, mock) ===="
# The mock ignores the device path. Hugepages are not used: the rootless
# SLURM/enroot container has no privilege to reserve them.
for op_type in READ WRITE; do
    ./bin/nixlbench --backend=DOCA_MEMOS --doca_memos_device_name=/dev/null \
        --start_block_size=4194304 --max_block_size=4194304 \
        --start_batch_size=128 --max_batch_size=128 --op_type=${op_type} \
        --total_buffer_size=31998345216 --num_threads=3 --progress_threads=1 --num_iter=3000
done
