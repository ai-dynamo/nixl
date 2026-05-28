#!/bin/sh
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

echo "==== Show memory info ===="
cat /proc/meminfo

ldd ./bin/nixlbench || true

echo "==== Building DOCA KV mock ===="
# NIXL's DOCA_MEMOS plugin was compiled against the real doca-kv at image-build
# time, but this CI node has no real NVMe-KV device. Run against a filesystem-backed
# mock that implements the same public DOCA KV API/ABI, swapped in at load time.
MOCK_SRC=/opt/doca_kv_mock
make -C "${MOCK_SRC}" shared

# Redirect the plugin's DT_NEEDED doca-kv soname to the mock, ahead of the real lib.
PLUGIN="${NIXL_PLUGIN_DIR}/libplugin_DOCA_MEMOS.so"
KV_SONAME=$(readelf -d "${PLUGIN}" | sed -n 's/.*NEEDED.*\[\(libdoca_kv[^]]*\)\].*/\1/p' | head -n1)
MOCK_LIB=$(find "${MOCK_SRC}" -name 'libdoca_kv*.so*' -type f | head -n1)
: "${KV_SONAME:?could not read a libdoca_kv soname from ${PLUGIN}}"
: "${MOCK_LIB:?mock libdoca_kv shared library not found under ${MOCK_SRC}}"
MOCK_STAGE=$(mktemp -d)
ln -sf "${MOCK_LIB}" "${MOCK_STAGE}/${KV_SONAME}"
export LD_LIBRARY_PATH="${MOCK_STAGE}:${LD_LIBRARY_PATH}"
echo "Using mock ${MOCK_LIB} as ${KV_SONAME}"

echo "==== Running Nixlbench tests (DOCA_MEMOS, mock) ===="
# Use the mock's /dev/null no-I/O mode: every KV op completes successfully without
# touching the filesystem. This smoke-tests the full plugin + nixlbench + DOCA KV
# API path without relying on the mock's io_uring file I/O, which is restricted in
# the rootless SLURM/enroot container. Hugepages are intentionally not used: that
# same rootless container has no privilege to reserve them.
./bin/nixlbench --backend=DOCA_MEMOS --doca_memos_device_name=/dev/null \
    --start_block_size=4194304 --max_block_size=4194304 \
    --start_batch_size=128 --max_batch_size=128 --op_type=READ,WRITE \
    --total_buffer_size=31998345216 --num_threads=3 --progress_threads=1 --num_iter=3000
