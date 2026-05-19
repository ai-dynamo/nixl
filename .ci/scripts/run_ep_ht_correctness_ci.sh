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
. "$(dirname "$0")/common.sh"

set -e
set -x
set -o pipefail

INSTALL_DIR=$1

if [ -z "${INSTALL_DIR}" ]; then
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
export NIXL_PREFIX=${INSTALL_DIR}
export NIXL_DEBUG_LOGGING=yes

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$(get_next_tcp_port)}
export WORLD_SIZE=1
export RANK=0

echo "==== Running NIXL EP HT 4-GPU correctness test ===="
echo "This check does not cover multi-node RDMA or performance."

python3 examples/device/ep/tests/test_ht.py \
    --num-processes "${NIXL_EP_HT_CI_NUM_PROCESSES:-4}" \
    --num-tokens "${NIXL_EP_HT_CI_NUM_TOKENS:-256}" \
    --hidden "${NIXL_EP_HT_CI_HIDDEN:-2048}" \
    --num-topk "${NIXL_EP_HT_CI_NUM_TOPK:-4}" \
    --num-experts "${NIXL_EP_HT_CI_NUM_EXPERTS:-64}" \
    --ci-correctness-only
