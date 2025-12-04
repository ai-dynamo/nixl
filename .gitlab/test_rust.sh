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


export PATH=$HOME/.cargo/bin:$PATH

which cargo

cargo --version

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins:/usr/local/lib:${INSTALL_DIR}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib64/pkgconfig:${INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib/pkgconfig:${INSTALL_DIR}/lib/$ARCH-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins
export NIXL_PREFIX=${INSTALL_DIR}
export NIXL_NO_STUBS_FALLBACK=1

cargo test -- --test-threads=1

# test that stubs and real wrapper defined APIs / symbols match
g++ -c ./src/bindings/rust/wrapper.cpp -o wrapper.o -I ./src/api/cpp/
g++ -c ./src/bindings/rust/stubs.cpp -o stubs.o

nm -C --defined-only wrapper.o | awk '$2 ~ /^T$/ {print $3}' | sort > wrapper_symbols.txt
nm -C --defined-only stubs.o | awk '$2 ~ /^T$/ {print $3}' | grep -v nixl_capi_stub_abort | sort > stubs_symbols.txt
diff wrapper_symbols.txt stubs_symbols.txt
diff_status=$?
rm wrapper_symbols.txt stubs_symbols.txt


if [ "$diff_status" -ne 0 ]; then
    echo "Stubs API and wrapper API differ"
    exit 1
else
    echo "Stubs API and wrapper API match"
fi


# test stubs build
cargo build --features stub-api

cargo package

