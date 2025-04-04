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

# Parse commandline arguments with first argument being the install directory
# and second argument being the UCX installation directory.
INSTALL_DIR=$1
UCX_INSTALL_DIR=$2

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir> <ucx_install_dir>"
    exit 1
fi

if [ -z "$UCX_INSTALL_DIR" ]; then
    UCX_INSTALL_DIR=$INSTALL_DIR
fi

apt-get -qq update
apt-get -qq install -y curl \
                             libnuma-dev \
                             numactl \
                             autotools-dev \
                             automake \
                             libtool \
                             libz-dev \
                             libiberty-dev \
                             flex \
                             build-essential \
                             cmake \
                             libibverbs-dev \
                             libgoogle-glog-dev \
                             libgtest-dev \
                             libjsoncpp-dev \
                             libpython3-dev \
                             libboost-all-dev \
                             libssl-dev \
                             libgrpc-dev \
                             libgrpc++-dev \
                             libprotobuf-dev \
                             meson \
                             ninja-build \
                             pkg-config \
                             protobuf-compiler-grpc \
                             pybind11-dev \
                             etcd-server \
                             net-tools \
                             pciutils \
                             libpci-dev \
                             uuid-dev \
                             ibverbs-utils \
                             libibmad-dev

curl -fSsL "https://github.com/openucx/ucx/tarball/v1.18.0" | tar xz
( \
  cd openucx-ucx* && \
  ./autogen.sh && \
  ./configure \
          --prefix=${UCX_INSTALL_DIR} \
          --enable-shared \
          --disable-static \
          --disable-doxygen-doc \
          --enable-optimizations \
          --enable-cma \
          --enable-devel-headers \
          --with-verbs \
          --with-dm \
          --enable-mt && \
        make -j && \
        make -j install-strip && \
        ldconfig \
)

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${INSTALL_DIR}/lib
export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH

# Disabling CUDA IPC not to use NVLINK, as it slows down local
# UCX transfers and can cause contention with local collectives.
export UCX_TLS=^cuda_ipc

meson setup nixl_build --prefix=${INSTALL_DIR} -Ducx_path=${UCX_INSTALL_DIR} -Dstatic_plugins=UCX
cd nixl_build && ninja && ninja install

# TODO(kapila): Copy the nixl.pc file to the install directory if needed.
# cp ${BUILD_DIR}/nixl.pc ${INSTALL_DIR}/lib/pkgconfig/nixl.pc
