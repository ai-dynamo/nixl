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
set -o pipefail

# Parse commandline arguments with first argument being the install directory
# and second argument being the UCX installation directory.
INSTALL_DIR=$1
UCX_INSTALL_DIR=$2
EXTRA_BUILD_ARGS=${3:-""}
NIXL_BUILD_DIR=${NIXL_BUILD_DIR:-nixl_build}
NIXLBENCH_BUILD_DIR=${NIXLBENCH_BUILD_DIR:-nixlbench_build}
# UCX_VERSION is the version of UCX to build override default with env variable.
UCX_VERSION=${UCX_VERSION:-v1.20.x}
# LIBFABRIC_VERSION is the version of libfabric to build override default with env variable.
LIBFABRIC_VERSION=${LIBFABRIC_VERSION:-v1.21.0}
# LIBFABRIC_INSTALL_DIR can be set via environment variable, defaults to INSTALL_DIR
LIBFABRIC_INSTALL_DIR=${LIBFABRIC_INSTALL_DIR:-$INSTALL_DIR}
# UCCL_COMMIT_SHA is the commit SHA of UCCL.
UCCL_COMMIT_SHA="a962f611021afc2e3c9358f6da4ae96539cbca0f"
TMPDIR=$(mktemp -d)

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir> <ucx_install_dir>"
    exit 1
fi

if [ -z "$UCX_INSTALL_DIR" ]; then
    UCX_INSTALL_DIR=$INSTALL_DIR
fi


# For running as user - check if running as root, if not set sudo variable
if [ "$(id -u)" -ne 0 ]; then
    SUDO=sudo
else
    SUDO=""
fi

ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"

if [ -n "$PRE_INSTALLED_ENV" ]; then
    echo "PRE_INSTALLED_ENV is set, skipping package installation"
else
    # Some docker images are with broken installations:
    $SUDO rm -rf /usr/lib/cmake/grpc /usr/lib/cmake/protobuf

    $SUDO apt-get -qq update
    $SUDO apt-get -qq install -y python3-dev \
                                 python3-pip \
                                 curl \
                                 wget \
                                 libnuma-dev \
                                 numactl \
                                 autotools-dev \
                                 automake \
                                 git \
                                 libtool \
                                 libz-dev \
                                 libiberty-dev \
                                 flex \
                                 build-essential \
                                 cmake \
                                 libgoogle-glog-dev \
                                 libgtest-dev \
                                 libgmock-dev \
                                 libjsoncpp-dev \
                                 libpython3-dev \
                                 libboost-all-dev \
                                 libssl-dev \
                                 libgrpc-dev \
                                 libgrpc++-dev \
                                 libprotobuf-dev \
                                 libcpprest-dev \
                                 libaio-dev \
                                 liburing-dev \
                                 libelf-dev \
                                 libgflags-dev \
                                 patchelf \
                                 meson \
                                 ninja-build \
                                 pkg-config \
                                 protobuf-compiler-grpc \
                                 pybind11-dev \
                                 etcd-server \
                                 net-tools \
                                 iproute2 \
                                 pciutils \
                                 libpci-dev \
                                 uuid-dev \
                                 libibmad-dev \
                                 doxygen \
                                 clang \
                                 hwloc \
                                 libhwloc-dev \
                                 libcurl4-openssl-dev zlib1g-dev # aws-sdk-cpp dependencies

    # Ubuntu 22.04 specific setup
    if grep -q "Ubuntu 22.04" /etc/os-release 2>/dev/null; then
        # Upgrade pip for '--break-system-packages' support
        $SUDO pip3 install --upgrade pip
    fi

    # Install python dependencies and upgrade to latest version
    $SUDO pip3 --no-cache-dir install --break-system-packages \
        meson meson-python pybind11 patchelf \
        click tabulate auditwheel tomlkit \
        pytest pytest-timeout zmq \
        mpmath typing-extensions sympy numpy \
        networkx MarkupSafe fsspec filelock jinja2 torch

    # Add DOCA repository and install packages
    ARCH_SUFFIX=$(if [ "${ARCH}" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi)
    MELLANOX_OS="$(. /etc/lsb-release; echo ${DISTRIB_ID}${DISTRIB_RELEASE} | tr A-Z a-z | tr -d .)"
    wget --tries=3 --waitretry=5 --no-verbose https://www.mellanox.com/downloads/DOCA/DOCA_v3.2.0/host/doca-host_3.2.0-125000-25.10-${MELLANOX_OS}_${ARCH_SUFFIX}.deb -O ${TMPDIR}/doca-host.deb
    $SUDO dpkg -i ${TMPDIR}/doca-host.deb
    $SUDO apt-get update
    $SUDO apt-get upgrade -y
    $SUDO apt-get install -y --no-install-recommends doca-sdk-gpunetio libdoca-sdk-gpunetio-dev libdoca-sdk-verbs-dev

    # Force reinstall of RDMA packages from DOCA repository
    # Reinstall needed to fix broken libibverbs-dev, which may lead to lack of Infiniband support.
    # Upgrade is not sufficient if the version is the same since apt skips the installation.
    $SUDO apt-get -qq -y install \
        --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
        libnuma-dev librdmacm-dev ibverbs-providers

    wget --tries=3 --waitretry=5 https://static.rust-lang.org/rustup/dist/${ARCH}-unknown-linux-gnu/rustup-init -O ${TMPDIR}/rustup-init
    chmod +x ${TMPDIR}/rustup-init
    ${TMPDIR}/rustup-init -y --default-toolchain 1.86.0

    wget --tries=3 --waitretry=5 "https://astral.sh/uv/install.sh" -O ${TMPDIR}/install_uv.sh
    chmod +x ${TMPDIR}/install_uv.sh
    ${TMPDIR}/install_uv.sh

    wget --tries=3 --waitretry=5 -O "${TMPDIR}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2" "https://github.com/ofiwg/libfabric/releases/download/${LIBFABRIC_VERSION}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2"
    tar xjf "${TMPDIR}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2" -C ${TMPDIR}
    rm "${TMPDIR}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2"
    ( \
      cd ${TMPDIR}/libfabric-* && \
      ./autogen.sh && \
      ./configure --prefix="${LIBFABRIC_INSTALL_DIR}" \
                  --disable-verbs \
                  --disable-psm3 \
                  --disable-opx \
                  --disable-usnic \
                  --disable-rstream \
                  --enable-efa && \
      make -j"$NPROC" && \
      make install && \
      $SUDO ldconfig \
    )

    ( \
      cd ${TMPDIR} && \
      git clone --depth 1 https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git && \
      cd etcd-cpp-apiv3 && \
      mkdir build && cd build && \
      cmake .. && \
      make -j"$NPROC" && \
      $SUDO make install && \
      $SUDO ldconfig \
    )

    ( \
      cd ${TMPDIR} && \
      git clone --recurse-submodules --depth 1 --shallow-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && \
      mkdir aws_sdk_build && \
      cd aws_sdk_build && \
      cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3;s3-crt" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
      make -j"$NPROC" && \
      $SUDO make install
    )

    ( \
      cd ${TMPDIR} && \
      git clone https://github.com/nvidia/gusli.git && \
      cd gusli && \
      $SUDO make all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0 VERBOSE=1 ALLOW_USE_URING=0 && \
      $SUDO ldconfig
    )

    ( \
      cd ${TMPDIR} && \
      git clone --depth 1 https://github.com/kvcache-ai/Mooncake.git && \
      cd Mooncake && \
      $SUDO bash dependencies.sh && \
      mkdir build && cd build && \
      cmake .. -DBUILD_SHARED_LIBS=ON && \
      make -j4 && \
      $SUDO make install && \
      $SUDO ldconfig
    )

    ( \
      cd ${TMPDIR} &&
      git clone --depth 1 https://github.com/google/gtest-parallel.git &&
      mkdir -p ${INSTALL_DIR}/bin &&
      cp ${TMPDIR}/gtest-parallel/* ${INSTALL_DIR}/bin/
    )
fi # PRE_INSTALLED_ENV end

if [ -n "$PRE_INSTALLED_UCX_ENV" ]; then
    echo "PRE_INSTALLED_UCX_ENV is set, skipping UCX compilation"
else
    if $HAS_GPU && test -d "$CUDA_HOME"; then
       ( \
        cd ${TMPDIR} && \
        git clone https://github.com/uccl-project/uccl.git && \
        cd uccl && git checkout -q "${UCCL_COMMIT_SHA}" && \
        cd p2p && \
        make -j"$NPROC" && \
        $SUDO make install && \
        $SUDO ldconfig
        )
    else
        echo "No NVIDIA GPU(s) detected. Skipping UCCL installation."
    fi
    curl -fSsL "https://github.com/openucx/ucx/tarball/${UCX_VERSION}" | tar xz -C ${TMPDIR}
    ( \
    cd ${TMPDIR}/openucx-ucx* && \
    ./autogen.sh && \
    ./contrib/configure-release-mt \
            --prefix="${UCX_INSTALL_DIR}" \
            --enable-shared \
            --disable-static \
            --disable-doxygen-doc \
            --enable-optimizations \
            --enable-cma \
            --enable-devel-headers \
            --with-verbs \
            --with-dm \
            ${UCX_CUDA_BUILD_ARGS} && \
          make -j"$NPROC" && \
          make -j install-strip && \
          $SUDO ldconfig \
    )
fi # PRE_INSTALLED_UCX_ENV end

$SUDO rm -rf ${TMPDIR}

export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib64:$LD_LIBRARY_PATH:${LIBFABRIC_INSTALL_DIR}/lib"
export CPATH="${INSTALL_DIR}/include:${LIBFABRIC_INSTALL_DIR}/include:$CPATH"
export PATH="${INSTALL_DIR}/bin:$HOME/.local/bin:/usr/local/bin:$HOME/.cargo/bin:$PATH"
export PKG_CONFIG_PATH="${INSTALL_DIR}/lib/pkgconfig:${INSTALL_DIR}/lib64/pkgconfig:${INSTALL_DIR}:${LIBFABRIC_INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH"
export NIXL_PLUGIN_DIR="${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins"
export CMAKE_PREFIX_PATH="${INSTALL_DIR}:${CMAKE_PREFIX_PATH}"

# Disabling CUDA IPC not to use NVLINK, as it slows down local
# UCX transfers and can cause contention with local collectives.
export UCX_TLS=^cuda_ipc

if [ -n "$PRE_INSTALLED_NIXL_ENV" ]; then
    echo "PRE_INSTALLED_NIXL_ENV is set, skipping compilation"
else
    # shellcheck disable=SC2086
    meson setup ${NIXL_BUILD_DIR} --prefix=${INSTALL_DIR} -Ducx_path=${UCX_INSTALL_DIR} -Dbuild_docs=true -Drust=false ${EXTRA_BUILD_ARGS} -Dlibfabric_path="${LIBFABRIC_INSTALL_DIR}" --buildtype=debug
    ninja -j"$NPROC" -C ${NIXL_BUILD_DIR} && ninja -j"$NPROC" -C ${NIXL_BUILD_DIR} install
    mkdir -p dist && cp ${NIXL_BUILD_DIR}/src/bindings/python/nixl-meta/nixl-*.whl dist/

    # TODO(kapila): Copy the nixl.pc file to the install directory if needed.
    # cp ${BUILD_DIR}/nixl.pc ${INSTALL_DIR}/lib/pkgconfig/nixl.pc

    cd benchmark/nixlbench
    meson setup ${NIXLBENCH_BUILD_DIR} -Dnixl_path=${INSTALL_DIR} -Dprefix=${INSTALL_DIR}
    ninja -j"$NPROC" -C ${NIXLBENCH_BUILD_DIR} && ninja -j"$NPROC" -C ${NIXLBENCH_BUILD_DIR} install
fi
