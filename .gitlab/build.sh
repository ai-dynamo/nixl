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

# Format elapsed seconds as human-readable string.
fmt_duration() {
    local secs=$1
    if [ "$secs" -ge 60 ]; then
        printf "%dm %ds" $((secs / 60)) $((secs % 60))
    else
        printf "%ds" "$secs"
    fi
}

# Suppress output on success, dump full log on failure.
# Prints a clear stage banner with elapsed time.
run_quiet() {
    local label="$1"; shift
    local logfile start_ts end_ts elapsed
    logfile=$(mktemp)
    start_ts=$(date +%s)
    { set +x; } 2>/dev/null
    echo ""
    echo "================================================================"
    echo "  START: ${label}"
    echo "================================================================"
    if "$@" > "$logfile" 2>&1; then
        rm -f "$logfile"
        end_ts=$(date +%s)
        elapsed=$((end_ts - start_ts))
        echo "----------------------------------------------------------------"
        echo "  OK: ${label}  [$(fmt_duration $elapsed)]"
        echo "================================================================"
        echo ""
        set -x
    else
        local rc=$?
        end_ts=$(date +%s)
        elapsed=$((end_ts - start_ts))
        echo "----------------------------------------------------------------"
        echo "  FAILED: ${label}  (exit ${rc})  [$(fmt_duration $elapsed)]"
        echo "================================================================"
        cat "$logfile"
        rm -f "$logfile"
        set -x
        return $rc
    fi
}

# Parse commandline arguments with first argument being the install directory
# and second argument being the UCX installation directory.
INSTALL_DIR=$1
UCX_INSTALL_DIR=$2
EXTRA_BUILD_ARGS=${3:-""}
NIXL_BUILD_DIR=${NIXL_BUILD_DIR:-nixl_build}
NIXLBENCH_BUILD_DIR=${NIXLBENCH_BUILD_DIR:-nixlbench_build}
# UCX_VERSION is the version of UCX to build override default with env variable.
UCX_VERSION=${UCX_VERSION:-v1.21.x}
# LIBFABRIC_VERSION is the version of libfabric to build override default with env variable.
LIBFABRIC_VERSION=${LIBFABRIC_VERSION:-v1.21.0}
# Abseil and gRPC versions for consistent toolchain build.
ABSL_TAG=${ABSL_TAG:-lts_2025_08_14}
GRPC_TAG=${GRPC_TAG:-v1.73.0}
# LIBFABRIC_INSTALL_DIR can be set via environment variable, defaults to INSTALL_DIR
LIBFABRIC_INSTALL_DIR=${LIBFABRIC_INSTALL_DIR:-$INSTALL_DIR}
# UCCL_COMMIT_SHA is the commit SHA of UCCL.
UCCL_COMMIT_SHA="2de728f1a27ea3f3b66059baf838f940e243ebc6"
AZURITE_VER="3.35.0"
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

export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib64:$LD_LIBRARY_PATH:${LIBFABRIC_INSTALL_DIR}/lib"
export CPATH="${INSTALL_DIR}/include:${LIBFABRIC_INSTALL_DIR}/include:$CPATH"
export PATH="${INSTALL_DIR}/bin:$HOME/.local/bin:/usr/local/bin:$HOME/.cargo/bin:$PATH"
export PKG_CONFIG_PATH="${INSTALL_DIR}/lib/pkgconfig:${INSTALL_DIR}/lib64/pkgconfig:${INSTALL_DIR}:${LIBFABRIC_INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH"
export NIXL_PLUGIN_DIR="${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins"
export CMAKE_PREFIX_PATH="${INSTALL_DIR}:${CMAKE_PREFIX_PATH}"

_build_total_start=$(date +%s)

if [ -n "$PRE_INSTALLED_ENV" ]; then
    echo "PRE_INSTALLED_ENV is set, skipping package installation"
else
    _phase_start=$(date +%s)
    # Some docker images are with broken installations:
    $SUDO rm -rf /usr/lib/cmake/grpc /usr/lib/cmake/protobuf

    run_quiet "Installing apt packages" bash -c "
        $SUDO apt-get -qq update &&
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
                                     libprotobuf-dev \
                                     libcpprest-dev \
                                     libaio-dev \
                                     libelf-dev \
                                     libgflags-dev \
                                     patchelf \
                                     meson \
                                     ninja-build \
                                     parallel \
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
                                     libxml2-dev \
                                     libcurl4-openssl-dev zlib1g-dev
    "
    $SUDO apt-mark hold liburing2 liburing-dev

    # Ubuntu 22.04 specific setup
    if grep -q "Ubuntu 22.04" /etc/os-release 2>/dev/null; then
        # Upgrade pip for '--break-system-packages' support
        $SUDO pip3 install --upgrade pip
    fi

    run_quiet "Installing Python packages" bash -c "
        $SUDO pip3 --no-cache-dir install --break-system-packages \
            meson meson-python pybind11 patchelf \
            click tabulate auditwheel tomlkit \
            pytest pytest-timeout zmq \
            mpmath typing-extensions sympy numpy \
            networkx MarkupSafe fsspec filelock jinja2
    "

    # Install torch from the CUDA-matched PyTorch index
    cuda_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d .)
    if [ -z "$cuda_version" ]; then
        echo "ERROR: unable to determine CUDA version from nvcc" >&2
        exit 1
    fi
    run_quiet "Installing PyTorch (cu${cuda_version})" bash -c "
        $SUDO pip3 --no-cache-dir install --break-system-packages \
            --index-url https://download.pytorch.org/whl/cu${cuda_version} torch
    "

    # Add DOCA repository and install packages
    ARCH_SUFFIX=$(if [ "${ARCH}" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi)
    MELLANOX_OS="$(. /etc/lsb-release; echo ${DISTRIB_ID}${DISTRIB_RELEASE} | tr A-Z a-z | tr -d .)"
    run_quiet "Installing DOCA packages" bash -c "
        wget --tries=3 --waitretry=5 --no-verbose https://www.mellanox.com/downloads/DOCA/DOCA_v3.2.0/host/doca-host_3.2.0-125000-25.10-${MELLANOX_OS}_${ARCH_SUFFIX}.deb -O ${TMPDIR}/doca-host.deb &&
        $SUDO dpkg -i ${TMPDIR}/doca-host.deb &&
        $SUDO apt-get update &&
        $SUDO apt-get upgrade -y &&
        $SUDO apt-get install -y --no-install-recommends doca-sdk-gpunetio libdoca-sdk-gpunetio-dev libdoca-sdk-verbs-dev
    "

    # Force reinstall of RDMA packages from DOCA repository
    # Reinstall needed to fix broken libibverbs-dev, which may lead to lack of Infiniband support.
    # Upgrade is not sufficient if the version is the same since apt skips the installation.
    run_quiet "Reinstalling RDMA packages" bash -c "
        $SUDO apt-get -qq -y install \
            --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
            libnuma-dev librdmacm-dev ibverbs-providers
    "

    run_quiet "Installing Rust" bash -c "
        wget --tries=3 --waitretry=5 https://static.rust-lang.org/rustup/dist/${ARCH}-unknown-linux-gnu/rustup-init -O ${TMPDIR}/rustup-init &&
        chmod +x ${TMPDIR}/rustup-init &&
        ${TMPDIR}/rustup-init -y --default-toolchain 1.86.0
    "

    run_quiet "Installing UV" bash -c "
        wget --tries=3 --waitretry=5 'https://astral.sh/uv/install.sh' -O ${TMPDIR}/install_uv.sh &&
        chmod +x ${TMPDIR}/install_uv.sh &&
        ${TMPDIR}/install_uv.sh
    "

    run_quiet "Installing Node.js and Azurite" bash -c "
        wget --tries=3 --waitretry=5 'https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh' -O ${TMPDIR}/install_nvm.sh &&
        chmod +x ${TMPDIR}/install_nvm.sh &&
        ${TMPDIR}/install_nvm.sh &&
        export NVM_DIR=${HOME}/.nvm &&
        [ -s \"\$NVM_DIR/nvm.sh\" ] && . \"\$NVM_DIR/nvm.sh\" &&
        nvm install --lts &&
        npm install -g azurite@${AZURITE_VER}
    "
    # Re-source nvm for subsequent commands
    export NVM_DIR=${HOME}/.nvm
    run_quiet "Sourcing nvm" bash -c "[ -s '${HOME}/.nvm/nvm.sh' ] && . '${HOME}/.nvm/nvm.sh'"

    run_quiet "Downloading libfabric" bash -c "
        wget --tries=3 --waitretry=5 -O '${TMPDIR}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2' 'https://github.com/ofiwg/libfabric/releases/download/${LIBFABRIC_VERSION}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2' &&
        tar xjf '${TMPDIR}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2' -C ${TMPDIR} &&
        rm '${TMPDIR}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2'
    "
    run_quiet "Building libfabric" bash -c "
        cd ${TMPDIR}/libfabric-* &&
        ./autogen.sh &&
        ./configure --prefix='${LIBFABRIC_INSTALL_DIR}' \
                    --disable-verbs \
                    --disable-psm3 \
                    --disable-opx \
                    --disable-usnic \
                    --disable-rstream \
                    --enable-efa &&
        make -j'$NPROC' &&
        make install &&
        $SUDO ldconfig &&
        cd .. &&
        rm -rf libfabric-*
    "

    run_quiet "Building Abseil" bash -c "
        cd ${TMPDIR} &&
        git clone https://github.com/abseil/abseil-cpp.git &&
        cd abseil-cpp &&
        git fetch --depth 1 origin '${ABSL_TAG}' &&
        git checkout '${ABSL_TAG}' &&
        mkdir -p build && cd build &&
        cmake .. \
            -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=ON \
            -DABSL_PROPAGATE_CXX_STD=ON \
            -DABSL_ENABLE_INSTALL=ON &&
        make -j'$NPROC' &&
        $SUDO make install &&
        $SUDO ldconfig &&
        cd ${TMPDIR} &&
        rm -rf abseil-cpp
    "

    run_quiet "Building gRPC" bash -c "
        cd ${TMPDIR} &&
        git clone --recurse-submodules -b '${GRPC_TAG}' --depth 1 --shallow-submodules https://github.com/grpc/grpc &&
        cd grpc &&
        mkdir -p cmake/build &&
        cd cmake/build &&
        cmake ../.. \
            -DgRPC_INSTALL=ON \
            -DgRPC_BUILD_TESTS=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_CXX_STANDARD=17 \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DCMAKE_PREFIX_PATH='${INSTALL_DIR}' \
            -Dabsl_DIR='${INSTALL_DIR}/lib/cmake/absl' \
            -DgRPC_SSL_PROVIDER=package \
            -DgRPC_ABSL_PROVIDER=package \
            -DgRPC_PROTOBUF_PROVIDER=module \
            -DgRPC_ZLIB_PROVIDER=package &&
        make -j'$NPROC' &&
        $SUDO make install &&
        $SUDO ldconfig &&
        cd ${TMPDIR} &&
        rm -rf grpc
    "

    run_quiet "Building etcd-cpp-apiv3" bash -c "
        cd ${TMPDIR} &&
        git clone --depth 1 https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git &&
        cd etcd-cpp-apiv3 &&
        sed -i '/^find_dependency(cpprestsdk)$/d' etcd-cpp-api-config.in.cmake &&
        mkdir build && cd build &&
        cmake .. \
            -DBUILD_ETCD_CORE_ONLY=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DETCD_CMAKE_CXX_STANDARD=17 \
            -DCMAKE_INSTALL_PREFIX='${INSTALL_DIR}' \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DCMAKE_PREFIX_PATH='${INSTALL_DIR}' &&
        make -j'$NPROC' &&
        $SUDO make install &&
        $SUDO ldconfig
    "

    run_quiet "Building aws-sdk-cpp" bash -c "
        cd ${TMPDIR} &&
        git clone --recurse-submodules --depth 1 --shallow-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.760 &&
        mkdir aws_sdk_build &&
        cd aws_sdk_build &&
        cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY='s3;s3-crt' -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local &&
        make -j'$NPROC' &&
        $SUDO make install &&
        cd .. &&
        rm -rf aws_sdk_build aws-sdk-cpp
    "

    run_quiet "Building gusli" bash -c "
        cd ${TMPDIR} &&
        git clone https://github.com/nvidia/gusli.git &&
        cd gusli &&
        $SUDO make all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0 VERBOSE=1 ALLOW_USE_URING=0 &&
        $SUDO ldconfig &&
        cd .. &&
        $SUDO rm -rf gusli
    "

    run_quiet "Building Mooncake" bash -c "
        cd ${TMPDIR} &&
        MOONCAKE_VERSION='${MOONCAKE_VERSION:-v0.3.10.post1}' &&
        echo \"MOONCAKE_VERSION: \${MOONCAKE_VERSION}\" &&
        git clone --depth 1 --branch \"\${MOONCAKE_VERSION}\" https://github.com/kvcache-ai/Mooncake.git &&
        cd Mooncake &&
        sed -i '/liburing-dev/d' dependencies.sh &&
        $SUDO bash dependencies.sh -y &&
        mkdir build && cd build &&
        cmake .. -DBUILD_SHARED_LIBS=ON -DWITH_STORE=OFF -G Ninja &&
        ninja &&
        $SUDO ninja install &&
        $SUDO ldconfig &&
        cd .. &&
        rm -rf Mooncake
    "

    run_quiet "Installing gtest-parallel" bash -c "
        cd ${TMPDIR} &&
        git clone --depth 1 https://github.com/google/gtest-parallel.git &&
        mkdir -p ${INSTALL_DIR}/bin &&
        cp ${TMPDIR}/gtest-parallel/* ${INSTALL_DIR}/bin/
    "

    run_quiet "Building Azure SDK" bash -c "
        cd ${TMPDIR} &&
        df -h &&
        curl -sL https://aka.ms/InstallAzureCLIDeb | $SUDO bash &&
        git clone --depth 1 https://github.com/Azure/azure-sdk-for-cpp.git --branch azure-storage-blobs_12.15.0 &&
        cd azure-sdk-for-cpp/ &&
        mkdir build && cd build &&
        AZURE_SDK_DISABLE_AUTO_VCPKG=1 cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DDISABLE_AMQP=ON -DDISABLE_AZURE_CORE_OPENTELEMETRY=ON &&
        cmake --build . --target azure-storage-blobs azure-identity &&
        $SUDO cmake --install sdk/core &&
        $SUDO cmake --install sdk/storage/azure-storage-common &&
        $SUDO cmake --install sdk/storage/azure-storage-blobs &&
        $SUDO cmake --install sdk/identity
    "
    { set +x; } 2>/dev/null
    _phase_end=$(date +%s)
    echo ""
    echo "################################################################"
    echo "  PHASE COMPLETE: Dependencies  [$(fmt_duration $((_phase_end - _phase_start)))]"
    echo "################################################################"
    echo ""
    set -x
fi # PRE_INSTALLED_ENV end

if [ -n "$PRE_INSTALLED_UCX_ENV" ]; then
    echo "PRE_INSTALLED_UCX_ENV is set, skipping UCX compilation"
else
    _phase_start=$(date +%s)
    if $HAS_GPU && test -d "$CUDA_HOME"; then
        run_quiet "Building UCCL" bash -c "
            cd ${TMPDIR} &&
            git clone https://github.com/uccl-project/uccl.git &&
            cd uccl && git checkout -q '${UCCL_COMMIT_SHA}' &&
            cd p2p &&
            make -j'$NPROC' &&
            $SUDO make install &&
            $SUDO ldconfig
        "
    else
        echo "No NVIDIA GPU(s) detected. Skipping UCCL installation."
    fi
    git clone https://github.com/openucx/ucx.git ${TMPDIR}/ucx
    run_quiet "Building UCX" bash -c "
        cd ${TMPDIR}/ucx &&
        git checkout '${UCX_VERSION}' &&
        ./autogen.sh &&
        ./contrib/configure-release-mt \
                --prefix='${UCX_INSTALL_DIR}' \
                --enable-shared \
                --disable-static \
                --disable-doxygen-doc \
                --enable-optimizations \
                --enable-cma \
                --enable-devel-headers \
                --with-verbs \
                --with-dm \
                --without-gdrcopy \
                ${UCX_CUDA_BUILD_ARGS} &&
        make -j'$NPROC' &&
        $SUDO make -j install-strip &&
        $SUDO ldconfig
    "
    { set +x; } 2>/dev/null
    _phase_end=$(date +%s)
    echo ""
    echo "################################################################"
    echo "  PHASE COMPLETE: UCX/UCCL  [$(fmt_duration $((_phase_end - _phase_start)))]"
    echo "################################################################"
    echo ""
    set -x
fi # PRE_INSTALLED_UCX_ENV end

$SUDO rm -rf ${TMPDIR}

# Disabling CUDA IPC not to use NVLINK, as it slows down local
# UCX transfers and can cause contention with local collectives.
export UCX_TLS=^cuda_ipc

if [ -n "$PRE_INSTALLED_NIXL_ENV" ]; then
    echo "PRE_INSTALLED_NIXL_ENV is set, skipping compilation"
else
    _phase_start=$(date +%s)
    { set +x; } 2>/dev/null
    echo ""
    echo "================================================================"
    echo "  START: Building NIXL"
    echo "================================================================"
    set -x
    _nixl_start=$(date +%s)

    # shellcheck disable=SC2086
    meson setup ${NIXL_BUILD_DIR} --prefix=${INSTALL_DIR} -Ducx_path=${UCX_INSTALL_DIR} -Dbuild_docs=true -Drust=false ${EXTRA_BUILD_ARGS} -Dlibfabric_path="${LIBFABRIC_INSTALL_DIR}" --buildtype=debug
    ninja -j"$NPROC" -C ${NIXL_BUILD_DIR} && ninja -j"$NPROC" -C ${NIXL_BUILD_DIR} install
    mkdir -p dist && cp ${NIXL_BUILD_DIR}/src/bindings/python/nixl-meta/nixl-*.whl dist/

    { set +x; } 2>/dev/null
    _nixl_end=$(date +%s)
    echo "----------------------------------------------------------------"
    echo "  OK: Building NIXL  [$(fmt_duration $((_nixl_end - _nixl_start)))]"
    echo "================================================================"
    echo ""
    set -x

    # TODO(kapila): Copy the nixl.pc file to the install directory if needed.
    # cp ${BUILD_DIR}/nixl.pc ${INSTALL_DIR}/lib/pkgconfig/nixl.pc

    { set +x; } 2>/dev/null
    echo ""
    echo "================================================================"
    echo "  START: Building nixlbench"
    echo "================================================================"
    set -x
    _bench_start=$(date +%s)

    cd benchmark/nixlbench
    meson setup ${NIXLBENCH_BUILD_DIR} -Dnixl_path=${INSTALL_DIR} -Dprefix=${INSTALL_DIR}
    ninja -j"$NPROC" -C ${NIXLBENCH_BUILD_DIR} && ninja -j"$NPROC" -C ${NIXLBENCH_BUILD_DIR} install

    { set +x; } 2>/dev/null
    _bench_end=$(date +%s)
    echo "----------------------------------------------------------------"
    echo "  OK: Building nixlbench  [$(fmt_duration $((_bench_end - _bench_start)))]"
    echo "================================================================"
    echo ""

    _phase_end=$(date +%s)
    echo "################################################################"
    echo "  PHASE COMPLETE: NIXL  [$(fmt_duration $((_phase_end - _phase_start)))]"
    echo "################################################################"
    echo ""
    set -x
fi

{ set +x; } 2>/dev/null
_build_total_end=$(date +%s)
echo ""
echo "################################################################"
echo "  TOTAL BUILD TIME: $(fmt_duration $((_build_total_end - _build_total_start)))"
echo "################################################################"
echo ""
set -x
