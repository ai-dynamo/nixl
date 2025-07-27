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
EXTRA_BUILD_ARGS=${3:-""}
# UCX_VERSION is the version of UCX to build override default with env variable.
UCX_VERSION=${UCX_VERSION:-v1.18.0}

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

# Some docker images are with broken installations:
$SUDO rm -rf /usr/lib/cmake/grpc /usr/lib/cmake/protobuf

$SUDO apt-get -qq update
$SUDO apt-get -qq install -y curl \
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
                             libibmad-dev \
                             doxygen \
                             libclang-dev \
                             libgflags-dev \
                             libcpprest-dev \
                             etcd-client \
                             libcurl4-openssl-dev \
                             zlib1g-dev

# Detect OS version using /etc/os-release
OS_VERSION="unknown"
if [ -f /etc/os-release ]; then
    # Read from /etc/os-release file
    OS_VERSION=$(grep '^VERSION_ID=' /etc/os-release | cut -d'"' -f2 2>/dev/null || echo "unknown")
fi

echo "Detected OS version: $OS_VERSION"

INSTALL_3FS=false
ARCH_DETECTED=$(uname -m)
if [ "$OS_VERSION" = "22.04" ] && [ "$ARCH_DETECTED" = "x86_64" ]; then
    INSTALL_3FS=true
    echo "Ubuntu 22.04 and x86_64 architecture detected - will install 3FS and its dependencies"
else
    echo "OS version $OS_VERSION and architecture $ARCH_DETECTED detected - skipping 3FS installation"
fi

# Install 3FS-specific dependencies only if needed
if [ "$INSTALL_3FS" = true ]; then
    echo "Installing 3FS-specific dependencies..."
    $SUDO apt-get -qq install -y libdouble-conversion-dev \
                                 libuv1-dev \
                                 liblz4-dev \
                                 liblzma-dev \
                                 libdwarf-dev \
                                 libunwind-dev \
                                 libgoogle-perftools-dev \
                                 google-perftools \
                                 clang-format-14 \
                                 clang-14 \
                                 clang-tidy-14 \
                                 lld-14 \
                                 libjemalloc-dev
fi

# Reinstall RDMA packages to ensure proper installation
$SUDO apt-get -qq install -y --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
                                         libnuma-dev librdmacm-dev ibverbs-providers

# Install FoundationDB, Rust, FUSE and 3FS only for Ubuntu 22.04
if [ "$INSTALL_3FS" = true ]; then
      # Create completely isolated build environment to avoid PyTorch interference
    # echo "Creating completely isolated build environment for 3FS..."
    # export BUILD_CC="$CC"
    # export BUILD_CXX="$CXX"
    # export BUILD_CFLAGS="$CFLAGS"
    # export BUILD_CXXFLAGS="$CXXFLAGS"
    # export BUILD_LDFLAGS="$LDFLAGS"
    # export BUILD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    # export BUILD_LIBRARY_PATH="$LIBRARY_PATH"
    # export BUILD_PATH="$PATH"
    # export BUILD_PKG_CONFIG_PATH="$PKG_CONFIG_PATH"
    # export BUILD_CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
    # Reset to minimal environment - completely isolate from PyTorch
    # unset CC CXX CFLAGS CXXFLAGS LDFLAGS
    # export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib"
    # export LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib"
    # export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"
    # export PKG_CONFIG_PATH=""
    # export CMAKE_PREFIX_PATH=""

    # Setup Rust environment
    export RUSTUP_HOME=/usr/local/rustup
    export CARGO_HOME=/usr/local/cargo
    export PATH=/usr/local/cargo/bin:$PATH
    export RUST_VERSION=1.86.0
    export RUSTARCH=${ARCH:-x86_64}-unknown-linux-gnu

    # Download and install rustup
    wget --tries=3 --waitretry=5 \
        "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" \
        "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init.sha256"
    sha256sum -c rustup-init.sha256
    chmod +x rustup-init
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH}
    rm rustup-init*
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

    echo "Installing FoundationDB for 3FS..."
    wget -q -O /tmp/foundationdb-server_7.1.67-1_amd64.deb https://github.com/apple/foundationdb/releases/download/7.1.67/foundationdb-server_7.1.67-1_amd64.deb
    wget -q -O /tmp/foundationdb-clients_7.1.67-1_amd64.deb https://github.com/apple/foundationdb/releases/download/7.1.67/foundationdb-clients_7.1.67-1_amd64.deb

    # Install FoundationDB clients first (this should work fine)
    echo "Installing FoundationDB clients..."
    dpkg -i /tmp/foundationdb-clients_7.1.67-1_amd64.deb

    # Install FoundationDB server with force options to skip service management
    echo "Installing FoundationDB server without service management..."
    # Use dpkg with --force-depends and --force-configure-any to skip service setup
    dpkg -i --force-depends --force-configure-any /tmp/foundationdb-server_7.1.67-1_amd64.deb || {
        echo "FoundationDB server installation had issues (expected in container), but continuing..."
        # Try to configure the package manually without running postinst script
        dpkg --configure -a || true
    }

    # Clean up FoundationDB packages
    rm -f /tmp/foundationdb-*.deb

    # Verify FoundationDB installation
    echo "Verifying FoundationDB installation..."
    if command -v fdbcli >/dev/null 2>&1; then
        echo "FoundationDB CLI installed successfully"
    else
        echo "Warning: FoundationDB CLI not found, but continuing..."
    fi

    # Build and install FUSE 3.16.2 for 3FS
    echo "Building and installing FUSE 3.16.2 for 3FS..."
    wget -q -O /tmp/fuse-3.16.2.tar.gz https://github.com/libfuse/libfuse/releases/download/fuse-3.16.2/fuse-3.16.2.tar.gz
    tar -zxf /tmp/fuse-3.16.2.tar.gz -C /tmp
    mkdir -p /tmp/fuse-3.16.2/build
    meson setup /tmp/fuse-3.16.2 /tmp/fuse-3.16.2/build
    ninja -C /tmp/fuse-3.16.2/build
    # Install FUSE but skip the problematic device node creation
    ninja -C /tmp/fuse-3.16.2/build install || {
        echo "FUSE installation had issues (expected in container), but continuing..."
        # Try to install just the libraries and headers manually
        cp /tmp/fuse-3.16.2/build/lib/libfuse3.so* /usr/local/lib/x86_64-linux-gnu/ 2>/dev/null || true
        cp /tmp/fuse-3.16.2/build/util/fusermount3 /usr/local/bin/ 2>/dev/null || true
        cp /tmp/fuse-3.16.2/include/fuse*.h /usr/local/include/fuse3/ 2>/dev/null || true
        cp /tmp/fuse-3.16.2/include/cuse_lowlevel.h /usr/local/include/fuse3/ 2>/dev/null || true
        echo "FUSE libraries and headers installed manually"
    }
    rm -rf /tmp/fuse-3.16.2*

    # Clone and build 3FS
    echo "Cloning and building 3FS..."
    git clone https://github.com/deepseek-ai/3fs /workspace/3fs
    git -C /workspace/3fs submodule update --init --recursive
    /workspace/3fs/patches/apply.sh
    # Unset any PyTorch-specific environment variables
    unset PYTHONPATH
    unset TORCH_HOME
    unset CUDA_HOME
    unset CUDA_PATH
    # Clean any existing build directory to avoid cached configuration
    rm -rf /workspace/3fs/build
    cmake -S /workspace/3fs -B /workspace/3fs/build \
            -G "Unix Makefiles" \
            -DCMAKE_CXX_COMPILER=clang++-14 \
            -DCMAKE_C_COMPILER=clang-14 \
            -DARROW_JEMALLOC_USE_STATIC=OFF \
            -DARROW_MIMALLOC=OFF \
            -DARROW_JEMALLOC=OFF \
            -DARROW_USE_SYSTEM_MALLOC=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DARROW_JEMALLOC_BUILD_JOBS=1
            # -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            # -DARROW_USE_SYSTEM_MALLOC=ON \
            # -DARROW_BUILD_STATIC=OFF \
            # -DARROW_BUILD_SHARED=ON \
            # -DARROW_DEPENDENCY_SOURCE=BUNDLED \
            # -DARROW_WITH_JEMALLOC=OFF \
            # -DARROW_WITH_MIMALLOC=OFF \
            # -DARROW_JEMALLOC_USE_SHARED=OFF \
            # -DARROW_WITH_ZSTD=OFF \
            # -DARROW_WITH_ZLIB=OFF \
            # -DARROW_WITH_SNAPPY=OFF \
            # -DARROW_WITH_LZ4=OFF \
            # -DARROW_WITH_BZ2=OFF \
            # -DARROW_WITH_BROTLI=OFF \
    cmake --build /workspace/3fs/build -j 16
    # # Restore original environment
    # if [ -n "$BUILD_CC" ]; then export CC="$BUILD_CC"; fi
    # if [ -n "$BUILD_CXX" ]; then export CXX="$BUILD_CXX"; fi
    # if [ -n "$BUILD_CFLAGS" ]; then export CFLAGS="$BUILD_CFLAGS"; fi
    # if [ -n "$BUILD_CXXFLAGS" ]; then export CXXFLAGS="$BUILD_CXXFLAGS"; fi
    # if [ -n "$BUILD_LDFLAGS" ]; then export LDFLAGS="$BUILD_LDFLAGS"; fi
    # if [ -n "$BUILD_LD_LIBRARY_PATH" ]; then export LD_LIBRARY_PATH="$BUILD_LD_LIBRARY_PATH"; fi
    # if [ -n "$BUILD_LIBRARY_PATH" ]; then export LIBRARY_PATH="$BUILD_LIBRARY_PATH"; fi
    # if [ -n "$BUILD_PATH" ]; then export PATH="$BUILD_PATH"; fi
    # if [ -n "$BUILD_PKG_CONFIG_PATH" ]; then export PKG_CONFIG_PATH="$BUILD_PKG_CONFIG_PATH"; fi
    # if [ -n "$BUILD_CMAKE_PREFIX_PATH" ]; then export CMAKE_PREFIX_PATH="$BUILD_CMAKE_PREFIX_PATH"; fi
    echo "3FS build completed successfully AAAAAAAAAAAAAAAAAAAAAAAA"
    cp /workspace/3fs/build/bin/* /usr/local/bin/
    mkdir -p /usr/include/hf3fs
    cp /workspace/3fs/src/lib/api/*.h /usr/include/hf3fs/
    cp /workspace/3fs/build/src/lib/api/libhf3fs_api_shared.so /usr/lib/
    mkdir -p /etc/3fs

    # Add 3FS binaries to PATH for easy access
    export PATH="/usr/local/bin:$PATH"
    echo "3FS installation completed successfully"
else
    echo "Skipping 3FS installation for OS version $OS_VERSION"
fi

curl -fSsL "https://github.com/openucx/ucx/tarball/${UCX_VERSION}" | tar xz
( \
  cd openucx-ucx* && \
  ./autogen.sh && \
  ./configure \
          --prefix="${UCX_INSTALL_DIR}" \
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
        $SUDO ldconfig \
)

( \
  cd /tmp && \
  git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git && \
  cd etcd-cpp-apiv3 && \
  mkdir build && cd build && \
  cmake .. && \
  make -j$(nproc) && \
  $SUDO make install && \
  $SUDO ldconfig \
)

( \
  git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && \
  mkdir aws_sdk_build && \
  cd aws_sdk_build && \
  cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
  make -j$(nproc) && \
  $SUDO make install
)

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${INSTALL_DIR}/lib
export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=${INSTALL_DIR}:${CMAKE_PREFIX_PATH}

# Disabling CUDA IPC not to use NVLINK, as it slows down local
# UCX transfers and can cause contention with local collectives.
export UCX_TLS=^cuda_ipc

meson setup nixl_build --prefix=${INSTALL_DIR} -Ducx_path=${UCX_INSTALL_DIR} -Dbuild_docs=true ${EXTRA_BUILD_ARGS}
cd nixl_build && ninja && ninja install

# TODO(kapila): Copy the nixl.pc file to the install directory if needed.
# cp ${BUILD_DIR}/nixl.pc ${INSTALL_DIR}/lib/pkgconfig/nixl.pc