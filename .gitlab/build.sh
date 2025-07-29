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

    # ========== COMPREHENSIVE DIAGNOSTIC SECTION ==========
    echo "========== 3FS BUILD DIAGNOSTICS START =========="

    # System information
    echo "=== SYSTEM INFO ==="
    echo "Hostname: $(hostname)"
    echo "Kernel: $(uname -a)"
    echo "CPU Info: $(nproc) cores"
    echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "Disk space: $(df -h /workspace | tail -1)"
    echo "Load average: $(uptime)"

    # Architecture and OS details
    echo "=== ARCHITECTURE & OS ==="
    echo "Architecture: $(uname -m)"
    echo "OS Release:"
    cat /etc/os-release | head -5
    echo "GCC Version: $(gcc --version | head -1)"
    echo "Clang-14 Version: $(clang-14 --version | head -1 2>/dev/null || echo 'clang-14 not found')"
    echo "CMake Version: $(cmake --version | head -1)"
    echo "Make Version: $(make --version | head -1)"

    # Environment variables
    echo "=== ENVIRONMENT VARIABLES ==="
    echo "PATH: $PATH"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo "LIBRARY_PATH: $LIBRARY_PATH"
    echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
    echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
    echo "CC: $CC"
    echo "CXX: $CXX"
    echo "CFLAGS: $CFLAGS"
    echo "CXXFLAGS: $CXXFLAGS"
    echo "LDFLAGS: $LDFLAGS"
    echo "RUSTUP_HOME: $RUSTUP_HOME"
    echo "CARGO_HOME: $CARGO_HOME"

    # Check critical dependencies
    echo "=== DEPENDENCY CHECKS ==="
    echo "Clang-14 location: $(which clang-14 2>/dev/null || echo 'NOT FOUND')"
    echo "Clang++-14 location: $(which clang++-14 2>/dev/null || echo 'NOT FOUND')"
    echo "Rust toolchain:"
    rustc --version 2>/dev/null || echo "Rust not found"
    cargo --version 2>/dev/null || echo "Cargo not found"
    echo "Git version: $(git --version)"

    # Check Rust installation specifically
    echo "=== RUST ENVIRONMENT ==="
    ls -la /usr/local/rustup 2>/dev/null || echo "Rustup directory not found"
    ls -la /usr/local/cargo 2>/dev/null || echo "Cargo directory not found"
    ls -la /usr/local/cargo/bin/ 2>/dev/null || echo "Cargo bin directory not found"
    echo "Rust version details:"
    rustc --version 2>/dev/null || echo "Rust compiler not found"
    cargo --version 2>/dev/null || echo "Cargo not found"

    # Check 3FS source state
    echo "=== 3FS SOURCE STATE ==="
    echo "3FS directory size: $(du -sh /workspace/3fs 2>/dev/null || echo 'Directory not found')"
    echo "Git status in 3FS:"
    cd /workspace/3fs && git status --porcelain | head -10
    echo "Git submodule status:"
    cd /workspace/3fs && git submodule status | head -10
    echo "Applied patches:"
    ls -la /workspace/3fs/patches/ 2>/dev/null || echo "No patches directory"

    # Check available libraries and their versions
    echo "=== LIBRARY VERSIONS ==="
    pkg-config --list-all | grep -E "(arrow|protobuf|grpc|folly|rocksdb)" | head -10 || echo "No relevant packages found"
    echo "Arrow library check:"
    ldconfig -p | grep arrow || echo "Arrow libraries not found in ldconfig"
    echo "FoundationDB check:"
    ls -la /usr/lib/libfdb* 2>/dev/null || echo "FoundationDB libraries not found"
    fdbcli --version 2>/dev/null || echo "FoundationDB CLI not working"

    # Memory and resource constraints
    echo "=== RESOURCE CONSTRAINTS ==="
    echo "Available memory:"
    free -m
    echo "Swap status:"
    swapon --show 2>/dev/null || echo "No swap configured"
    echo "Ulimits:"
    ulimit -a | grep -E "(open files|max memory|stack size)"

    # Additional resource analysis for CI comparison
    echo "=== DETAILED RESOURCE ANALYSIS ==="
    echo "Total system memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
    echo "CPU cores: $(nproc)"
    echo "CPU info:"
    grep -E "model name|cpu cores|siblings" /proc/cpuinfo | head -6
    echo "Load average details: $(cat /proc/loadavg)"
    echo "Disk usage for /workspace:"
    df -h /workspace 2>/dev/null || df -h /
    echo "Available inodes:"
    df -i /workspace 2>/dev/null || df -i /

    # Check for previous build artifacts that might interfere
    echo "=== POTENTIAL BUILD CONFLICTS ==="
    echo "Existing CMake cache files:"
    find /workspace/3fs -name "CMakeCache.txt" 2>/dev/null || echo "No CMake cache files found"
    echo "Existing build directories:"
    find /workspace/3fs -type d -name "build*" 2>/dev/null || echo "No build directories found"

    # Check CMAKE configuration that will be used
    echo "=== CMAKE CONFIGURATION PREVIEW ==="
    echo "CMake will be called with these flags:"
    echo "  Source: /workspace/3fs"
    echo "  Build: /workspace/3fs/build"
    echo "  Generator: Unix Makefiles"
    echo "  CXX Compiler: clang++-14"
    echo "  C Compiler: clang-14"
    echo "  ARROW_JEMALLOC_USE_STATIC: OFF"
    echo "  ARROW_MIMALLOC: OFF"
    echo "  ARROW_JEMALLOC: OFF"
    echo "  ARROW_USE_SYSTEM_MALLOC: ON"
    echo "  CMAKE_BUILD_TYPE: Release"
    echo "  ARROW_JEMALLOC_BUILD_JOBS: 1"

    # Test basic compilation capability
    echo "=== COMPILATION TEST ==="
    echo "Testing basic C++ compilation with clang++-14:"
    echo '#include <iostream>
int main() { std::cout << "Hello World" << std::endl; return 0; }' > /tmp/test.cpp
    clang++-14 -o /tmp/test /tmp/test.cpp 2>&1 && echo "Basic C++ compilation: SUCCESS" || echo "Basic C++ compilation: FAILED"
    rm -f /tmp/test /tmp/test.cpp

    echo "========== 3FS BUILD DIAGNOSTICS END =========="
    # ========== END DIAGNOSTIC SECTION ==========
    echo "========== CMAKE CONFIGURATION PHASE START =========="
    echo "Running CMake configuration..."

    # Temporarily hide jemalloc libraries from CMake
    echo "=== MASKING JEMALLOC LIBRARIES ==="
    JEMALLOC_BACKUP_DIR="/tmp/jemalloc_backup"
    mkdir -p "$JEMALLOC_BACKUP_DIR"

    # Move jemalloc libraries temporarily
    for lib in /usr/lib/x86_64-linux-gnu/libjemalloc* /usr/lib/libjemalloc* /lib/x86_64-linux-gnu/libjemalloc* /lib/libjemalloc*; do
        if [ -f "$lib" ]; then
            echo "Temporarily moving: $lib"
            mv "$lib" "$JEMALLOC_BACKUP_DIR/" 2>/dev/null || true
        fi
    done

    # Also hide jemalloc pkg-config files
    for pc in /usr/lib/x86_64-linux-gnu/pkgconfig/jemalloc* /usr/lib/pkgconfig/jemalloc* /usr/share/pkgconfig/jemalloc*; do
        if [ -f "$pc" ]; then
            echo "Temporarily moving pkg-config: $pc"
            mv "$pc" "$JEMALLOC_BACKUP_DIR/" 2>/dev/null || true
        fi
    done

    CMAKE_CONFIG_CMD="cmake -S /workspace/3fs -B /workspace/3fs/build \\
            -G \"Unix Makefiles\" \\
            -DCMAKE_CXX_COMPILER=clang++-14 \\
            -DCMAKE_C_COMPILER=clang-14 \\
            -DARROW_JEMALLOC=OFF \\
            -DARROW_JEMALLOC_USE_STATIC=OFF \\
            -DARROW_MIMALLOC=OFF \\
            -DARROW_USE_SYSTEM_MALLOC=ON \\
            -DCMAKE_BUILD_TYPE=Release \\
            -DARROW_JEMALLOC_BUILD_JOBS=1 \\
            -DARROW_BUILD_STATIC=OFF \\
            -DARROW_BUILD_SHARED=ON \\
            -DJEMALLOC_INCLUDE_DIR= \\
            -DJEMALLOC_LIB= \\
            -DARROW_DEPENDENCY_USE_SHARED=OFF \\
            -DARROW_THIRDPARTY_DEPENDENCIES=BUNDLED \\
            -DARROW_WITH_JEMALLOC=OFF \\
            -DJEMALLOC_ROOT= \\
            -DJEMALLOC_LIBRARIES= \\
            -DJEMALLOC_FOUND=FALSE \\
            -DJemalloc_FOUND=FALSE \\
            -DCMAKE_DISABLE_FIND_PACKAGE_jemalloc=TRUE \\
            -DCMAKE_DISABLE_FIND_PACKAGE_Jemalloc=TRUE \\
            -DCMAKE_C_FLAGS='-DARROW_JEMALLOC=0' \\
            -DCMAKE_CXX_FLAGS='-DARROW_JEMALLOC=0'"
            # -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            # -DARROW_USE_SYSTEM_MALLOC=ON \
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

    echo "CMAKE command: $CMAKE_CONFIG_CMD"
    echo "Starting CMAKE configuration at: $(date)"

    eval $CMAKE_CONFIG_CMD
    CMAKE_CONFIG_EXIT_CODE=$?

    echo "CMAKE configuration completed at: $(date)"
    echo "CMAKE configuration exit code: $CMAKE_CONFIG_EXIT_CODE"

    if [ $CMAKE_CONFIG_EXIT_CODE -ne 0 ]; then
        echo "ERROR: CMAKE configuration failed with exit code $CMAKE_CONFIG_EXIT_CODE"
        echo "Checking CMake error logs:"
        if [ -f /workspace/3fs/build/CMakeFiles/CMakeError.log ]; then
            echo "=== CMAKE ERROR LOG ==="
            cat /workspace/3fs/build/CMakeFiles/CMakeError.log
        fi
        if [ -f /workspace/3fs/build/CMakeFiles/CMakeOutput.log ]; then
            echo "=== CMAKE OUTPUT LOG ==="
            tail -50 /workspace/3fs/build/CMakeFiles/CMakeOutput.log
        fi
        exit $CMAKE_CONFIG_EXIT_CODE
    fi

    echo "========== CMAKE CONFIGURATION PHASE END =========="
    echo ""
    echo "========== BUILD PHASE START =========="
    echo "Starting build at: $(date)"
    echo "Build command: cmake --build /workspace/3fs/build -j 16"

    # Store build output and monitor progress
    cmake --build /workspace/3fs/build -j 16 2>&1 | tee /tmp/3fs_build.log
    BUILD_EXIT_CODE=${PIPESTATUS[0]}

    echo "Build completed at: $(date)"
    echo "Build exit code: $BUILD_EXIT_CODE"

    if [ $BUILD_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Build failed with exit code $BUILD_EXIT_CODE"
        echo "=== LAST 100 LINES OF BUILD OUTPUT ==="
        tail -100 /tmp/3fs_build.log
        echo ""
        echo "=== BUILD ERROR ANALYSIS ==="
        echo "Searching for common error patterns in build log..."
        grep -i "error\|fail\|fatal" /tmp/3fs_build.log | tail -20 || echo "No obvious error patterns found"
        echo ""
        echo "=== MEMORY USAGE DURING BUILD ==="
        free -m
        echo ""
        echo "=== DISK SPACE AFTER BUILD ==="
        df -h /workspace
        exit $BUILD_EXIT_CODE
    fi

    echo "========== BUILD PHASE END =========="

    # Restore jemalloc libraries regardless of build result
    echo "=== RESTORING JEMALLOC LIBRARIES ==="
    if [ -d "$JEMALLOC_BACKUP_DIR" ]; then
        for file in "$JEMALLOC_BACKUP_DIR"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                echo "Restoring: $filename"
                # Try to restore to original location
                if [[ "$filename" == *"pkgconfig"* ]]; then
                    mv "$file" "/usr/lib/x86_64-linux-gnu/pkgconfig/" 2>/dev/null || \
                    mv "$file" "/usr/lib/pkgconfig/" 2>/dev/null || \
                    mv "$file" "/usr/share/pkgconfig/" 2>/dev/null || true
                else
                    mv "$file" "/usr/lib/x86_64-linux-gnu/" 2>/dev/null || \
                    mv "$file" "/usr/lib/" 2>/dev/null || \
                    mv "$file" "/lib/x86_64-linux-gnu/" 2>/dev/null || \
                    mv "$file" "/lib/" 2>/dev/null || true
                fi
            fi
        done
        rmdir "$JEMALLOC_BACKUP_DIR" 2>/dev/null || true
    fi
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