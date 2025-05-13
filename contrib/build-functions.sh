#!/bin/bash

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


missing_requirement() {
    error "ERROR: $1 requires an argument."
}

# Error handling functions
error() {
    printf '%s %s\n' "$1" "$2" >&2
    return 1
}

# Show help message
function show_help() {
    set +x
    local function_name=$1
    echo "usage: source contrib/build.env && <command> [options]"
    echo "Examples: "
    echo "  source contrib/build.env && install_build_dependencies"
    echo "  source contrib/build.env && build_nixl --install-dir /opt/nvidia/nixl"
    echo "  source contrib/build.env && install_test_dependencies"
    echo "  source contrib/build.env && run_cpp_tests --install-dir /opt/nvidia/nixl"
    echo "  source contrib/build.env && run_python_tests --install-dir /opt/nvidia/nixl"
    echo ""
    case $function_name in
        build_nixl)
            echo "Build the NIXL project"
            echo ""
            echo "Description:"
            echo "  Builds the NIXL project using Meson build system. This function configures"
            echo "  and compiles NIXL with the specified options and installs it to the target"
            echo "  directory."
            echo ""
            echo "Options:"
            echo "  --install-dir DIR           Installation directory for NIXL (default: $NIXL_INSTALL_DIR)"
            echo "  --ucx-install-dir DIR       UCX installation directory (default: $UCX_INSTALL_DIR)"
            echo "  --build-dir DIR             Build directory (default: $NIXL_BUILD_DIR)"
            echo "  --static-plugins LIST       Comma-separated list of plugins to build statically"
            echo "  --enable-doxygen            Build API documentation using Doxygen (default: false)"
            echo "  --no-enable-doxygen         Skip building documentation"
            echo "  --disable-gds-backend       Disable NVIDIA GPUDirect Storage backend (default: false)"
            echo "  --no-disable-gds-backend    Enable NVIDIA GPUDirect Storage backend"
            echo "  --enable-install-headers    Install development header files (default: true)"
            echo "  --no-enable-install-headers Skip installing header files"
            echo "  --install                   Run installation step after build (default: true)"
            echo "  --no-install                Skip installation step"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic build and install"
            echo "  build_nixl --install-dir /opt/nvidia/nixl --ucx-install-dir /opt/nvidia/ucx"
            echo ""
            echo "  # Build with static plugins and documentation"
            echo "  build_nixl --install-dir /opt/nvidia/nixl \\"
            echo "            --static-plugins ucx,gds \\"
            echo "            --enable-doxygen"
            echo ""
            echo "Returns:"
            echo "  0 on success"
            echo "  1 on failure"
            ;;
        install_build_dependencies)
            echo "Install required system dependencies for building NIXL"
            echo ""
            echo "Options:"
            echo "  None"
            echo ""
            echo "This function installs all required system packages and dependencies"
            echo "for building NIXL on Ubuntu systems, including:"
            echo "  - Build tools (gcc, g++, make, ninja-build, meson)"
            echo "  - Development packages (libtool, autoconf, automake)"
            echo "  - Required libraries (libnuma-dev, libibverbs-dev)"
            echo "  - CUDA development tools"
            echo ""
            echo "Note: Must be run with root/sudo privileges"
            echo ""
            echo "Example:"
            echo "  sudo -E bash -c 'source contrib/build.env && install_build_dependencies'"
            ;;
        install_test_dependencies)
            echo "Install required system dependencies"
            echo ""
            echo "Options:"
            echo "  None"
            echo ""
            echo "This function installs all required system packages and dependencies"
            echo "for running the NIXL test suite on Ubuntu systems. It must be run with"
            echo "root/sudo privileges."
            ;;
        build_ucx)
            echo "Build and install UCX (Universal Communication X) library"
            echo ""
            echo "Description:"
            echo "  This function downloads, builds and installs the UCX library with optimized settings."
            echo "  UCX is a communication framework optimized for high-performance computing and"
            echo "  machine learning applications. The build is configured with shared libraries,"
            echo "  verbs support, device memory, and multi-threading enabled by default."
            echo ""
            echo "Options:"
            echo "  --ucx-version          UCX version to build (default: $UCX_VERSION)"
            echo "  --install-dir DIR      Directory to install UCX (default: $UCX_INSTALL_DIR)"
            echo "  --enable-shared        Enable shared libraries (default: yes)"
            echo "  --no-enable-shared     Disable shared libraries"
            echo "  --enable-static        Enable static libraries (default: no)"
            echo "  --no-enable-static     Disable static libraries"
            echo "  --enable-doxygen       Enable Doxygen documentation (default: no)"
            echo "  --no-enable-doxygen    Disable Doxygen documentation"
            echo "  --enable-optimizations Enable optimizations (default: yes)"
            echo "  --no-enable-optimizations Disable optimizations"
            echo "  --enable-cma            Enable CMA support (default: yes)"
            echo "  --no-enable-cma         Disable CMA support"
            echo "  --enable-devel-headers  Enable development headers (default: yes)"
            echo "  --no-enable-devel-headers Disable development headers"
            echo "  --with-verbs          Enable verbs support (default: yes)"
            echo "  --without-verbs       Disable verbs support"
            echo "  --with-dm             Enable device memory support (default: yes)"
            echo "  --without-dm          Disable device memory support"
            echo "  --enable-mt           Enable multi-threading support (default: yes)"
            echo "  --no-enable-mt        Disable multi-threading support"
            echo "  --help                Show help message"
            echo ""
            echo "Examples:"
            echo "  # Basic build with default options"
            echo "  build_ucx --install-dir /opt/nvidia/ucx"
            echo ""
            echo "  # Build with static libraries and without device memory support"
            echo "  build_ucx --install-dir /opt/nvidia/ucx --enable-static --without-dm"
            echo ""
            echo "Returns:"
            echo "  0 on success"
            echo "  1 on failure"
            ;;
        run_cpp_tests)
            echo "Run C++ tests and examples"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR       Directory containing installed NIXL binaries (default: $NIXL_INSTALL_DIR)"
            echo "  --ucx-install-dir DIR   Directory containing installed UCX binaries (default: $UCX_INSTALL_DIR)"
            echo "  --help                  Show this help message"
            echo ""
            echo "This function executes various C++ test binaries and examples from the NIXL project."
            echo "It runs unit tests, backend tests, and a client-server test scenario."
            echo ""
            echo "Example:"
            echo "  run_cpp_tests --install-dir /opt/nvidia/nixl --ucx-install-dir /opt/nvidia/ucx"
            echo ""
            echo "The following tests are executed:"
            echo "- desc_example: Descriptor example"
            echo "- agent_example: Agent example"
            echo "- nixl_example: NIXL API example"
            echo "- ucx_backend_test: UCX backend tests"
            echo "- ucx_mo_backend_test: UCX memory ordering backend tests"
            echo "- ucx_backend_multi: Multi-threaded UCX backend tests"
            echo "- serdes_test: Serialization/deserialization tests"
            echo ""
            echo "The following tests are disabled by default:"
            echo "- md_streamer: Memory domain streamer test"
            echo "- nixl_test: NIXL client-server test"
            echo "- p2p_test: Peer-to-peer test"
            echo "- ucx_worker_test: UCX worker test"
            echo ""
            echo "Environment variables used:"
            echo "- LD_LIBRARY_PATH: Updated to include NIXL, UCX and CUDA library paths"
            echo "- CPATH: Updated to include NIXL and UCX header paths"
            echo "- PATH: Updated to include NIXL and UCX binary paths"
            echo "- PKG_CONFIG_PATH: Updated for NIXL and UCX"
            echo "- NIXL_PLUGIN_DIR: Set to NIXL and UCX plugins directory"
            echo ""
            echo "Returns:"
            echo "  0 on success, non-zero on failure"
            ;;
        run_python_tests)
            echo "Run Python tests and examples"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR        Directory containing installed NIXL binaries (default: $NIXL_INSTALL_DIR)"
            echo "  --ucx-install-dir DIR    Directory containing installed UCX binaries (default: $UCX_INSTALL_DIR)"
            echo "  --help                   Show this help message"
            echo ""
            echo "This function executes the Python test suite including:"
            echo "- Installs required Python packages (pytest, zmq)"
            echo "- Sets up environment variables for library paths and plugins"
            echo "- Installs the NIXL Python package locally"
            echo "- Runs Python unit tests and examples"
            echo ""
            echo "Environment variables used:"
            echo "- LD_LIBRARY_PATH: Updated to include NIXL, UCX and CUDA library paths"
            echo "- CPATH: Updated to include NIXL and UCX header paths"
            echo "- PATH: Updated to include NIXL and UCX binary paths"
            echo "- PKG_CONFIG_PATH: Updated for NIXL and UCX"
            echo "- NIXL_PLUGIN_DIR: Set to NIXL and UCX plugins directory"
            echo ""
            echo "Example:"
            echo "  run_python_tests --install-dir /opt/nvidia/nixl --ucx-install-dir /opt/nvidia/ucx"
            echo ""
            echo "Returns:"
            echo "  0 on success, non-zero on failure"
            ;;
        *)
            echo "Available commands:"
            echo "  build_nixl"
            echo "  build_ucx"
            echo "  install_build_dependencies"
            echo "  install_test_dependencies"
            echo "  run_cpp_tests"
            echo "  run_python_tests"
            ;;
    esac
}

function install_build_dependencies() {
    while :; do
        case $1 in
            -h|-\?|--help)
                show_help install_test_dependencies
                return 0
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    if ! grep -q "Ubuntu" /etc/os-release; then
        error "This script only supports Ubuntu"
        return 1
    fi
    # Check if script is run as root, use sudo if not
    local SUDO=""
    if [ "$EUID" -ne 0 ]; then
        SUDO="sudo"
    fi
    $SUDO apt-get -qq update
    $SUDO apt-get -qq install -y \
    automake \
        autotools-dev \
        build-essential \
        cmake \
        curl \
        doxygen \
        etcd-server \
        flex \
        ibverbs-utils \
        libboost-all-dev \
        libgoogle-glog-dev \
        libgrpc++-dev \
        libgrpc-dev \
        libgtest-dev \
        libiberty-dev \
        libibmad-dev \
        libibverbs-dev \
        libjsoncpp-dev \
        libnuma-dev \
        libpci-dev \
        libprotobuf-dev \
        libpython3-dev \
        libssl-dev \
        libz-dev \
        libtool \
        meson \
        net-tools \
        ninja-build \
        numactl \
        pciutils \
        pkg-config \
        protobuf-compiler-grpc \
        pybind11-dev \
        uuid-dev
    return $?
}

function install_test_dependencies() {
    while :; do
        case $1 in
            -h|-\?|--help)
                show_help install_test_dependencies
                return 0
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    if ! grep -q "Ubuntu" /etc/os-release; then
        error "This script only supports Ubuntu"
        return 1
    fi
    # Check if script is run as root, use sudo if not
    local SUDO=""
    if [ "$EUID" -ne 0 ]; then
        SUDO="sudo"
    fi
    $SUDO apt-get -qq update
    $SUDO apt-get -qq install -y \
        libaio-dev \
        liburing-dev
    return $?
}

function build_ucx() {
    local install_dir=${UCX_INSTALL_DIR:-}
    local ucx_version=${UCX_VERSION:-}
    local enable_shared="yes"
    local enable_static="no"
    local enable_doxygen="no"
    local enable_optimizations="yes"
    local enable_cma="yes"
    local enable_devel_headers="yes"
    local with_verbs="yes"
    local with_dm="yes"
    local enable_mt="yes"

    while :; do
        case $1 in
            -h|-\?|--help)
                show_help build_ucx
                return 0
                ;;
            --ucx-version)
                if [ "$2" ]; then
                    ucx_version="$2"
                    shift
                else
                    missing_requirement "$1"
                    return 1
                fi
                ;;
            --install-dir)
                if [ "$2" ]; then
                    install_dir="$2"
                    shift
                else
                    missing_requirement "$1"
                    return 1
                fi
                ;;
            --enable-shared)
                enable_shared="yes"
                ;;
            --no-enable-shared)
                enable_shared="no"
                ;;
            --enable-static)
                enable_static="yes"
                ;;
            --no-enable-static)
                enable_static="no"
                ;;
            --enable-doxygen)
                enable_doxygen="yes"
                ;;
            --no-enable-doxygen)
                enable_doxygen="no"
                ;;
            --enable-optimizations)
                enable_optimizations="yes"
                ;;
            --no-enable-optimizations)
                enable_optimizations="no"
                ;;
            --enable-cma)
                enable_cma="yes"
                ;;
            --no-enable-cma)
                enable_cma="no"
                ;;
            --enable-devel-headers)
                enable_devel_headers="yes"
                ;;
            --no-enable-devel-headers)
                enable_devel_headers="no"
                ;;
            --with-verbs)
                with_verbs="yes"
                ;;
            --without-verbs)
                with_verbs="no"
                ;;
            --with-dm)
                with_dm="yes"
                ;;
            --without-dm)
                with_dm="no"
                ;;
            --enable-mt)
                enable_mt="yes"
                ;;
            --no-enable-mt)
                enable_mt="no"
                ;;
            -?*|?*)
                error 'Unknown option: ' "$1"
                return 1
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    curl -fSsL "https://github.com/openucx/ucx/tarball/v${ucx_version}" | tar xz
    (
        cd openucx-ucx* || { error 'cannot cd to' 'openucx-ucx dir' ; return 1; }
        ./autogen.sh
        ./configure \
            --prefix="${install_dir}" \
            "$([ "$enable_shared" = "yes" ] && echo "--enable-shared" || echo "--disable-shared")" \
            "$([ "$enable_static" = "yes" ] && echo "--enable-static" || echo "--disable-static")" \
            "$([ "$enable_doxygen" = "yes" ] && echo "--enable-doxygen-doc" || echo "--disable-doxygen-doc")" \
            "$([ "$enable_optimizations" = "yes" ] && echo "--enable-optimizations" || echo "--disable-optimizations")" \
            "$([ "$enable_cma" = "yes" ] && echo "--enable-cma" || echo "--disable-cma")" \
            "$([ "$enable_devel_headers" = "yes" ] && echo "--enable-devel-headers" || echo "--disable-devel-headers")" \
            "$([ "$with_verbs" = "yes" ] && echo "--with-verbs" || echo "--without-verbs")" \
            "$([ "$with_dm" = "yes" ] && echo "--with-dm" || echo "--without-dm")" \
            "$([ "$enable_mt" = "yes" ] && echo "--enable-mt" || echo "--disable-mt")"

        if ! make -j ; then
            error "make failed"
            return 1
        fi
        if ! make -j install-strip ; then
            error "make install-strip failed"
            return 1
        fi
        ldconfig
    )
}

function build_nixl() {
    local install_dir=${NIXL_INSTALL_DIR:-}
    local ucx_install_dir=${UCX_INSTALL_DIR:-}
    local build_dir=${NIXL_BUILD_DIR:-}
    local enable_doxygen="false"
    local enable_install_headers="true"
    local disable_gds_backend="false"
    local static_plugins=""
    local do_install="true"

    while :; do
        case $1 in
        -h|-\?|--help)
            show_help build_nixl
            return 0
            ;;
        --build-dir)
            if [ "$2" ]; then
                build_dir="$2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --install-dir)
            if [ "$2" ]; then
                install_dir="$2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --ucx-install-dir)
            if [ "$2" ]; then
                ucx_install_dir=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --static-plugins)
            if [ "$2" ]; then
                static_plugins="$2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --enable-doxygen)
            enable_doxygen="true"
            ;;
        --no-enable-doxygen)
            enable_doxygen="false"
            ;;
        --disable-gds-backend)
            disable_gds_backend="true"
            ;;
        --no-disable-gds-backend)
            disable_gds_backend="false"
            ;;
        --enable-install-headers)
            enable_install_headers="true"
            ;;
        --no-enable-install-headers)
            enable_install_headers="false"
            ;;
        --install)
            do_install="true"
            ;;
        --no-install)
            do_install="false"
            ;;
        -?*|?*)
            error 'Unknown option: ' "$1"
            return 1
            ;;
        *)
            break
            ;;
        esac
        shift
    done

    # Disabling CUDA IPC not to use NVLINK, as it slows down local
    # UCX transfers and can cause contention with local collectives.
    export UCX_TLS=^cuda_ipc

    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    build_cmd="meson setup $build_dir \
        --prefix=${install_dir} \
        -Ducx_path=${ucx_install_dir} \
        $([ -n "$static_plugins" ] && echo "-Dstatic_plugins=${static_plugins}") \
        -Ddisable_gds_backend=${disable_gds_backend} \
        -Dinstall_headers=${enable_install_headers} \
        -Dbuild_docs=${enable_doxygen}"

    echo "Running build command:"
    echo "$build_cmd"

    if ! eval "$build_cmd" ; then
        error "meson setup failed"
        return 1
    fi

    if ! ninja -C "$build_dir" ; then
        error "ninja failed"
        return 1
    fi

    if [ "$do_install" = "true" ]; then
        if ! ninja -C "$build_dir" install ; then
            error "ninja install failed"
            return 1
        fi
    fi
}

function run_cpp_tests() {
    local install_dir=${NIXL_INSTALL_DIR:-}
    local ucx_install_dir=${UCX_INSTALL_DIR:-}
    local -r TEXT_YELLOW="\033[1;33m"
    local -r TEXT_CLEAR="\033[0m"
    local -r DEFAULT_IP="127.0.0.1"
    local -r DEFAULT_PORT="1234"
    local rc=0
    local -r tests=(
        "desc_example"
        "agent_example"
        "nixl_example"
        "ucx_backend_test"
        "ucx_mo_backend_test"
        "ucx_backend_multi"
        "serdes_test"
    )
    local -r disabled_tests=(
        "md_streamer"
        "nixl_test"
        "p2p_test"
        "ucx_worker_test"
    )

    while :; do
        case $1 in
        -h|-\?|--help)
            show_help run_cpp_tests
            return 0
            ;;
        --install-dir)
            if [ "$2" ]; then
                install_dir="$2"
                shift
            else
                missing_requirement "$1"
                return 1
            fi
            ;;
        --ucx-install-dir)
            if [ "$2" ]; then
                ucx_install_dir="$2"
                shift
            else
                missing_requirement "$1"
                return 1
            fi
            ;;
        -?*|?*)
            error 'Unknown option: ' "$1"
            return 1
            ;;
        *)
            break
            ;;
        esac
        shift
    done

    export LD_LIBRARY_PATH=${install_dir}/lib:${install_dir}/lib/x86_64-linux-gnu:${install_dir}/lib/x86_64-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${ucx_install_dir}/lib:${ucx_install_dir}/lib/x86_64-linux-gnu:${ucx_install_dir}/lib/x86_64-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

    export CPATH=${install_dir}/include:${ucx_install_dir}/include:$CPATH
    export PATH=${install_dir}/bin:${ucx_install_dir}/bin:$PATH
    export PKG_CONFIG_PATH=${install_dir}/lib/pkgconfig:${ucx_install_dir}/lib/pkgconfig:$PKG_CONFIG_PATH
    export NIXL_PLUGIN_DIR=${install_dir}/lib/x86_64-linux-gnu/plugins:${ucx_install_dir}/lib/x86_64-linux-gnu/plugins

    echo "==== Running C++ tests ===="
    pushd "${install_dir}" || { error "cannot cd to ${install_dir}" ; return 1; }

    for test in "${tests[@]}"; do
        ./bin/"${test}"
        rc=$((rc+$?))
    done

    # Run NIXL client-server test
    ./bin/nixl_test initiator "${DEFAULT_IP}" "${DEFAULT_PORT}"&
    sleep 1
    ./bin/nixl_test target "${DEFAULT_IP}" "${DEFAULT_PORT}"
    rc=$((rc+$?))

    echo "${TEXT_YELLOW}==== Disabled tests==="
    for test in "${disabled_tests[@]}"; do
        echo "./bin/${test} disabled"
    done
    echo "${TEXT_CLEAR}"
    popd || error "cannot pop from ${install_dir}"
    return $rc
}

function run_python_tests() {
    local install_dir=${NIXL_INSTALL_DIR:-}
    local ucx_install_dir=${UCX_INSTALL_DIR:-}
    local -r DEFAULT_IP="127.0.0.1"
    local -r DEFAULT_PORT="1234"
    local rc=0

    while :; do
        case $1 in
        -h|-\?|--help)
            show_help run_python_tests
            return 0
            ;;
        --install-dir)
            if [ "$2" ]; then
                install_dir="$2"
                shift
            else
                missing_requirement "$1"
                return 1
            fi
            ;;
        --ucx-install-dir)
            if [ "$2" ]; then
                ucx_install_dir="$2"
                shift
            else
                missing_requirement "$1"
                return 1
            fi
            ;;
        -?*|?*)
            error 'Unknown option: ' "$1"
            return 1
            ;;
        *)
            break
            ;;
        esac
        shift
    done

    export LD_LIBRARY_PATH=${ucx_install_dir}/lib:${install_dir}/lib:${install_dir}/lib/x86_64-linux-gnu:${install_dir}/lib/x86_64-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
    export CPATH=${install_dir}/include:${ucx_install_dir}/include:$CPATH
    export PATH=${install_dir}/bin:${ucx_install_dir}/bin:$PATH
    export PKG_CONFIG_PATH=${install_dir}/lib/pkgconfig:${ucx_install_dir}/lib/pkgconfig:$PKG_CONFIG_PATH
    export NIXL_PLUGIN_DIR=${install_dir}/lib/x86_64-linux-gnu/plugins:${ucx_install_dir}/lib/x86_64-linux-gnu/plugins

    if ! pip3 install --break-system-packages .; then
        error "pip3 install failed"
        return 1
    fi
    if ! pip3 install --break-system-packages pytest; then
        error "pip3 install pytest failed"
        return 1
    fi
    if ! pip3 install --break-system-packages zmq; then
        error "pip3 install zmq failed"
        return 1
    fi

    echo "==== Running python tests ===="
    python3 examples/python/nixl_api_example.py
    rc=$((rc+$?))
    pytest test/python
    rc=$((rc+$?))

    echo "==== Running python example ===="
    pushd examples/python || { error "cannot cd to examples/python" ; return 1; }
    python3 blocking_send_recv_example.py --name="B" --mode="target" --zmq_ip=${DEFAULT_IP} --zmq_port=${DEFAULT_PORT}&
    sleep 1
    python3 blocking_send_recv_example.py --name="A" --mode="initiator" --zmq_ip=${DEFAULT_IP} --zmq_port=${DEFAULT_PORT}
    rc=$((rc+$?))
    popd || error "cannot pop from examples/python"
    return $rc
}

# Check if script is being sourced
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    # Script is being executed directly
    error "This script is not meant to be executed directly"
    exit 1
fi
