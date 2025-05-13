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

# shellcheck source=common-functions.sh
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common-functions.sh"


function nixl_install_build_dependencies() {
    while :; do
        case ${1:-} in
            -h|-\?|--help)
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
                echo "  sudo -E bash -c 'source contrib/build.env && nixl_install_build_dependencies'"

                return 0
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    _is_valid_os_type || return 1

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
        libaio-dev \
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

function nixl_build_ucx() {
    local install_dir=${_NIXL_UCX_INSTALL_DIR:-}
    local ucx_version=${_NIXL_UCX_VERSION:-}
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
        case ${1:-} in
            -h|-\?|--help)
                echo "Build and install UCX (Universal Communication X) library"
                echo ""
                echo "Description:"
                echo "  This function downloads, builds and installs the UCX library with optimized settings."
                echo "  UCX is a communication framework optimized for high-performance computing and"
                echo "  machine learning applications. The build is configured with shared libraries,"
                echo "  verbs support, device memory, and multi-threading enabled by default."
                echo ""
                echo "Options:"
                echo "  --ucx-version VERSION  UCX version to build (default: $_NIXL_UCX_VERSION)"
                echo "  --install-dir DIR      Directory to install UCX (default: $_NIXL_UCX_INSTALL_DIR)"
                echo "  --[no-]shared          Enable shared libraries (default: true)"
                echo "  --[no-]static          Enable static libraries (default: false)"
                echo "  --[no-]doxygen         Enable Doxygen documentation (default: false)"
                echo "  --[no-]optimizations   Enable optimizations (default: true)"
                echo "  --[no-]cma             Enable CMA support (default: true)"
                echo "  --[no-]devel-headers   Enable development headers (default: true)"
                echo "  --[no-]verbs           Enable verbs support (default: true)"
                echo "  --[no-]dm              Enable device memory support (default: true)"
                echo "  --[no-]mt              Enable multi-threading support (default: true)"
                echo "  --help                 Show help message"
                echo ""
                echo "Examples:"
                echo "  # Basic build with default options"
                echo "  nixl_build_ucx --install-dir /opt/nvidia/ucx"
                echo ""
                echo "  # Build with static libraries and without device memory support"
                echo "  nixl_build_ucx --install-dir /opt/nvidia/ucx --static --dm"
                echo ""
                echo "Returns:"
                echo "  0 on success"
                echo "  1 on failure"

                return 0
                ;;
            --ucx-version)
                if [ "$2" ]; then
                    ucx_version="$2"
                    shift
                else
                    _missing_requirement "$1"
                    return 1
                fi
                ;;
            --install-dir)
                if [ "$2" ]; then
                    install_dir="$2"
                    shift
                else
                    _missing_requirement "$1"
                    return 1
                fi
                ;;
            --shared)
                enable_shared="yes"
                ;;
            --no-shared)
                enable_shared="no"
                ;;
            --static)
                enable_static="yes"
                ;;
            --no-static)
                enable_static="no"
                ;;
            --doxygen)
                enable_doxygen="yes"
                ;;
            --no-doxygen)
                enable_doxygen="no"
                ;;
            --optimizations)
                enable_optimizations="yes"
                ;;
            --no-optimizations)
                enable_optimizations="no"
                ;;
            --cma)
                enable_cma="yes"
                ;;
            --no-cma)
                enable_cma="no"
                ;;
            --devel-headers)
                enable_devel_headers="yes"
                ;;
            --no-devel-headers)
                enable_devel_headers="no"
                ;;
            --verbs)
                with_verbs="yes"
                ;;
            --no-verbs)
                with_verbs="no"
                ;;
            --dm)
                with_dm="yes"
                ;;
            --no-dm)
                with_dm="no"
                ;;
            --mt)
                enable_mt="yes"
                ;;
            --no-mt)
                enable_mt="no"
                ;;
            -?*|?*)
                _error 'Unknown option: ' "$1"
                return 1
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    # clean old ucx downloads
    rm -rf openucx-ucx*

    curl -fSsL "https://github.com/openucx/ucx/tarball/v${ucx_version}" | tar xz
    (
        pushd openucx-ucx* || { _error 'cannot cd to' 'openucx-ucx dir' ; return 1; }

        if ! ./autogen.sh; then
            _error "autogen.sh failed"
            return 1
        fi

        if ! ./configure \
            --prefix="${install_dir}" \
            --enable-shared=$enable_shared \
            --enable-static=$enable_static \
            --enable-doxygen-doc=$enable_doxygen \
            --enable-optimizations=$enable_optimizations \
            --enable-cma=$enable_cma \
            --enable-devel-headers=$enable_devel_headers \
            --with-verbs=$with_verbs \
            --with-dm=$with_dm \
            --enable-mt=$enable_mt; then
            _error "configure failed"
            return 1
        fi

        if ! make -j ; then
            _error "make failed"
            return 1
        fi
        if ! make -j install-strip ; then
            _error "make install-strip failed"
            return 1
        fi
        if ! ldconfig; then
            _error "ldconfig failed"
            return 1
        fi
    )
}

function nixl_build() {
    local install_dir=${_NIXL_INSTALL_DIR:-}
    local ucx_install_dir=${_NIXL_UCX_INSTALL_DIR:-}
    local build_dir=${_NIXL_BUILD_DIR:-}
    local enable_doxygen="false"
    local enable_install_headers="true"
    local disable_gds_backend="false"
    local static_plugins=""
    local do_install="true"

    while :; do
        case ${1:-} in
        -h|-\?|--help)
            echo "Build the NIXL project"
            echo ""
            echo "Description:"
            echo "  Builds the NIXL project using Meson build system. This function configures"
            echo "  and compiles NIXL with the specified options and installs it to the target"
            echo "  directory."
            echo ""
            echo "Options:"
            echo "  --install-dir DIR           Installation directory for NIXL (default: $_NIXL_INSTALL_DIR)"
            echo "  --ucx-install-dir DIR       UCX installation directory (default: $_NIXL_UCX_INSTALL_DIR)"
            echo "  --build-dir DIR             Build directory (default: $_NIXL_BUILD_DIR)"
            echo "  --static-plugins LIST       Comma-separated list of plugins to build statically"
            echo "  --[no-]doxygen              Build API documentation using Doxygen (default: false)"
            echo "  --[no-]gds-backend          Enable NVIDIA GPUDirect Storage backend (default: true)"
            echo "  --[no-]install-headers      Install development header files (default: true)"
            echo "  --[no-]install              Run installation step after build (default: true)"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic build and install"
            echo "  nixl_build --install-dir /opt/nvidia/nixl --ucx-install-dir /opt/nvidia/ucx"
            echo ""
            echo "  # Build with static plugins and documentation"
            echo "  nixl_build --install-dir /opt/nvidia/nixl \\"
            echo "            --static-plugins ucx,gds \\"
            echo "            --doxygen"
            echo ""
            echo "Returns:"
            echo "  0 on success"
            echo "  1 on failure"

            return 0
            ;;
        --build-dir)
            if [ "$2" ]; then
                build_dir="$2"
                shift
            else
                _missing_requirement "$1"
            fi
            ;;
        --install-dir)
            if [ "$2" ]; then
                install_dir="$2"
                shift
            else
                _missing_requirement "$1"
            fi
            ;;
        --ucx-install-dir)
            if [ "$2" ]; then
                ucx_install_dir=$2
                shift
            else
                _missing_requirement "$1"
            fi
            ;;
        --static-plugins)
            if [ "$2" ]; then
                static_plugins="-Dstatic_plugins=${2}"
                shift
            else
                _missing_requirement "$1"
            fi
            ;;
        --doxygen)
            enable_doxygen="true"
            ;;
        --no-doxygen)
            enable_doxygen="false"
            ;;
        --gds-backend)
            disable_gds_backend="false"
            ;;
        --no-gds-backend)
            disable_gds_backend="true"
            ;;
        --install-headers)
            enable_install_headers="true"
            ;;
        --no-install-headers)
            enable_install_headers="false"
            ;;
        --install)
            do_install="true"
            ;;
        --no-install)
            do_install="false"
            ;;
        -?*|?*)
            _error 'Unknown option: ' "$1"
            return 1
            ;;
        *)
            break
            ;;
        esac
        shift
    done

    _setup_environment "${install_dir}"

    # Disabling CUDA IPC not to use NVLINK, as it slows down local
    # UCX transfers and can cause contention with local collectives.
    export UCX_TLS=^cuda_ipc

    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    build_cmd="meson setup $build_dir \
        --prefix=${install_dir} \
        -Ducx_path=${ucx_install_dir} \
        $static_plugins \
        -Ddisable_gds_backend=${disable_gds_backend} \
        -Dinstall_headers=${enable_install_headers} \
        -Dbuild_docs=${enable_doxygen}"

    echo "Running build command:"
    echo "$build_cmd"

    if ! eval "$build_cmd" ; then
        _error "meson setup failed"
        return 1
    fi

    if ! ninja -C "$build_dir" ; then
        _error "ninja failed"
        return 1
    fi

    if [ "$do_install" = "true" ]; then
        if ! ninja -C "$build_dir" install ; then
            _error "ninja install failed"
            return 1
        fi
    fi
}

# Check if script is being sourced
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    # Script is being executed directly
    _error "This script is not meant to be executed directly"
    exit 1
fi
