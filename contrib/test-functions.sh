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

function nixl_install_test_dependencies() {
    while :; do
        case ${1:-} in
            -h|-\?|--help)
                echo "Install required system dependencies"
                echo ""
                echo "Options:"
                echo "  None"
                echo ""
                echo "This function installs all required system packages and dependencies"
                echo "for running the NIXL test suite on Ubuntu systems. It must be run with"
                echo "root/sudo privileges."
                echo ""
                echo "Example:"
                echo "  sudo -E bash -c 'source contrib/build.env && nixl_install_test_dependencies'"

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
        liburing-dev
    return $?
}

function nixl_cpp_tests() {
    local install_dir=${_NIXL_INSTALL_DIR:-}
    local rc=0
    local -r DEFAULT_IP="127.0.0.1"
    local -r DEFAULT_PORT="1234"
    local -r test_list=(
        "./bin/desc_example"
        "./bin/agent_example"
        "./bin/nixl_example"
        "./bin/ucx_backend_test"
        "./bin/ucx_mo_backend_test"
        "./bin/ucx_backend_multi"
        "./bin/serdes_test"
        "./bin/gtest"
        "./bin/test_plugin"
        "./bin/nixl_posix_test -n 128 -s 1048576"
        "#./bin/md_streamer"
        "#./bin/p2p_test"
        "#./bin/ucx_worker_test"
    )

    while :; do
        case ${1:-} in
        -h|-\?|--help)
            echo "Run C++ tests and examples"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR       Directory containing installed NIXL binaries (default: $_NIXL_INSTALL_DIR)"
            echo "  --help                  Show this help message"
            echo ""
            echo "This function executes various C++ test binaries and examples from the NIXL project."
            echo "It runs unit tests, backend tests, and a client-server test scenario."
            echo ""
            echo "Example:"
            echo "  nixl_cpp_tests --install-dir /opt/nvidia/nixl"
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
            return 0
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

    # Start timing
    local start_time=$SECONDS

    _setup_environment "${install_dir}"

    echo "==== Show system info ===="
    env
    nvidia-smi topo -m || true
    ibv_devinfo || true
    uname -a || true

    echo "==== Running C++ tests ===="
    pushd "${install_dir}" || { _error "cannot cd to ${install_dir}" ; return 1; }

    for test in "${test_list[@]}"; do
        echo "==== Running test: ${test} ===="
        if ! eval "${test}"; then
            echo "==== Test: ${test} failed ===="
            rc=1
            break
        fi
    done

    if [ $rc -eq 0 ]; then
        echo "==== Running test: NIXL client-server test ===="
        ./bin/nixl_test target "${DEFAULT_IP}" "${DEFAULT_PORT}"&
        sleep 1
        if ! ./bin/nixl_test initiator "${DEFAULT_IP}" "${DEFAULT_PORT}"; then
            _error "nixl_test initiator failed"
            rc=1
        fi
    fi

    echo "==== Disabled tests ===="
    for test in "${test_list[@]}"; do
        if [[ "${test}" == \#* ]]; then
            echo "test: ${test#\#} is disabled"
        fi
    done

    popd || _error "cannot pop from ${install_dir}"

    # Calculate and print execution time
    local duration=$((SECONDS - start_time))
    echo "==== C++ tests execution time: ${duration} seconds ===="

    return $rc
}

function nixl_python_tests() {
    local install_dir=${_NIXL_INSTALL_DIR:-}
    local -r DEFAULT_IP="127.0.0.1"
    local -r DEFAULT_PORT="1234"
    local rc=0

    while :; do
        case ${1:-} in
        -h|-\?|--help)
            echo "Run Python tests and examples"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR        Directory containing installed NIXL binaries (default: $_NIXL_INSTALL_DIR)"
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
            echo "  nixl_python_tests --install-dir /opt/nvidia/nixl"
            echo ""
            echo "Returns:"
            echo "  0 on success, non-zero on failure"

            return 0
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

    # Start timing
    local start_time=$SECONDS

    _setup_environment "${install_dir}"

    for pkg in "." "pytest" "pytest-timeout" "zmq"; do
        if ! pip3 install --break-system-packages "$pkg"; then
            _error "pip3 install $pkg failed"
            return 1
        fi
    done

    echo "==== Running python tests ===="
    if ! python3 examples/python/nixl_api_example.py; then
        _error "python3 examples/python/nixl_api_example.py failed"
        return 1
    fi

    if ! pytest test/python; then
        _error "pytest test/python failed"
        return 1
    fi

    echo "==== Running python example ===="
    pushd examples/python || { _error "cannot cd to examples/python" ; return 1; }

    python3 blocking_send_recv_example.py --mode="target" --ip=${DEFAULT_IP} --port=${DEFAULT_PORT} &
    sleep 5
    if ! python3 blocking_send_recv_example.py --mode="initiator" --ip=${DEFAULT_IP} --port=${DEFAULT_PORT}; then
        _error "python3 blocking_send_recv_example.py failed"
        return 1
    fi

    if ! python3 partial_md_example.py; then
        _error "python3 partial_md_example.py failed"
        return 1
    fi

    popd || _error "cannot pop from examples/python"

    # Calculate and print execution time
    local duration=$((SECONDS - start_time))
    echo "==== Python tests execution time: ${duration} seconds ===="

    return $rc
}

# Check if script is being sourced
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    # Script is being executed directly
    _error "This script is not meant to be executed directly"
    exit 1
fi
