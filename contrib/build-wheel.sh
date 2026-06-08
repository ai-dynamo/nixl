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

# Parse arguments
PYTHON_VERSION="3.12"
ARCH=$(uname -m)
WHL_PLATFORM="manylinux_2_39_$ARCH"
UCX_PLUGINS_DIR="/usr/lib64/ucx"
NIXL_PLUGINS_DIR="/usr/local/nixl/lib/$ARCH-linux-gnu/plugins"
OUTPUT_DIR="dist"
BUILD_NIXL_EP="false"
WHEEL_NAME=""
BACKEND_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION=$2
            shift
            shift
            ;;
        --platform)
            WHL_PLATFORM=$2
            shift
            shift
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift
            shift
            ;;
        --ucx-plugins-dir)
            UCX_PLUGINS_DIR=$2
            shift
            shift
            ;;
        --nixl-plugins-dir)
            NIXL_PLUGINS_DIR=$2
            shift
            shift
            ;;
        --wheel-name)
            WHEEL_NAME=$2
            shift
            shift
            ;;
        --backend-dir)
            BACKEND_DIR=$2
            shift
            shift
            ;;
        --help)
            echo "Usage: $0 [--python-version <python-version>] [--platform <platform>] [--output-dir <output-dir>] [--ucx-plugins-dir <ucx-plugins-dir>] [--nixl-plugins-dir <nixl-plugins-dir>] [--wheel-name <wheel-name>] [--backend-dir <backend-dir>]"
            echo "  --python-version: Python version to build the wheel for (default: $PYTHON_VERSION)"
            echo "  --platform: Platform to build the wheel for (default: $WHL_PLATFORM)"
            echo "  --output-dir: Directory to output the wheel to (default: $OUTPUT_DIR)"
            echo "  --ucx-plugins-dir: Directory to find UCX plugins in (default: $UCX_PLUGINS_DIR)"
            echo "  --nixl-plugins-dir: Directory to find NIXL plugins in (default: $NIXL_PLUGINS_DIR)"
            echo "  --wheel-name: Python wheel distribution name (default: auto-detected CUDA wheel)"
            echo "  --backend-dir: Python import package directory (default: wheel name with '-' replaced by '_')"
            echo "  --build-nixl-ep: Build wheel with nixl_ep package included (requires a CUDA sm_90 or newer target environment)"
            echo "  --help: Show this help message"
            echo ""
            echo "Must be executed from the root of the NIXL repository."
            exit 0
            ;;
        --build-nixl-ep)
            BUILD_NIXL_EP="true"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

set -e
set -x

# Build the wheel
TMP_DIR=$(mktemp -d)

if [ -z "$WHEEL_NAME" ]; then
    CUDA_MAJOR=$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | cut -d'.' -f1)
    # Must be 12 or 13
    if [ "$CUDA_MAJOR" -ne 12 ] && [ "$CUDA_MAJOR" -ne 13 ]; then
        echo "Invalid CUDA_MAJOR: '$CUDA_MAJOR'"
        exit 1
    fi
    WHEEL_NAME="nixl-cu${CUDA_MAJOR}"
    BACKEND_DIR="nixl_cu${CUDA_MAJOR}"
elif [ -z "$BACKEND_DIR" ]; then
    BACKEND_DIR="${WHEEL_NAME//-/_}"
fi

./contrib/tomlutil.py --wheel-name "$WHEEL_NAME" pyproject.toml

UV_BUILD_ARGS=(
    --wheel
    --out-dir "$TMP_DIR"
    --python "$PYTHON_VERSION"
    -Csetup-args="-Dpython_backend_dir=$BACKEND_DIR"
    -Csetup-args="-Dpython_backend_package=$WHEEL_NAME"
)

if [ "$BUILD_NIXL_EP" = "true" ]; then
    UV_BUILD_ARGS+=(
        -Csetup-args=-Dbuild_nixl_ep=true
        -Csetup-args=-Dbuild_examples=true
    )
fi

uv build "${UV_BUILD_ARGS[@]}"

# Bundle libraries
mkdir "$TMP_DIR/dist"
auditwheel repair \
    --exclude 'libcuda*' --exclude 'libcufile*' \
    --exclude 'libssl*' --exclude 'libcrypto*' \
    --exclude 'libefa*' --exclude 'libhwloc*' --exclude 'libfabric*' \
    --exclude 'libtorch*' --exclude 'libc10*' --exclude 'libdoca*' \
    --exclude 'libred_client*' --exclude 'libred_async*' --exclude 'liblz4*' \
    --exclude 'libamdhip64*' --exclude 'libhsa-runtime64*' \
    --exclude 'libroc*' --exclude 'librocm*' --exclude 'libhip*' --exclude 'libamd_comgr*' \
    "$TMP_DIR"/nixl*.whl --plat "$WHL_PLATFORM" --wheel-dir "$TMP_DIR/dist"
./contrib/wheel_add_ucx_plugins.py --ucx-plugins-dir "$UCX_PLUGINS_DIR" --nixl-plugins-dir "$NIXL_PLUGINS_DIR" "$TMP_DIR"/dist/*.whl
cp "$TMP_DIR"/dist/*.whl "$OUTPUT_DIR"

# Clean up
rm -rf "$TMP_DIR"
