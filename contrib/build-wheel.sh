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
TORCH_VERSIONS=""

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
        --help)
            echo "Usage: $0 [--python-version <python-version>] [--platform <platform>] [--output-dir <output-dir>] [--ucx-plugins-dir <ucx-plugins-dir>] [--nixl-plugins-dir <nixl-plugins-dir>]"
            echo "  --python-version: Python version to build the wheel for (default: $PYTHON_VERSION)"
            echo "  --platform: Platform to build the wheel for (default: $WHL_PLATFORM)"
            echo "  --output-dir: Directory to output the wheel to (default: $OUTPUT_DIR)"
            echo "  --ucx-plugins-dir: Directory to find UCX plugins in (default: $UCX_PLUGINS_DIR)"
            echo "  --nixl-plugins-dir: Directory to find NIXL plugins in (default: $NIXL_PLUGINS_DIR)"
            echo "  --build-nixl-ep: Build wheel with nixl_ep package included (requires CUDA sm90-compatible environment)"
            echo "  --help: Show this help message"
            echo ""
            echo "Must be executed from the root of the NIXL repository."
            exit 0
            ;;
        --build-nixl-ep)
            BUILD_NIXL_EP="true"
            shift
            ;;
        --torch-versions)
            TORCH_VERSIONS=$2
            shift
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

CUDA_MAJOR=$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | cut -d'.' -f1)
# Must be 12 or 13
if [ "$CUDA_MAJOR" -ne 12 ] && [ "$CUDA_MAJOR" -ne 13 ]; then
    echo "Invalid CUDA_MAJOR: '$CUDA_MAJOR'"
    exit 1
fi
AUDITWHEEL_EXCLUDES="--exclude libcuda* --exclude libcufile* --exclude libssl* --exclude libcrypto* --exclude libefa* --exclude libhwloc* --exclude libfabric* --exclude libtorch* --exclude libc10* --exclude libdoca*"

PKG_NAME="nixl-cu${CUDA_MAJOR}"
CU_TAG="cu$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | tr -d .)"
./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml

# Pin the torch build dep in pyproject.toml so uv build's isolated env resolves it.
pin_torch() {
    local VER=$1
    ./contrib/tomlutil.py --torch-version "$VER" pyproject.toml
}

# PyPI stays primary (for meson-python etc); PyTorch indexes are extras for torch.
UV_BUILD_INDEX_FLAGS=(
    --extra-index-url "https://download.pytorch.org/whl/${CU_TAG}"
    --extra-index-url "https://download.pytorch.org/whl/nightly/${CU_TAG}"
    --index-strategy unsafe-best-match
    --prerelease allow
)

# Check whether torch==${VER}.* is resolvable from the configured indexes
# for the target Python version. Echoes "yes" on success, nothing otherwise.
torch_available() {
    local VER=$1
    local CHECK_VENV
    CHECK_VENV=$(mktemp -d)/venv
    uv venv "$CHECK_VENV" --python "$PYTHON_VERSION" >/dev/null 2>&1 || return
    # shellcheck disable=SC1090
    source "$CHECK_VENV/bin/activate"
    if uv pip install --dry-run --pre \
        --extra-index-url "https://download.pytorch.org/whl/${CU_TAG}" \
        --extra-index-url "https://download.pytorch.org/whl/nightly/${CU_TAG}" \
        --index-strategy unsafe-best-match \
        "torch==${VER}.*" >/dev/null 2>&1; then
        echo "yes"
    fi
    deactivate
    rm -rf "$(dirname "$CHECK_VENV")"
}

build_wheel() {
    local OUT_DIR=$1
    if [ "$BUILD_NIXL_EP" = "true" ]; then
        uv build --wheel --out-dir "$OUT_DIR" --python $PYTHON_VERSION \
            "${UV_BUILD_INDEX_FLAGS[@]}" \
            -Csetup-args=-Dbuild_nixl_ep=true \
            -Csetup-args=-Dbuild_examples=true
    else
        uv build --wheel --out-dir "$OUT_DIR" --python $PYTHON_VERSION \
            "${UV_BUILD_INDEX_FLAGS[@]}"
    fi
}

repair_wheel() {
    local IN_DIR=$1
    local OUT_DIR=$2
    mkdir -p "$OUT_DIR"
    auditwheel repair $AUDITWHEEL_EXCLUDES "$IN_DIR"/nixl*.whl --plat $WHL_PLATFORM --wheel-dir "$OUT_DIR"
    ./contrib/wheel_add_ucx_plugins.py --ucx-plugins-dir $UCX_PLUGINS_DIR --nixl-plugins-dir $NIXL_PLUGINS_DIR "$OUT_DIR"/*.whl
}

if [ "$BUILD_NIXL_EP" = "true" ] && [ -n "$TORCH_VERSIONS" ]; then
    # Multi-torch: build full wheel with first torch, then merge extra .so from others.
    IFS=',' read -ra TV_REQUESTED <<< "$TORCH_VERSIONS"

    # Filter to torch versions actually resolvable for this (Python, CUDA) combo.
    TV_ARRAY=()
    SKIPPED=()
    for TV in "${TV_REQUESTED[@]}"; do
        if [ -n "$(torch_available "$TV")" ]; then
            TV_ARRAY+=("$TV")
        else
            SKIPPED+=("$TV")
        fi
    done

    if [ ${#SKIPPED[@]} -gt 0 ]; then
        echo "=== Skipping torch versions (no wheel on index for Python ${PYTHON_VERSION} + ${CU_TAG}): ${SKIPPED[*]} ==="
    fi
    if [ ${#TV_ARRAY[@]} -eq 0 ]; then
        echo "ERROR: none of the requested torch versions (${TV_REQUESTED[*]}) are available for Python ${PYTHON_VERSION} + ${CU_TAG}"
        exit 1
    fi
    echo "=== Building for torch versions: ${TV_ARRAY[*]} ==="

    FIRST_TORCH="${TV_ARRAY[0]}"
    echo "=== Building wheel with torch ${FIRST_TORCH} ==="
    pin_torch "$FIRST_TORCH"
    build_wheel "$TMP_DIR"
    repair_wheel "$TMP_DIR" "$TMP_DIR/dist"
    BASE_WHL=$(ls "$TMP_DIR"/dist/*.whl)

    for ((i=1; i<${#TV_ARRAY[@]}; i++)); do
        TV="${TV_ARRAY[$i]}"
        echo "=== Building nixl_ep .so for torch ${TV} ==="
        pin_torch "$TV"

        EP_TMP=$(mktemp -d)
        build_wheel "$EP_TMP"

        # Extract torch-versioned .so from new wheel, inject into base wheel
        EP_EXTRACT=$(mktemp -d)
        unzip -o "$EP_TMP"/nixl*.whl -d "$EP_EXTRACT"
        BASE_EXTRACT=$(mktemp -d)
        unzip -o "$BASE_WHL" -d "$BASE_EXTRACT"

        TORCH_MM=$(echo "$TV" | tr -d '.')
        find "$EP_EXTRACT" -name "nixl_ep_cpp_torch${TORCH_MM}*" -exec cp {} "$BASE_EXTRACT"/nixl_ep_cu${CUDA_MAJOR}/ \;

        # Regenerate RECORD
        DIST_INFO=$(ls -d "$BASE_EXTRACT"/*.dist-info)
        (cd "$BASE_EXTRACT" && find . -type f ! -name RECORD -printf '%P\n' | while read f; do
            hash=$(python3 -c "import hashlib,base64; d=open('$f','rb').read(); print('sha256=' + base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode())")
            size=$(stat -c%s "$f")
            echo "$f,$hash,$size"
        done > "$DIST_INFO/RECORD"
        echo "$(basename $DIST_INFO)/RECORD,," >> "$DIST_INFO/RECORD")

        rm -f "$BASE_WHL"
        (cd "$BASE_EXTRACT" && zip -r "$BASE_WHL" .)
        rm -rf "$EP_TMP" "$EP_EXTRACT" "$BASE_EXTRACT"
    done

    cp "$BASE_WHL" "$OUTPUT_DIR"
else
    build_wheel "$TMP_DIR"
    repair_wheel "$TMP_DIR" "$TMP_DIR/dist"
    cp "$TMP_DIR"/dist/*.whl "$OUTPUT_DIR"
fi

# Clean up
rm -rf "$TMP_DIR"
