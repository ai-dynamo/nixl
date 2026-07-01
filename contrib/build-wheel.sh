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

PYTHON_VERSION="3.12"
ARCH=$(uname -m)
WHL_PLATFORM="manylinux_2_39_$ARCH"
UCX_PLUGINS_DIR="/usr/lib64/ucx"
NIXL_PLUGINS_DIR="/usr/local/nixl/lib/$ARCH-linux-gnu/plugins"
OUTPUT_DIR="dist"
BUILD_NIXL_EP="false"
TORCH_VERSIONS=""
WHEEL_NAME=""
BACKEND_DIR=""

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
            echo "  --torch-versions: Comma-separated list of torch versions to build the wheel for (default: $TORCH_VERSIONS)"
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

if [ "$BUILD_NIXL_EP" = "true" ] && [ -z "$TORCH_VERSIONS" ]; then
    echo "ERROR: --build-nixl-ep requires --torch-versions (e.g. --torch-versions 2.11,2.12)" >&2
    exit 1
fi

set -e
set -x

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

# ROCm (or any explicit --wheel-name) build: nixl and torch are already built
# and installed in the build stage, so do one self-contained wheel build and
# skip the CUDA-only nvcc / multi-torch machinery below.
if [ -z "$CUDA_MAJOR" ]; then
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
    rm -rf "$TMP_DIR"
    exit 0
fi

AUDITWHEEL_EXCLUDES="--exclude libcuda* --exclude libcufile* --exclude libssl* --exclude libcrypto* --exclude libefa* --exclude libhwloc* --exclude libfabric* --exclude libtorch* --exclude libc10* --exclude libdoca* --exclude libred_client* --exclude libred_async* --exclude liblz4*"

PKG_NAME="nixl-cu${CUDA_MAJOR}"
CU_TAG="cu$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | tr -d .)"
./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml

TORCH_STABLE_INDEX="https://download.pytorch.org/whl/${CU_TAG}"
TORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/${CU_TAG}"

# Build deps for the per-iteration venv; torch is installed separately.
BUILD_DEPS=(
    "meson"
    "meson-python"
    "pybind11"
    "patchelf"
    "pyyaml"
    "types-PyYAML"
    "setuptools>=80.9.0"
)

# Slugify a dotted version (e.g. "2.13" -> "213", "3.10" -> "310") so it can
# be used unambiguously as a path component.
slug() { echo "${1//./}"; }

# Cache torch channel per version for this script run. Populated by
# ensure_torch_venv(); cleared when the venv is torn down after a build.
declare -A TORCH_CHANNEL_CACHE

# Path for a per-iteration build venv. One venv per (python, torch) tuple
# so torch's transitive footprint (nvidia-*, triton, sympy, …) never bleeds
# across torch versions. Lives in /workspace, not /tmp, so it inherits the
# image's UV_CACHE_DIR layout and is visible to debugging.
venv_path() {
    local VER=${1:-}
    if [ -n "$VER" ]; then
        echo "/workspace/venv-torch$(slug "$VER")-py$(slug "$PYTHON_VERSION")"
    else
        echo "/workspace/venv-py$(slug "$PYTHON_VERSION")"
    fi
}

torch_venv_ready() {
    local VENV_PATH=$1
    [ -x "$VENV_PATH/bin/python" ] && \
        "$VENV_PATH/bin/python" -c 'import torch' >/dev/null 2>&1
}

# Install torch from the cu index, isolated from PyPI: with PyPI as a
# fallback its plain `torch==X.Y.0` beats cu nightly's `X.Y.0.dev*+cuXX`
# (PEP 440: final > pre-release).
install_torch() {
    local VENV_PATH=$1
    local VER=$2
    local CHANNEL=$3
    local MAJOR="${VER%%.*}"
    local MINOR="${VER##*.}"

    if [ "$CHANNEL" = "nightly" ]; then
        uv pip install \
            --python "$VENV_PATH/bin/python" \
            --index-url "$TORCH_NIGHTLY_INDEX" \
            --pre \
            "torch>=${MAJOR}.${MINOR}.0.dev0,<${MAJOR}.$((MINOR + 1))"
    else
        uv pip install \
            --python "$VENV_PATH/bin/python" \
            --index-url "$TORCH_STABLE_INDEX" \
            "torch==${VER}.*"
    fi
}

# Create or reuse a venv with torch installed. uv's --dry-run does not
# always reject wheels with a mismatched platform tag, so we probe with a
# real install once and reuse the venv for the wheel build.
# Sets TORCH_CHANNEL_CACHE and returns 0 on success, 1 if unavailable.
ensure_torch_venv() {
    local VER=$1
    local VENV_PATH
    VENV_PATH=$(venv_path "$VER")
    local CHANNEL="${TORCH_CHANNEL_CACHE[$VER]:-}"

    if [ "$CHANNEL" = "unavailable" ]; then
        return 1
    fi
    if [ -n "$CHANNEL" ] && torch_venv_ready "$VENV_PATH"; then
        return 0
    fi

    rm -rf "$VENV_PATH"
    if ! uv venv "$VENV_PATH" --python "$PYTHON_VERSION"; then
        TORCH_CHANNEL_CACHE[$VER]="unavailable"
        return 1
    fi

    if install_torch "$VENV_PATH" "$VER" "stable" 2>/dev/null; then
        CHANNEL="stable"
    elif install_torch "$VENV_PATH" "$VER" "nightly" 2>/dev/null; then
        CHANNEL="nightly"
    else
        rm -rf "$VENV_PATH"
        TORCH_CHANNEL_CACHE[$VER]="unavailable"
        return 1
    fi

    TORCH_CHANNEL_CACHE[$VER]="$CHANNEL"
    return 0
}

torch_channel() {
    echo "${TORCH_CHANNEL_CACHE[$1]:-unavailable}"
}

destroy_torch_venv() {
    local VER=${1:-}
    local VENV_PATH
    VENV_PATH=$(venv_path "$VER")
    rm -rf "$VENV_PATH"
    if [ -n "$VER" ]; then
        unset "TORCH_CHANNEL_CACHE[$VER]"
    fi
}

# Build the wheel for the current PYTHON_VERSION (and optional torch VER).
# Each iteration uses a fresh venv so torch's dependencies
# (nvidia-* wheels, triton, sympy, …) do not leak across iterations.
build_wheel() {
    local OUT_DIR=$1
    local VER=${2:-}

    local VENV_PATH
    VENV_PATH=$(venv_path "$VER")
    local CHANNEL="stable"
    if [ -n "$VER" ]; then
        ensure_torch_venv "$VER" || {
            echo "ERROR: torch ${VER} is not installable for Python ${PYTHON_VERSION} + ${CU_TAG} on $(uname -m)" >&2
            exit 1
        }
        CHANNEL=$(torch_channel "$VER")
    else
        rm -rf "$VENV_PATH"
        uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
    fi

    echo "=== Provisioning ${VENV_PATH} (python ${PYTHON_VERSION}${VER:+, torch ${VER} [${CHANNEL}]}) ==="
    uv pip install --python "$VENV_PATH/bin/python" "${BUILD_DEPS[@]}"

    # Activate so meson's `find_installation('python3')` resolves to this
    # venv's interpreter (which has the right torch).
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"

    local BUILD_ARGS=(
        --wheel
        --no-build-isolation
        --out-dir "$OUT_DIR"
        --python "$VENV_PATH/bin/python"
    )
    if [ "$BUILD_NIXL_EP" = "true" ]; then
        BUILD_ARGS+=(
            -Csetup-args=-Dbuild_nixl_ep=true
            -Csetup-args=-Dbuild_examples=true
        )
    fi
    uv build "${BUILD_ARGS[@]}"

    deactivate
    # torch + nvidia-* in each venv is several GB; tear down so the docker
    # layer does not get too large across the (python, torch) matrix.
    destroy_torch_venv "$VER"
}

repair_wheel() {
    local IN_DIR=$1
    local OUT_DIR=$2
    mkdir -p "$OUT_DIR"
    auditwheel repair $AUDITWHEEL_EXCLUDES "$IN_DIR"/nixl*.whl --plat "$WHL_PLATFORM" --wheel-dir "$OUT_DIR"
    ./contrib/wheel_add_ucx_plugins.py --ucx-plugins-dir "$UCX_PLUGINS_DIR" --nixl-plugins-dir "$NIXL_PLUGINS_DIR" "$OUT_DIR"/*.whl
}

# Echo the path of the single .whl in $1, or exit if the count is not 1.
get_wheel_path() {
    local dir=$1 wheels
    shopt -s nullglob
    wheels=("$dir"/*.whl)
    shopt -u nullglob
    if [ ${#wheels[@]} -ne 1 ]; then
        echo "expected 1 wheel in $dir, got ${#wheels[@]}: ${wheels[*]}" >&2
        exit 1
    fi
    echo "${wheels[0]}"
}

if [ "$BUILD_NIXL_EP" = "true" ] && [ -n "$TORCH_VERSIONS" ]; then
    # Multi-torch: build the full wheel with the first torch, then merge
    # the per-torch .so from the others into it.
    IFS=',' read -ra TORCH_REQUESTED <<< "$TORCH_VERSIONS"

    # Filter to torch versions actually resolvable for this (Python, CUDA) combo.
    TORCH_ARRAY=()
    SKIPPED=()
    for TORCH in "${TORCH_REQUESTED[@]}"; do
        if ensure_torch_venv "$TORCH"; then
            TORCH_ARRAY+=("$TORCH")
        else
            SKIPPED+=("$TORCH")
        fi
    done

    if [ ${#SKIPPED[@]} -gt 0 ]; then
        echo "=== Skipping torch versions (no wheel on index for Python ${PYTHON_VERSION} + ${CU_TAG}): ${SKIPPED[*]} ==="
    fi
    if [ ${#TORCH_ARRAY[@]} -eq 0 ]; then
        echo "ERROR: none of the requested torch versions (${TORCH_REQUESTED[*]}) are available for Python ${PYTHON_VERSION} + ${CU_TAG}"
        exit 1
    fi
    echo "=== Building for torch versions: ${TORCH_ARRAY[*]} ==="

    FIRST_TORCH="${TORCH_ARRAY[0]}"
    echo "=== Building wheel with torch ${FIRST_TORCH} ==="
    build_wheel "$TMP_DIR" "$FIRST_TORCH"
    repair_wheel "$TMP_DIR" "$TMP_DIR/dist"
    BASE_WHL=$(get_wheel_path "$TMP_DIR/dist")

    for ((i=1; i<${#TORCH_ARRAY[@]}; i++)); do
        TORCH="${TORCH_ARRAY[$i]}"
        echo "=== Building nixl_ep .so for torch ${TORCH} ==="

        EP_TMP=$(mktemp -d)
        build_wheel "$EP_TMP" "$TORCH"
        repair_wheel "$EP_TMP" "$EP_TMP/dist"

        # Merge only the torch-versioned .so. Both wheels were built
        # against the same outer C++ build, so its DT_NEEDED entries
        # (libucp-<hash>.so etc.) match what auditwheel already bundled
        # into $BASE_WHL.
        TORCH_MM=$(echo "$TORCH" | tr -d '.')
        EP_WHL=$(get_wheel_path "$EP_TMP/dist")
        ./contrib/wheel_merge.py \
            --base-wheel "$BASE_WHL" \
            --source-wheel "$EP_WHL" \
            --pattern "nixl_ep_cpp_torch${TORCH_MM}.*" \
            --target-dir "nixl_ep_cu${CUDA_MAJOR}"

        rm -rf "$EP_TMP"
    done

    cp "$BASE_WHL" "$OUTPUT_DIR"
else
    build_wheel "$TMP_DIR"
    repair_wheel "$TMP_DIR" "$TMP_DIR/dist"
    cp "$(get_wheel_path "$TMP_DIR/dist")" "$OUTPUT_DIR"
fi

# Clean up
rm -rf "$TMP_DIR"
