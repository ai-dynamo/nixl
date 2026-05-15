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

# Index URLs for the resolved CUDA tag. Stable cu index hosts released
# wheels; nightly index hosts dev/pre-release wheels.
TORCH_STABLE_INDEX="https://download.pytorch.org/whl/${CU_TAG}"
TORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/${CU_TAG}"

# Build deps installed into the per-iteration venv (torch is added
# separately with channel-appropriate constraints).
BUILD_DEPS=(
    "meson"
    "meson-python"
    "pybind11"
    "patchelf"
    "pyyaml"
    "types-PyYAML"
    "setuptools>=80.9.0"
)

# Channel cache for repeated lookups within a single script run.
declare -A TORCH_CHANNEL_CACHE

# Slugify a dotted version (e.g. "2.13" -> "213", "3.10" -> "310") so it can
# be used unambiguously as a path component.
slug() { echo "${1//./}"; }

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

# Determine whether torch==${VER}.* has a stable release on the stable cu
# index for the target Python version. Echoes "stable" or "nightly".
# A torch version is treated as stable iff a non-pre-release wheel resolves
# from the stable cu index alone (no nightly index, no --pre).
torch_channel() {
    local VER=$1
    if [ -n "${TORCH_CHANNEL_CACHE[$VER]:-}" ]; then
        echo "${TORCH_CHANNEL_CACHE[$VER]}"
        return
    fi
    local CHANNEL="nightly"
    local CHECK_VENV="/workspace/venv-probe-py$(slug "$PYTHON_VERSION")"
    rm -rf "$CHECK_VENV"
    if uv venv "$CHECK_VENV" --python "$PYTHON_VERSION" >/dev/null 2>&1; then
        if uv pip install --dry-run \
            --python "$CHECK_VENV/bin/python" \
            --index-url "$TORCH_STABLE_INDEX" \
            "torch==${VER}.*" >/dev/null 2>&1; then
            CHANNEL="stable"
        fi
    fi
    rm -rf "$CHECK_VENV"
    TORCH_CHANNEL_CACHE[$VER]="$CHANNEL"
    echo "$CHANNEL"
}

# Echo the torch requirement spec for the given (version, channel). Stable
# stays as `torch==X.Y.*`; nightly uses an explicit `.dev0` lower bound so
# pre-release/dev wheels are admissible without --prerelease=allow.
torch_spec() {
    local VER=$1
    local CHANNEL=$2
    if [ "$CHANNEL" = "nightly" ]; then
        local MAJOR="${VER%%.*}"
        local MINOR="${VER##*.}"
        echo "torch>=${MAJOR}.${MINOR}.0.dev0,<${MAJOR}.$((MINOR + 1))"
    else
        echo "torch==${VER}.*"
    fi
}

# Echo the uv index/resolution flags appropriate for the given channel,
# one flag per line. Stable builds use only the stable cu index and no
# pre-release allowance; nightly builds add the nightly index and
# --prerelease=allow.
torch_uv_flags() {
    local CHANNEL=$1
    if [ "$CHANNEL" = "nightly" ]; then
        printf '%s\n' \
            --extra-index-url "$TORCH_STABLE_INDEX" \
            --extra-index-url "$TORCH_NIGHTLY_INDEX" \
            --index-strategy unsafe-best-match \
            --prerelease allow
    else
        printf '%s\n' \
            --extra-index-url "$TORCH_STABLE_INDEX" \
            --index-strategy unsafe-best-match
    fi
}

# Check whether torch==${VER}.* is resolvable from any configured index
# (stable or nightly) for the target Python version. Echoes "yes" on
# success, nothing otherwise.
torch_available() {
    local VER=$1
    local CHECK_VENV="/workspace/venv-probe-py$(slug "$PYTHON_VERSION")"
    rm -rf "$CHECK_VENV"
    uv venv "$CHECK_VENV" --python "$PYTHON_VERSION" >/dev/null 2>&1 || return
    if uv pip install --dry-run --pre \
        --python "$CHECK_VENV/bin/python" \
        --extra-index-url "$TORCH_STABLE_INDEX" \
        --extra-index-url "$TORCH_NIGHTLY_INDEX" \
        --index-strategy unsafe-best-match \
        "torch==${VER}.*" >/dev/null 2>&1; then
        echo "yes"
    fi
    rm -rf "$CHECK_VENV"
}

# Build the wheel for the current PYTHON_VERSION (and optional torch VER).
# Creates a fresh venv at venv_path, installs build deps + torch with the
# channel-appropriate flags, runs `uv build --no-build-isolation`, and
# tears the venv down so the next iteration starts from a clean slate.
# Doing it this way instead of `pip install --reinstall torch` avoids
# orphan packages from the previous torch's transitive footprint
# (nvidia-* wheels, triton, sympy, …) bleeding across iterations.
build_wheel() {
    local OUT_DIR=$1
    local VER=${2:-}

    local VENV_PATH
    VENV_PATH=$(venv_path "$VER")
    local CHANNEL="stable"
    [ -n "$VER" ] && CHANNEL=$(torch_channel "$VER")

    local UV_FLAGS=()
    while IFS= read -r f; do UV_FLAGS+=("$f"); done < <(torch_uv_flags "$CHANNEL")

    local TORCH_PKG=()
    [ -n "$VER" ] && TORCH_PKG+=("$(torch_spec "$VER" "$CHANNEL")")

    echo "=== Provisioning ${VENV_PATH} (python ${PYTHON_VERSION}${VER:+, torch ${VER} [${CHANNEL}]}) ==="
    rm -rf "$VENV_PATH"
    uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
    uv pip install \
        --python "$VENV_PATH/bin/python" \
        "${UV_FLAGS[@]}" \
        "${BUILD_DEPS[@]}" \
        "${TORCH_PKG[@]}"

    # Activate so meson's `find_installation('python3')` resolves to this
    # venv's interpreter (which has the right torch). Deactivate before
    # returning so the caller's auditwheel keeps using the orchestration
    # venv on PATH.
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
    # Free disk: torch + nvidia-* wheels in a venv add up to several GB;
    # 3 torches × 5 pythons would otherwise blow the docker layer budget.
    rm -rf "$VENV_PATH"
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
    IFS=',' read -ra TORCH_REQUESTED <<< "$TORCH_VERSIONS"

    # Filter to torch versions actually resolvable for this (Python, CUDA) combo.
    TORCH_ARRAY=()
    SKIPPED=()
    for TORCH in "${TORCH_REQUESTED[@]}"; do
        if [ -n "$(torch_available "$TORCH")" ]; then
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
    BASE_WHL=$(ls "$TMP_DIR"/dist/*.whl)

    for ((i=1; i<${#TORCH_ARRAY[@]}; i++)); do
        TORCH="${TORCH_ARRAY[$i]}"
        echo "=== Building nixl_ep .so for torch ${TORCH} ==="

        EP_TMP=$(mktemp -d)
        build_wheel "$EP_TMP" "$TORCH"
        # Repair so the .so passes auditwheel for the manylinux tag before we extract it
        repair_wheel "$EP_TMP" "$EP_TMP/dist"

        # Extract torch-versioned .so from repaired wheel, inject into base wheel
        EP_EXTRACT=$(mktemp -d)
        unzip -o "$EP_TMP"/dist/*.whl -d "$EP_EXTRACT"
        BASE_EXTRACT=$(mktemp -d)
        unzip -o "$BASE_WHL" -d "$BASE_EXTRACT"

        TORCH_MM=$(echo "$TORCH" | tr -d '.')
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
