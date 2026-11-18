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
TORCH_CUDA_INDEX=""
GPU_STACK="cuda"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ROCM_VERSION=""
WHL_FILE_PREFIX="nixl"

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
            echo "  --build-nixl-ep: Build wheel with nixl_ep package included (requires a CUDA sm_90 or newer target environment)"
            echo "  --torch-versions: Comma-separated list of torch versions to build the wheel for (default: $TORCH_VERSIONS)"
            echo "  --torch-cuda-index: torch wheel index, e.g. cu132 (default: cu132 for CUDA 13, cu<major><minor> of the nvcc toolkit for CUDA 12)"
            echo "  --gpu-stack: GPU stack to build the wheel for, cuda or rocm (default: $GPU_STACK)"
            echo "  --rocm-path: ROCm installation prefix, only used with --gpu-stack rocm (default: $ROCM_PATH)"
            echo "  --rocm-version: ROCm version X.Y[.Z], only used with --gpu-stack rocm (default: read from the ROCm installation)"
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
        --torch-cuda-index)
            TORCH_CUDA_INDEX=$2
            shift
            shift
            ;;
        --gpu-stack)
            GPU_STACK=$2
            shift
            shift
            ;;
        --rocm-path)
            ROCM_PATH=$2
            shift
            shift
            ;;
        --rocm-version)
            ROCM_VERSION=$2
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
    echo "ERROR: --build-nixl-ep requires --torch-versions" >&2
    exit 1
fi

# The ROCm runtime comes from the host installation and must never be vendored.
ROCM_RUNTIME_LIBS=(
    "libamdhip64"
    "libhipfile"
    "libhsa-runtime64"
    "librocm_sysdeps"
    "librocprofiler-register"
    "libamd_comgr"
)
ROCM_PREFIX_LIBS=()

# Names of the shared libraries at the top level of <rocm-path>/lib, without
# the .so suffix. A name the system library directories also provide is left
# out: excluding it would stop auditwheel from vendoring the system copy too.
rocm_prefix_libs() {
    local lib name sysdir known
    shopt -s nullglob
    for lib in "$ROCM_PATH"/lib/lib*.so*; do
        name=$(basename "$lib")
        name=${name%%.so*}
        known="false"
        for sysdir in /lib64 /usr/lib64 /lib /usr/lib "/usr/lib/$ARCH-linux-gnu" /usr/local/lib64 /usr/local/lib; do
            if compgen -G "$sysdir/$name.so*" >/dev/null; then
                known="true"
                break
            fi
        done
        if [ "$known" = "false" ]; then
            echo "$name"
        fi
    done | sort -u
    shopt -u nullglob
}

case "$GPU_STACK" in
    cuda) ;;
    rocm)
        if [ "$ARCH" != "x86_64" ]; then
            echo "ERROR: --gpu-stack rocm is only supported on x86_64 (this machine is $ARCH)" >&2
            exit 1
        fi
        if [ "$BUILD_NIXL_EP" = "true" ]; then
            echo "ERROR: --build-nixl-ep is not supported with --gpu-stack rocm (nixl_ep is CUDA-only)" >&2
            exit 1
        fi
        if [ -n "$TORCH_VERSIONS" ]; then
            echo "ERROR: --torch-versions is not supported with --gpu-stack rocm (the ROCm wheel is not built against torch)" >&2
            exit 1
        fi
        if [ -n "$TORCH_CUDA_INDEX" ]; then
            echo "ERROR: --torch-cuda-index is not supported with --gpu-stack rocm" >&2
            exit 1
        fi
        if [ ! -d "$ROCM_PATH/lib" ]; then
            echo "ERROR: no ROCm installation at '$ROCM_PATH' (set --rocm-path or ROCM_PATH)" >&2
            exit 1
        fi
        if [ -z "$ROCM_VERSION" ]; then
            for ROCM_VERSION_FILE in "$ROCM_PATH/.info/version" "$ROCM_PATH/core/.info/version"; do
                if [ -r "$ROCM_VERSION_FILE" ]; then
                    ROCM_VERSION=$(grep -Eom1 '^[0-9]+\.[0-9]+(\.[0-9]+)?' "$ROCM_VERSION_FILE")
                    [ -n "$ROCM_VERSION" ] && break
                fi
            done
            if [ -z "$ROCM_VERSION" ]; then
                echo "ERROR: cannot determine the ROCm version from $ROCM_PATH/.info/version or $ROCM_PATH/core/.info/version; pass --rocm-version X.Y[.Z]" >&2
                exit 1
            fi
        fi
        if ! [[ "$ROCM_VERSION" =~ ^([0-9]+)\.([0-9]+)(\.[0-9]+)?$ ]]; then
            echo "ERROR: invalid ROCm version '$ROCM_VERSION' (expected X.Y[.Z])" >&2
            exit 1
        fi
        ROCM_MAJOR="${BASH_REMATCH[1]}"
        ROCM_MINOR="${BASH_REMATCH[2]}"
        ROCM_VARIANT="rocm${ROCM_MAJOR}_${ROCM_MINOR}"
        mapfile -t ROCM_PREFIX_LIBS < <(rocm_prefix_libs)
        ;;
    *)
        echo "ERROR: invalid --gpu-stack '$GPU_STACK' (expected cuda or rocm)" >&2
        exit 1
        ;;
esac

set -e
set -x

TMP_DIR=$(mktemp -d)

if [ "$GPU_STACK" = "rocm" ]; then
    AUDITWHEEL_EXCLUDES="--exclude libssl* --exclude libcrypto* --exclude libefa* --exclude libhwloc* --exclude libfabric* --exclude libdoca* --exclude libred_client* --exclude libred_async* --exclude liblz4*"
    for LIB in "${ROCM_RUNTIME_LIBS[@]}"; do
        AUDITWHEEL_EXCLUDES+=" --exclude ${LIB}*"
    done
    for LIB in "${ROCM_PREFIX_LIBS[@]}"; do
        AUDITWHEEL_EXCLUDES+=" --exclude ${LIB}.so*"
    done
    # Distribution nixl-rocm<X>.<Y>; its normalized name, which prefixes the wheel
    # file, is the import package nixl_rocm<X>_<Y>.
    PKG_NAME="nixl-rocm${ROCM_MAJOR}.${ROCM_MINOR}"
    WHL_FILE_PREFIX="nixl_${ROCM_VARIANT}-"
    echo "=== Building ${PKG_NAME} for ROCm ${ROCM_VERSION} (import package nixl_${ROCM_VARIANT}) ==="
    echo "=== auditwheel excludes: ${AUDITWHEEL_EXCLUDES} ==="

    ./contrib/tomlutil.py --wheel-name "$PKG_NAME" --remove-dependency torch pyproject.toml
else
    CUDA_MAJOR=$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | cut -d'.' -f1)
    # Must be 12 or 13
    if [ "$CUDA_MAJOR" -ne 12 ] && [ "$CUDA_MAJOR" -ne 13 ]; then
        echo "Invalid CUDA_MAJOR: '$CUDA_MAJOR'"
        exit 1
    fi
    AUDITWHEEL_EXCLUDES="--exclude libcuda* --exclude libcufile* --exclude libcuobjclient* --exclude libssl* --exclude libcrypto* --exclude libefa* --exclude libhwloc* --exclude libfabric* --exclude libtorch* --exclude libc10* --exclude libdoca* --exclude libred_client* --exclude libred_async* --exclude liblz4*"

    PKG_NAME="nixl-cu${CUDA_MAJOR}"
    # torch index: --torch-cuda-index, else cu132 for CUDA 13 (PyTorch has no stable
    # cu134 wheels) or cu<major><minor> of the toolkit; the CUDA major must match.
    if [ -n "$TORCH_CUDA_INDEX" ]; then
        CU_TAG="$TORCH_CUDA_INDEX"
    elif [ "$CUDA_MAJOR" -eq 13 ]; then
        CU_TAG="cu132"
    else
        CU_TAG="cu$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | tr -d .)"
    fi
    case "$CU_TAG" in
        cu${CUDA_MAJOR}[0-9]*) ;;
        *)
            echo "ERROR: torch index '$CU_TAG' does not match the CUDA ${CUDA_MAJOR} toolkit" >&2
            exit 1
            ;;
    esac
    ./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml

    TORCH_STABLE_INDEX="https://download.pytorch.org/whl/${CU_TAG}"
    TORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/${CU_TAG}"
fi

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
    # Toolchain evidence for the CI build manifest; never fatal.
    mkdir -p build-info && uv pip list --python "$VENV_PATH/bin/python" --format json \
        > "build-info/pip-py${PYTHON_VERSION}-torch${VER:-base}.json" || true

    # Activate so meson's `find_installation('python3')` resolves to this
    # venv's interpreter (which has the right torch).
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"

    local BUILD_ARGS=(
        --wheel
        --no-build-isolation
        --out-dir "$OUT_DIR"
        --python "$VENV_PATH/bin/python"
        -Ccompile-args=-j"${NPROC:-$(nproc)}"
    )
    if [ "$BUILD_NIXL_EP" = "true" ]; then
        BUILD_ARGS+=(
            -Csetup-args=-Dbuild_nixl_ep=true
            -Csetup-args=-Dbuild_examples=true
        )
    fi
    if [ "$GPU_STACK" = "rocm" ]; then
        BUILD_ARGS+=(
            -Csetup-args=-Drocm_path="$ROCM_PATH"
            -Csetup-args=-Dwheel_variant="$ROCM_VARIANT"
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
    auditwheel repair $AUDITWHEEL_EXCLUDES "$IN_DIR"/"$WHL_FILE_PREFIX"*.whl --plat "$WHL_PLATFORM" --wheel-dir "$OUT_DIR"
    ./contrib/wheel_add_ucx_plugins.py --ucx-plugins-dir "$UCX_PLUGINS_DIR" --nixl-plugins-dir "$NIXL_PLUGINS_DIR" "$OUT_DIR"/*.whl
    if [ "$GPU_STACK" = "rocm" ]; then
        check_rocm_wheel "$OUT_DIR"/"$WHL_FILE_PREFIX"*.whl
    fi
}

# Fail if the repaired wheel vendors a ROCm runtime or DOCA library, its import package
# is not nixl_<variant>, or it carries what wheel_merge_rocm.py adds to the
# rocm-nixl wheel (the nixl dispatcher) or what belongs to CUDA builds (nixl_ep).
check_rocm_wheel() {
    python3 - "$1" "$ROCM_VARIANT" "${ROCM_RUNTIME_LIBS[*]}" "${ROCM_PREFIX_LIBS[*]}" <<'EOF'
import os
import re
import sys
import zipfile

wheel, variant, runtime_libs, prefix_libs = sys.argv[1:5]
with zipfile.ZipFile(wheel) as whl:
    names = whl.namelist()

patterns = [re.compile(re.escape(lib)) for lib in runtime_libs.split()]
# auditwheel renames what it vendors to <name>-<hash>.so[.N].
patterns += [
    re.compile(re.escape(lib) + r"(-[0-9a-f]{8})?\.so")
    for lib in prefix_libs.split()
]
vendored = sorted(
    n for n in names if any(p.match(os.path.basename(n)) for p in patterns)
)

top_dirs = {n.split("/", 1)[0] for n in names if "/" in n}
# \w* keeps out nixl_<variant>.libs and nixl_<variant>-<version>.dist-info.
backends = sorted(d for d in top_dirs if re.fullmatch(r"nixl_(rocm|cu)\w*", d))
expected = "nixl_" + variant

errors = []
if "nixl" in top_dirs:
    errors.append("the nixl dispatcher package must not be in a ROCm backend wheel")
if "nixl_meta_utils.py" in names:
    errors.append("nixl_meta_utils.py must not be in a ROCm backend wheel")
ep = sorted(d for d in top_dirs if d == "nixl_ep" or d.startswith("nixl_ep_"))
if ep:
    errors.append("nixl_ep must not be in a ROCm backend wheel: " + ", ".join(ep))
if vendored:
    errors.append("ROCm runtime libraries were vendored: " + ", ".join(vendored))
doca = sorted(n for n in names if os.path.basename(n).startswith("libdoca"))
if doca:
    errors.append("DOCA libraries were vendored: " + ", ".join(doca))
if backends != [expected]:
    errors.append(
        "expected import package '%s', found: %s"
        % (expected, ", ".join(backends) or "none")
    )
for error in errors:
    print("ERROR: %s: %s" % (os.path.basename(wheel), error), file=sys.stderr)
sys.exit(1 if errors else 0)
EOF
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
