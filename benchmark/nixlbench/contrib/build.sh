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

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")
BUILD_CONTEXT_ARGS=""
NIXL_SRC=$(readlink -f "${SOURCE_DIR}/../../..")
NIXL_BUILD_CONTEXT_ARGS="--build-context nixl=$NIXL_SRC"
NIXL_BENCH_BUILD_CONTEXT_ARGS="--build-context nixlbench=$BUILD_CONTEXT/"
DOCKER_FILE="${SOURCE_DIR}/Dockerfile"
UCX_SRC=""
UCX_BUILD_CONTEXT_ARGS=""
BUILD_TYPE="release"
EFA_VERSION=""
commit_id=$(git rev-parse --short HEAD)

# Get latest TAG and add COMMIT_ID for dev
latest_tag=$(git describe --tags --abbrev=0 $(git rev-list --tags --max-count=1 main) | sed 's/^v//') || true
if [ -z ${latest_tag} ]; then
    latest_tag="0.0.1"
    echo "No git release tag found, setting to unknown version: ${latest_tag}"
fi

# Empty by default and forwarded independently below, so whichever one is left
# unset comes from the Dockerfile's own ARG default rather than an empty string.
BASE_IMAGE=""
BASE_IMAGE_TAG=""
ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"
WHL_BASE=manylinux_2_39
WHL_PLATFORM=${WHL_BASE}_${ARCH}
WHL_PYTHON_VERSIONS="3.12"
NPROC=${NPROC:-$(nproc)}
APT_MIRROR=""
# VALKEY / GLIDE:
#   --enable-valkey           build valkey-glide FFI inside the image (needs network)
#   --glide-path <prefix>     use a prebuilt prefix instead (skips cargo in Docker)
#   --valkey-glide-ref <tag>  git tag/branch for in-image build (default v2.6.0-rc2)
GLIDE_PATH=""
ENABLE_VALKEY=0
GLIDE_FROM_SOURCE=0
VALKEY_GLIDE_REF="v2.6.0-rc2"
GLIDE_BUILD_CONTEXT_ARGS=""

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
        --base-image)
            if [ "$2" ]; then
                BASE_IMAGE="$2"
                shift
            else
                missing_requirement $1
            fi
        ;;
        --base-image-tag)
            if [ "$2" ]; then
                BASE_IMAGE_TAG=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --build-type)
            if [ "$2" ]; then
                BUILD_TYPE=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --aws)
            DOCKER_FILE="${SOURCE_DIR}/Dockerfile.aws-amzn2023"
            ;;
        --efa-version)
            if [ "$2" ]; then
                EFA_VERSION=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --nixl)
            if [ "$2" ]; then
                NIXL_BUILD_CONTEXT_ARGS="--build-context nixl=$2"
                NIXL_SRC=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --nixlbench)
            if [ "$2" ]; then
                NIXL_BENCH_BUILD_CONTEXT_ARGS="--build-context nixlbench=$2"
                commit_id=$(git --git-dir=$2/.git --work-tree=$2 rev-parse --short HEAD)
                shift
            else
                missing_requirement $1
            fi
            ;;
        --enable-valkey)
            ENABLE_VALKEY=1
            GLIDE_FROM_SOURCE=1
            ;;
        --glide-path)
            if [ "$2" ]; then
                GLIDE_PATH=$(readlink -f "$2")
                ENABLE_VALKEY=1
                GLIDE_FROM_SOURCE=0
                shift
            else
                missing_requirement $1
            fi
            ;;
        --valkey-glide-ref)
            if [ "$2" ]; then
                VALKEY_GLIDE_REF=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --no-cache)
            NO_CACHE=" --no-cache"
            ;;
        --python-versions)
            if [ "$2" ]; then
                WHL_PYTHON_VERSIONS=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --tag)
            if [ "$2" ]; then
                TAG="--tag $2"
                shift
            else
                missing_requirement $1
            fi
            ;;
        --ucx)
            if [ "$2" ]; then
                UCX_SRC=$2
                UCX_BUILD_CONTEXT_ARGS="--build-context ucx=$2 --build-arg UCX=custom"
                shift
            else
                missing_requirement $1
            fi
            ;;
        --arch)
            if [ "$2" ]; then
                ARCH=$2
                WHL_PLATFORM=${WHL_BASE}_${ARCH}
                shift
            else
                missing_requirement $1
            fi
            ;;
        --apt-mirror)
            if [ "$2" ]; then
                APT_MIRROR=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --)
            shift
            break
            ;;
         -?*)
            error 'ERROR: Unknown option: ' $1
            ;;
         ?*)
            error 'ERROR: Unknown option: ' $1
            ;;
        *)
            break
            ;;
        esac
        shift
    done

    if [ -z "$NIXL_BUILD_CONTEXT_ARGS" ]; then
        error "ERROR: --nixl <path to nixl source> is required"
    fi

    if [ "$ENABLE_VALKEY" = "1" ] && [ -n "$GLIDE_PATH" ]; then
        if [ ! -f "$GLIDE_PATH/lib/libglide_ffi.a" ]; then
            error "ERROR: --glide-path missing lib/libglide_ffi.a under" "$GLIDE_PATH"
        fi
        if [ ! -f "$GLIDE_PATH/include/glide_ffi.h" ]; then
            error "ERROR: --glide-path missing include/glide_ffi.h under" "$GLIDE_PATH"
        fi
        if [ ! -f "$GLIDE_PATH/proto/connection_request.proto" ]; then
            error "ERROR: --glide-path missing proto/connection_request.proto under" "$GLIDE_PATH"
        fi
        GLIDE_FROM_SOURCE=0
        GLIDE_BUILD_CONTEXT_ARGS="--build-context glide=$GLIDE_PATH"
    else
        # Named context is always required by the Dockerfile COPY --from=glide.
        # From-source builds ignore the stub and clone valkey-glide in the image.
        GLIDE_STUB=$(mktemp -d)
        touch "$GLIDE_STUB/.keep"
        trap 'rm -rf "$GLIDE_STUB"' EXIT
        GLIDE_BUILD_CONTEXT_ARGS="--build-context glide=$GLIDE_STUB"
    fi

    BUILD_CONTEXT_ARGS="$NIXL_BUILD_CONTEXT_ARGS $NIXL_BENCH_BUILD_CONTEXT_ARGS $UCX_BUILD_CONTEXT_ARGS $GLIDE_BUILD_CONTEXT_ARGS"

    VERSION=v$latest_tag.dev.$commit_id
    if [ -z "$TAG" ]; then
        TAG="--tag nixlbench:${VERSION}"
        echo $TAG
    fi
}

show_build_options() {
    echo ""
    echo "Building NIXLBench Image"
    echo "Dockerfile: ${DOCKER_FILE}"
    echo "NIXL Source: ${NIXL_SRC}"
    echo "UCX Source: ${UCX_SRC} (optional)"
    echo "VALKEY: ENABLE_VALKEY=${ENABLE_VALKEY} GLIDE_FROM_SOURCE=${GLIDE_FROM_SOURCE}"
    echo "GLIDE Path: ${GLIDE_PATH:-<none>}  (ref=${VALKEY_GLIDE_REF})"
    echo "Build Type: ${BUILD_TYPE}"
    echo "Image Tag: ${TAG}"
    echo "Build Context: ${BUILD_CONTEXT}"
    echo "Build Context Args: ${BUILD_CONTEXT_ARGS}"
    # Each component is overridden independently; whichever is unset comes
    # from the Dockerfile's own ARG default.
    echo "Base Image: ${BASE_IMAGE:-<Dockerfile default>}:${BASE_IMAGE_TAG:-<Dockerfile default>}"
    if [ -n "$EFA_VERSION" ]; then
        echo "EFA Version: ${EFA_VERSION}"
    fi
    echo "Container arch: ${ARCH}"
    echo "Python Versions for wheel build: ${WHL_PYTHON_VERSIONS}"
    echo "Wheel Platform: ${WHL_PLATFORM}"
}

show_help() {
    echo "usage: build.sh --nixl <path to nixl source>"
    echo "  [--base-image base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--aws build with Dockerfile.aws-amzn2023]"
    echo "  [--efa-version EFA installer version]"
    echo "  [--nixlbench path/to/nixlbench/source/dir]"
    echo "  [--ucx path/to/ucx/source/dir]"
    echo "  [--enable-valkey]  # build valkey-glide FFI in-image and enable VALKEY plugin"
    echo "  [--glide-path path/to/glide/prefix]  # optional prebuilt prefix instead of in-image cargo"
    echo "  [--valkey-glide-ref tag]  # default v2.6.0-rc2 (used with --enable-valkey)"
    echo "  [--build-type [debug|release] to select build type]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--python-versions python versions to build for, comma separated]"
    echo "  [--tag tag for image]"
    echo "  [--arch [x86_64|aarch64] to select target architecture]"
    echo "  [--apt-mirror base URL of an apt mirror to use instead of the public Ubuntu archive]"
    echo "  [DOCKER_BUILD_EXTRA_ARGS env var: extra arguments passed through to docker build, e.g. --secret id=ghconfig,src=<file>; word-split, so paths with spaces are not supported]"
    exit 0
}

missing_requirement() {
    error "ERROR: $1 requires an argument."
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

BUILD_ARGS+="${BASE_IMAGE:+ --build-arg BASE_IMAGE=$BASE_IMAGE}"
BUILD_ARGS+="${BASE_IMAGE_TAG:+ --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG}"
BUILD_ARGS+=" --build-arg WHL_PYTHON_VERSIONS=$WHL_PYTHON_VERSIONS"
BUILD_ARGS+=" --build-arg WHL_PLATFORM=$WHL_PLATFORM"
BUILD_ARGS+=" --build-arg ARCH=$ARCH"
BUILD_ARGS+=" --build-arg BUILD_TYPE=$BUILD_TYPE"
BUILD_ARGS+=" --build-arg NPROC=$NPROC"
BUILD_ARGS+=" --build-arg ENABLE_VALKEY=$ENABLE_VALKEY"
BUILD_ARGS+=" --build-arg GLIDE_FROM_SOURCE=$GLIDE_FROM_SOURCE"
BUILD_ARGS+=" --build-arg VALKEY_GLIDE_REF=$VALKEY_GLIDE_REF"
if [ -n "$EFA_VERSION" ]; then
    BUILD_ARGS+=" --build-arg EFA_VERSION=$EFA_VERSION"
fi
BUILD_ARGS+="${APT_MIRROR:+ --build-arg APT_MIRROR=$APT_MIRROR}"

show_build_options

docker build --platform linux/$ARCH -f $DOCKER_FILE $BUILD_ARGS ${DOCKER_BUILD_EXTRA_ARGS:-} $TAG $NO_CACHE $BUILD_CONTEXT_ARGS $BUILD_CONTEXT --progress plain
