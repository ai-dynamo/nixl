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
DOCKER_FILE="${SOURCE_DIR}/Dockerfile"
commit_id=$(git rev-parse --short HEAD)

# Get latest TAG and add COMMIT_ID for dev
latest_tag=$(git describe --tags --abbrev=0 $(git rev-list --tags --max-count=1 main) | sed 's/^v//') || true
if [[ -z ${latest_tag} ]]; then
    latest_tag="0.0.1"
    echo "No git release tag found, setting to unknown version: ${latest_tag}"
fi
VERSION=v$latest_tag.dev.$commit_id

BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base
BASE_IMAGE_TAG=25.10-cuda13.0-devel-ubuntu24.04
CUDA_IMAGE=nvcr.io/nvidia/cuda
CUDA_IMAGE_TAG=13.2.1-devel-ubi8
CUDA_VERSION=13.2
MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28
MANYLINUX_IMAGE_TAG=2026.06.06-1
ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"
WHL_BASE=manylinux_2_39
WHL_BASE_EXPLICIT=false
WHL_PLATFORM=${WHL_BASE}_${ARCH}
WHL_PYTHON_VERSIONS="3.12"
UCX_REPO=${UCX_REPO:-https://github.com/openucx/ucx.git}
UCX_REF=${UCX_REF:-v1.22.x}
UCX_SONAME_SUFFIX=${UCX_SONAME_SUFFIX:-}
PRIVATE_UCX_SONAME_SUFFIX="nixl"
BUILD_NIXL_EP="true"
OS="ubuntu24"
NPROC=${NPROC:-$(nproc)}
GRPC_NPROC=${GRPC_NPROC:-$(nproc)}
BUILD_TYPE="release"
BUILD_INFINIA="false"
INFINIA_LIBS_IMAGE="harbor.mellanox.com/nixl/infinia-libs:v2.4.0-beta.1"
BUILD_UCX_SPCX_PLUGIN="false"
UCX_SPCX_PLUGIN_REF="main"

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
        --cuda-image)
            if [ "$2" ]; then
                CUDA_IMAGE=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --cuda-image-tag)
            if [ "$2" ]; then
                CUDA_IMAGE_TAG=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --cuda-version)
            if [ "$2" ]; then
                CUDA_VERSION=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --manylinux-image)
            if [ "$2" ]; then
                MANYLINUX_IMAGE=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --manylinux-image-tag)
            if [ "$2" ]; then
                MANYLINUX_IMAGE_TAG=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --wheel-base)
            if [ "$2" ]; then
                WHL_BASE=$2
                WHL_BASE_EXPLICIT=true
                shift
            else
                missing_requirement $1
            fi
            ;;
        --os)
            if [ "$2" ]; then
                OS=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --no-cache)
            NO_CACHE=" --no-cache"
            ;;
        --build-type)
            if [ "$2" ]; then
                BUILD_TYPE=$2
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
        --dockerfile)
            if [ "$2" ]; then
                DOCKER_FILE="$2"
                shift
            else
                missing_requirement $1
            fi
            ;;
        --python-versions)
            if [ "$2" ]; then
                WHL_PYTHON_VERSIONS=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --torch-versions)
            if [ "$2" ]; then
                WHL_TORCH_VERSIONS=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --ucx-repo)
            if [ "$2" ]; then
                UCX_REPO=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --ucx-ref)
            if [ "$2" ]; then
                UCX_REF=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --ucx-soname-suffix)
            if [ "$2" ]; then
                UCX_SONAME_SUFFIX=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --private-ucx)
            UCX_SONAME_SUFFIX=${UCX_SONAME_SUFFIX:-$PRIVATE_UCX_SONAME_SUFFIX}
            ;;
        --build-nixl-ep)
            BUILD_NIXL_EP=true
            ;;
        --build-ucx-spcx-plugin)
            BUILD_UCX_SPCX_PLUGIN="true"
            ;;
        --ucx-spcx-plugin-ref)
            if [ "$2" ]; then
                UCX_SPCX_PLUGIN_REF=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --arch)
            if [ "$2" ]; then
                ARCH=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --wheel-base-image)
            if [ "$2" ]; then
                WHEEL_BASE_IMAGE=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --build-infinia)
            BUILD_INFINIA="true"
            ;;
        --infinia-image)
            if [ "$2" ]; then
                INFINIA_LIBS_IMAGE=$2
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

    if [[ $OS == "ubuntu22" ]]; then
        BASE_IMAGE=nvidia/cuda
        BASE_IMAGE_TAG=13.0.1-devel-ubuntu22.04
        WHL_BASE=${WHL_BASE:-manylinux_2_34}
    fi

    if [[ "$DOCKER_FILE" == *Dockerfile.manylinux ]] && [ "$WHL_BASE_EXPLICIT" = "false" ]; then
        WHL_BASE=manylinux_2_28
    fi

    WHL_PLATFORM=${WHL_BASE}_${ARCH}

    case "$ARCH" in
        x86_64) TARGET_PLATFORM="linux/amd64" ;;
        aarch64) TARGET_PLATFORM="linux/arm64" ;;
        *) error "ERROR:" "unsupported architecture: $ARCH" ;;
    esac

    if [ -z "$TAG" ]; then
        TAG="--tag nixl:${VERSION}"
    fi
}

show_build_options() {
    echo ""
    echo "Building NIXL Image"
    echo "Image Tag: ${TAG}"
    echo "Build Context: ${BUILD_CONTEXT}"
    if [[ "$DOCKER_FILE" == *Dockerfile.manylinux ]]; then
        echo "CUDA Source Image: ${CUDA_IMAGE}:${CUDA_IMAGE_TAG}"
        echo "Manylinux Image: ${MANYLINUX_IMAGE}_${ARCH}:${MANYLINUX_IMAGE_TAG}"
        echo "CUDA Version: ${CUDA_VERSION}"
    else
        echo "Base Image: ${BASE_IMAGE}:${BASE_IMAGE_TAG}"
    fi
    echo "Container arch: ${ARCH}"
    echo "Docker platform: ${TARGET_PLATFORM}"
    echo "Python Versions for wheel build: ${WHL_PYTHON_VERSIONS}"
    echo "Wheel Platform: ${WHL_PLATFORM}"
    echo "UCX Repo: ${UCX_REPO}"
    echo "UCX Ref: ${UCX_REF}"
    if [ -n "$UCX_SONAME_SUFFIX" ]; then
        echo "UCX SONAME suffix: ${UCX_SONAME_SUFFIX}"
        echo "UCX module deepbind: Enabled"
    else
        echo "UCX SONAME suffix: Disabled"
        echo "UCX module deepbind: Disabled"
    fi
    if [ "$BUILD_NIXL_EP" = "true" ]; then
        echo "NIXL EP: Enabled"
    else
        echo "NIXL EP: Disabled"
    fi
    if [ "$BUILD_INFINIA" = "true" ]; then
        echo "Infinia DDN plugin: Enabled (image: ${INFINIA_LIBS_IMAGE})"
    else
        echo "Infinia DDN plugin: Disabled"
    fi
    if [ "$BUILD_UCX_SPCX_PLUGIN" = "true" ]; then
        echo "UCX spcx plugin: Enabled (ref: ${UCX_SPCX_PLUGIN_REF})"
    else
        echo "UCX spcx plugin: Disabled"
    fi
    echo "Build Type: ${BUILD_TYPE}"
}

show_help() {
    echo "usage: build-container.sh"
    echo "  [--base-image base image for non-manylinux builds]"
    echo "  [--base-image-tag base image tag for non-manylinux builds]"
    echo "  [--cuda-image CUDA source image for manylinux builds (default: ${CUDA_IMAGE})]"
    echo "  [--cuda-image-tag CUDA source image tag for manylinux builds (default: ${CUDA_IMAGE_TAG})]"
    echo "  [--cuda-version CUDA toolkit major.minor copied into the manylinux image]"
    echo "  [--manylinux-image PyPA manylinux image prefix (default: ${MANYLINUX_IMAGE})]"
    echo "  [--manylinux-image-tag pinned PyPA manylinux image tag (default: ${MANYLINUX_IMAGE_TAG})]"
    echo "  [--wheel-base base platform for wheel builds]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--os [ubuntu24|ubuntu22] to select Ubuntu version]"
    echo "  [--build-type [debug|release] to select build type (default: release)]"
    echo "  [--tag tag for image]"
    echo "  [--python-versions python versions to build for, comma separated]"
    echo "  [--ucx-repo ucx git repository URL]"
    echo "  [--ucx-ref ucx git reference (branch, tag, or sha)]"
    echo "  [--ucx-soname-suffix suffix to pass to UCX --with-soname-suffix]"
    echo "  [--private-ucx shortcut for --ucx-soname-suffix ${PRIVATE_UCX_SONAME_SUFFIX}; requires a UCX ref with --with-soname-suffix and --enable-module-deepbind]"
    echo "  [--build-nixl-ep build NIXL with NIXL EP support (requires UCX >= 1.21)]"
    echo "  [--build-ucx-spcx-plugin build and bundle the UCX spcx external plugin (requires NIXL_GITLAB_TOKEN and NIXL_SPCX_PLUGIN_REPO_URL in the environment) (requires --dockerfile contrib/Dockerfile.manylinux)]"
    echo "  [--ucx-spcx-plugin-ref git ref of ucx-spcx-plugin to build (default: ${UCX_SPCX_PLUGIN_REF})]"
    echo "  [--arch [x86_64|aarch64] to select target architecture]"
    echo "  [--dockerfile path to a dockerfile to use]"
    echo "  [--torch-versions torch versions to build for, comma separated (default: uses Dockerfile ARG default)]"
    echo "  [--wheel-base-image pre-built wheel base image URL; skips wheel_base stage and builds only the wheel stage]"
    echo "  [--build-infinia build and bundle the Infinia DDN plugin (requires --dockerfile contrib/Dockerfile.manylinux; harbor.mellanox.com must be reachable)]"
    echo "  [--infinia-image full image reference for infinia-libs (default: ${INFINIA_LIBS_IMAGE})]"
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

if [ -d "$NIXL_DIR/build" ]; then
    echo "Please delete the build directory before creating container"
    exit 1
fi

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG"
BUILD_ARGS+=" --build-arg CUDA_IMAGE=$CUDA_IMAGE --build-arg CUDA_IMAGE_TAG=$CUDA_IMAGE_TAG"
BUILD_ARGS+=" --build-arg CUDA_VERSION=$CUDA_VERSION"
BUILD_ARGS+=" --build-arg MANYLINUX_IMAGE=$MANYLINUX_IMAGE --build-arg MANYLINUX_IMAGE_TAG=$MANYLINUX_IMAGE_TAG"
BUILD_ARGS+=" --build-arg WHL_PYTHON_VERSIONS=$WHL_PYTHON_VERSIONS"
BUILD_ARGS+="${WHL_TORCH_VERSIONS:+ --build-arg WHL_TORCH_VERSIONS=$WHL_TORCH_VERSIONS}"
BUILD_ARGS+=" --build-arg WHL_PLATFORM=$WHL_PLATFORM"
BUILD_ARGS+=" --build-arg ARCH=$ARCH"
BUILD_ARGS+=" --build-arg UCX_REPO=$UCX_REPO"
BUILD_ARGS+=" --build-arg UCX_REF=$UCX_REF"
BUILD_ARGS+=" --build-arg UCX_SONAME_SUFFIX=$UCX_SONAME_SUFFIX"
BUILD_ARGS+=" --build-arg BUILD_NIXL_EP=$BUILD_NIXL_EP"
BUILD_ARGS+=" --build-arg NPROC=$NPROC"
BUILD_ARGS+=" --build-arg GRPC_NPROC=$GRPC_NPROC"
BUILD_ARGS+=" --build-arg OS=$OS"
BUILD_ARGS+=" --build-arg BUILD_TYPE=$BUILD_TYPE"
BUILD_ARGS+=" --build-arg BUILD_INFINIA=$BUILD_INFINIA"
BUILD_ARGS+=" --build-arg BUILD_UCX_SPCX_PLUGIN=$BUILD_UCX_SPCX_PLUGIN"

# The plugin source is fetched on the host and placed inside the build
# context (ucx-spcx-plugin-src/), where the Dockerfile's plugin RUN builds
# it from the copied context. This keeps credentials and BuildKit-specific
# syntax out of the Dockerfile entirely, so the default build works with
# any docker builder. The token stays out of URLs and argv (git credential
# helper reading the environment) so git errors cannot leak it.
SPCX_SRC_DIR="$BUILD_CONTEXT/ucx-spcx-plugin-src"
rm -rf "$SPCX_SRC_DIR"
if [ "$BUILD_UCX_SPCX_PLUGIN" = "true" ]; then
    case "$DOCKER_FILE" in
        *Dockerfile.manylinux) ;;
        *) error "ERROR:" "--build-ucx-spcx-plugin requires --dockerfile contrib/Dockerfile.manylinux (the default Dockerfile does not consume it)" ;;
    esac
    if [ -z "${NIXL_GITLAB_TOKEN:-}" ]; then
        error "ERROR:" "--build-ucx-spcx-plugin requires the NIXL_GITLAB_TOKEN environment variable"
    fi
    if [ -z "${NIXL_SPCX_PLUGIN_REPO_URL:-}" ]; then
        error "ERROR:" "--build-ucx-spcx-plugin requires the NIXL_SPCX_PLUGIN_REPO_URL environment variable"
    fi
    trap 'rm -rf "$SPCX_SRC_DIR"' EXIT
    mkdir -p "$SPCX_SRC_DIR"
    (
        set -e
        cd "$SPCX_SRC_DIR"
        git init -q
        git remote add origin "$NIXL_SPCX_PLUGIN_REPO_URL"
        # shellcheck disable=SC2016
        GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
            -c credential.helper='!f() { echo "username=oauth2"; echo "password=${NIXL_GITLAB_TOKEN}"; }; f' \
            fetch -q --depth 1 origin "$UCX_SPCX_PLUGIN_REF"
        git checkout -q FETCH_HEAD
        echo "ucx-spcx-plugin ref ${UCX_SPCX_PLUGIN_REF} -> commit $(git rev-parse HEAD)"
        rm -rf .git
    ) || error "ERROR:" "failed to fetch ucx-spcx-plugin at ref ${UCX_SPCX_PLUGIN_REF}"
fi

if [ -n "$WHEEL_BASE_IMAGE" ]; then
    BUILD_ARGS+=" --build-arg wheel_base=$WHEEL_BASE_IMAGE"
    # Not named BUILD_TARGET: Jenkins exports a BUILD_TARGET job parameter
    # (nixl/nixlbench) that would leak into the docker command line.
    DOCKER_BUILD_TARGET="--target wheel"
fi

# Infinia DDN libs: pre-pulled from harbor on the host and placed flat into
# infinia-libs/ in the build context. The Dockerfile's BUILD_INFINIA RUN block
# copies them to /opt/ddn/red/ — no harbor reference in the Dockerfile. The whole
# block is guarded so external docker build runs are unaffected when
# BUILD_INFINIA=false (default): no filesystem writes and no EXIT trap installed.
if [ "$BUILD_INFINIA" = "true" ]; then
    case "$DOCKER_FILE" in
        *Dockerfile.manylinux) ;;
        *) error "ERROR:" "--build-infinia requires --dockerfile contrib/Dockerfile.manylinux" ;;
    esac
    INFINIA_LIBS_DIR="$BUILD_CONTEXT/infinia-libs"
    rm -rf "$INFINIA_LIBS_DIR"
    mkdir -p "$INFINIA_LIBS_DIR"
    trap 'rm -rf "$INFINIA_LIBS_DIR"' EXIT
    (
        set -e
        echo "Pulling Infinia libs image: ${INFINIA_LIBS_IMAGE}"
        docker pull "$INFINIA_LIBS_IMAGE"
        cid=$(docker create "$INFINIA_LIBS_IMAGE")
        trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
        docker cp "$cid:/infinia/$ARCH/." "$INFINIA_LIBS_DIR/"
        echo "Infinia libs staged from ${INFINIA_LIBS_IMAGE} (arch: ${ARCH})"
    ) || error "ERROR:" "failed to stage Infinia libs from ${INFINIA_LIBS_IMAGE}"
    # Fail fast on an empty extraction (wrong $ARCH, misconfigured image): otherwise
    # the wheel would silently build without libplugin_INFINIA.so.
    [ -n "$(ls -A "$INFINIA_LIBS_DIR")" ] || \
        error "ERROR:" "no Infinia libs found for arch $ARCH in ${INFINIA_LIBS_IMAGE}"
fi

show_build_options

docker build --platform "$TARGET_PLATFORM" -f $DOCKER_FILE $BUILD_ARGS $TAG $NO_CACHE ${DOCKER_BUILD_TARGET:-} $BUILD_CONTEXT
