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

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")
BUILD_CONTEXT_ARGS=""
NIXL_SRC=$(readlink -f "${SOURCE_DIR}/../../..")
NIXL_BUILD_CONTEXT_ARGS="--build-context nixl=$NIXL_SRC"
NIXL_BENCH_BUILD_CONTEXT_ARGS="--build-context nixlbench=$BUILD_CONTEXT/"
DOCKER_FILE="${SOURCE_DIR}/Dockerfile"
UCX_SRC=""
UCX_BUILD_CONTEXT_ARGS=""
commit_id=$(git rev-parse --short HEAD)

# Get latest TAG and add COMMIT_ID for dev
latest_tag=$(git describe --tags --abbrev=0 $(git rev-list --tags --max-count=1 main) | sed 's/^v//') || true
if [ -z ${latest_tag} ]; then
    latest_tag="0.0.1"
    echo "No git release tag found, setting to unknown version: ${latest_tag}"
fi

BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base
BASE_IMAGE_TAG=25.03-cuda12.8-devel-ubuntu24.04
ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"
WHL_BASE=manylinux_2_39
WHL_PLATFORM=${WHL_BASE}_${ARCH}
WHL_PYTHON_VERSIONS="3.12"
OS="ubuntu24"
NPROC=${NPROC:-$(nproc)}

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

    BUILD_CONTEXT_ARGS="$NIXL_BUILD_CONTEXT_ARGS $NIXL_BENCH_BUILD_CONTEXT_ARGS $UCX_BUILD_CONTEXT_ARGS"

    VERSION=v$latest_tag.dev.$commit_id
    if [ -z "$TAG" ]; then
        TAG="--tag nixlbench:${VERSION}"
        echo $TAG
    fi
}

show_build_options() {
    echo ""
    echo "Building NIXLBench Image"
    echo "NIXL Source: ${NIXL_SRC}"
    echo "UCX Source: ${UCX_SRC} (optional)"
    echo "Image Tag: ${TAG}"
    echo "Build Context: ${BUILD_CONTEXT}"
    echo "Build Context Args: ${BUILD_CONTEXT_ARGS}"
    echo "Base Image: ${BASE_IMAGE}:${BASE_IMAGE_TAG}"
    echo "Container arch: ${ARCH}"
    echo "Python Versions for wheel build: ${WHL_PYTHON_VERSIONS}"
    echo "Wheel Platform: ${WHL_PLATFORM}"
}

show_help() {
    echo "usage: build.sh --nixl <path to nixl source>"
    echo "  [--base base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--nixlbench path/to/nixlbench/source/dir]"
    echo "  [--ucx path/to/ucx/source/dir]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--os [ubuntu24|ubuntu22] to select Ubuntu version]"
    echo "  [--python-versions python versions to build for, comma separated]"
    echo "  [--tag tag for image]"
    echo "  [--arch [x86_64|aarch64] to select target architecture]"
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

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG"
BUILD_ARGS+=" --build-arg WHL_PYTHON_VERSIONS=$WHL_PYTHON_VERSIONS"
BUILD_ARGS+=" --build-arg WHL_PLATFORM=$WHL_PLATFORM"
BUILD_ARGS+=" --build-arg ARCH=$ARCH"
BUILD_ARGS+=" --build-arg NPROC=$NPROC"

show_build_options

docker build --platform linux/$ARCH -f $DOCKER_FILE $BUILD_ARGS $TAG $NO_CACHE $BUILD_CONTEXT_ARGS $BUILD_CONTEXT --progress plain
