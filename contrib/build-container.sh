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
BASE_IMAGE_TAG=25.03-cuda12.8-devel-ubuntu24.04
ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"
WHL_BASE=manylinux_2_39
WHL_PLATFORM=${WHL_BASE}_${ARCH}
WHL_PYTHON_VERSIONS="3.12"
UCX_REF=v1.19.x
OS="ubuntu24"

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
        --ucx-upstream)
            # Master branch (v1.20) also containing EFA SRD support
            UCX_REF=7ec95b95e524a87e81cac92f5ca8523e3966b16b
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

    if [[ $OS == "ubuntu22" ]]; then
        BASE_IMAGE_TAG=24.10-cuda12.6-devel-ubuntu22.04
        WHL_PLATFORM=manylinux_2_34_${ARCH}
    fi

    if [ -z "$TAG" ]; then
        TAG="--tag nixl:${VERSION}"
    fi
}

show_build_options() {
    echo ""
    echo "Building NIXL Image"
    echo "Image Tag: ${TAG}"
    echo "Build Context: ${BUILD_CONTEXT}"
    echo "Base Image: ${BASE_IMAGE}:${BASE_IMAGE_TAG}"
    echo "Container arch: ${ARCH}"
    echo "Python Versions for wheel build: ${WHL_PYTHON_VERSIONS}"
    echo "Wheel Platform: ${WHL_PLATFORM}"
    echo "UCX Ref: ${UCX_REF}"
}

show_help() {
    echo "usage: build-container.sh"
    echo "  [--base base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--os [ubuntu24|ubuntu22] to select Ubuntu version]"
    echo "  [--tag tag for image]"
    echo "  [--python-versions python versions to build for, comma separated]"
    echo "  [--ucx-upstream use ucx master branch]"
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

if [ -d "$NIXL_DIR/build" ]; then
    echo "Please delete the build directory before creating container"
    exit 1
fi

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG"
BUILD_ARGS+=" --build-arg WHL_PYTHON_VERSIONS=$WHL_PYTHON_VERSIONS"
BUILD_ARGS+=" --build-arg WHL_PLATFORM=$WHL_PLATFORM"
BUILD_ARGS+=" --build-arg ARCH=$ARCH"
BUILD_ARGS+=" --build-arg UCX_REF=$UCX_REF"

show_build_options

docker build --platform linux/$ARCH -f $DOCKER_FILE $BUILD_ARGS $TAG $NO_CACHE $BUILD_CONTEXT
