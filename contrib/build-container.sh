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
ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"
WHL_BASE=manylinux_2_39
WHL_PLATFORM=${WHL_BASE}_${ARCH}
WHL_PYTHON_VERSIONS="3.12"
UCX_REF=${UCX_REF:-v1.21.x}
BUILD_NIXL_EP="true"
OS="ubuntu24"
NPROC=${NPROC:-$(nproc)}
GRPC_NPROC=${GRPC_NPROC:-$(nproc)}
BUILD_TYPE="release"
# CUDA toolkit version (e.g. 12.9 / 13.0). Option B (manylinux on public PyPA base)
# no longer inherits this from the base image's ENV, so it's passed in. Defaults to
# 13.0 to match the default BASE_IMAGE_TAG (cuda13.0); override with --cuda-version.
CUDA_VERSION=${CUDA_VERSION:-13.0}
# INFINIA build variant passed to Dockerfile.manylinux: "bundled" pulls the private DDN libs
# image; "none" (via --no-infinia) skips it so external builds without access still work.
INFINIA_VARIANT=${INFINIA_VARIANT:-bundled}

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
        --wheel-base)
            if [ "$2" ]; then
                WHL_BASE=$2
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
        --cuda-version)
            if [ "$2" ]; then
                CUDA_VERSION=$2
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
        --ucx-ref)
            if [ "$2" ]; then
                UCX_REF=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --build-nixl-ep)
            BUILD_NIXL_EP=true
            ;;
        --no-infinia)
            INFINIA_VARIANT=none
            ;;
        --arch)
            if [ "$2" ]; then
                ARCH=$2
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

    WHL_PLATFORM=${WHL_BASE}_${ARCH}

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
    if [ "$BUILD_NIXL_EP" = "true" ]; then
        echo "NIXL EP: Enabled"
    else
        echo "NIXL EP: Disabled"
    fi
    echo "Build Type: ${BUILD_TYPE}"
}

show_help() {
    echo "usage: build-container.sh"
    echo "  [--base base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--cuda-version CUDA version, e.g. 12.9 (manylinux builds)]"
    echo "  [--wheel-base base platform for wheel builds]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--os [ubuntu24|ubuntu22] to select Ubuntu version]"
    echo "  [--build-type [debug|release] to select build type (default: release)]"
    echo "  [--tag tag for image]"
    echo "  [--python-versions python versions to build for, comma separated]"
    echo "  [--ucx-ref ucx git reference (branch, tag, or sha)]"
    echo "  [--build-nixl-ep build NIXL with NIXL EP support (requires UCX >= 1.21)]"
    echo "  [--no-infinia skip the INFINIA plugin (manylinux: don't pull the private DDN libs image)]"
    echo "  [--arch [x86_64|aarch64] to select target architecture]"
    echo "  [--dockerfile path to a dockerfile to use]"
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

# The manylinux build path requires CUDA_VERSION in MAJOR.MINOR form (e.g. 12.9, 13.0):
# Dockerfile.manylinux derives the torch index (cu<major><minor>) and the cu12/cu13
# meta-wheel split from it. Empty yields a broken index (https://download.pytorch.org/
# whl/cu) and a skipped meta-wheel; a major-only value like "12" yields a nonexistent
# "cu12" index instead of e.g. "cu129". Validate the shape (not just presence) up front.
case "$DOCKER_FILE" in
    *manylinux*)
        if ! echo "$CUDA_VERSION" | grep -qE '^[0-9]+\.[0-9]+$'; then
            echo "ERROR: --cuda-version must be MAJOR.MINOR for manylinux builds (e.g. --cuda-version 12.9); got '${CUDA_VERSION}'" >&2
            exit 1
        fi
        ;;
esac

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG"
BUILD_ARGS+=" --build-arg CUDA_VERSION=$CUDA_VERSION"
BUILD_ARGS+=" --build-arg WHL_PYTHON_VERSIONS=$WHL_PYTHON_VERSIONS"
BUILD_ARGS+=" --build-arg WHL_PLATFORM=$WHL_PLATFORM"
BUILD_ARGS+=" --build-arg ARCH=$ARCH"
BUILD_ARGS+=" --build-arg UCX_REF=$UCX_REF"
BUILD_ARGS+=" --build-arg BUILD_NIXL_EP=$BUILD_NIXL_EP"
BUILD_ARGS+=" --build-arg NPROC=$NPROC"
BUILD_ARGS+=" --build-arg GRPC_NPROC=$GRPC_NPROC"
BUILD_ARGS+=" --build-arg OS=$OS"
BUILD_ARGS+=" --build-arg BUILD_TYPE=$BUILD_TYPE"
BUILD_ARGS+=" --build-arg INFINIA_VARIANT=$INFINIA_VARIANT"

show_build_options

# Disable buildx's default provenance/sbom attestations (the attestation-manifest export
# was tipping the ARM build over the job timeout; CI images don't need them). Use the env
# var, which buildx honors and the classic docker build frontend ignores -- unlike the
# --provenance/--sbom flags, which classic docker rejects as unknown.
export BUILDX_NO_DEFAULT_ATTESTATIONS=1
docker build --platform linux/$ARCH -f $DOCKER_FILE $BUILD_ARGS $TAG $NO_CACHE $BUILD_CONTEXT
