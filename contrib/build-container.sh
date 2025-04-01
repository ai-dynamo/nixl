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
FULLCOMMIT=$(git rev-parse HEAD)

# Get latest TAG and add COMMIT_ID for dev
latest_tag=$(git describe --tags --abbrev=0 $(git rev-list --tags --max-count=1 main) | sed 's/^v//') || true
if [[ -z ${latest_tag} ]]; then
    latest_tag="0.0.1"
    echo "No git release tag found, setting to unknown version: ${latest_tag}"
fi

BASE_IMAGE=nvcr.io/nvidia/pytorch
BASE_IMAGE_TAG=25.02-py3
VERSION=v$latest_tag.dev.$commit_id
UBUNTUOS="24.04"
USE_LOCAL_DIR=0
NIXL_DIR=/tmp/nixl

NIXL_COMMIT=ec6345c5279142a3805ab1a0e876f954d079dbf7
NIXL_REPO=ai-dynamo/nixl.git

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
        --base-imge-tag)
            if [ "$2" ]; then
                BASE_IMAGE_TAG=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
	--os)
            if [ "$2" ]; then
                UBUNTUOS=$2
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
	--use-local)
	    USE_LOCAL_DIR=1
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

    if [[ $UBUNTUOS == "22.04" ]]; then
	BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base
	BASE_IMAGE_TAG=24.10-cuda12.6-devel-ubuntu22.04
    fi

    if [ -z "$TAG" ]; then
        TAG="--tag nixl:${VERSION}"
    fi

    if [ $USE_LOCAL_DIR -eq 1 ]; then
	NIXL_DIR=$BUILD_CONTEXT
	NIXL_COMMIT=$FULLCOMMIT
    fi
}

show_build_options() {
    echo ""
    echo "Building NIXL Image: '${TAG}' for Ubuntu${UBUNTUOS}"
    echo "Using local nixl source: ${NIXL_DIR}"
    echo "    Build Context: ${BUILD_CONTEXT}"
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base base image]"
    echo "  [--base-imge-tag base image tag]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--os [24.04|22.04] to select Ubuntu version]"
    echo "  [--tag tag for image]"
    echo "  [--use-local copy current source dir to container]"
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

show_build_options

if [ $USE_LOCAL_DIR -eq 0 ]; then
    if [ -d "$NIXL_DIR" ]; then
	echo "Warning: $NIXL_DIR exists, skipping clone"
    else
	git clone https://github.com/${NIXL_REPO} ${NIXL_DIR}
    fi

    if ! git checkout ${NIXL_COMMIT}; then
	echo "ERROR: Failed to checkout commit ${NIXL_COMMIT}."
	echo "Please delete $NIXL_DIR and retry."
	exit 1
    fi
else
    if [ -d "$NIXL_DIR/build" ]; then
	echo "Please delete the build directory before creating container"
	exit 1
    fi
fi

BUILD_CONTEXT_ARGS+=" --build-context nixl=$NIXL_DIR"
BUILD_ARGS+=" --build-arg NIXL_COMMIT=${NIXL_COMMIT} --build-arg NIXL_REPO=${NIXL_REPO}"
BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG"
BUILD_ARGS+=" --build-arg UBUNTUOS=$UBUNTUOS"

docker build -f $DOCKER_FILE $BUILD_ARGS $TAG $NO_CACHE $BUILD_CONTEXT_ARGS $BUILD_CONTEXT
