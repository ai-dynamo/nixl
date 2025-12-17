#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exit on error and print commands
set -e
set -x

CUDA_VERSIONS="12.8.1"
PYTHON_VERSIONS="3.10,3.11,3.12,3.13,3.14"

arch=$(uname -m)
[ "$arch" = "arm64" ] && arch="aarch64"

# Remove any existing dist and wheels directories
rm -rf dist/*
rm -rf wheels/*
mkdir -p wheels

# Remove any existing container
docker rm temp-nixl || true

for cuda_version in ${CUDA_VERSIONS}
do
    tag="nixl-wheels-${cuda_version}"
    ./contrib/build-container.sh \
        --base-image 'nvcr.io/nvidia/cuda' \
        --base-image-tag "${cuda_version}-devel-rockylinux8" \
        --wheel-base manylinux_2_28 \
        --python-versions "${PYTHON_VERSIONS}" \
        --tag $tag \
        --arch $arch \
        --dockerfile contrib/Dockerfile.rockylinux8
    docker create --name temp-nixl $tag
    docker cp temp-nixl:/workspace/nixl/dist/ wheels/
    # Move all .whl files from wheels/dist subdirectories at any depth to wheels/
    find wheels/dist -type f -name '*.whl' -exec mv {} wheels/ \;
    rm -rf wheels/dist
    docker rm temp-nixl
done
