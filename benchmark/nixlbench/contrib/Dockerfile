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

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
ARG BASE_IMAGE_TAG="25.03-cuda12.8-devel-ubuntu24.04"

# UCX argument is either "upstream" (default installed in base image) or "custom" (build from source)
ARG UCX="upstream"
ARG DEFAULT_PYTHON_VERSION="3.12"

# --- Stage 1: Use common base image ---
FROM nixl-base:latest AS base_stage

# Re-declare for use in this stage
ARG ARCH="x86_64"
ARG DEFAULT_PYTHON_VERSION
ARG UCX
ARG NPROC

# --- Stage 2a: Represents using UCX from the base image ---
FROM base_stage AS ucx_upstream_image
RUN echo "INFO: Using UCX from base image (UCX=${UCX})."

# --- Stage 2b: Represents building UCX from source ---
FROM base_stage AS ucx_custom_image
RUN mkdir -p /workspace/ucx
COPY --from=ucx . /workspace/ucx

RUN echo "INFO: Starting custom UCX build..." && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        --reinstall autoconf automake libtool pkg-config make g++ \
        libnuma-dev librdmacm-dev ibverbs-providers libibverbs-dev rdma-core \
        ibverbs-utils libibumad-dev && \
    echo "INFO: Removing pre-existing UCX installations..." && \
    rm -rf /usr/lib/ucx /opt/hpcx/ucx && \
    cd /workspace/ucx && \
    ./autogen.sh && \
    echo "INFO: Building UCX..." && \
    ./contrib/configure-release --with-cuda=/usr/local/cuda \
                                --enable-mt \
                                --without-go && \
    make -j${NPROC:-$(nproc)} && \
    make install && \
    cd / && \
    echo "INFO: Finished building and installing UCX."

# --- Stage 3: UCX Image Selection ---
# This stage selects the correct UCX image based on the UCX argument
FROM ucx_${UCX}_image AS ucx_image

# --- Stage 4: Final Image Assembly ---
# Re-declare ARGs needed in this final stage
ARG ARCH="x86_64"
ARG DEFAULT_PYTHON_VERSION
ARG WHL_PYTHON_VERSIONS="3.12"
ARG WHL_PLATFORM="manylinux_2_39_$ARCH"
ARG NPROC

COPY --from=nixl . /workspace/nixl
COPY --from=nixlbench . /workspace/nixlbench

WORKDIR /workspace/nixl

ENV VIRTUAL_ENV=/workspace/nixl/.venv
RUN uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION && \
    # pybind11 pip install needed for ubuntu 22.04
    uv pip install --upgrade meson pybind11 patchelf pyYAML click tabulate

RUN rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=/usr/local/nixl && \
    cd build && \
    ninja && \
    ninja install

ENV NIXL_PLUGIN_DIR=/usr/local/nixl/lib/$ARCH-linux-gnu/plugins
RUN echo "/usr/local/nixl/lib/$ARCH-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
    echo "/usr/local/nixl/lib/$ARCH-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

# Create the wheel
# No need to specifically add path to libcuda.so here, meson finds the stubs and links them
RUN IFS=',' read -ra PYTHON_VERSIONS <<< "$WHL_PYTHON_VERSIONS" && \
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do \
        uv build --wheel --out-dir /tmp/dist --python $PYTHON_VERSION; \
    done

# Exclude libcuda.so.1 due to compatibility issues, should link with cuda driver library on host
RUN uv pip install auditwheel && \
    uv run auditwheel repair --exclude libcuda.so.1 /tmp/dist/nixl-*cp31*.whl --plat $WHL_PLATFORM --wheel-dir /workspace/nixl/dist

RUN uv pip install dist/nixl-*cp${DEFAULT_PYTHON_VERSION//./}*.whl

WORKDIR /workspace/nixlbench

RUN ls /usr/local/lib
RUN echo $LD_LIBRARY_PATH
RUN ldconfig

RUN ls -ll /workspace/nixlbench

RUN rm -rf build && \
    mkdir build && \
    uv run meson setup build -Dnixl_path=/usr/local/nixl/ -Dprefix=/usr/local/nixlbench && \
    cd build && ninja && ninja install

WORKDIR /workspace/nixl

RUN ls -ll benchmark/kvbench

# Install dependencies for benchmarks
RUN uv pip install -e benchmark/kvbench

ENV PATH=/usr/local/nixlbench/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/nixlbench/lib:$LD_LIBRARY_PATH
ENV PYTHON_PATH=/usr/local/nixlbench/lib/python3/dist-packages/nixlbench/

# Fix for etcd proto dependencies
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
WORKDIR /workspace/nixl/benchmark/kvbench
