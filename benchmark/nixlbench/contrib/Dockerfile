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

# --- Stage 1: Common OS setup ---
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS os_setup_stage

# Re-declare for use in this stage
ARG ARCH="x86_64"
ARG DEFAULT_PYTHON_VERSION
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ninja-build \
    pybind11-dev \
    libclang-dev \
    cmake \
    libgflags-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    libaio-dev \
    liburing-dev \
    protobuf-compiler-grpc \
    libcpprest-dev \
    etcd-server \
    etcd-client \
    autotools-dev \
    automake \
    libtool \
    libz-dev \
    flex \
    libgtest-dev \
    build-essential

# Add Mellanox repository and install packages
RUN ARCH_SUFFIX=$(if [ "${ARCH}" = "aarch64" ]; then echo "arm64-sbsa"; else echo "${ARCH}"; fi) && \
    export PKG_CONFIG_PATH="/opt/mellanox/doca/lib/${ARCH_SUFFIX}-linux-gnu/pkgconfig:/opt/mellanox/dpdk/lib/${ARCH_SUFFIX}-linux-gnu/pkgconfig:$PKG_CONFIG_PATH" && \
    curl -fsSL https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu24.04/${ARCH_SUFFIX}/GPG-KEY-Mellanox.pub | \
    gpg --dearmor | tee /usr/share/keyrings/mellanox-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/mellanox-archive-keyring.gpg] https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu24.04/${ARCH_SUFFIX} ./" | \
    tee /etc/apt/sources.list.d/mellanox.list && \
    DEBIAN_FRONTEND=noninteractive apt update -y && \
    apt install -y --no-install-recommends \
        mlnx-dpdk mlnx-dpdk-dev \
        doca-sdk-common doca-sdk-dma doca-sdk-dpdk-bridge \
        doca-sdk-eth doca-sdk-flow doca-sdk-rdma doca-all \
        doca-sdk-gpunetio libdoca-sdk-gpunetio-dev

# --- Stage 2a: Represents using UCX from the base image ---
FROM os_setup_stage AS ucx_upstream_image
RUN echo "INFO: Using UCX from base image (UCX=${UCX})."

# --- Stage 2b: Represents building UCX from source ---
FROM os_setup_stage AS ucx_custom_image
RUN mkdir -p /workspace/ucx
COPY --from=ucx . /workspace/ucx

RUN echo "INFO: Starting custom UCX build..." && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        --reinstall autoconf automake libtool pkg-config make g++ \
        libnuma-dev librdmacm-dev ibverbs-providers libibverbs-dev rdma-core \
        ibverbs-utils libibumad-dev && \
    echo "INFO: Removing pre-existing UCX installations..." && \
    rm -rf /usr/local/ucx /opt/hpcx/ucx && \
    cd /workspace/ucx && \
    ./autogen.sh && \
    echo "INFO: Building UCX..." && \
    ./contrib/configure-release --prefix=/usr/local/ucx \
                                --with-cuda=/usr/local/cuda \
                                --enable-mt \
                                --without-go && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    echo "INFO: Finished building and installing UCX to /usr/local/ucx."

# --- Stage 3: UCX Image Selection ---
# This stage selects the correct UCX image based on the UCX argument
FROM ucx_${UCX}_image AS ucx_image

# --- Stage 4: Final Image Assembly ---
# Re-declare ARGs needed in this final stage
ARG ARCH="x86_64"
ARG DEFAULT_PYTHON_VERSION
ARG WHL_PYTHON_VERSIONS="3.12"
ARG WHL_PLATFORM="manylinux_2_39_$ARCH"

WORKDIR /workspace
RUN git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git &&\
	cd etcd-cpp-apiv3 && mkdir build && cd build && \
	cmake .. && make -j$(nproc) && make install

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=nixl . /workspace/nixl
COPY --from=nixlbench . /workspace/nixlbench

WORKDIR /workspace/nixl

# This is only to find libcuda.so.1 correctly for build, removed once done
RUN ln -sf "/usr/local/cuda/compat/lib.real" "/usr/local/cuda/compat/lib"
ARG OLD_LD_PATH=${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/lib:$LD_LIBRARY_PATH

ENV VIRTUAL_ENV=/workspace/nixl/.venv
RUN uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION && \
    # pybind11 pip install needed for ubuntu 22.04
    uv pip install --upgrade meson pybind11 patchelf pyYAML

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
RUN IFS=',' read -ra PYTHON_VERSIONS <<< "$WHL_PYTHON_VERSIONS" && \
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do \
        uv build --wheel --out-dir /tmp/dist --python $PYTHON_VERSION; \
    done
RUN uv pip install auditwheel && \
    uv run auditwheel repair /tmp/dist/nixl-*cp31*.whl --plat $WHL_PLATFORM --wheel-dir /workspace/nixl/dist

RUN uv pip install dist/nixl-*cp${DEFAULT_PYTHON_VERSION//./}*.whl

# Remove the cuda setup
RUN rm -f /usr/local/cuda/compat/lib
ENV LD_LIBRARY_PATH=$OLD_LD_PATH

WORKDIR /workspace/nixlbench

RUN ls /usr/local/lib
RUN echo $LD_LIBRARY_PATH
RUN ldconfig

RUN ls -ll /workspace/nixlbench
RUN rm -rf build && \
    mkdir build && \
    uv run meson setup build -Dnixl_path=/usr/local/nixl/ && \
    cd build && ninja
ENV PATH=/workspace/nixlbench/build:$PATH
WORKDIR /workspace/nixl/benchmark/kvbench
