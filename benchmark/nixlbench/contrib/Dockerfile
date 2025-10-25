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
    hwloc \
    libhwloc-dev \
    build-essential

# Add DOCA repository and install packages
RUN ARCH_SUFFIX=$(if [ "${ARCH}" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) && \
    MELLANOX_OS="$(. /etc/lsb-release; echo ${DISTRIB_ID}${DISTRIB_RELEASE} | tr A-Z a-z | tr -d .)" && \
    wget --tries=3 --waitretry=5 --no-verbose https://www.mellanox.com/downloads/DOCA/DOCA_v3.1.0/host/doca-host_3.1.0-091000-25.07-${MELLANOX_OS}_${ARCH_SUFFIX}.deb -O doca-host.deb && \
    dpkg -i doca-host.deb && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends doca-sdk-gpunetio libdoca-sdk-gpunetio-dev libdoca-sdk-verbs-dev

# Force reinstall of RDMA packages from DOCA repository
# Reinstall needed to fix broken libibverbs-dev, which may lead to lack of Infiniband support.
# Upgrade is not sufficient if the version is the same since apt skips the installation.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y install \
    --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
    libnuma-dev librdmacm-dev ibverbs-providers

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

# --- Stage 2a: Represents using UCX from the base image ---
FROM os_setup_stage AS ucx_upstream_image
RUN echo "INFO: Using UCX from base image (UCX=${UCX})."

# --- Stage 2b: Represents building UCX from source ---
FROM os_setup_stage AS ucx_custom_image
ARG BUILD_TYPE="release"
ARG NPROC
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
                                $(if [ "$BUILD_TYPE" = "debug" ]; then echo "--enable-debug"; fi) \
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
ARG BUILD_TYPE="release"
ARG LIBFABRIC_VERSION="v2.3.0"
ARG NPROC

WORKDIR /workspace

# Build libfabric from source
RUN wget --tries=3 --waitretry=5 --timeout=30 --read-timeout=60 \
    "https://github.com/ofiwg/libfabric/releases/download/${LIBFABRIC_VERSION}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2" -O libfabric.tar.bz2 && \
    tar xjf libfabric.tar.bz2 && rm libfabric.tar.bz2 && \
    cd libfabric-* && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local \
                --disable-verbs \
                --disable-psm3 \
                --disable-opx \
                --disable-usnic \
                --disable-rstream \
                --enable-efa \
                --with-cuda=/usr/local/cuda \
                --enable-cuda-dlopen \
                --with-gdrcopy \
                --enable-gdrcopy-dlopen && \
    make -j${NPROC:-$(nproc)} && \
    make install && \
    ldconfig

RUN git clone --depth 1 https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git && \
    cd etcd-cpp-apiv3 && \
    sed -i '/^find_dependency(cpprestsdk)$/d' etcd-cpp-api-config.in.cmake && \
    mkdir build && cd build && \
    cmake .. -DBUILD_ETCD_CORE_ONLY=ON -DCMAKE_BUILD_TYPE=Release && make -j${NPROC:-$(nproc)} && make install

# Install AWS SDK C++ dependencies and build
RUN apt-get update && apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev hwloc libhwloc-dev

RUN git clone --recurse-submodules --depth 1 --shallow-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && \
    mkdir sdk_build && \
    cd sdk_build && \
    cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j && \
    make install

RUN git clone https://github.com/nvidia/gusli.git && \
    cd gusli && \
    make all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0 VERBOSE=1 ALLOW_USE_URING=0 && \
    cd ..

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# By default, uv downloads python packages to $HOME/.cache/uv and hard links them
# from the virtual environment. This means that the files reside in /root/.cache/uv,
# which is not what we want since some systems mount user home dir into /root,
# in which case the venv is broken when the container is started.
# Set a custom cache directory inside /workspace to avoid this.
ENV UV_CACHE_DIR=/workspace/.cache/uv
RUN mkdir -p $UV_CACHE_DIR
# Disable build isolation, i.e. uv should not create a new virtual environment for
# building wheels.
ENV UV_NO_BUILD_ISOLATION=1
# Disable syncing, i.e. uv will not download packages outside uv pip commands.
ENV UV_NO_SYNC=1
# Create a new virtual environment
ENV VIRTUAL_ENV=/workspace/.venv
RUN rm -rf $VIRTUAL_ENV && uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION
# Activate the virtual environment
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install python dependencies
RUN uv pip install --upgrade meson meson-python pybind11 patchelf pyYAML click tabulate auditwheel tomlkit
# Install PyTorch
RUN CUDA_SHORT_VERSION=cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d .) && \
    FLAGS="--index-url https://download.pytorch.org/whl/$CUDA_SHORT_VERSION" && \
    uv pip install $FLAGS torch torchvision torchaudio

COPY --from=nixl . /workspace/nixl
COPY --from=nixlbench . /workspace/nixlbench

WORKDIR /workspace/nixl

RUN rm -rf build && \
    mkdir build && \
    meson setup build --prefix=/usr/local/nixl --buildtype=$BUILD_TYPE && \
    cd build && \
    ninja && \
    ninja install

RUN echo "/usr/local/nixl/lib/$ARCH-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
    echo "/usr/local/nixl/lib/$ARCH-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

RUN uv pip install .

WORKDIR /workspace/nixlbench

RUN ls /usr/local/lib
RUN echo $LD_LIBRARY_PATH
RUN ldconfig

RUN ls -ll /workspace/nixlbench

RUN rm -rf build && \
    mkdir build && \
    meson setup build -Dnixl_path=/usr/local/nixl/ -Dprefix=/usr/local/nixlbench --buildtype=$BUILD_TYPE && \
    cd build && ninja && ninja install

WORKDIR /workspace/nixl

RUN ls -ll benchmark/kvbench

# Install dependencies for benchmarks
RUN uv pip install -e benchmark/kvbench

ENV PATH=/usr/local/nixlbench/bin:/usr/local/nixl/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/nixlbench/lib:$LD_LIBRARY_PATH
ENV PYTHON_PATH=/usr/local/nixlbench/lib/python3/dist-packages/nixlbench/

# Fix for etcd proto dependencies
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
WORKDIR /workspace/nixl/benchmark/kvbench
