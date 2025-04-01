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

ARG BASE_IMAGE="nvcr.io/nvidia/pytorch"
ARG BASE_IMAGE_TAG="25.02-py3"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

ARG MOFED_VERSION=24.10-1.1.4.0
ARG UBUNTUOS=24.04

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb

ARG NIXL_COMMIT
ARG NIXL_REPO

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install curl \
    git \
    libnuma-dev \
    numactl \
    wget \
    autotools-dev \
    automake \
    libtool \
    libz-dev \
    libiberty-dev \
    flex \
    build-essential \
    cmake \
    libibverbs-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libpython3-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    pybind11-dev \
    python3-full \
    python3-pip \
    python3-numpy \
    etcd-server \
    net-tools \
    pciutils \
    libpci-dev \
    vim \
    tmux \
    screen \
    ibverbs-utils \
    libibmad-dev \
    googletest

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y linux-tools-common linux-tools-generic ethtool iproute2
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y dkms linux-headers-generic
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y meson ninja-build uuid-dev gdb

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install -y wget libglib2.0-0
RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

RUN cd /usr/local/src && \
    curl -fSsL "https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu${UBUNTUOS}-x86_64.tgz" -o mofed.tgz && \
    tar -xf /usr/local/src/mofed.tgz && \
    cd MLNX_OFED_LINUX-* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ./DEBS/libibverbs* ./DEBS/ibverbs-providers* ./DEBS/librdmacm* ./DEBS/libibumad* && \
    rm -rf /var/lib/apt/lists/* /usr/local/src/*

ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs

ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/gdrcopy.git
RUN PREFIX=/usr/local DESTLIB=/usr/local/lib make -C /workspace/gdrcopy lib_install
RUN cp gdrcopy/src/libgdrapi.so.2.* /usr/lib/x86_64-linux-gnu/
RUN ldconfig

ARG UCX_VERSION=v1.18.0

RUN cd /usr/local/src && \
    curl -fSsL "https://github.com/openucx/ucx/tarball/${UCX_VERSION}" | tar xz && \
    cd openucx-ucx* && \
    ./autogen.sh && ./configure     \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-cuda=/usr/local/cuda \
        --with-verbs                \
        --with-dm                   \
        --with-gdrcopy=/usr/local   \
        --enable-mt &&              \
    make -j &&                      \
    make -j install-strip &&        \
    ldconfig

ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
ENV CPATH=/usr/include:$CPATH
ENV PATH=/usr/bin:$PATH
ENV PKG_CONFIG_PATH=/usr/lib/pkgconfig:$PKG_CONFIG_PATH
SHELL ["/bin/bash", "-c"]

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/nixl/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
ENV PYTHONPATH=/usr/local/nixl/lib/python3/dist-packages/:$PYTHONPATH

# Required for Ubuntu 22.04
RUN pip3 install --upgrade meson torch

# Make sure to build the container from the cloned git repository
WORKDIR /opt/nixl
COPY --from=nixl . .
RUN echo "NIXL commit: ${NIXL_COMMIT}" > /opt/nixl/commit.txt
RUN cd /opt/nixl && \
    mkdir build && \
    meson setup build/ --prefix=/usr/local/nixl && \
    cd build/ && \
    ninja && \
    ninja install

RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl.conf
RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf
ENV NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins
RUN ldconfig

# Create the wheel
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN pip3 install patchelf auditwheel
RUN cd /opt/nixl && uv build --wheel --out-dir /workspace/dist
RUN auditwheel repair /workspace/dist/nixl-*cp31*.whl --plat manylinux_2_39_x86_64
