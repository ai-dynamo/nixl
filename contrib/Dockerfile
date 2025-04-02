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
ARG BASE_IMAGE_TAG="24.10-cuda12.6-devel-ubuntu22.04"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

ARG MOFED_VERSION=24.10-1.1.4.0
ARG UBUNTUOS=22.04
ARG DEFAULT_PYTHON_VERSION="3.12"

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
    autotools-dev \
    automake \
    libtool \
    libz-dev \
    libiberty-dev \
    flex \
    cmake \
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libboost-all-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    etcd-server \
    net-tools \
    pciutils \
    libpci-dev \
    tmux \
    screen \
    libibmad-dev \
    googletest \
    linux-tools-common \
    linux-tools-generic \
    ethtool \
    iproute2 \
    dkms \
    linux-headers-generic \
    ninja-build \
    uuid-dev \
    gdb \
    python3-pip \
    pybind11-dev

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

RUN git clone https://github.com/NVIDIA/gdrcopy.git
RUN PREFIX=/usr/local DESTLIB=/usr/local/lib make -C gdrcopy lib_install
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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace/nixl
ENV VIRTUAL_ENV=/workspace/nixl/.venv

RUN uv venv $VIRTUAL_ENV --python 3.12 && \
    # pybind11 pip install needed for ubuntu 22.04
    uv pip install --upgrade meson pybind11 patchelf

COPY . /workspace/nixl

RUN cd /workspace/nixl && \
    mkdir build && \
    uv run meson setup build/ --prefix=/usr/local/nixl && \
    cd build && \
    ninja && \
    ninja install

RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl.conf
RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf
ENV NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins
RUN ldconfig

# Create the wheel
ARG WHL_PYTHON_VERSIONS="3.12"
ARG WHL_PLATFORM="manylinux_2_34_x86_64"
RUN IFS=',' read -ra PYTHON_VERSIONS <<< "$WHL_PYTHON_VERSIONS" && \
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do \
        uv build --wheel --out-dir /tmp/dist --python $PYTHON_VERSION; \
    done
RUN uv pip install auditwheel && \
    uv run auditwheel repair /tmp/dist/nixl-*cp31*.whl --plat $WHL_PLATFORM --wheel-dir /workspace/nixl/dist
