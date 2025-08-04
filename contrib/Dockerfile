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
ARG OS

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

# Set default OS if not provided
ARG OS=${OS:-ubuntu24}
ARG ARCH="x86_64"
ARG DEFAULT_PYTHON_VERSION="3.12"
ARG UCX_REF="v1.19.x"
ARG UCX_PREFIX="/usr"
ARG UCX_PLUGIN_DIR="$UCX_PREFIX/lib/ucx"
ARG NIXL_PREFIX="/usr/local/nixl"
ARG NIXL_PLUGIN_DIR="$NIXL_PREFIX/lib/$ARCH-linux-gnu/plugins"
ARG NPROC
ARG WHL_DEFAULT_PYTHON_VERSIONS="3.12"

RUN apt-get update -y && \
    apt-get install -y ubuntu-keyring && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ninja-build \
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
    build-essential \
    python3.12-dev \
    libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev # aws-sdk-cpp dependencies

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
    --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
    libnuma-dev librdmacm-dev ibverbs-providers

WORKDIR /workspace
RUN git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git && \
    cd etcd-cpp-apiv3 && \
    sed -i '/^find_dependency(cpprestsdk)$/d' etcd-cpp-api-config.in.cmake && \
    mkdir build && cd build && \
    cmake .. -DBUILD_ETCD_CORE_ONLY=ON -DCMAKE_BUILD_TYPE=Release && make -j${NPROC:-$(nproc)} && make install

RUN git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && \
    mkdir aws_sdk_build && cd aws_sdk_build && \
    cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j${NPROC:-$(nproc)} && make install

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.86.0 \
    RUSTARCH=${ARCH}-unknown-linux-gnu

# Download rustup-init and its checksum for the target architecture
RUN wget --tries=3 --waitretry=5 \
    "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" \
    "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init.sha256" && \
    sha256sum -c rustup-init.sha256 && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init* && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# Add Mellanox repository and install packages
RUN ARCH_SUFFIX=$(if [ "${ARCH}" = "aarch64" ]; then echo "arm64-sbsa"; else echo "${ARCH}"; fi) && \
    # Map OS names to Mellanox repository format
    MELLANOX_OS=$(if [ "${OS}" = "ubuntu22" ]; then echo "ubuntu22.04"; elif [ "${OS}" = "ubuntu24" ]; then echo "ubuntu24.04"; else echo "${OS}"; fi) && \
    export PKG_CONFIG_PATH="/opt/mellanox/doca/lib/${ARCH_SUFFIX}-linux-gnu/pkgconfig:/opt/mellanox/dpdk/lib/${ARCH_SUFFIX}-linux-gnu/pkgconfig:$PKG_CONFIG_PATH" && \
    curl -fsSL https://linux.mellanox.com/public/repo/doca/3.0.0/${MELLANOX_OS}/${ARCH_SUFFIX}/GPG-KEY-Mellanox.pub | \
    gpg --dearmor | tee /usr/share/keyrings/mellanox-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/mellanox-archive-keyring.gpg] https://linux.mellanox.com/public/repo/doca/3.0.0/${MELLANOX_OS}/${ARCH_SUFFIX} ./" | \
    tee /etc/apt/sources.list.d/mellanox.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        mlnx-dpdk mlnx-dpdk-dev \
        doca-sdk-common doca-sdk-dma doca-sdk-dpdk-bridge \
        doca-sdk-eth doca-sdk-flow doca-sdk-rdma doca-all

RUN if [ "$OS" = "ubuntu24" ]; then \
        apt-get install -y --no-install-recommends \
            doca-sdk-gpunetio libdoca-sdk-gpunetio-dev; \
    fi

WORKDIR /workspace/nixl
COPY . /workspace/nixl

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ENV VIRTUAL_ENV=/workspace/nixl/.venv
RUN rm -rf $VIRTUAL_ENV && uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION && \
    uv pip install --upgrade "meson>=0.64.0" patchelf

# Install pybind11 via apt
RUN apt-get update && apt-get install -y --no-install-recommends pybind11-dev

RUN rm -rf /usr/lib/ucx
RUN rm -rf /opt/hpcx/ucx

RUN cd /usr/local/src && \
     git clone https://github.com/openucx/ucx.git && \
     cd ucx && 			     \
     git checkout $UCX_REF &&        \
     ./autogen.sh && ./configure     \
         --prefix=$UCX_PREFIX        \
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
         --with-efa                  \
         --enable-mt &&              \
     make -j${NPROC:-$(nproc)} &&                \
     make -j${NPROC:-$(nproc)} install-strip &&  \
     ldconfig

ENV NIXL_PREFIX=$NIXL_PREFIX
RUN rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=$NIXL_PREFIX && \
    cd build && \
    ninja && \
    ninja install

RUN echo "$NIXL_PREFIX/lib/$ARCH-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
    echo "$NIXL_PLUGIN_DIR" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

RUN cd src/bindings/rust && cargo build --release --locked

# Build wheel using the build-wheel.sh script for better UCX plugin bundling and library management
RUN ./contrib/build-wheel.sh \
    --python-version $DEFAULT_PYTHON_VERSION \
    --platform manylinux_2_39_$ARCH \
    --ucx-plugins-dir $UCX_PLUGIN_DIR \
    --nixl-plugins-dir $NIXL_PLUGIN_DIR \
    --output-dir /workspace/nixl/dist

RUN uv pip install dist/nixl-*cp${DEFAULT_PYTHON_VERSION//./}*.whl
