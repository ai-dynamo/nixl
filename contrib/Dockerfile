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

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

ARG UBUNTUOS=24.04
ARG DEFAULT_PYTHON_VERSION="3.12"

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ninja-build \
    uuid-dev \
    gdb \
    python3-pip \
    pybind11-dev \
    cmake

RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace/nixl
ENV VIRTUAL_ENV=/workspace/nixl/.venv

RUN uv venv $VIRTUAL_ENV --python 3.12 && \
    # pybind11 pip install needed for ubuntu 22.04
    uv pip install --upgrade meson pybind11 patchelf

COPY . /workspace/nixl

RUN cd /workspace/nixl && \
    mkdir build && \
    uv run meson setup build/ \
        --prefix=/usr/local/nixl \
        --Ducx_path=/opt/hpcx/ucx && \
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
