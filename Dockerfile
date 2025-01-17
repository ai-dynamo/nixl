FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 as base
ARG MOFED_VERSION=24.10-1.1.4.0

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb

RUN apt-get update -y && apt-get -y install curl \
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
                                            libibmad-dev

RUN apt-get install -y linux-tools-common linux-tools-generic ethtool iproute2
RUN apt-get install -y dkms linux-headers-generic
RUN apt-get install -y meson ninja-build uuid-dev gdb

# Use nightly cuda 12.8 based pytorch since these are not yet released
RUN pip3 install --break-system-packages --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

RUN apt-get update && apt install -y wget libglib2.0-0
RUN wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

RUN cd /usr/local/src && \
    curl -fSsL "https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu24.04-x86_64.tgz" -o mofed.tgz && \
    tar -xf /usr/local/src/mofed.tgz && \
    cd MLNX_OFED_LINUX-* && \
    apt-get update && apt-get install -y --no-install-recommends \
        ./DEBS/libibverbs* ./DEBS/ibverbs-providers* ./DEBS/librdmacm* ./DEBS/libibumad* && \
    rm -rf /var/lib/apt/lists/* /usr/local/src/*

ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/gdrcopy.git
RUN PREFIX=/usr/local DESTLIB=/usr/local/lib make -C /workspace/gdrcopy lib_install
RUN cp gdrcopy/src/libgdrapi.so.2.* /usr/lib/x86_64-linux-gnu/
RUN ldconfig

ARG OPENMPI_VERSION=5.0.6
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
        --enable-mt                 \
        --with-mlx5-dv &&           \
    make -j &&                      \
    make -j install-strip &&        \
    ldconfig

ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
ENV CPATH=/usr/include:$CPATH
ENV PATH=/usr/bin:$PATH
ENV PKG_CONFIG_PATH=/usr/lib/pkgconfig:$PKG_CONFIG_PATH
SHELL ["/bin/bash", "-c"]

RUN cd /usr/local/src && \
    curl -fSsL "https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSION:0:3}/downloads/openmpi-${OPENMPI_VERSION}.tar.gz" | tar xz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure                          \
        --prefix=/usr/local/ompi         \
        --with-cuda=/usr/local/cuda      \
        --enable-mpi1-compatibility --with-slurm --without-ofi --with-libevent=internal --with-pmix=internal --with-prrte=internal --enable-hwloc=internal && \
    make -j &&                           \
    make -j install-strip &&             \
    ldconfig

WORKDIR /workspace

ENV LD_LIBRARY_PATH=/usr/local/ompi/lib:$LD_LIBRARY_PATH
ENV CPATH=/usr/local/ompi/include:$CPATH
ENV PATH=/usr/local/ompi/bin:$PATH
ENV PKG_CONFIG_PATH=/usr/local/ompi/lib/pkgconfig:$PKG_CONFIG_PATH

ARG USERNAME
ARG PASSWORD

RUN git clone https://${USERNAME}:${PASSWORD}@gitlab-master.nvidia.com/adv-dev-team/nixl.git && cd nixl && git branch
RUN cd nixl && mkdir build && meson setup build/ --prefix=/usr/local/nixl && cd build/ && ninja && ninja install

RUN cd nixl && PYTHONPATH=./build/src/pybind python3 test/python/nixl_wrapper_test.py
