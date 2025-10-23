<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL Benchmark

A comprehensive benchmarking tool for the NVIDIA Inference Xfer Library (NIXL) that uses ETCD for coordination. NIXLBench provides performance testing across multiple communication backends and storage systems, making it ideal for evaluating high-performance data transfer scenarios in distributed computing environments.

## Table of Contents

1. [Features](#features)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Building](#building)
5. [Usage](#usage)
6. [Backend-Specific Examples](#backend-specific-examples)
7. [Troubleshooting](#troubleshooting)
8. [Support and Resources](#support-and-resources)

## Features

- **Multiple Communication Backends**: UCX, GPUNETIO, Mooncake, Libfabric for network communication
- **Storage Backend Support**: GDS, GDS_MT, POSIX, HF3FS, OBJ (S3), GUSLI for storage operations
- **Flexible Communication Patterns**:
  - **Pairwise**: Point-to-point communication between pairs
  - **Many-to-one**: Multiple initiators to single target
  - **One-to-many**: Single initiator to multiple targets
  - **TP (Tensor Parallel)**: Optimized for distributed training workloads
- **Memory Type Support**: Tests both CPU (DRAM) and GPU (VRAM) memory transfers
- **Multiple Worker Types**:
  - **NIXL worker**: Full-featured with all backend support
  - **NVSHMEM worker**: GPU-focused with VRAM-only transfers
- **Cloud-Native Coordination**: Uses ETCD for worker coordination - ideal for containerized environments
- **Performance Features**:
  - Multi-threading support with configurable progress threads
  - VMM memory allocation support for CUDA Fabric
  - Comprehensive performance metrics with latency percentiles
  - Data consistency validation for reliability testing

## System Requirements

### Hardware Requirements
- **CPU**: x86_64 or aarch64 architecture
- **Memory**: Minimum 8GB RAM (16GB+ recommended for compilation)
- **Storage**: At least 20GB free disk space
- **GPU**: NVIDIA GPU with CUDA support (for GPU features)
- **Network**:
  - ***InfiniBand/Ethernet Adapters*** - UCX/GPUNetIO/Mooncake Backends
  - ***Elastic Fabric Adapters (EFA) in AWS*** - UCX/Libfabric Backends

### Software Requirements
- **Operating System**: Ubuntu 22.04/24.04 LTS (recommended) or RHEL-based
- **Docker**: Version 20.10+ (for container builds)
- **Git**: For source code management
- **CUDA Toolkit**: 12.8+ (for GPU features)
- **Python**: 3.12+ (for benchmark utilities)

## Quick Start

### Using Docker (Recommended)

The fastest way to get started is using the pre-built Docker container:

```bash
# Clone the NIXL repository
git clone https://github.com/ai-dynamo/nixl.git
cd nixl/benchmark/nixlbench/contrib

# Build the container with default settings
./build.sh

# Start ETCD server for coordination
docker run -d --name etcd-server \
  -p 2379:2379 -p 2380:2380 \
  quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --data-dir=/etcd-data \
  --listen-client-urls=http://0.0.0.0:2379 \
  --advertise-client-urls=http://0.0.0.0:2379 \
  --listen-peer-urls=http://0.0.0.0:2380 \
  --initial-advertise-peer-urls=http://0.0.0.0:2380 \
  --initial-cluster=default=http://0.0.0.0:2380

# Run a basic benchmark
docker run -it --gpus all --network host nixlbench:latest \
  nixlbench --etcd_endpoints http://localhost:2379 --backend UCX
```

### Native Installation

For development or when Docker is not available:

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libgflags-dev libgrpc-dev libprotobuf-dev \
  etcd-server etcd-client python3-dev python3-pip

# Build and install NIXL first
cd /path/to/nixl
meson setup build --prefix=/usr/local/nixl --buildtype=release
cd build && ninja && sudo ninja install

# Build NIXLBench
cd /path/to/nixlbench
meson setup build -Dnixl_path=/usr/local/nixl --buildtype=release
cd build && ninja && sudo ninja install
```

## Building

### Docker Container Build (Recommended)

The Docker approach handles all dependencies automatically and is the most reliable method.

#### Prerequisites - Docker Installation

**Ubuntu/Debian:**
```bash
# Install Docker using convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

**RHEL/CentOS/Fedora:**
```bash
# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io

# Start Docker
sudo systemctl start docker && sudo systemctl enable docker
sudo usermod -aG docker $USER
```

#### Building the Container

```bash
# Navigate to the build directory
cd nixl/benchmark/nixlbench/contrib

# Basic build
./build.sh

# Advanced build options
./build.sh --build-type debug --arch aarch64 --python-versions "3.10,3.11,3.12"
```

**Available Build Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--nixl <path>` | Path to NIXL source directory | Parent NIXL Directory |
| `--nixlbench <path>` | Path to NIXLBench source directory | Current directory |
| `--ucx <path>` | Path to custom UCX source (optional) | Uses base image UCX |
| `--build-type <type>` | Build type: `debug` or `release` | `release` |
| `--base-image <image>` | Base Docker image | `nvcr.io/nvidia/cuda-dl-base` |
| `--base-image-tag <tag>` | Base image tag | `25.03-cuda12.8-devel-ubuntu24.04` |
| `--arch <arch>` | Target architecture: `x86_64` or `aarch64` | Auto-detected |
| `--python-versions <versions>` | Python versions (comma-separated) | `3.12` |
| `--tag <tag>` | Custom Docker image tag | Auto-generated |
| `--no-cache` | Disable Docker build cache | Cache enabled |

### Native Build

For development environments or when Docker is not available.

#### Core Dependencies

**Required:**
- **NIXL**: Core communication library
- **UCX**: Unified Communication X library
- **CUDA**: NVIDIA CUDA Toolkit (≥12.8)
- **CMake**: Build system (≥3.20)
- **Meson**: Build system for NIXL/NIXLBench
- **Ninja**: Build backend
- **etcd-cpp-api**: C++ client library for etcd (required for metadata exchange)
- **GFlags**: Command line flag processing
- **OpenMP**: Multi-threading support

**Optional:**
- **LibFabric**: Fabric communication library
- **DOCA**: NVIDIA DOCA SDK for GPUNetIO
- **AWS SDK C++**: For S3 object storage backend
- **GDS**: NVIDIA GPUDirect Storage
- **GUSLI**: G3+ User Space Access Library for direct block device access
- **NVSHMEM**: Required for NVSHMEM worker type
- **hwloc**: Hardware locality detection (required for Libfabric only)

**Python Dependencies:**
- **PyTorch**: For KV-cache benchmarks
- **NumPy**: Numerical computing
- **PyYAML**: YAML configuration parsing
- **Click**: Command-line interface
- **Tabulate**: Table formatting

#### System Dependencies Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  autotools-dev automake libtool libz-dev flex \
  libgtest-dev hwloc libhwloc-dev libgflags-dev \
  libgrpc-dev libgrpc++-dev libprotobuf-dev \
  libaio-dev liburing-dev protobuf-compiler-grpc \
  libcpprest-dev etcd-server etcd-client \
  pybind11-dev libclang-dev libcurl4-openssl-dev \
  libssl-dev uuid-dev zlib1g-dev python3-dev python3-pip

# Install RDMA/InfiniBand packages
sudo apt-get reinstall -y --no-install-recommends \
  autoconf automake libtool pkg-config make g++ \
  libnuma-dev librdmacm-dev ibverbs-providers \
  libibverbs-dev rdma-core ibverbs-utils libibumad-dev
```

#### CUDA Toolkit Installation
```bash
# Download and install CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.15_linux.run
sudo sh cuda_12.8.0_550.54.15_linux.run

# Set environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### UCX Installation
```bash
# Option 1: Use system UCX
sudo apt-get install -y libucx-dev

# Option 2: Build from source
git clone https://github.com/openucx/ucx.git
cd ucx
./autogen.sh
./contrib/configure-release --with-cuda=/usr/local/cuda --enable-mt
make -j$(nproc) && sudo make install
```

#### etcd-cpp-api Installation (Required)
```bash
# Clone and build etcd-cpp-api
git clone --depth 1 https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
cd etcd-cpp-apiv3

# Remove cpprestsdk dependency from CMake config (already installed via apt)
sed -i '/^find_dependency(cpprestsdk)$/d' etcd-cpp-api-config.in.cmake

# Build and install
mkdir build && cd build
cmake .. \
  -DBUILD_ETCD_CORE_ONLY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install
sudo ldconfig
```

#### Optional Dependencies

**LibFabric:**
```bash
wget https://github.com/ofiwg/libfabric/releases/download/v2.3.0/libfabric-2.3.0.tar.bz2
tar xjf libfabric-2.3.0.tar.bz2 && cd libfabric-2.3.0
./configure --prefix=/usr/local --with-cuda=/usr/local/cuda --enable-cuda-dlopen --enable-efa
make -j$(nproc) && sudo make install
```

**AWS SDK C++ (for S3 backend):**
```bash
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581
mkdir sdk_build && cd sdk_build
cmake ../aws-sdk-cpp/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_ONLY="s3" \
  -DENABLE_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install
```

**DOCA (Optional):**
```bash
# Add Mellanox repository and install DOCA
wget https://www.mellanox.com/downloads/DOCA/DOCA_v3.1.0/host/doca-host_3.1.0-091000-25.07-ubuntu2404_amd64.deb
sudo dpkg -i doca-host_3.1.0-091000-25.07-ubuntu2404_amd64.deb
sudo apt-get update && sudo apt-get install -y doca-sdk-gpunetio libdoca-sdk-gpunetio-dev
```

**GUSLI (Optional - for GUSLI backend):**
```bash
# Clone and build GUSLI
git clone https://github.com/nvidia/gusli.git
cd gusli
make all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0 VERBOSE=1 ALLOW_USE_URING=0

# Install library and headers
sudo cp libgusli_clnt.so /usr/lib/
sudo cp gusli_*.hpp /usr/include/
sudo ldconfig
```

**Note**: GUSLI must be built before building NIXL. See [GUSLI Plugin README](../../src/plugins/gusli/README.md) for detailed installation and usage instructions.

#### Python Environment Setup
```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install Python dependencies
uv pip install meson pybind11 patchelf pyYAML click tabulate torch
```

#### Building NIXL and NIXLBench

```bash
# Build NIXL first
cd /path/to/nixl
rm -rf build && mkdir build
uv run meson setup build --prefix=/usr/local/nixl --buildtype=release
cd build && ninja && sudo ninja install

# Update library paths
echo "/usr/local/nixl/lib/x86_64-linux-gnu" | sudo tee /etc/ld.so.conf.d/nixl.conf
echo "/usr/local/nixl/lib/x86_64-linux-gnu/plugins" | sudo tee -a /etc/ld.so.conf.d/nixl.conf
sudo ldconfig

# Build NIXLBench
cd /path/to/nixlbench
rm -rf build && mkdir build
uv run meson setup build \
  -Dnixl_path=/usr/local/nixl/ \
  -Dprefix=/usr/local/nixlbench \
  --buildtype=release
cd build && ninja && sudo ninja install

# Update PATH
export PATH=/usr/local/nixlbench/bin:/usr/local/nixl/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nixlbench/lib:$LD_LIBRARY_PATH
```

#### Available Meson Build Options
- `nixl_path`: Path to NIXL installation (default: /usr/local)
- `cudapath_inc`: Include path for CUDA
- `cudapath_lib`: Library path for CUDA
- `cudapath_stub`: Extra stub path for CUDA
- `etcd_inc_path`: Path to ETCD C++ client includes
- `etcd_lib_path`: Path to ETCD C++ client library
- `nvshmem_inc_path`: Path to NVSHMEM include directory
- `nvshmem_lib_path`: Path to NVSHMEM library directory
- `buildtype`: Build type: `debug`, `release`, `debugoptimized` (default: release)
- `prefix`: Installation prefix (default: /usr/local)

## Usage

### ETCD Coordination Setup

NIXLBench uses ETCD for worker coordination, which is essential for multi-node benchmarking.

**Start ETCD server:**
```bash
# Option 1: Using Docker
docker run -d --name etcd-server \
  -p 2379:2379 -p 2380:2380 \
  quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --data-dir=/etcd-data \
  --listen-client-urls=http://0.0.0.0:2379 \
  --advertise-client-urls=http://0.0.0.0:2379 \
  --listen-peer-urls=http://0.0.0.0:2380 \
  --initial-advertise-peer-urls=http://0.0.0.0:2380 \
  --initial-cluster=default=http://0.0.0.0:2380

# Option 2: Native installation
sudo apt install etcd-server
sudo systemctl start etcd && sudo systemctl enable etcd
```

### Basic Usage Examples

```bash
# Basic UCX benchmark with VRAM transfers
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --initiator_seg_type VRAM --target_seg_type VRAM

# Storage benchmark with GDS backend
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GDS --filepath /mnt/storage/testfile

# S3 object storage benchmark
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend OBJ --obj_bucket_name my-bucket --obj_access_key $AWS_ACCESS_KEY_ID --obj_secret_key $AWS_SECRET_ACCESS_KEY

# Multi-threaded benchmark with progress threads
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --num_threads 4 --enable_pt --progress_threads 2
```

### Command Line Options

#### Core Configuration
```
--runtime_type NAME        # Type of runtime to use [ETCD] (default: ETCD)
--worker_type NAME         # Worker to use to transfer data [nixl, nvshmem] (default: nixl)
--backend NAME             # Communication backend [UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ, GUSLI] (default: UCX)
--benchmark_group NAME     # Name of benchmark group for parallel runs (default: default)
--etcd_endpoints URL       # ETCD server URL for coordination (default: http://localhost:2379)
```

#### Memory and Transfer Configuration
```
--initiator_seg_type TYPE  # Memory segment type for initiator [DRAM, VRAM] (default: DRAM)
--target_seg_type TYPE     # Memory segment type for target [DRAM, VRAM] (default: DRAM)
--scheme NAME              # Communication scheme [pairwise, manytoone, onetomany, tp] (default: pairwise)
--mode MODE                # Process mode [SG (Single GPU per proc), MG (Multi GPU per proc)] (default: SG)
--op_type TYPE             # Operation type [READ, WRITE] (default: WRITE)
--check_consistency        # Enable consistency checking
--total_buffer_size SIZE   # Total buffer size across devices per process (default: 8GiB)
--start_block_size SIZE    # Starting block size (default: 4KiB)
--max_block_size SIZE      # Maximum block size (default: 64MiB)
--start_batch_size SIZE    # Starting batch size (default: 1)
--max_batch_size SIZE      # Maximum batch size (default: 1)
```

#### Performance and Threading
```
--num_iter NUM             # Number of iterations (default: 1000)
--warmup_iter NUM          # Number of warmup iterations (default: 100)
--large_blk_iter_ftr NUM   # Factor to reduce transfer iteration for block size above 1MB (default: 16)
--num_threads NUM          # Number of threads used by benchmark (default: 1)
--num_initiator_dev NUM    # Number of devices in initiator processes (default: 1)
--num_target_dev NUM       # Number of devices in target processes (default: 1)
--enable_pt                # Enable progress thread (only used with nixl worker)
--progress_threads NUM     # Number of progress threads (default: 0)
--enable_vmm               # Enable VMM memory allocation when DRAM is requested
```

#### Device and Network Configuration
```
--device_list LIST         # Comma-separated device names (default: all)
--etcd_endpoints URL       # ETCD server URL for coordination (optional for storage backends)
```

#### Storage Backend Options (GDS, GDS_MT, POSIX, HF3FS, OBJ)
```
--filepath PATH            # File path for storage operations
--num_files NUM            # Number of files used by benchmark (default: 1)
--storage_enable_direct    # Enable direct I/O for storage operations
```

#### Backend-Specific Options

**GDS Backend:**
```
--gds_batch_pool_size NUM  # Batch pool size for GDS operations (default: 32)
--gds_batch_limit NUM      # Batch limit for GDS operations (default: 128)
```

**GDS_MT Backend:**
```
--gds_mt_num_threads NUM   # Number of threads used by GDS MT plugin (default: 1)
```

**POSIX Backend:**
```
--posix_api_type TYPE      # API type for POSIX operations [AIO, URING] (default: AIO)
```

**GPUNETIO Backend:**
```
--gpunetio_device_list LIST # Comma-separated GPU CUDA device id for GPUNETIO
```

**OBJ (S3) Backend:**
```
--obj_access_key KEY       # Access key for S3 backend
--obj_secret_key KEY       # Secret key for S3 backend
--obj_session_token TOKEN  # Session token for S3 backend
--obj_bucket_name NAME     # Bucket name for S3 backend
--obj_scheme SCHEME        # HTTP scheme for S3 backend [http, https] (default: http)
--obj_region REGION        # Region for S3 backend (default: eu-central-1)
--obj_use_virtual_addressing # Use virtual addressing for S3 backend
--obj_endpoint_override URL # Endpoint override for S3 backend
--obj_req_checksum TYPE    # Required checksum for S3 backend [supported, required] (default: supported)
```

**GUSLI Backend:**
```
--device_list LIST                     # Device specs in format 'id:type:path' (e.g., '11:F:./store0.bin,27:K:/dev/nvme0n1')
                                       # Type: F (file), K (kernel device), N (networked server with t/u prefix)
--gusli_client_name NAME               # Client identifier (default: NIXLBench)
--gusli_max_simultaneous_requests NUM  # Concurrent request limit (default: 32)
--gusli_device_security LIST           # Comma-separated security flags per device (e.g., 'sec=0x3,sec=0x71')
--gusli_bdev_byte_offset BYTES         # Starting LBA offset in bytes (default: 1048576)
--gusli_config_file CONTENT            # Custom config file content (auto-generated if not provided)

Note: storage_enable_direct is automatically enabled for GUSLI backend
```

### Using ETCD for Coordination

NIXL Benchmark uses an ETCD key-value store for coordination between benchmark workers. This is useful in containerized or cloud-native environments.

**ETCD Requirements:**
- **Required**: Network backends (UCX, GPUNETIO, Mooncake, Libfabric) and multi-node setups
- **Optional**: Storage backends (GDS, GDS_MT, POSIX, HF3FS, OBJ, GUSLI) running as single instances
- **Required**: Storage backends when `--etcd_endpoints` is explicitly specified

**For multi-node benchmarks:**

1. Ensure ETCD server is running (e.g., `docker run -p 2379:2379 quay.io/coreos/etcd`
2. Launch multiple nixlbench instances pointing to the same ETCD server

**For single-instance storage benchmarks:**
```bash
# No ETCD needed - just run directly
./nixlbench --backend GDS --filepath /mnt/storage/testfile

# Or with explicit ETCD if coordination is needed
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GDS --filepath /mnt/storage/testfile
```

Note: etcd can be installed directly on host as well:
```bash
apt install etcd-server
```

Example:
```bash
# On host 1
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --initiator_seg_type VRAM --target_seg_type VRAM

# On host 2
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --initiator_seg_type VRAM --target_seg_type VRAM
```

The workers automatically coordinate ranks through ETCD as they connect.

### Backend-Specific Examples

#### Network Backends

**UCX Backend (Default)**
```bash
# Basic UCX benchmark
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX

# UCX with specific devices
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --device_list mlx5_0,mlx5_1
```

**GPUNETIO Backend:**
```bash
# DOCA GPUNetIO with specific GPU devices
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GPUNETIO --gpunetio_device_list 0,1
```

### Storage Backends

**GDS (GPU Direct Storage):**
```bash
# Basic GDS benchmark (no ETCD needed for single instance)
./nixlbench --backend GDS --filepath /mnt/storage/testfile --storage_enable_direct

# GDS with ETCD coordination
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GDS --filepath /mnt/storage/testfile --storage_enable_direct

# GDS with custom batch settings
./nixlbench --backend GDS --filepath /mnt/storage/testfile --gds_batch_pool_size 64 --gds_batch_limit 256
```

**GDS_MT (Multi-threaded GDS):**
```bash
# Multi-threaded GDS (no ETCD needed for single instance)
./nixlbench --backend GDS_MT --filepath /mnt/storage/testfile --gds_mt_num_threads 8
```

**POSIX Backend:**
```bash
# POSIX with AIO (no ETCD needed for single instance)
./nixlbench --backend POSIX --filepath /mnt/storage/testfile --posix_api_type AIO

# POSIX with io_uring
./nixlbench --backend POSIX --filepath /mnt/storage/testfile --posix_api_type URING --storage_enable_direct
```

**GUSLI Backend (G3+ User Space Access Library):**

GUSLI provides direct user-space access to block storage devices, supporting local files, kernel block devices, and networked GUSLI servers.

**Note**: Direct I/O is automatically enabled when GUSLI backend is selected (no need to specify `--storage_enable_direct`).

```bash
# Basic GUSLI benchmark - single file device
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin" \
           --num_initiator_dev=1 \
           --num_target_dev=1 \
           --op_type=WRITE

# NVMe device with custom security
./nixlbench --backend=GUSLI \
           --device_list="27:K:/dev/nvme0n1" \
           --gusli_device_security="sec=0x7" \
           --num_initiator_dev=1 \
           --num_target_dev=1 \
           --op_type=READ

# Multi-device configuration
./nixlbench --backend=GUSLI \
           --device_list="11:F:./store0.bin,14:K:/dev/zero,27:K:/dev/nvme0n1" \
           --gusli_device_security="sec=0x3,sec=0x71,sec=0x7" \
           --num_initiator_dev=3 \
           --num_target_dev=3 \
           --op_type=WRITE

# Networked GUSLI server (TCP)
./nixlbench --backend=GUSLI \
           --device_list="20:N:t192.168.1.100" \
           --gusli_device_security="sec=0x10" \
           --num_initiator_dev=1 \
           --num_target_dev=1 \
           --op_type=WRITE

# High concurrency with multiple threads
./nixlbench --backend=GUSLI \
           --device_list="27:K:/dev/nvme0n1" \
           --gusli_max_simultaneous_requests=128 \
           --num_threads=8 \
           --total_buffer_size=$((16*1024*1024*1024)) \
           --op_type=WRITE
```

**GUSLI Device Types:**
- `F`: File-backed storage (e.g., `11:F:./store0.bin`)
- `K`: Kernel block device (e.g., `27:K:/dev/nvme0n1`, `14:K:/dev/zero`)
- `N`: Networked GUSLI server with protocol prefix (e.g., `20:N:t192.168.1.100` for TCP, `21:N:u10.0.0.5` for UDP)

**GUSLI-Specific Parameters:**
- `--gusli_client_name`: Client identifier (default: "NIXLBench")
- `--gusli_max_simultaneous_requests`: Concurrent request limit (default: 32)
- `--gusli_device_security`: Comma-separated security flags per device (default: "sec=0x3" for each device)
- `--gusli_bdev_byte_offset`: Starting LBA offset in bytes (default: 1MB)
- `--gusli_config_file`: Custom config file content override

**Notes**:
- Number of devices in `--device_list` must match `--num_initiator_dev` and `--num_target_dev`
- Direct I/O is automatically enabled for GUSLI (no need to specify `--storage_enable_direct`)

### Worker Types

**NVSHMEM Worker:**
```bash
# NVSHMEM (GPU-only, VRAM required)
./nixlbench --etcd_endpoints http://etcd-server:2379 --worker_type nvshmem --initiator_seg_type VRAM --target_seg_type VRAM
```

### S3 Object Storage Backend

For OBJ plugin benchmarking, ETCD is optional for single instances.

```bash
# Basic S3 benchmark using environment variables (no ETCD needed)
AWS_ACCESS_KEY_ID=<access_key> AWS_SECRET_ACCESS_KEY=<secret_key> AWS_DEFAULT_REGION=<region> \
./nixlbench --backend OBJ --obj_bucket_name <bucket_name>

# S3 benchmark using command line flags (no ETCD needed)
./nixlbench --backend OBJ \
  --obj_access_key <access_key> \
  --obj_secret_key <secret_key> \
  --obj_region <region> \
  --obj_bucket_name <bucket_name>

# S3 benchmark with ETCD coordination (if needed)
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend OBJ --obj_bucket_name <bucket_name>
```

**Performance Considerations:**
Transfer times are higher than local storage, so consider reducing iterations:

```bash
./nixlbench --backend OBJ \
  --obj_bucket_name test-bucket \
  --warmup_iter 32 --num_iter 32 --large_blk_iter_ftr 2
```

**Testing Options:**
- Test read operations: `--op_type READ`
- Validate data consistency: `--check_consistency`

### Multi-Node Coordination

Launch multiple nixlbench instances pointing to the same ETCD server:

```bash
# On host 1
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --initiator_seg_type VRAM --target_seg_type VRAM

# On host 2
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --initiator_seg_type VRAM --target_seg_type VRAM
```

The workers automatically coordinate ranks through ETCD as they connect.

## Troubleshooting

### Common Build Issues

#### CUDA Not Found
```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
```

#### UCX Build Failures
```bash
# Install missing dependencies
# Only required if UCX cannot detect RDMA libraries correctly
sudo apt-get reinstall -y libibverbs-dev librdmacm-dev rdma-core

# Clean and rebuild
make clean && make -j$(nproc)
```

#### etcd-cpp-api Build Issues
```bash
# If cpprestsdk errors occur
sudo apt-get install -y libcpprest-dev

# If protobuf version conflicts
sudo apt-get install -y libprotobuf-dev protobuf-compiler

# If etcd server is not available for testing
sudo systemctl start etcd && sudo systemctl enable etcd
```

#### Python Environment Issues
```bash
# Reset virtual environment
rm -rf .venv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

#### Docker Build Failures
```bash
# Clear Docker cache
docker system prune -a

# Build with verbose output
docker build --progress=plain --no-cache ...
```

### Runtime Issues

#### Library Not Found Errors
```bash
# Update library cache
sudo ldconfig

# Check library paths
ldd /usr/local/nixlbench/bin/nixlbench
```

#### GPU Access Issues
```bash
# Verify GPU access
nvidia-smi

# Check GPU / NIC Topology
nvidia-smi topo -m
# GPU / NIC closest to each other have PIX/PXB as the value

# Check CUDA driver
cat /proc/driver/nvidia/version
```

#### Network Backend Issues
```bash
# List available devices
ibv_devices  # For RDMA devices
ibv_devinfo -v # Detailed Info on all RDMA devices

ip link show  # For Ethernet devices

# Test UCX
ucx_info -d  # List UCX devices

export UCX_LOG_LEVEL=DEBUG # Verbose UCX logging

export UCX_PROTO_INFO=y # See transport used by UCX
```

### Performance Tuning

#### CPU Affinity
```bash
# Bind to specific cores
taskset -c 0-7 nixlbench ...

# Use numactl for NUMA systems
numactl --cpunodebind=0 --membind=0 nixlbench ...
```

#### Network Tuning
```bash
# Increase buffer sizes
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Support and Resources

- **Documentation**: https://github.com/ai-dynamo/nixl/tree/main/benchmark/nixlbench
- **Issues**: https://github.com/ai-dynamo/nixl/issues
- **NVIDIA Developer Forums**: https://forums.developer.nvidia.com/
- **UCX Documentation**: https://openucx.readthedocs.io/

---

*This guide covers NIXLBench build and usage procedures as of 2025. For the latest updates, please refer to the official repository.*