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

A benchmarking tool for the NVIDIA Inference Xfer Library (NIXL) that uses ETCD for coordination.

## Features

- Benchmarks NIXL performance across multiple backends:
  - **Network backends**: UCX, UCX_MO, GPUNETIO, Mooncake
  - **Storage backends**: GDS, GDS_MT, POSIX, HF3FS, OBJ (S3)
- Supports multiple communication patterns:
  - **Pairwise**: Point-to-point communication between pairs
  - **Many-to-one**: Multiple initiators to single target
  - **One-to-many**: Single initiator to multiple targets
  - **TP (Tensor Parallel)**: Optimized for distributed training workloads
- Tests both CPU (DRAM) and GPU (VRAM) memory transfers
- Support for multiple worker types:
  - **NIXL worker**: Full-featured with all backend support
  - **NVSHMEM worker**: GPU-focused with VRAM-only transfers
- Uses ETCD for worker coordination - ideal for containerized and cloud-native environments
- Multi-threading support with configurable progress threads
- VMM memory allocation support for CUDA Fabric
- Comprehensive performance metrics with latency percentiles
- Data consistency validation for reliability testing

## Building

### Prerequisites

#### Required Dependencies
- **NIXL Library** - NVIDIA Inference Xfer Library
- **GFlags** - Command line flag processing
- **OpenMP** - Multi-threading support
- **ETCD C++ client** - Coordination runtime (https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3)

#### Optional Dependencies
- **CUDA Toolkit** - Required for VRAM operations and GPU backends
- **NVSHMEM** - Required for NVSHMEM worker type
- **Backend-specific libraries** as needed:
  - UCX for network communication
  - GDS/cuFile for GPU Direct Storage
  - io_uring for POSIX URING operations

### Building with Meson

Basic build:
```bash
# Configure build
meson setup build

# Build
cd build
meson compile

# Install (optional)
meson install
```

#### Custom Dependency Paths

If dependencies are installed in non-standard locations, you can specify their paths:

```bash
# With custom dependency paths
meson setup build \
  -Dnixl_path=/path/to/nixl/installation \
  -Dcudapath_inc=/path/to/cuda/include \
  -Dcudapath_lib=/path/to/cuda/lib64 \
  -Detcd_inc_path=/path/to/etcd/include \
  -Detcd_lib_path=/path/to/etcd/lib \
  -Dnvshmem_inc_path=/path/to/nvshmem/include \
  -Dnvshmem_lib_path=/path/to/nvshmem/lib

# To view all available meson project options
meson configure build
```

#### Available Build Options
- `nixl_path`: Path to NIXL installation (default: /usr/local)
- `cudapath_inc`: Include path for CUDA
- `cudapath_lib`: Library path for CUDA
- `cudapath_stub`: Extra stub path for CUDA
- `etcd_inc_path`: Path to ETCD C++ client includes
- `etcd_lib_path`: Path to ETCD C++ client library
- `nvshmem_inc_path`: Path to NVSHMEM include directory
- `nvshmem_lib_path`: Path to NVSHMEM library directory

## Usage

### Basic Usage

```bash
# Run basic UCX benchmark with VRAM transfers
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --initiator_seg_type VRAM --target_seg_type VRAM

# Run storage benchmark with GDS backend
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GDS --filepath /mnt/storage/testfile

# Run S3 object storage benchmark
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend OBJ --obj_bucket_name my-bucket --obj_access_key $AWS_ACCESS_KEY_ID --obj_secret_key $AWS_SECRET_ACCESS_KEY

# Run multi-threaded benchmark with progress threads
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX --num_threads 4 --enable_pt --progress_threads 2
```

### Command Line Options

#### Core Options
```
--runtime_type NAME        # Type of runtime to use [ETCD] (default: ETCD)
--worker_type NAME         # Worker to use to transfer data [nixl, nvshmem] (default: nixl)
--backend NAME             # Communication backend [UCX, UCX_MO, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ] (default: UCX)
--benchmark_group NAME     # Name of benchmark group for parallel runs (default: default)
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
--etcd_endpoints URL       # ETCD server URL for coordination (default: http://localhost:2379)
```

#### Storage Backend Options (GDS, GDS_MT, POSIX, HF3FS, OBJ)
```
--filepath PATH            # File path for storage operations
--num_files NUM            # Number of files used by benchmark (default: 1)
--storage_enable_direct    # Enable direct I/O for storage operations
```

#### GDS Backend Specific Options
```
--gds_batch_pool_size NUM  # Batch pool size for GDS operations (default: 32)
--gds_batch_limit NUM      # Batch limit for GDS operations (default: 128)
```

#### GDS_MT Backend Specific Options
```
--gds_mt_num_threads NUM   # Number of threads used by GDS MT plugin (default: 1)
```

#### POSIX Backend Specific Options
```
--posix_api_type TYPE      # API type for POSIX operations [AIO, URING] (default: AIO)
```

#### GPUNETIO Backend Specific Options
```
--gpunetio_device_list LIST # Comma-separated GPU CUDA device id for GPUNETIO
```

#### OBJ (S3) Backend Specific Options
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

### Using ETCD for Coordination

NIXL Benchmark uses an ETCD key-value store for coordination between benchmark workers. This is useful in containerized or cloud-native environments.

To run the benchmark:

1. Ensure ETCD server is running (e.g., `docker run -p 2379:2379 quay.io/coreos/etcd`
2. Launch multiple nixlbench instances pointing to the same ETCD server

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

# UCX Memory-Only variant
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend UCX_MO
```

**GPUNETIO Backend**
```bash
# DOCA GPUNetIO with specific GPU devices
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GPUNETIO --gpunetio_device_list 0,1
```

#### Storage Backends

**GDS (GPU Direct Storage)**
```bash
# Basic GDS benchmark
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GDS --filepath /mnt/storage/testfile --storage_enable_direct

# GDS with custom batch settings
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GDS --filepath /mnt/storage/testfile --gds_batch_pool_size 64 --gds_batch_limit 256
```

**GDS_MT (Multi-threaded GDS)**
```bash
# Multi-threaded GDS
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend GDS_MT --filepath /mnt/storage/testfile --gds_mt_num_threads 8
```

**POSIX Backend**
```bash
# POSIX with AIO
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend POSIX --filepath /mnt/storage/testfile --posix_api_type AIO

# POSIX with io_uring
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend POSIX --filepath /mnt/storage/testfile --posix_api_type URING --storage_enable_direct
```

#### Worker Types

**NVSHMEM Worker**
```bash
# NVSHMEM (GPU-only, VRAM required)
./nixlbench --etcd_endpoints http://etcd-server:2379 --worker_type nvshmem --initiator_seg_type VRAM --target_seg_type VRAM
```

### Benchmarking the OBJ (S3) Backend

For OBJ plugin benchmarking run etcd-server and a single nixlbench instance.

Example:
```bash
# Basic S3 benchmark using environment variables
AWS_ACCESS_KEY_ID=<access_key> AWS_SECRET_ACCESS_KEY=<secret_key> AWS_DEFAULT_REGION=<region> \
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend OBJ --obj_bucket_name <bucket_name>

# S3 benchmark using command line flags
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend OBJ \
  --obj_access_key <access_key> \
  --obj_secret_key <secret_key> \
  --obj_region <region> \
  --obj_bucket_name <bucket_name>
```

**Performance Considerations:**
Transfer times are higher than local storage, so consider reducing iterations:

```bash
./nixlbench --etcd_endpoints http://etcd-server:2379 --backend OBJ \
  --obj_bucket_name test-bucket \
  --warmup_iter 32 --num_iter 32 --large_blk_iter_ftr 2
```

**Testing Options:**
- Test read operations: `--op_type READ`
- Validate data consistency: `--check_consistency`
