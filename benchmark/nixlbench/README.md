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

- Benchmarks NIXL performance across different backends
- Supports multiple communication patterns
- Tests both CPU (DRAM) and GPU (VRAM) memory transfers
- Uses ETCD for worker coordination - ideal for containerized and cloud-native environments

## Building

### Prerequisites

- NIXL Library
- CUDA Toolkit
- GFlags
- ETCD C++ client (https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3)

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

If NIXL is installed in a non-standard location, you can specify the path:

```bash
# With custom NIXL path
meson setup /path/to/build/dir -Dnixl_path=/path/to/nixl/installation

# To view all meson project options
meson configure /path/to/build/dir
```

## Usage

### Basic Usage

```bash
# Run the benchmark with default settings
# using ETCD runtime and ETCD server
./nixlbench --etcd-endpoints http://etcd-server:2379 --backend UCX --initiator_seg_type VRAM
```

### Command Line Options

```
--backend NAME             # Communication backend [UCX, UCX_MO] (default: UCX)
--worker_type NAME         # Worker to use to transfer data [nixl, nvshmem] (default: nixl)
--initiator_seg_type TYPE  # Memory segment type for initiator [DRAM, VRAM] (default: DRAM)
--target_seg_type TYPE     # Memory segment type for target [DRAM, VRAM] (default: DRAM)
--scheme NAME              # Communication scheme [pairwise, manytoone, onetomany, tp] (default: pairwise)
--mode MODE                # Process mode [SG (Single GPU per proc), MG (Multi GPU per proc)] (default: SG)
--op_type TYPE             # Operation type [READ, WRITE] (default: WRITE)
--check_consistency        # Enable consistency checking
--total_buffer_size SIZE   # Total buffer size (default: 8GiB)
--start_block_size SIZE    # Starting block size (default: 4KiB)
--max_block_size SIZE      # Maximum block size (default: 64MiB)
--start_batch_size SIZE    # Starting batch size (default: 1)
--max_batch_size SIZE      # Maximum batch size (default: 1)
--num_iter NUM             # Number of iterations (default: 1000)
--warmup_iter NUM          # Number of warmup iterations (default: 100)
--num_threads NUM          # Number of threads used by benchmark (default: 1)
--num_initiator_dev NUM    # Number of devices in initiator processes (default: 1)
--num_target_dev NUM       # Number of devices in target processes (default: 1)
--enable_pt                # Enable progress thread
--device_list LIST         # Comma-separated device names (default: all)
--runtime_type NAME        # Type of runtime to use [ETCD] (default: ETCD)
--etcd-endpoints URL       # ETCD server URL for coordination (default: http://localhost:2379)
--enable_vmm               # Enable VMM memory allocation when DRAM is requested
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
./nixlbench --runtime_type=ETCD --etcd-endpoints http://etcd-server:2379 --backend UCX --seg_type VRAM

# On host 2
./nixlbench --runtime_type=ETCD --etcd-endpoints http://etcd-server:2379 --backend UCX --seg_type VRAM
```

The workers automatically coordinate ranks through ETCD as they connect.
