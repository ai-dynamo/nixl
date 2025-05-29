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

# NVIDIA Inference Xfer Library (NIXL)

NVIDIA Inference Xfer Library (NIXL) is targeted for accelerating point to point communications in AI inference frameworks such as NVIDIA Dynamo, while providing an abstraction over various types of memory (e.g., CPU and GPU) and storage (e.g., file, block and object store) through a modular plug-in architecture.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/nixl)](https://github.com/ai-dynamo/nixl/releases/latest)

## Pre-build Distributions
### PyPI Wheel

The nixl python API and libraries, including UCX, are available directly through PyPI:

```
pip install nixl
```

## Prerequisites for source build
### Ubuntu:

`$ sudo apt install build-essential cmake pkg-config`

### Fedora:

`$ sudo dnf install gcc-c++ cmake pkg-config`

### Python

`$ pip3 install meson ninja pybind11`

### UCX

NIXL was tested with UCX version 1.18.0.

[GDRCopy](https://github.com/NVIDIA/gdrcopy) is available on Github and is necessary for maximum performance, but UCX and NIXL will work without it.

```
$ wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
$ tar xzf ucx-1.18.0.tar.gz
$ cd ucx-1.18.0
$ ./configure                          \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-cuda=<cuda install>         \
    --with-verbs                       \
    --with-dm                          \
    --with-gdrcopy=<gdrcopy install>   \
    --enable-mt
$ make -j
$ make -j install-strip
$ ldconfig
```

### ETCD (Optional)
NIXL can use ETCD for metadata distribution and coordination between nodes in distributed environments. To use ETCD with NIXL:
#### ETCD Server and Client
 ```
$ sudo apt install etcd etcd-server etcd-client

# Or use Docker
$ docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.1
```

#### ETCD CPP API
Installed from https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3

```
$ sudo apt install libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
$ sudo apt install libcpprest-dev
$ git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
$ cd etcd-cpp-apiv3
$ mkdir build && cd build
$ cmake ..
$ make -j$(nproc) && make install
```

### Additional plugins

Some plugins may have additional build requirements, see them here:

- [Mooncake](src/plugins/mooncake/README.md)
- [POSIX](src/plugins/posix/README.md)
- [GDS](src/plugins/cuda_gds/README.md)

## Getting started
### Build & install

```
$ meson setup <name_of_build_dir>
$ cd <name_of_build_dir>
$ ninja
$ ninja install
```

### Build Options

NIXL supports several build options that can be specified during the meson setup phase:

```bash
# Basic build setup with default options
$ meson setup <name_of_build_dir>

# Setup with custom options (example)
$ meson setup <name_of_build_dir> \
    -Dbuild_docs=true \           # Build Doxygen documentation
    -Ducx_path=/path/to/ucx \     # Custom UCX installation path
    -Dinstall_headers=true \      # Install development headers
    -Ddisable_gds_backend=false   # Enable GDS backend
```

Common build options:
- `build_docs`: Build Doxygen documentation (default: false)
- `ucx_path`: Path to UCX installation (default: system path)
- `install_headers`: Install development headers (default: true)
- `disable_gds_backend`: Disable GDS backend (default: false)
- `cudapath_inc`, `cudapath_lib`: Custom CUDA paths
- `static_plugins`: Comma-separated list of plugins to build statically

### Building Documentation

If you have Doxygen installed, you can build the documentation:

```bash
# Configure with documentation enabled
$ meson setup <name_of_build_dir> -Dbuild_docs=true
$ cd <name_of_build_dir>
$ ninja

# Documentation will be generated in <name_of_build_dir>/html
# After installation (ninja install), documentation will be available in <prefix>/share/doc/nixl/
```

### pybind11 Python Interface
The pybind11 bindings for the public facing NIXL API are available in src/bindings/python. These bindings implement the headers in the src/api/cpp directory.

The preferred way is to build it through meson-python, which will just let it be installed with pip. This can be done from the root nixl directory:

` $ pip install .`

### Rust Bindings
```bash
# Build with default NIXL installation (/opt/nvidia/nvda_nixl)
$ cd src/bindings/rust
$ cargo build --release

# Or specify custom NIXL location
$ NIXL_PREFIX=/path/to/nixl cargo build --release

# Run tests
$ cargo test
```

Use in your project by adding to `Cargo.toml`:
```toml
[dependencies]
nixl-sys = { path = "path/to/nixl/bindings/rust" }
```

### Other build options
See [contrib/README.md](contrib/README.md) for more build options.

### Building Docker container
To build the docker container, first clone the current repository. Also make sure you are able to pull docker images to your machine before attempting to build the container.

Run the following from the root folder of the cloned NIXL repository:
```
$ ./contrib/build-container.sh
```

By default, the container is built with Ubuntu 24.04. To build a container for Ubuntu 22.04 use the --os option as follows:
```
$ ./contrib/build-container.sh --os ubuntu22
```

To see all the options supported by the container use:
```
$ ./contrib/build-container.sh -h
```

The container also includes a prebuilt python wheel in /workspace/dist if required for installing/distributing. Also, the wheel can be built with a separate script (see below).

### Building the python wheel
The contrib folder also includes a script to build the python wheel with the UCX dependencies. Note, that UCX and other NIXL dependencies are required to be installed.
```
$ ./contrib/build-wheel.sh
```

## Running with ETCD
NIXL can use ETCD for metadata exchange between distributed nodes. This is especially useful in containerized or cloud-native environments.

### Environment Setup
To use ETCD with NIXL, set the following environment variables:

```bash
# Set ETCD endpoints (required) - replace localhost with the hostname of the etcd server
export NIXL_ETCD_ENDPOINTS="http://localhost:2379"

# Set ETCD namespace (optional, defaults to /nixl/agents)
export NIXL_ETCD_NAMESPACE="/nixl/agents"
```

### Running the ETCD Example
NIXL includes an example demonstrating metadata exchange and data transfer using ETCD:

```bash
# Start an ETCD server if not already running
# For example:
# docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.1

# Set the ETCD env variables as above

# Run the example. The two agents in the example will exchange metadata through ETCD
# and perform data transfers
./<nixl_build_path>/examples/nixl_etcd_example
```

### nixlbench Benchmark
For more comprehensive testing, the nixlbench benchmarking tool supports ETCD for worker coordination:

```bash
# Build nixlbench (see benchmark/nixlbench/README.md for details)
cd benchmark/nixlbench
meson setup build && cd build && ninja

# Run benchmark with ETCD
./nixlbench --etcd-endpoints http://localhost:2379 --backend UCX --initiator_seg_type VRAM
```

## Examples

* [C++ examples](https://github.com/ai-dynamo/nixl/tree/main/examples/cpp)

* [Python examples](https://github.com/ai-dynamo/nixl/tree/main/examples/python)

## Third-Party Components

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
