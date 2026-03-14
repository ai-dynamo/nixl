<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Dell Technologies Inc. All rights reserved.
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

# NIXL libblkio Plugin

This backend provides high-performance block device I/O operations using the libblkio library. libblkio offers a unified interface to various block device backends including io_uring, virtio-blk-vhost-user, and virtio-blk-vhost-vdpa.

## Dependencies

To use the libblkio plugin, you need to install the libblkio package:

```bash
# Ubuntu/Debian
sudo apt-get install libblkio-dev

# RHEL/CentOS/Fedora
sudo dnf install libblkio-devel
```

## Supported API Types

The libblkio plugin currently supports:

- **io_uring**: High-performance asynchronous I/O using Linux io_uring (default and recommended)
- **virtio-blk-vhost-user**: Virtio block device over vhost-user protocol (defines kept for future support)
- **virtio-blk-vhost-vdpa**: Virtio block device over vhost-vdpa protocol (defines kept for future support)

## Configuration Parameters

The libblkio plugin accepts the following backend parameters:

### Required Parameters
- `api_type`: API type to use
  - `"IO_URING"`: Use io_uring backend (default)
  - `"VHOST_USER"`: Use virtio-blk-vhost-user (future support)
  - `"VHOST_VDPA"`: Use virtio-blk-vhost-vdpa (future support)
- `device_list`: Comma-separated list of devices in format `id:type:path` (e.g., `1:B:/dev/loop0,2:B:/dev/loop1`)

### Optional Parameters
- `direct_io`: Enable direct I/O (default: `0`/false)
  - `"1"` or `"true"`: Enable direct I/O
  - `"0"` or `"false"`: Disable direct I/O
- `io_polling`: Enable I/O polling (default: `false`)
- `num_queues`: Number of I/O queues (default: 1, vhost-specific)
- `queue_size`: Queue size (default: 128, vhost-specific)

## Usage Guide

### Basic Usage

```cpp
nixlAgent agent("client_name", nixlAgentConfig(true));
nixl_b_params_t params;

// Configure libblkio backend
params["api_type"] = "IO_URING";
params["device_list"] = "1:B:/dev/loop0,2:B:/dev/loop1";
params["direct_io"] = "1";

nixlBackendH* backend = nullptr;
nixl_status_t status = agent.createBackend("LIBBLKIO", params, backend);
```

## Using libblkio with NIXLBench

NIXLBench supports libblkio backend for storage performance benchmarking. The benchmark tool allows configurable device specifications and API types.

### Basic Usage

```bash
# Single device with io_uring and direct I/O
./nixlbench --backend=LIBBLKIO \
           --libblkio_api_type=IO_URING \
           --device_list="1:B:/dev/loop0" \
           --op_type=WRITE \
           --storage_enable_direct \
           --num_initiator_dev=1 \
           --num_target_dev=1

# Multi-device setup
./nixlbench --backend=LIBBLKIO \
           --libblkio_api_type=IO_URING \
           --device_list="1:B:/dev/loop0,2:B:/dev/loop1" \
           --op_type=READ \
           --num_initiator_dev=2 \
           --num_target_dev=2

# Disable direct I/O
./nixlbench --backend=LIBBLKIO \
           --libblkio_api_type=IO_URING \
           --device_list="1:B:/dev/loop0" \
           --op_type=WRITE \
           --num_initiator_dev=1 \
           --num_target_dev=1
```

### Device List Format

Devices are specified using the `--device_list` parameter with format `id:type:path`:
- `id`: Numeric device identifier (e.g., 1, 2, 3)
- `type`: Device type (currently only `B` for block device is supported)
- `path`: Device path (e.g., `/dev/loop0`, `/dev/nvme0n1`)

**Important**: The number of devices in `--device_list` must match `--num_initiator_dev` and `--num_target_dev`.

### libblkio-Specific Parameters

**Required**:
- `--device_list`: Device specs in format `id:type:path` (e.g., `1:B:/dev/loop0,2:B:/dev/loop1`)
- `--num_initiator_dev`: Must match number of devices in `--device_list`
- `--num_target_dev`: Must match number of devices in `--device_list`

**Optional**:
- `--libblkio_api_type`: API type to use (default: `IO_URING`)
- `--storage_enable_direct`: Enable direct I/O (default: disabled)

### Advanced Examples

```bash
# High-performance NVMe benchmark with direct I/O
./nixlbench --backend=LIBBLKIO \
           --libblkio_api_type=IO_URING \
           --device_list="1:B:/dev/nvme0n1" \
           --storage_enable_direct \
           --num_initiator_dev=1 \
           --num_target_dev=1 \
           --num_threads=8 \
           --total_buffer_size=$((8*1024*1024*1024)) \
           --op_type=READ

# Multi-device load distribution
./nixlbench --backend=LIBBLKIO \
           --libblkio_api_type=IO_URING \
           --device_list="1:B:/dev/loop0,2:B:/dev/loop1,3:B:/dev/loop2" \
           --storage_enable_direct \
           --num_initiator_dev=3 \
           --num_target_dev=3 \
           --op_type=WRITE

# Buffered I/O (no direct I/O)
./nixlbench --backend=LIBBLKIO \
           --libblkio_api_type=IO_URING \
           --device_list="1:B:/dev/loop0" \
           --num_initiator_dev=1 \
           --num_target_dev=1 \
           --op_type=WRITE
```

## Running libblkio with Docker

Docker by default may block certain syscalls required by libblkio. These need to be explicitly enabled when running NIXL agents that use the libblkio plugin in Docker.

### Create a seccomp json file

```bash
# Download default seccomp profile
wget https://github.com/moby/moby/blob/master/profiles/seccomp/default.json

# Add the following to the section, syscalls:names in default.json
# "io_uring_setup",
# "io_uring_enter", 
# "io_uring_register",
# "io_uring_sync"

# Run docker with the new seccomp json file
docker run --security-opt seccomp=default.json -it --runtime=runc ... <imageid>
```

## Building and Testing

### Prerequisites

Install GoogleTest development libraries (required for gtest targets):

```bash
sudo apt-get install libgtest-dev libgmock-dev
```

### Configure the Build

Run from the repository root. Use `debug` build type — tests are skipped for `release` builds:

```bash
meson setup build \
    -Dbuildtype=debug \
    -Dbuild_tests=true \
    -Dtest_all_plugins=true
```

To reconfigure an existing build directory:

```bash
meson setup --reconfigure build \
    -Dbuildtype=debug \
    -Dbuild_tests=true \
    -Dtest_all_plugins=true
```

Use `--wipe` instead of `--reconfigure` if dependency changes are not being picked up:

```bash
meson setup --wipe build \
    -Dbuildtype=debug \
    -Dbuild_tests=true \
    -Dtest_all_plugins=true
```

### Build All Plugins and Tests

```bash
ninja -C build
```

### Build Specific Targets

**libblkio plugin only:**

```bash
ninja -C build src/plugins/libblkio/libplugin_LIBBLKIO.so
```

**Conventional (legacy) test binary:**

```bash
ninja -C build test/unit/plugins/libblkio/nixl_libblkio_test
```

**GTest unit test binary** (no device required):

```bash
ninja -C build test/gtest/unit/unit
```

**GTest integration test binary** (requires real block device):

```bash
ninja -C build test/gtest/plugins/libblkio/libblkio_gtest
```

### Running Tests

#### Conventional (Legacy) Test

Requires a real block device. Set `NIXL_LIBBLKIO_PATH` to the device path:

```bash
sudo NIXL_PLUGIN_DIR=build/src/plugins/libblkio \
     NIXL_LIBBLKIO_PATH=/dev/loop0 \
     build/test/unit/plugins/libblkio/nixl_libblkio_test
```

#### GTest Unit Tests

No real device required. Tests cover backend creation and memory registration:

```bash
sudo NIXL_PLUGIN_DIR=build/src/plugins/libblkio \
     build/test/gtest/unit/unit --gtest_filter='Libblkio*'
```

To list all available libblkio unit test cases:

```bash
sudo NIXL_PLUGIN_DIR=build/src/plugins/libblkio \
     build/test/gtest/unit/unit --gtest_list_tests
```

#### GTest Integration Tests

Requires a real block device. Set `NIXL_LIBBLKIO_PATH` to the device path:

```bash
sudo NIXL_PLUGIN_DIR=build/src/plugins/libblkio \
     NIXL_LIBBLKIO_PATH=/dev/loop0 \
     build/test/gtest/plugins/libblkio/libblkio_gtest
```

#### Run All Registered Tests via Meson

Runs all tests registered with `meson test`, including conventional and gtest targets:

```bash
meson test -C build
```

To run only libblkio-related tests:

```bash
meson test -C build --test-args '--gtest_filter=Libblkio*'
```

To view verbose test output:

```bash
meson test -C build -v
```

## Current Limitations

1. Only io_uring API type is actively supported. VHOST_USER and VHOST_VDPA types are defined for future support but will result in errors if used.
2. The plugin requires block device paths in the device_list format.
3. Direct I/O is disabled by default and must be explicitly enabled via `--storage_enable_direct` or the `direct_io` parameter.

## Future Enhancements

- Full support for virtio-blk-vhost-user and virtio-blk-vhost-vdpa API types
- Additional device types beyond block devices
- Enhanced error handling and logging
