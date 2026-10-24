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

The libblkio plugin supports three drivers:

- **IO_URING**: High-performance asynchronous I/O using Linux io_uring (default). Uses direct buffer access with no memory region mapping overhead.
- **VHOST_USER**: Virtio block device over vhost-user protocol. Connects to a vhost-user-blk Unix socket (e.g., qemu-storage-daemon). Uses bounce buffers with fd-backed memory for vhost protocol compliance.
- **VDPA**: Virtio block device over vhost-vdpa kernel interface. Connects to `/dev/vhost-vdpa-*` devices for hardware or software vDPA. Uses bounce buffers similar to VHOST_USER.

## Configuration Parameters

The libblkio plugin accepts the following backend parameters:

### Optional Parameters

- `api_type`: API type to use (default: `"IO_URING"`)
  - `"IO_URING"`: Use io_uring backend
  - `"VHOST_USER"`: Use virtio-blk-vhost-user (vhost-user socket path)
  - `"VDPA"`: Use virtio-blk-vhost-vdpa (vhost-vdpa device path)
- `direct_io`: Enable direct I/O (default: `0`/false)
  - `"1"` or `"true"`: Enable direct I/O
  - `"0"` or `"false"`: Disable direct I/O
- `io_polling`: Enable I/O polling (default: `false`)
- `num_queues`: Number of I/O queues (default: 1)
- `queue_size`: Queue depth (default: 128). Mapped to `num-entries` for IO_URING and `queue-size` for VHOST_USER/VDPA.

### Device Resolution

The plugin supports two device resolution modes:

**Multi-Device Mode (device_list)**:
- `device_list`: Comma-separated list of devices in format `id:type:path` (e.g., `1:B:/dev/loop0,2:B:/dev/loop1`)
  - Each device is assigned a specific devId for precise control

**Single-Device Mode (no device_list)**:
- When `device_list` is not provided, the plugin resolves the device path using:
  1. `metaInfo` field from memory descriptor (highest priority)
  2. `NIXL_LIBBLKIO_PATH` environment variable (fallback)
- Suitable for simple single-device deployments

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

## Running libblkio with Docker

Docker by default may block certain syscalls required by libblkio. These need to be explicitly enabled when running NIXL agents that use the libblkio plugin in Docker.

### Create a seccomp json file

```bash
# Download default seccomp profile
wget -O default.json https://raw.githubusercontent.com/moby/moby/master/profiles/seccomp/default.json

# Add the following to the section, syscalls:names in default.json
# "io_uring_setup",
# "io_uring_enter",
# "io_uring_register"

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

**GTest integration test binary** (requires real block device):

```bash
ninja -C build test/gtest/plugins/libblkio/libblkio_gtest
```

### Running Tests

#### Conventional (Legacy) Test

Requires a real block device. Set `NIXL_LIBBLKIO_TEST_DEVICE` or use `--device`:

```bash
sudo NIXL_PLUGIN_DIR=build/src/plugins/libblkio \
     NIXL_LIBBLKIO_TEST_DEVICE=/dev/loop0 \
     build/test/unit/plugins/libblkio/nixl_libblkio_test
```

#### GTest Integration Tests

Requires a real block device. Set `NIXL_LIBBLKIO_TEST_DEVICE` to the device path:

```bash
sudo NIXL_PLUGIN_DIR=build/src/plugins/libblkio \
     NIXL_LIBBLKIO_TEST_DEVICE=/dev/loop0 \
     build/test/gtest/plugins/libblkio/libblkio_gtest
```

#### Run All Registered Tests via Meson

Runs all tests registered with `meson test`, including conventional and gtest targets:

```bash
meson test -C build
```

To run only libblkio-related tests:

```bash
meson test -C build libblkio_plugin_test libblkio_gtest
```

To view verbose test output:

```bash
meson test -C build -v
```

## Current Limitations

1. For IO_URING, direct I/O is disabled by default. Enable it with the `direct_io` parameter.
2. For VHOST_USER and VDPA drivers, bounce buffers are used for DRAM I/O to satisfy fd-backed memory requirements. This adds a memory copy overhead compared to io_uring.
3. I/O polling for io_uring requires `direct_io=1` to be enabled.

## Future Enhancements

- Zero-copy support for VHOST_USER/VDPA when users provide memfd-backed memory
- Additional device types beyond block devices
- Enhanced error handling and logging
