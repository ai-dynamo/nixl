---
title: Building NIXLBench
description: Build NIXLBench using Docker or natively from source with Meson.
---

NIXLBench requires a NIXL installation -- see [Building NIXL from Source](../../building-nixl/index.md) for instructions.

## System Requirements

### Hardware

- **CPU:** x86_64 or aarch64 architecture
- **Memory:** 8 GB RAM minimum, 16 GB or more recommended for compilation
- **Storage:** 20 GB free disk space
- **GPU:** NVIDIA GPU with CUDA support (for GPU features)
- **Network:** InfiniBand or Ethernet adapters (for network backends)

### Software

- **OS:** Ubuntu 22.04 or 24.04 LTS (recommended), or RHEL-based distributions
- **Docker:** 20.10 or later (for container builds)
- **CUDA Toolkit:** 12.8 or later
- **Python:** 3.12 or later

## Build Instructions

<Tabs>
<Tab title="Docker">

The Docker build handles all dependencies automatically and is the recommended approach.

Clone the repository and navigate to the build directory:

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl/benchmark/nixlbench/contrib
```

Build the container with default settings:

```bash
./build.sh
```

Common build options:

```bash
# Debug build
./build.sh --build-type debug

# Build for aarch64
./build.sh --arch aarch64

# Combine options
./build.sh --build-type debug --arch aarch64
```

For the full list of build options, see the [NIXLBench README](https://github.com/ai-dynamo/nixl/tree/main/benchmark/nixlbench).

Run a quick benchmark to verify the build:

```bash
docker run -it --gpus all --network host nixlbench:latest \
  nixlbench --etcd_endpoints http://localhost:2379 --backend UCX
```

</Tab>
<Tab title="Native">

Build NIXLBench natively when Docker is not available or for development workflows. NIXL must be built and installed first -- see [Building NIXL from Source](../../building-nixl/index.md).

### Core Dependencies

- [UCX](../../backends/ucx.md)
- CUDA Toolkit (12.8 or later)
- Meson
- Ninja
- etcd-cpp-api
- GFlags
- OpenMP

### Build Steps

```bash
cd benchmark/nixlbench
meson setup build -Dnixl_path=/usr/local/nixl --buildtype=release
cd build && ninja && sudo ninja install
```

### Meson Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `nixl_path` | Path to NIXL installation | `/usr/local` |
| `buildtype` | Build type: `debug`, `release`, `debugoptimized` | `release` |
| `prefix` | Installation prefix | `/usr/local` |
| `etcd_inc_path` | Path to etcd C++ client headers | `""` |
| `etcd_lib_path` | Path to etcd C++ client library | `""` |

### Post-Install Setup

Add NIXLBench to your environment:

```bash
export PATH=/usr/local/nixlbench/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nixlbench/lib:$LD_LIBRARY_PATH
```

</Tab>
</Tabs>
