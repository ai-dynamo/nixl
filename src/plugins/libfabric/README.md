# NIXL Libfabric Plugin

This plugin provides a high-performance RDMA backend for NIXL using the OpenFabrics Interfaces (OFI) Libfabric library.

## Overview

The Libfabric plugin provides a high-performance RDMA communication backend with the following key capabilities:

- **Multi-Rail RDMA**: Automatic discovery and utilization of multiple network devices for increased bandwidth
- **GPU Direct Support**: Zero-copy transfers between GPU memory (VRAM) and remote systems with CUDA integration. GDR (GPU Direct RDMA) support is currently required.
- **Scalable Connection Management**: Efficient multi-agent connectivity with robust state tracking and automatic reconnection
- **Asynchronous Processing**: Non-blocking RDMA operations with pre-allocated request pools and completion processing
- **Thread-Safe Concurrency**: Background progress threads with lock-free data structures and configurable threading patterns
- **Topology-Aware Optimization**: Hardware-aware GPU-to-EFA and NUMA-to-EFA mapping using hwloc for optimal performance (EFA-specific)

## Dependencies

### Required Dependencies

- **Libfabric**
  - Many systems will have libfabric already installed. If not, custom libfabric installation is available via https://ofiwg.github.io/libfabric/ - Minimum required version: `v1.21.0`
  - For EFA enabled AWS instances, it is recommended to install through AWS EFA installer: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html - Recommend to use the latest version

- **hwloc**
  - hwloc is used to understand the underlying architecture to optimize application performance. Suggested version: 2.10.0 or newer

### Network Hardware Requirements

Validated compatibility with:

- **AWS EFA** (Elastic Fabric Adapter)

Any other Libfabric providers should also work but have not been validated in production environments. Community validation and feedback are highly appreciated!

## Build Instructions

```bash
# Basic build setup with default options
$ meson setup <name_of_build_dir>

# Setup with custom options (example)
$ meson setup <name_of_build_dir> \
    -Dlibfabric_path=/path/to/libfabric

# Build and install
$ cd build
$ ninja && ninja install
```

## API Reference

### Core Classes

- **`nixlLibfabricEngine`** - Main backend engine providing multi-rail RDMA operations with GPU Direct support
- **`nixlLibfabricRailManager`** - Manages multiple network rails with topology-aware selection and striping strategies
- **`nixlLibfabricRail`** - Individual network rail handling libfabric resources and completion processing
- **`nixlLibfabricTopology`** - Hardware topology discovery for optimal GPU-to-EFA and NUMA-to-EFA mapping
- **`nixlLibfabricBackendH`** - Request handle for tracking multi-request transfer completion with atomic counters
- **`nixlLibfabricConnection`** - Multi-rail connection metadata for remote agents with state management

## Troubleshooting

### Debug Information

Enable debug logging by setting environment variables:

```bash
# Libfabric debug logging
export FI_LOG_LEVEL=debug
export FI_LOG_PROV=efa  # or verbs, tcp, etc.

# NIXL debug logging
export NIXL_LOG_LEVEL=debug
```

### Common Issues

**No network devices detected:**

```bash
# Check available fabric interfaces
fi_info -l

# For checking specific devices (e.g. EFA as an example)
fi_info -p efa
```

For additional support, check the NIXL documentation and Libfabric provider-specific guides.
