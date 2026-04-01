---
title: Libfabric
description: Libfabric (OFI) network backend for VRAM-to-VRAM, VRAM-to-DRAM, and DRAM-to-DRAM transfers as an alternative to UCX.
---

## Overview

The Libfabric backend uses OpenFabrics Interfaces (OFI) for VRAM and DRAM transfers across nodes, supporting GPU-to-GPU, GPU-to-CPU, and CPU-to-CPU paths. It serves as an alternative to the UCX backend for environments where libfabric is the primary networking fabric, supporting verbs, EFA, and TCP providers.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ VRAM; VRAM ↔ DRAM; DRAM ↔ DRAM |
| **Protocol** | OFI (verbs, EFA, TCP providers) |
| **Best For** | Environments with libfabric as the primary networking fabric |

## Installation

Libfabric is an optional backend. Install it if your deployment requires OFI-based networking or if you are running on AWS with EFA.

### Install from Source

```bash
git clone https://github.com/ofiwg/libfabric.git
cd libfabric
./autogen.sh
./configure --enable-verbs --enable-tcp
make -j
make install
ldconfig
```

Adjust the `--enable-*` flags based on your required providers:

- `--enable-verbs` -- InfiniBand and RoCE support
- `--enable-tcp` -- TCP fallback transport
- `--enable-efa` -- AWS Elastic Fabric Adapter support

### Additional Dependencies

- **hwloc** (2.10.0+) -- Required for topology-aware optimization
- **libnuma** (`libnuma-dev` on Debian/Ubuntu, `libnuma-devel` on RPM-based) -- Required for NUMA-aware rail selection

### Verify Installation

```bash
fi_info --version
```

This should print the libfabric version and list available providers.

See [Configuration](#configuration) for build options.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-libfabric.mdx" />

### Build Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `libfabric_path` | String (path) | System default | Path to the libfabric installation directory. |

## When to Use

- AWS EFA environments where libfabric is the standard networking stack
- Non-EFA clusters where libfabric is the preferred networking fabric
- Deployments where UCX is unavailable or libfabric offers better performance for your provider
