---
title: UCX
description: UCX transfer backend for high-performance VRAM and DRAM transfers via RDMA and TCP.
---

## Overview

UCX is the general-purpose high-performance network transport backend in NIXL. It supports RoCE, InfiniBand, and TCP, making it the default backend for VRAM and DRAM transfers between nodes. UCX is automatically selected when no specific backend is requested and both agents have it initialized.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ VRAM; VRAM ↔ DRAM; DRAM ↔ DRAM |
| **Protocol** | RoCE, InfiniBand, TCP |
| **Best For** | GPU-to-GPU and CPU-to-CPU transfers between nodes |

## Installation

UCX is the default transfer backend and is included automatically with the `pip install nixl` package. For source builds, UCX must be built before NIXL.

### Build from Source

NIXL is tested with UCX version 1.22.x.

```bash
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.22.x
./autogen.sh
./contrib/configure-release-mt       \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-cuda=<cuda install>         \
    --with-verbs                       \
    --with-dm                          \
    --with-gdrcopy=<gdrcopy install>
make -j
make -j install-strip
ldconfig
```

Replace `<cuda install>` with the path to your CUDA installation (e.g., `/usr/local/cuda`) and `<gdrcopy install>` with the path to your GDRCopy installation if available.

<Tip>
[GDRCopy](https://github.com/NVIDIA/gdrcopy) is optional but recommended for maximum GPU memory registration performance. UCX and NIXL work without it, but performance may be reduced for GPU-to-GPU transfers.
</Tip>

See [Configuration](#configuration) for build options.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-ucx.mdx" />

### UCX Tuning

The UCX backend reads UCX settings from `NIXL_UCX_<X>` environment variables, falling back to the global `UCX_<X>` ones. For each setting, `NIXL_UCX_<X>` takes precedence over `UCX_<X>`, which takes precedence over the `ucx.conf` file and the UCX defaults. This applies to every UCX setting, including transport-level ones such as `NIXL_UCX_IB_PCI_RELAXED_ORDERING`.

Prefer the `NIXL_UCX_<X>` form. `UCX_<X>` variables are also read by every other UCX user in the same process, such as MPI or UCC, while `NIXL_UCX_<X>` only affects NIXL:

```bash
# Restricts NIXL's transports only; MPI and other UCX users keep their defaults
export NIXL_UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp
export NIXL_UCX_NET_DEVICES=mlx5_0:1
```

NIXL applies its own UCX defaults (for example `RNDV_THRESH=inf` and `MAX_RMA_RAILS=2`) only when neither `NIXL_UCX_<X>` nor `UCX_<X>` is set in the environment, so either form overrides them.

Process-wide UCX options, such as `UCX_LOG_LEVEL`, are shared by all UCX users and are read only from `UCX_<X>`.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ucx_path` | System path | Path to UCX installation. |

## When to Use

- **GPU-to-GPU transfers via RDMA** -- UCX leverages RoCE or InfiniBand for high-bandwidth, low-latency GPU memory transfers.
- **CPU-to-CPU with InfiniBand or RoCE** -- Standard high-performance network transport for host memory.
- **General-purpose fallback** -- UCX supports both VRAM and DRAM, making it suitable for most transfer scenarios.
