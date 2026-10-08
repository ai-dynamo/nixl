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

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ucx_path` | System path | Path to UCX installation. |

## Transfer Completion and Buffer Lifetime

A failed operation does not complete other operations in the same transfer. UCX keeps the transfer in `NIXL_IN_PROG` until its outstanding requests have finished, then reports the recorded error.

For WRITEs, local PUT completion is not remote completion. Every attempted WRITE must be flushed, including a PUT that returns an immediate error after partially posting. A failed endpoint flush is retried during polling, and the transfer remains in progress until a flush succeeds. The original error is still reported after draining; no success notification is sent for a failed transfer.

Invalidating remote metadata is not cancellation. Keep the transfer buffers and handle alive, continue polling, and retry `releaseXferReq` if it returns `NIXL_ERR_REPOST_ACTIVE`. This does not guarantee bounded completion for a stalled peer. If a WRITE's endpoint cannot be flushed successfully, it remains in progress and cannot be released; an endpoint error alone is not proof that remote buffers are safe to reuse.

## When to Use

- **GPU-to-GPU transfers via RDMA** -- UCX leverages RoCE or InfiniBand for high-bandwidth, low-latency GPU memory transfers.
- **CPU-to-CPU with InfiniBand or RoCE** -- Standard high-performance network transport for host memory.
- **General-purpose fallback** -- UCX supports both VRAM and DRAM, making it suitable for most transfer scenarios.
