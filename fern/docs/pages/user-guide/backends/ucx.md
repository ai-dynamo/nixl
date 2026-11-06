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

## Affine SGL WRITE requests

When `NIXL_UCX_SGL_ENABLE=1` and NIXL is built with UCX SGL support, a WRITE
request can describe regular rows without constructing one NIXL descriptor per
row. Set `nixl_opt_args_t::customParam` when calling `createXferReq`:

```cpp
nixl_opt_args_t options;
options.customParam = "ucx_affine_write_v1=27,64,576,2304,576";
```

The five values are `run_count,rows_per_run,row_bytes,local_stride_bytes,remote_stride_bytes`.
The first `run_count` local and remote descriptors each name the first row of a
run and have logical length `rows_per_run * row_bytes`. Descriptors after those
runs, such as a ready marker, are transferred normally. The example describes
27 runs of 64 rows, with a 576-byte row every 2304 bytes locally and every
576 bytes remotely. The same geometry applies to every run in one request.
Both strides must be at least `row_bytes`.

The backend checks the full expanded span of each run against its registered
local and remote memory regions before posting. Keep the registration and
remote metadata valid until the request completes. UCX still builds its SGL
and retains its flush, completion, and notification behavior. The option is
valid only for WRITE with UCX SGL enabled; requests without it use the ordinary
path. An optional `;worker_id=N` suffix selects a UCX shared worker.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-ucx.mdx" />

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ucx_path` | System path | Path to UCX installation. |

## When to Use

- **GPU-to-GPU transfers via RDMA** -- UCX leverages RoCE or InfiniBand for high-bandwidth, low-latency GPU memory transfers.
- **CPU-to-CPU with InfiniBand or RoCE** -- Standard high-performance network transport for host memory.
- **General-purpose fallback** -- UCX supports both VRAM and DRAM, making it suitable for most transfer scenarios.
