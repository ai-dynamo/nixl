---
title: GDS
description: GPUDirect Storage backend for direct GPU-to-file transfers without CPU bounce buffers.
---

## Overview

GDS (GPUDirect Storage) enables direct transfers between GPU memory and files without CPU bounce buffers, using the NVIDIA cuFile API. It uses cuFile batch transfers by default and can select the multi-threaded engine with `mode=mt`. The default behavior of `createBackend("GDS")` remains unchanged. This backend requires GDS-capable hardware and drivers.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ File; DRAM ↔ File |
| **Protocol** | GPUDirect Storage (cuFile API) |
| **Best For** | Direct GPU-to-NVMe/filesystem transfers |

## Installation

### Prerequisites

- A GPU with GPUDirect Storage support (NVIDIA data center GPUs)
- CUDA Toolkit 11.4 or later installed
- cuFile driver and libraries (included with CUDA Toolkit 11.4+)
- A compatible filesystem (ext4, XFS, or a parallel filesystem with GDS support)

### Verify GDS Installation

The cuFile libraries required for GDS are included with the CUDA Toolkit. Verify they are present:

```bash
ls /usr/local/cuda/lib64/libcufile*
```

You should see `libcufile.so` and related library files.

Run the GDS compatibility check:

```bash
/usr/local/cuda/gds/tools/gdscheck -p
```

This verifies GPU support, kernel driver compatibility, and filesystem readiness.

## Configuration

### Backend Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `batch` | Transfer engine: `batch` or `mt`. |
| `batch_pool_size` | `16` | Number of preallocated cuFile batch handles. Applies only when `mode=batch`. |
| `batch_limit` | `128` | Maximum entries in one cuFile batch. Applies only when `mode=batch`. |
| `max_request_size` | `16777216` | Maximum bytes in one prepared I/O chunk. Applies only when `mode=batch`. |
| `thread_count` | `max(1, hardware concurrency / 2)` | Number of persistent TaskFlow workers. Applies only when `mode=mt`. |

### Environment Variables

<Markdown src="/snippets/env-vars-gds.mdx" />

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `gds_path` | `/usr/local/cuda/` | Path to GDS cuFile installation. |
| `disable_gds_backend` | `false` | Disable GDS backend entirely. Also disables GDS_MT. |

## When to Use

- **Direct GPU-to-file on NVMe storage** -- Bypass CPU bounce buffers for maximum throughput on NVMe drives.
- **Checkpoint save/load from GPU memory** -- Write GPU tensors directly to storage without staging through host memory.
- **Eliminating CPU bounce buffer overhead** -- Remove the CPU memory copy step in GPU-to-file transfers.
- **Batch I/O by default** -- Use `GDS` without parameters, or with `mode=batch`, for cuFile batch transfers.
- **Multi-threaded I/O** -- Use `GDS` with `mode=mt` to distribute I/O across persistent TaskFlow workers.
