---
title: GDS
description: GPUDirect Storage backend for direct GPU-to-file transfers without CPU bounce buffers.
---

## Overview

GDS (GPUDirect Storage) enables direct transfers between GPU memory and files without CPU bounce buffers, using the NVIDIA cuFile API. This backend requires GDS-capable hardware and drivers.

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
- **Single-threaded GDS workloads** -- Use GDS when a single I/O thread is sufficient for your file operations.
