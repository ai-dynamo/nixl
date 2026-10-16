---
title: HF3FS
description: HuggingFace 3FS backend for file operations on 3FS fuse-based filesystems.
---

## Overview

The HF3FS backend enables file operations on HuggingFace 3FS fuse-based filesystems, bridging NIXL transfers and 3FS-mounted storage through the fuse layer.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ File |
| **Protocol** | 3FS fuse filesystem |
| **Best For** | Environments using HuggingFace 3FS for storage |

## Installation

The HF3FS backend requires the 3FS libraries and headers to be installed before building NIXL.

### Prerequisites

1. Build and install [3FS](https://github.com/deepseek-ai/3FS/)
2. Ensure that `hf3fs_usrbio.so` and `libhf3fs_api_shared.so` are installed under `/usr/lib/`
3. Ensure that 3FS headers are installed under `/usr/include/hf3fs`

After installing 3FS, rebuild NIXL to enable the HF3FS backend.

<Tip>
For best performance, provide page-aligned memory with a size that is a multiple of the page size to `nixlAgent.registerMem()`. This enables zero-copy shared memory between the application and the 3FS backend process.
</Tip>

## Configuration

The HF3FS backend has no backend-specific environment variables or build options. It requires a mounted 3FS fuse filesystem to be accessible on the system.

## When to Use

- **HuggingFace infrastructure with 3FS storage** -- Use when 3FS is the primary storage layer.
- **File-to-DRAM transfers through 3FS-mounted paths** -- Read and write data on 3FS fuse mounts.
- **NIXL-managed transfers on 3FS-mounted storage** -- Integrate 3FS into NIXL's transfer framework for any workload.
