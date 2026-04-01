---
title: GDS_MT
description: Multi-threaded GPUDirect Storage backend for higher throughput on parallel file operations.
---

## Overview

GDS_MT is the multi-threaded variant of the GDS backend. It uses the same NVIDIA cuFile API and hardware requirements as GDS but distributes I/O across multiple threads for higher throughput on parallel file operations.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ File; DRAM ↔ File |
| **Protocol** | GPUDirect Storage (cuFile API, multi-threaded) |
| **Best For** | Parallel GPU-to-file transfers |

## Installation

GDS_MT shares the same prerequisites and installation requirements as the [GDS](./gds) backend. See the [GDS Installation](./gds#installation) section for prerequisites, cuFile verification, and build options.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-gds.mdx" />

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `gds_path` | `/usr/local/cuda/` | Path to GDS cuFile installation. Shared with the GDS backend. |
| `disable_gds_backend` | `false` | Disable GDS backend entirely. Disables both GDS and GDS_MT. |

## When to Use

- **Parallel file operations** -- Distribute GPU-to-file I/O across multiple threads for higher aggregate throughput.
- **Concurrent checkpoint writes** -- Write multiple GPU tensors to storage simultaneously.
- **When single-threaded GDS does not saturate storage bandwidth** -- Switch to GDS_MT when a single I/O thread leaves storage bandwidth underutilized.
