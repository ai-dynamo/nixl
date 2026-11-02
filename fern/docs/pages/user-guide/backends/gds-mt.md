---
title: GDS_MT
description: Multi-threaded GPUDirect Storage backend for higher throughput on parallel file operations.
---

## Overview

GDS_MT is a compatibility name for the same multi-threaded engine selected by `GDS` with `mode=mt`. It uses the same NVIDIA cuFile API and hardware requirements as GDS but distributes I/O across multiple threads for higher throughput on parallel file operations. The default behavior of `createBackend("GDS_MT")` remains unchanged.

`GDS_MT` will be removed in a future update. New integrations should use `GDS` with `mode=mt`.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ File; DRAM ↔ File |
| **Protocol** | GPUDirect Storage (cuFile API, multi-threaded) |
| **Best For** | Parallel GPU-to-file transfers |

## Installation

GDS_MT shares the same prerequisites and installation requirements as the [GDS](/nixl/user-guide/backend-selection/gds) backend. See the [GDS Installation](/nixl/user-guide/backend-selection/gds#installation) section for prerequisites, cuFile verification, and build options.

## Configuration

`GDS_MT` accepts only the multi-threaded engine configuration. The batch parameters documented for `GDS` do not apply.

### Backend Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thread_count` | `max(1, hardware concurrency / 2)` | Number of persistent TaskFlow workers. |

### Environment Variables

<Markdown src="/snippets/env-vars-gds.mdx" />

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `gds_path` | `/usr/local/cuda/` | Path to GDS cuFile installation. Shared with the GDS backend. |
| `disable_gds_backend` | `false` | Disable GDS backend entirely. Disables both GDS and GDS_MT. |

## When to Use

- **Existing integrations using the compatibility name** -- `GDS_MT` continues to select the multi-threaded engine until the compatibility name is removed.
- **New multi-threaded integrations** -- Use `GDS` with `mode=mt` instead of adopting `GDS_MT`.
