---
title: AMD Infinity Storage (AIS)
description: GPU direct-storage backend for transfers between AMD GPU memory and files via hipFile.
---

## Overview

The AIS backend is the ROCm counterpart of the [GDS](/nixl/user-guide/backend-selection/gds)
backend: it moves data between AMD GPU memory (`VRAM_SEG`) and files
(`FILE_SEG`) using the AMD Infinity Storage hipFile APIs, bypassing staged
copies through host memory.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ File |
| **Protocol** | hipFile (AMD Infinity Storage) |
| **Best For** | GPU direct storage on AMD platforms |

The backend is local-only: no remote transfers and no notifications. Exactly
one side of a transfer must be the file side; memory-to-memory and
file-to-file requests are rejected.

## Installation

Requires ROCm 7.14 or later (the first release shipping hipFile) with
`libamdhip64` and `libhipfile.so`, plus TaskFlow (provided by the Meson
subproject). The plugin is skipped silently when HIP or hipFile is not found;
request it explicitly to make missing dependencies a configure error.

```bash
meson setup build -Drocm_path=/opt/rocm
meson compile -C build
```

HIP is detected independently of CUDA, so `AIS` can be built alongside the
CUDA plugins when both toolchains are present.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `mt` | Submission engine. Only `mt` (multi-threaded, one blocking hipFile read/write per request) is supported today; a `batch` mode will follow once hipFile's batch API is adopted. |
| `thread_count` | half of hardware concurrency (min 1) | Number of persistent TaskFlow workers. |

Files can be registered by fd (fd in the descriptor `devId`) or by path
(`"<modes>:<path>"` in `metaInfo`; the backend owns the open/close). Repeated
registrations of the same open fd share one reference-counted hipFile handle.

<Note>
hipFile rejects host memory registration today, so only `VRAM_SEG` buffers are
supported; `DRAM_SEG` support will follow. hipFile also refuses filesystems it
does not recognize as AIS-capable — `HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true`
allows them (for example, for unit tests on a scratch directory).
</Note>

## When to Use

- **GPU direct storage on AMD** — read and write files directly from ROCm
  device memory without host staging.
- **GDS-parity pipelines** — reuse code written against the GDS backend on
  AMD hardware.

See the [plugin README](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/rocm_ais/README.md)
for the engine architecture, the full parameter reference, and an end-to-end
test example.
