---
title: AMD Infinity Storage (AIS)
description: GPU direct-storage backend for transfers between AMD GPU memory and files via hipFile.
---

## Overview

The `AIS` backend transfers data between AMD GPU memory (`VRAM_SEG`) and files
(`FILE_SEG`) by passing the device buffer directly to hipFile. It is the ROCm
counterpart of [GDS](/nixl/user-guide/backend-selection/gds).

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ File |
| **Protocol** | hipFile (AMD Infinity Storage) |
| **Best For** | GPU direct storage on AMD platforms |

AIS supports local transfers only and provides no notifications. Exactly one
side of a transfer must be a file. The engine rejects memory-to-memory and
file-to-file requests.

## Installation

Set `rocm_path` to a ROCm installation with HIP (`libamdhip64` and
`hip/hip_runtime.h`) and hipFile (`libhipfile.so` and `hipfile.h`). The
project's ROCm container defaults to 7.14.0. Meson also requires `hipify-perl`
for ROCm builds and looks for Taskflow with a subproject fallback.

```bash
meson setup build -Drocm_path=/opt/rocm -Denable_plugins=AIS
meson compile -C build
```

| Build condition | Default plugin selection | Explicit `-Denable_plugins=AIS` |
|-----------------|--------------------------|---------------------------------|
| Empty `rocm_path` (default) | ROCm disabled, AIS skipped | Configuration error |
| Nonempty `rocm_path`, HIP library or header missing | Configuration error | Configuration error |
| HIP found, hipFile library or header missing | AIS skipped | Configuration error |
| Valid ROCm installation, `disable_rocm_ais_backend=true` | AIS skipped | Configuration error |

HIP is detected independently of CUDA, so `AIS` can be built alongside the
CUDA plugins when both toolchains are present.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `mt` | Multi-threaded submission. Any value other than `mt` fails backend creation. |
| `thread_count` | `max(1, hardware_concurrency() / 2)` | Number of persistent Taskflow workers. A value of `0` uses the default. |

Each memory/file descriptor pair becomes one Taskflow task. If a blocking
`hipFileRead` or `hipFileWrite` transfers only part of the data, the worker
continues from the next byte. A zero-byte result with bytes outstanding fails
the transfer.

- **Open file descriptor** — put the fd in `devId` and leave `metaInfo` empty.
  The caller owns the fd. Registrations of the same open fd share one
  reference-counted hipFile handle.
- **Path** — put `<modes>:<path>` in `metaInfo`, for example
  `rw:/data/cache.bin` for an existing file. The backend opens and closes the
  file. Choose a unique `devId` for each active path registration on this
  backend, even for the same path. Reusing an active ID makes the AIS engine
  return `NIXL_ERR_INVALID_PARAM`. When AIS is the only backend tried,
  `nixlAgent::registerMem` reports this as `NIXL_ERR_BACKEND`. Deregistration
  releases the ID for reuse.

<Note>
AIS advertises `VRAM_SEG` and `FILE_SEG`. The engine rejects `DRAM_SEG` by
default. Its unadvertised `allow_dram` test option accepts a host descriptor
without registering its buffer with hipFile.

To let hipFile attempt I/O on an unrecognized scratch filesystem, set
`HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true`. The AIS benchmark disables hipFile's
POSIX compatibility path when exercising direct I/O.
</Note>

## When to Use

- **GPU direct storage on AMD** — read or write ROCm device buffers on
  AIS-capable storage.
- **Porting a GDS pipeline** — select `AIS` for local GPU-buffer and file
  transfers through the same NIXL API.

For the engine design, see the [plugin README](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/rocm_ais/README.md).
For a C++ example that writes and reads back GPU buffers, see the
[AIS transfer test](https://github.com/ai-dynamo/nixl/blob/main/test/unit/plugins/ais/nixl_ais_test.cpp).
