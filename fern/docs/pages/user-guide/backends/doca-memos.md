---
title: DOCA_MEMOS
description: DOCA_MEMOS backend for key-value transfers between DRAM and an NVMe KV device on BlueField.
---

## Overview

The DOCA_MEMOS backend stores and retrieves values on an NVMe key-value device using the NVIDIA DOCA KV library. Each `OBJ_SEG` descriptor names one key, and each transfer moves one whole value between that key and a `DRAM_SEG` buffer.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ Object |
| **Technology** | DOCA KV over the kernel NVMe driver |
| **Best For** | KV cache offload to BlueField-attached KV storage |

## Installation

### Prerequisites

- **DOCA SDK** -- Version 4.4 or later, from the [NVIDIA DOCA SDK](https://developer.nvidia.com/doca-downloads) page. The plugin links `libdoca_kv` and `libdoca_common`.
- **Hardware** -- NVIDIA BlueField-3 or newer, exposing an NVMe KV namespace.
- **libb2** -- Blake2 hashing, used by the optional `convert_key_to_128bit` key mode.
- **Hugepages** -- `DRAM_SEG` buffers must be backed by hugepages. The device receives buffers through the kernel NVMe driver by user address, and a value spread across 4 KiB pages can exceed the device's per-command segment limit. Reserve 2 MiB hugepages before starting the application, for example `sysctl vm.nr_hugepages=1024`, and allocate buffers with `mmap(MAP_HUGETLB)`.

Meson finds DOCA through `pkg-config` (`doca-kv`, `doca-common`). Point `PKG_CONFIG_PATH` at the DOCA install if it isn't on the default search path. When a dependency is missing the plugin is skipped, or, if you enabled it explicitly with `-Denable_plugins=DOCA_MEMOS`, configuration fails.

## Configuration

Create the backend with parameters passed to `createBackend("DOCA_MEMOS", params)`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `device_name` | Path to the NVMe KV device. Required. | None |
| `num_tasks` | DOCA task pool size, clamped to the device maximum | `8192` |
| `nguid` | 32-character hex NVMe namespace NGUID | All zeros |
| `convert_key_to_128bit` | Hash `metaInfo` strings to 128-bit keys instead of parsing them as hex | `false` |

```cpp
nixl_b_params_t params = {
    {"device_name", "/dev/nvme0n1"},
    {"num_tasks", "256"},
};
agent.createBackend("DOCA_MEMOS", params);
```

## Keys

Device keys are at most 16 bytes. The backend derives the key for an `OBJ_SEG` descriptor from its `metaInfo`:

- **Empty `metaInfo`** -- The raw bytes of `devId` are the key. `devId == 0` is a valid key.
- **Default mode** -- `metaInfo` must be a hex string of up to 32 characters, which is decoded into the key. Any other string is rejected at registration.
- **`convert_key_to_128bit=true`** -- Every non-empty `metaInfo` is hashed with unkeyed 16-byte Blake2b. Strings that look like hex are hashed, not decoded. Use this mode when application object names are longer than 16 bytes.

```cpp
nixlBlobDesc remote_desc;
remote_desc.devId = 0;
remote_desc.metaInfo = "636b70745f30303031";
agent.registerMem(remote_desc, OBJ_SEG);
```

## Transfers

- **`NIXL_WRITE`** stores the local buffer as the key's value.
- **`NIXL_READ`** retrieves the key's value. The read fails with `NIXL_ERR_BACKEND` if the key is missing or if the stored value's length differs from the local buffer length. Read with the same length that was written.
- **`queryMem()`** issues an EXIST per descriptor. A response is set if the key exists and `nullopt` if it doesn't.

A transfer that hits an error stops submitting its remaining descriptors and reports the first error once in-flight descriptors complete. The backend does not support notifications.

## Threading

With the agent's progress thread enabled, a dedicated thread makes every DOCA call and other threads hand it work through a queue. Without it, `checkXfer()` and `queryMem()` drive progress on the calling thread, serialized by a lock that follows the agent's `syncMode`.

See the [plugin README](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/doca_memos/README.md) for the full API behavior and a complete usage example.
