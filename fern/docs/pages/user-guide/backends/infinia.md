---
title: INFINIA
description: DDN INFINIA backend for asynchronous transfers between DRAM or VRAM and object storage.
---

## Overview

The INFINIA backend connects NIXL to DDN INFINIA object storage. It uses the INFINIA Client and its C++20 coroutine-based asynchronous API to read and write objects directly from registered host or GPU memory.

| Property | Value |
|----------|-------|
| **Transfer Types** | DRAM ↔ Object, VRAM ↔ Object |
| **Technology** | DDN INFINIA Client and Async libraries |
| **Best For** | Checkpoints and other large-object workflows on DDN INFINIA |

## Installation

The backend requires an INFINIA installation containing these libraries and headers:

- `libred_client.so`
- `libred_async.so`
- `red/red_async.hpp`
- `red/red_status.h`

Configure NIXL with the path to the INFINIA installation:

```bash
meson setup build -Dinfinia_path=/path/to/infinia
meson compile -C build
```

The path must contain `lib/` and `include/` directories.

## Configuration

Create the backend with parameters passed to `createBackend("INFINIA", params)`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cluster` | INFINIA cluster name | `cluster1` |
| `tenant` | Tenant name | `red` |
| `subtenant` | Subtenant name | `red` |
| `dataset` | Dataset name | `nixl` |
| `sthreads` | Number of service threads | `8` |
| `num_buffers` | Number of operation buffers | `512` |
| `num_ring_entries` | Number of ring entries | `512` |
| `coremasks` | CPU affinity mask | `0x2` |
| `max_retries` | Maximum retries for failed operations | Library default |
| `config_file` | Path to a `key=value` configuration file | None |

The `RED_CLUSTER`, `RED_TENANT`, and `RED_DATASET` environment variables override backend parameters. `RED_TENANT` may include a subtenant as `tenant/subtenant`.

```cpp
nixl_b_params_t params = {
    {"cluster", "mycluster"},
    {"tenant", "mytenant"},
    {"subtenant", "mysubtenant"},
    {"dataset", "mydataset"},
};
agent.createBackend("INFINIA", params);
```

Alternatively, pass a configuration file:

```text
cluster=mycluster
tenant=mytenant
subtenant=mysubtenant
dataset=mydataset
sthreads=8
num_buffers=512
num_ring_entries=512
coremasks=0x2
```

```cpp
nixl_b_params_t params = {{"config_file", "/path/to/infinia.conf"}};
agent.createBackend("INFINIA", params);
```

## Memory Registration

Register local buffers as `DRAM_SEG` or `VRAM_SEG`. Register the remote object as `OBJ_SEG`; its descriptor `metaInfo` is used as the object key. If `metaInfo` is empty, the backend uses the object descriptor's device ID as the key.

Transfers are asynchronous. Poll them with `checkXfer`, and release completed request handles with `releaseReqH`.
