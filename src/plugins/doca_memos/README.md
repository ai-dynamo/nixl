<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL DOCA_MEMOS Plugin

A NIXL backend plugin that provides hardware-accelerated key-value storage operations using NVIDIA's DOCA KV library (`libdoca_kv`, headers `doca_kvdev.h`, `doca_kvdev_io.h`, `doca_nvme_kernel_kvdev.h`, `doca_nvme_kernel_kvdev_io.h`) for BlueField devices. The NIXL backend name is **DOCA_MEMOS**; the underlying DOCA package may be renamed in the future.

## Overview

The DOCA_MEMOS plugin enables high-performance key-value operations (store, retrieve, exist) on BlueField devices through NVIDIA's DOCA framework.

## Requirements

- **DOCA SDK**: Version 4.4 or later
- **Hardware**: NVIDIA BlueField-3 DPU or newer
- **Libraries**:
  - `libdoca_kv` - DOCA Key-Value library
  - `libdoca_common` - DOCA Core library
  - `libdoca_nvme_kernel_kvdev` - DOCA NVMe kernel device support

## Configuration

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `device_name` | Path to the NVMe KV device | `/dev/nvme0n1` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_tasks` | Maximum number of concurrent tasks in the DOCA task pool (clamped to `doca_kvdev_get_max_tasks()`) | `8192` |
| `nguid` | 32-character hex NVMe namespace NGUID passed to `doca_kvdev_set_nguid()` | all zeros |
| `query_mem_mode` | Controls `queryMem()` behavior: `assume_success` returns success for every descriptor without querying the device; `actual` issues real DOCA EXIST operations | `assume_success` |
| `ignore_read_not_found` | When `true`, retrieve operations on missing keys complete successfully (buffer contents are undefined) instead of failing with `NIXL_ERR_BACKEND` | `false` |

### Example Configuration

```cpp
#include "nixl.h"

// Create DOCA_MEMOS backend with required device_name
nixl_b_params_t params = {
    {"device_name", "/dev/nvme0n1"},  // Required
    {"num_tasks", "256"}               // Optional
};
agent.createBackend("DOCA_MEMOS", params);
```

## Supported Operations

### Memory Registration

- **DRAM_SEG**: Local system memory buffers for data transfer
- **OBJ_SEG**: Key-value pairs stored on the NVMe KV device
  - If `metaInfo` is a valid hex string of up to `2 * DOCA_MEMOS_MAX_OBJECT_KEY_LEN` (32) characters, it is decoded into the object key
  - Otherwise, if `metaInfo` is non-empty and at most `DOCA_MEMOS_MAX_OBJECT_KEY_LEN` (16) bytes, it is used as the raw key
  - If `metaInfo` is empty, the raw bytes of `devId` (`sizeof(uint64_t)`) are used as the key

### Transfer Operations

- **NIXL_WRITE**: Store data to a key (DOCA STORE operation)
- **NIXL_READ**: Retrieve data from a key (DOCA RETRIEVE operation)

### Query Operations

- **queryMem()**: Check if a key exists
  - In `assume_success` mode (default): immediately returns success for every descriptor without contacting the device
  - In `actual` mode: issues real DOCA EXIST operations; returns success if the key exists, nullopt if not found or on error

## Architecture

### DOCA Resources

The plugin manages the following DOCA resources:

- **Progress Engine (PE)**: Handles task submission and completion polling
- **NVMe Kernel KV Device (nvmeKvdev)**: Backend-specific NVMe-kernel KV device (path + NGUID configured before start)
- **Generic KV Device (kvdev)**: View of the NVMe kernel KV device through the `doca_kvdev` API for capability queries
- **NVMe Kernel KV IO Context (kkvIo) / Generic KV IO Context (kvIo)**: Per-engine I/O context; configured via `doca_kvdev_io_set_num_tasks()`, `doca_kvdev_io_set_task_completion_cb()`, and `doca_kvdev_io_set_task_error_cb()` before start
- **Context (ctx)**: DOCA context connected to the progress engine

All DOCA resources are initialized in the constructor and cleaned up in the destructor following RAII principles.

### Threading Model

The plugin selects one of two progress-engine implementations based on `enableProgTh`:

#### With Progress Thread (`enableProgTh = true`)

A dedicated background thread is the sole caller of DOCA APIs. Other threads submit transfers, cancel requests, and post queries by pushing entries into a producer queue guarded by a short-lived mutex held only for a pointer/vector swap. The progress thread drains the queue and submits tasks to DOCA outside the lock, keeping the caller-facing hot path effectively lock-free.

#### Without Progress Thread (`enableProgTh = false`)

No background thread is created. Progress is driven by the caller inside `checkXfer()` and `queryMem()`. A single mutex serializes all DOCA calls (submission, progress polling, cancellation) so the engine remains safe under concurrent callers; callers that can tolerate a dedicated thread should prefer the threaded mode for lower contention.

### Request Handling

Each NIXL transfer or query is represented by an opaque `nixlDocaMemosBackendReqH`. The handle aggregates the status of every DOCA task it spawned: callers observe completion through `checkXfer()`, which returns `NIXL_IN_PROG` until every submitted task has reported a result, then returns either `NIXL_SUCCESS` or the first non-success status seen. Completion is published with release ordering, so the caller does not need any external locking to read it.

Failures are sticky: once any task reports a hard error, the request stays in that error state until released, even if later tasks succeed.

### Error Handling

The plugin uses fail-fast error handling for batch operations:

- **On first error**: Stops submitting new tasks immediately
- **In-flight tasks**: Waits for already-submitted tasks to complete
- **Overall status**: The entire batch operation is marked as failed
- **Rationale**: NIXL has no API to report per-task status, so partial success cannot be conveyed

## API Behavior

### registerMem()

Registers memory with the plugin. For OBJ_SEG memory the resolved key (see [Memory Registration](#memory-registration)) is stored in the returned metadata object and retrieved by `postXfer()`. No DOCA call is made.

### prepXfer()

Creates a request handle with the expected number of tasks. No DOCA operations are performed.

### postXfer()

Submits DOCA tasks for the transfer operation. In threaded mode, `postXfer()` pushes an entry onto the producer queue and returns immediately; the progress thread performs the allocation and submission. In no-thread mode, `postXfer()` holds the engine mutex and drives submission inline. If the DOCA task pool is full, the remaining descriptors are queued internally and retried — by the progress thread in threaded mode, or by the next caller-driven `checkXfer()` / `queryMem()` in no-thread mode.

If any descriptor hits a non-recoverable error during submission, no further descriptors are submitted for that request and the overall status is set to `NIXL_ERR_BACKEND` once the in-flight tasks drain.

Each task's `task_user_data` is a per-descriptor `docaMemosTaskContext` pointing back at the request handle.

### checkXfer()

Reports whether a transfer has completed. In no-thread mode it also polls the DOCA progress engine and retries any submissions that were previously queued because the task pool was full. Returns `NIXL_IN_PROG` while tasks remain outstanding, `NIXL_SUCCESS` once every task has completed without error, or the first error status observed otherwise.

### queryMem()

Behavior depends on the `query_mem_mode` configuration parameter:

**`assume_success` mode (default):**
Returns success for every descriptor immediately, without contacting the device. Useful when the caller already knows that keys exist (e.g., because it just wrote them) and wants to avoid the latency of actual EXIST operations.

**`actual` mode:**
Synchronously checks if keys exist on the device by submitting EXIST tasks for each key and busy-polling (`std::this_thread::yield()`) until completion. Returns a vector with success (key exists) or nullopt (key missing or error).

### releaseReqH()

Releases the request handle. If any tasks are still in flight, the handle is marked cancelled and deletion is deferred until the last callback fires. It is safe to call regardless of whether the transfer has completed.

### deregisterMem()

Releases the metadata object returned by `registerMem()`. Does not delete data from the device.

## Usage Example

```cpp
#include "nixl.h"

// Create agent
nixl_agent_config config;
config.enable_prog_thread = true;  // Enable progress thread
nixl_agent agent("kv_agent", config);

// Create DOCA KV backend
nixl_b_params_t params = {{"device_name", "/dev/nvme0n1"}};
agent.createBackend("DOCA_MEMOS", params);

// Register OBJ_SEG memory with a key (metaInfo must be <= 16 raw bytes or
// a hex string of <= 32 chars; see "Memory Registration" above)
nixlBlobDesc remote_desc;
remote_desc.devId = 100;
remote_desc.metaInfo = "ckpt_0001";
agent.registerMem(remote_desc, OBJ_SEG);

// Register DRAM_SEG memory for local buffer
char buffer[4096] = "checkpoint data";
nixlBlobDesc local_desc;
local_desc.devId = 0;
local_desc.baseAddr = reinterpret_cast<uint64_t>(buffer);
local_desc.length = 4096;
agent.registerMem(local_desc, DRAM_SEG);

// Create descriptor lists
auto local_dlist = agent.createDlist({local_desc}, DRAM_SEG);
auto remote_dlist = agent.createDlist({remote_desc}, OBJ_SEG);

// Write data to key
auto write_handle = agent.postXfer(NIXL_WRITE, local_dlist, remote_dlist, "kv_agent");
while (agent.checkXfer(write_handle) == NIXL_IN_PROG) {
    // Wait for completion
}
agent.releaseReqH(write_handle);

// Check if key exists
std::vector<nixl_query_resp_t> query_results;
auto query_dlist = agent.createDlist({remote_desc}, OBJ_SEG);
nixl_status_t status = agent.queryMem(query_dlist, query_results);
if (status == NIXL_SUCCESS && query_results[0].has_value()) {
    std::cout << "Key exists!" << std::endl;
}

// Read data from key
char read_buffer[4096] = {0};
local_desc.baseAddr = reinterpret_cast<uint64_t>(read_buffer);
local_dlist = agent.createDlist({local_desc}, DRAM_SEG);

auto read_handle = agent.postXfer(NIXL_READ, local_dlist, remote_dlist, "kv_agent");
while (agent.checkXfer(read_handle) == NIXL_IN_PROG) {
    // Wait for completion
}
agent.releaseReqH(read_handle);

// Cleanup
agent.deregisterMem(remote_desc);
agent.deregisterMem(local_desc);
```

## Performance Considerations

### Task Pool Size

The `num_tasks` parameter controls the DOCA task pool size:
- Larger values support more concurrent operations
- Smaller values reduce memory overhead
- Default of 8192 is suitable for most workloads
- Set based on expected maximum in-flight operations

### Progress Thread vs. Manual Polling

**Use progress thread when:**
- Running a multi-threaded application
- Submitting operations from multiple threads
- Need low-latency completions without explicit polling

**Disable progress thread when:**
- Running a single-threaded application
- Want explicit control over when I/O progresses
- Building an async runtime with custom event loop
- Minimizing thread count and lock contention

### Key Naming

- Keys are at most 16 bytes (`DOCA_MEMOS_MAX_OBJECT_KEY_LEN`)
- Pass hex strings for explicit binary keys, or short ASCII strings to use the raw-bytes path

## Limitations

- **No notifications**: The plugin does not support NIXL's notification mechanism
- **Single progress engine**: All operations share one progress engine, which may limit scalability under extreme concurrency

## Thread Safety

- All public API methods are thread-safe in both progress-engine modes
- With progress thread: DOCA calls run only on the progress thread; other threads communicate through a queue guarded by a short-held mutex
- Without progress thread: DOCA calls are serialized by a mutex held across submission, progress, and cancellation
- Request-handle completion state uses atomic operations, allowing `checkXfer()` to poll without acquiring the engine mutex

## Troubleshooting

### "Failed to create DOCA NVMe kernel KV device"

- Verify the device path is correct (`ls /dev/nvme*`)
- Ensure the device supports KV operations (BlueField-3 or newer)
- Check device permissions

### "Failed to allocate DOCA KV task"

- Task pool is exhausted (all tasks in use)
- Increase `num_tasks` parameter
- Ensure tasks are being completed and handles released

### Timeout in queryMem

- Progress engine is not being polled (if no progress thread)
- Device is unresponsive or overloaded
- Check DOCA logs for hardware issues
