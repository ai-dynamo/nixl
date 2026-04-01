---
title: Device API Reference
description: CUDA device-side API for GPU-initiated data transfers in NIXL.
---

This is the CUDA device-side API reference for NIXL. The Device API enables GPU kernels to initiate and monitor data transfers directly from device code, without returning to the host. For the host-side C++ API, see the [C++ API Reference](./cpp-api).

To use the NIXL Device API, include the device header:

```cpp
#include "nixl_device.cuh"
```

## Types, Enums and Defines

### nixl_gpu_level_t

An enumeration of GPU execution levels for transfer requests.

| Value | Description |
|-------|-------------|
| `THREAD` | Thread-level transfer request (default) |
| `WARP` | Warp-level transfer request |
| `BLOCK` | Block-level transfer request |
| `GRID` | Grid-level transfer request |

The level template parameter controls the granularity of the transfer operation. Higher levels (WARP, BLOCK, GRID) can batch operations for better throughput but require all threads in the group to participate.

```cpp
// Thread-level (default)
nixlPut(src, dst, size, channel_id);

// Warp-level
nixlPut<nixl_gpu_level_t::WARP>(src, dst, size, channel_id);
```

### nixlGpuXferStatusH

Transfer status handle for tracking asynchronous operations.

| Field | Type | Description |
|-------|------|-------------|
| `device_request` | `ucp_device_request_t` | Internal UCX device request handle |

Pass a pointer to a `nixlGpuXferStatusH` to transfer functions, then query completion with `nixlGpuGetXferStatus`.

### nixlMemViewElem

Memory view element referencing a buffer in a prepared memory view.

| Field | Type | Description |
|-------|------|-------------|
| `mvh` | `nixlMemViewH` | Memory view handle (from host-side `nixlAgent::prepMemView`) |
| `index` | `size_t` | Index in the memory view |
| `offset` | `size_t` | Byte offset within the buffer |

Memory views must be prepared on the host using `nixlAgent::prepMemView` before device code can access them. See the [C++ API Reference](./cpp-api) for details.

## Flags

### nixl_gpu_flags

Namespace containing transfer flag constants and utilities.

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `defer` | `uint64_t` | `1` | Defer the transfer (do not send immediately). When set, the transfer is batched until the next non-deferred operation or explicit flush. |

#### to_ucp_flags

Device-side helper function that converts NIXL flags to internal UCP flags.

```cpp
__device__ inline uint64_t to_ucp_flags(uint64_t nixl_flags) noexcept;
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `nixl_flags` | `uint64_t` | NIXL transfer flags (combination of `nixl_gpu_flags` constants) |

Returns `uint64_t` -- Converted UCP flags for internal use.

<Note>
This is an internal helper. Application code typically passes flags directly to `nixlPut` or `nixlAtomicAdd`.
</Note>

## Functions

### nixlPut

Post a single-region memory transfer from local to remote GPU.

```cpp
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `const nixlMemViewElem &` | Source memory view element (local) |
| `dst` | `const nixlMemViewElem &` | Destination memory view element (remote) |
| `size` | `size_t` | Number of bytes to transfer |
| `channel_id` | `unsigned` | Channel ID for the transfer (default: `0`) |
| `flags` | `uint64_t` | Transfer flags from `nixl_gpu_flags` (default: `0`) |
| `xfer_status` | `nixlGpuXferStatusH *` | Optional status handle for tracking completion (default: `nullptr`) |

**Template parameter:** `level` -- GPU execution level (default: `nixl_gpu_level_t::THREAD`).

**Returns:**

| Value | Description |
|-------|-------------|
| `NIXL_IN_PROG` | Transfer posted successfully |
| `NIXL_ERR_BACKEND` | An error occurred in the UCX backend |

### nixlAtomicAdd

Atomic add to remote GPU memory.

```cpp
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `uint64_t` | Value to add to the remote counter |
| `counter` | `const nixlMemViewElem &` | Counter memory view element (remote) |
| `channel_id` | `unsigned` | Channel ID for the transfer (default: `0`) |
| `flags` | `uint64_t` | Transfer flags from `nixl_gpu_flags` (default: `0`) |
| `xfer_status` | `nixlGpuXferStatusH *` | Optional status handle for tracking completion (default: `nullptr`) |

**Template parameter:** `level` -- GPU execution level (default: `nixl_gpu_level_t::THREAD`).

**Returns:**

| Value | Description |
|-------|-------------|
| `NIXL_IN_PROG` | Atomic add posted successfully |
| `NIXL_ERR_BACKEND` | An error occurred in the UCX backend |

<Note>
The atomic increment is visible only after previous writes complete.
</Note>

### nixlGetPtr

Get a local pointer to remote memory.

```cpp
__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mvh` | `nixlMemViewH` | Memory view handle (remote buffers) |
| `index` | `size_t` | Index in the memory view |

Returns `void *` -- Pointer to the mapped memory, or `nullptr` if not available.

<Note>
The memory view must be prepared on the host using `nixlAgent::prepMemView`.
</Note>

### nixlGpuGetXferStatus

Get the status of a transfer request.

```cpp
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `xfer_status` | `nixlGpuXferStatusH &` | Status handle passed to a previous transfer function |

**Template parameter:** `level` -- GPU execution level (default: `nixl_gpu_level_t::THREAD`).

**Returns:**

| Value | Description |
|-------|-------------|
| `NIXL_SUCCESS` | The request has completed, no more operations are in progress |
| `NIXL_IN_PROG` | One or more operations in the request have not completed |
| `NIXL_ERR_BACKEND` | An error occurred in the UCX backend |
