# NIXL Device API Kernel Guide

This guide explains the GPU kernel patterns used in the 2proc example and how they demonstrate the NIXL device API for GPU-initiated communication.

## What is GPU-Initiated Communication?

Traditionally, GPUs rely on the CPU to orchestrate data transfers:
- **CPU-initiated**: Host calls `cudaMemcpy()` or UCX APIs to move data between GPUs
- **GPU-initiated**: GPU threads directly trigger RDMA transfers using the NIXL device API

**Why use GPU-initiated transfers?**
- **Lower latency**: GPU can initiate transfer the moment it's ready, without round-trip to CPU
- **Better overlap**: GPU kernel continues while transfer happens in background
- **Finer-grained control**: Different GPU threads can initiate different transfers
- **Simplified programming**: No need to synchronize back to CPU just to start a transfer

**When to use GPU-initiated transfers:**
- GPU produces data that another GPU needs immediately
- Communication is intermixed with computation
- Want to overlap communication with computation
- Building collective operations or complex communication patterns

The NIXL device API exposes GPU-side functions that allow CUDA kernels to directly post RDMA operations.

## NIXL Device API Core Functions

The 2proc example demonstrates these key device API functions:

### 1. `nixlGpuPostSingleWriteXferReq<LEVEL>(req, ...)`
Posts a GPU-initiated RDMA write from GPU thread(s).

```cuda
template<nixl_gpu_level_t LEVEL>
void nixlGpuPostSingleWriteXferReq(
    nixlGpuXferReqH req,     // GPU transfer request handle
    uint32_t dst_offset,     // Destination offset (bytes)
    uint32_t src_offset,     // Source offset (bytes)
    uint32_t dst_rkey_idx,   // Destination remote key index
    size_t size,             // Transfer size (bytes)
    uint32_t flags,          // Operation flags
    bool wait_completion     // Wait for completion
);
```

**What it does**: Initiates an RDMA write from GPU memory to remote GPU memory.

**Cooperation Levels** (`LEVEL`):
- `nixl_gpu_level_t::THREAD` - Single thread posts the request
- `nixl_gpu_level_t::WARP` - All 32 threads in a warp cooperate to post

### 2. `nixlGpuPostSignalXferReq<LEVEL>(req, ...)`
Posts a GPU-initiated signal (atomic increment) to remote memory.

```cuda
template<nixl_gpu_level_t LEVEL>
void nixlGpuPostSignalXferReq(
    nixlGpuXferReqH req,     // Signal request handle
    uint32_t offset,         // Signal memory offset
    uint64_t value,          // Increment value
    uint32_t dst_rkey_idx,   // Destination remote key index
    uint32_t flags,          // Operation flags
    bool wait_completion     // Wait for completion
);
```

**What it does**: Atomically increments a counter in remote GPU memory, typically used to signal completion.

### 3. `nixlGpuReadSignal<LEVEL>(signal_ptr)`
Reads a local signal value from GPU memory.

```cuda
template<nixl_gpu_level_t LEVEL>
uint64_t nixlGpuReadSignal(const void* signal_ptr);
```

**What it does**: Reads a signal value, typically used in a busy-wait loop to wait for remote operations to complete.

## Thread vs Warp Cooperation Levels

The device API supports two cooperation patterns for GPU threads.

### THREAD Level
**One thread** handles the entire operation independently.

```cuda
__global__ void thread_level_kernel(...) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Single thread posts write
        nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::THREAD>(
            data_req, 0, 0, 0, size, 0, true
        );
        // Single thread posts signal
        nixlGpuPostSignalXferReq<nixl_gpu_level_t::THREAD>(
            signal_req, 0, 1, 0, 0, true
        );
    }
}
```

**When to use**:
- Simple patterns with one transfer
- Prototyping and debugging
- When only one thread has the data or decision to transfer

### WARP Level
**All 32 threads in a warp** cooperate on the operation.

```cuda
__global__ void warp_level_kernel(...) {
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        unsigned lane = threadIdx.x;
        // All 32 threads cooperate on write
        nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::WARP>(
            data_req, 0, 0, 0, size, 0, true
        );
        // Lane 0 posts signal
        if (lane == 0) {
            nixlGpuPostSignalXferReq<nixl_gpu_level_t::WARP>(
                signal_req, 0, 1, 0, 0, true
            );
        }
    }
}
```

**When to use**:
- Multiple threads naturally working together
- Matching CUDA's warp-based execution model
- When you have multiple handles (one per lane) for parallel transfers

**Key difference**: WARP level requires all 32 threads in a warp to call the function together, while THREAD level can be called by a single thread.

## 2proc Kernel Patterns

The 2proc example demonstrates the fundamental write-and-signal pattern.

### Pattern 1: Write + Signal (Initiator Side)

The initiator sends data and signals completion:

```cuda
__global__ void post_write_and_signal_kernel_warp(
    uintptr_t data_req_handles_ptr,
    int data_req_count,
    uintptr_t signal_req_handle,
    uint64_t *signal_ptr,
    size_t size,
    uint64_t signal_inc,
    int pipeline
) {
    if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) {
        auto *data_req_handles =
            reinterpret_cast<const uintptr_t *>(data_req_handles_ptr);
        nixlGpuXferReqH data_req =
            reinterpret_cast<nixlGpuXferReqH>(data_req_handles[0]);
        nixlGpuXferReqH signal_req =
            reinterpret_cast<nixlGpuXferReqH>(signal_req_handle);

        unsigned lane = threadIdx.x;

        // Post the write
        nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::WARP>(
            data_req, 0, 0, 0, size, 0, true
        );

        // Signal completion (lane 0 only)
        if (lane == 0) {
            nixlGpuPostSignalXferReq<nixl_gpu_level_t::WARP>(
                signal_req, 0, signal_inc, 0, 0, true
            );
        }
    }
}
```

**Key points**:
- Data write happens first
- Signal increment happens second (tells receiver "data is ready")
- Only one lane posts the signal to avoid multiple increments
- Target can wait on signal value to know when data arrives

**Why this pattern**: Provides synchronization without CPU involvement. The signal acts as a completion notification.

### Pattern 2: Signal Wait (Target Side)

The target waits for the signal to reach expected value:

```cuda
__global__ void wait_for_signal_kernel_thread(
    const void* signal_ptr,
    uint64_t expected_value
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Busy-wait until signal reaches expected value
        while (nixlGpuReadSignal<nixl_gpu_level_t::THREAD>(signal_ptr)
               < expected_value) {
            // GPU polling - no CPU involvement
        }
    }
}
```

**Key points**:
- GPU actively polls signal memory
- Loop continues until remote write + signal completes
- No CPU wakeup or notification needed
- GPU can do other work in parallel (multi-stream)

**Why this pattern**: Enables GPU-to-GPU synchronization without going through the CPU.

## The Write-and-Signal Pattern

This is the fundamental coordination pattern in GPU-initiated communication:

```
Initiator GPU:                    Target GPU:
1. Write data         -------->   [data buffer]
2. Increment signal   -------->   [signal memory]
                                  3. Wait for signal
                                  4. Use data
```

**Why separate write and signal?**
- **Ordering**: Signal ensures write is visible before use
- **Notification**: Target knows when new data is available
- **Flexibility**: Can batch multiple writes per signal
- **Atomicity**: Signal provides atomic counter for tracking multiple operations

**Real-world usage**: This pattern is used in MoE models (like NIXL EP) where GPUs exchange expert activations without CPU coordination.

## Device API Setup Flow

Before kernels can use these functions, the host must prepare:

1. **Initialize DeviceHost**
   ```python
   host = device_host.DeviceHost("name", "UCX", port)
   ```

2. **Register GPU memory**
   ```python
   reg_id = host.register_vram(ptr, size, device_id)
   ```

3. **Exchange metadata with peer**
   ```python
   local_meta = host.get_metadata()
   # Exchange via TCP/etcd/other mechanism
   peer_name = host.add_remote_agent(remote_meta)
   ```

4. **Create transfer descriptors**
   ```python
   descs = host.serialize_xfer_descs(ptr, size, device_id)
   # Send to peer via out-of-band mechanism
   ```

5. **Create GPU request handles**
   ```python
   req = host.create_write_req(ptr, size, device_id, remote_descs, peer_name)
   gpu_req = host.create_gpu_req(req)  # Handle usable in kernel
   ```

6. **Launch kernel with GPU handle**
   ```python
   kernel<<<...>>>(gpu_req, ...)
   ```

**Why this setup?** The host establishes connections and prepares metadata, then gives GPU threads handles they can use directly. Once set up, GPU can post transfers without host involvement.

## From 2proc to EP

The 2proc example shows **device API building blocks**. The EP framework builds on these for production use:

### 2proc (Building Blocks)
- **Purpose**: Teach device API patterns
- **Pattern**: Simple write + signal
- **Topology**: Two processes, fixed
- **Use case**: Learning and prototyping

### EP (Production Framework)
- **Purpose**: MoE expert-parallel communication
- **Pattern**: Complex dispatch/combine with token routing
- **Topology**: Dynamic ranks with elastic scaling
- **Use case**: Production inference workloads

**Learning path**:
1. Understand 2proc kernels (basic patterns)
2. Understand how patterns compose (EP framework)
3. Build your own device API applications

**Key insight**: EP uses the same `nixlGpuPost*()` functions you see in 2proc, just orchestrated into more complex communication patterns for MoE workloads.

## Common Use Cases for Device API

**When device API is a good fit:**
- **MoE models**: Expert routing between GPUs (like NIXL EP)
- **Custom collectives**: Building allreduce, allgather with GPU control
- **Streaming pipelines**: GPU produces data for next GPU in pipeline
- **Dynamic communication**: Transfer patterns depend on computation results
- **Low-latency communication**: Avoiding CPU round-trip overhead

**When host-initiated API might be better:**
- Static communication patterns known ahead of time
- CPU controls the overall workflow
- Mixed CPU/GPU memory transfers
- Simpler programming model is preferred

## Summary

The 2proc kernels demonstrate:
1. **GPU-initiated RDMA writes** via `nixlGpuPostSingleWriteXferReq`
2. **GPU-initiated signaling** via `nixlGpuPostSignalXferReq`
3. **GPU-side synchronization** via `nixlGpuReadSignal`
4. **Thread vs Warp cooperation** patterns
5. **Write-and-signal** coordination primitive

These are the building blocks for GPU-to-GPU communication without CPU involvement. Master these patterns to build efficient distributed GPU applications.

## Further Reading

- **NIXL documentation**: [NIXL repository](https://github.com/ai-dynamo/nixl)
- **EP README**: `../ep/README.md` - See device API in production MoE context
- **Device host API**: `csrc/device_host.cpp` - Host-side setup details
- **Kernel source**: `csrc/kernels.cu` - Full implementation (~115 lines)
