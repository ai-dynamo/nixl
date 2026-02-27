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

## NIXL Device API V2 Core Functions

The 2proc example demonstrates the modern Device API V2:

### 1. `nixlPut<LEVEL>(src_desc, dst_desc, size, ...)`
Posts a GPU-initiated RDMA write from GPU thread(s).

```cuda
template<nixl_gpu_level_t LEVEL>
void nixlPut(
    nixlMemDesc src_desc,    // Source memory descriptor
    nixlMemDesc dst_desc,    // Destination memory descriptor
    size_t size,             // Transfer size (bytes)
    uint32_t channel_id,     // Channel ID (usually 0)
    unsigned flags,          // Operation flags (e.g., NO_DELAY)
    void *status             // Optional status pointer
);
```

**What it does**: Initiates an RDMA write from local GPU memory to remote GPU memory.

**Memory Descriptor** (`nixlMemDesc`):
```cuda
struct nixlMemDesc {
    void *mvh;        // Memory view handle (from prepMemoryView)
    size_t index;     // Buffer index
    size_t offset;    // Offset within buffer
};
```

**Cooperation Levels** (`LEVEL`):
- `nixl_gpu_level_t::THREAD` - Single thread posts the request
- `nixl_gpu_level_t::WARP` - All 32 threads in a warp cooperate

### 2. `nixlAtomicAdd<LEVEL>(value, counter_desc, ...)`
Posts a GPU-initiated atomic add to remote memory.

```cuda
template<nixl_gpu_level_t LEVEL>
void nixlAtomicAdd(
    uint64_t value,          // Value to add
    nixlMemDesc counter_desc, // Remote counter descriptor
    uint32_t channel_id,     // Channel ID (usually 0)
    unsigned flags,          // Operation flags (e.g., NO_DELAY)
    void *status             // Optional status pointer
);
```

**What it does**: Atomically adds a value to a counter in remote GPU memory, typically used to signal completion.

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
__global__ void thread_level_kernel(nixlMemDesc src_desc, nixlMemDesc dst_desc,
                                     nixlMemDesc signal_desc, size_t size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Single thread posts write
        nixlPut<nixl_gpu_level_t::THREAD>(
            src_desc, dst_desc, size, 0,
            static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr
        );
        // Single thread posts signal
        nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
            1, signal_desc, 0,
            static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr
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
__global__ void warp_level_kernel(nixlMemDesc src_desc, nixlMemDesc dst_desc,
                                   nixlMemDesc signal_desc, size_t size) {
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        unsigned lane = threadIdx.x;
        // All 32 threads cooperate on write
        nixlPut<nixl_gpu_level_t::WARP>(
            src_desc, dst_desc, size, 0,
            static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr
        );
        // Lane 0 posts signal
        if (lane == 0) {
            nixlAtomicAdd<nixl_gpu_level_t::WARP>(
                1, signal_desc, 0,
                static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr
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

The initiator sends data and signals completion using Device API V2:

```cuda
__global__ void post_write_and_signal_kernel_warp(
    const nixlMemDesc *src_descs,
    const nixlMemDesc *dst_descs,
    nixlMemDesc signal_desc,
    size_t size
) {
    if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) {
        unsigned lane = threadIdx.x;

        // Post the write
        nixlPut<nixl_gpu_level_t::WARP>(
            src_descs[0], dst_descs[0], size, 0,
            static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr
        );

        // Signal completion (lane 0 only)
        if (lane == 0) {
            nixlAtomicAdd<nixl_gpu_level_t::WARP>(
                1, signal_desc, 0,
                static_cast<unsigned>(nixl_gpu_flags_t::NO_DELAY), nullptr
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

## Device API V2 Setup Flow

Before kernels can use these functions, the host must prepare:

1. **Initialize NIXL Agent**
   ```cpp
   nixlAgent agent("name", config);
   agent.createBackend("UCX", params, backend);
   ```

2. **Register GPU memory**
   ```cpp
   nixl_reg_dlist_t reg(VRAM_SEG);
   reg.addDesc(nixlBlobDesc((uintptr_t)ptr, size, device_id, ""));
   agent.registerMem(reg, &params);
   ```

3. **Exchange metadata with peer**
   ```cpp
   std::string local_meta;
   agent.getLocalMD(local_meta);
   // Exchange via TCP/etcd/other mechanism
   agent.loadRemoteMD(remote_meta, remote_name);
   ```

4. **Prepare memory views**
   ```cpp
   nixlMemoryViewH local_mvh = nullptr;
   agent.prepMemoryView(local_descs, local_mvh, &params);

   nixlMemoryViewH remote_mvh = nullptr;
   agent.prepMemoryView(remote_descs, remote_mvh, &params);
   ```

5. **Create memory descriptors**
   ```cpp
   nixlMemDesc src_desc{local_mvh, 0, 0};
   nixlMemDesc dst_desc{remote_mvh, 0, 0};
   ```

6. **Launch kernel with descriptors**
   ```cpp
   kernel<<<...>>>(src_desc, dst_desc, size, ...)
   ```

**Why this setup?** The host establishes connections and prepares memory view handles, then creates descriptors that GPU threads can use directly. Once set up, GPU can post transfers without host involvement.

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

**Key insight**: EP uses the same Device API V2 functions (`nixlPut`, `nixlAtomicAdd`) you see in 2proc, just orchestrated into more complex communication patterns for MoE workloads.

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
1. **GPU-initiated RDMA writes** via `nixlPut`
2. **GPU-initiated signaling** via `nixlAtomicAdd`
3. **GPU-side synchronization** via `nixlGpuReadSignal`
4. **Thread vs Warp cooperation** patterns
5. **Write-and-signal** coordination primitive

These are the building blocks for GPU-to-GPU communication without CPU involvement. Master these patterns to build efficient distributed GPU applications.

## Further Reading

- **NIXL documentation**: [NIXL repository](https://github.com/ai-dynamo/nixl)
- **EP README**: `../ep/README.md` - See device API in production MoE context
- **Device host API**: `csrc/device_host.cpp` - Host-side setup details
- **Kernel source**: `csrc/kernels.cu` - Full implementation (~115 lines)
