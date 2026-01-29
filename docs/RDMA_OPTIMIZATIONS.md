# NIXL RDMA Performance Optimizations

This document describes the RDMA performance optimizations implemented for the NIXL libfabric backend, based on research from Kalia et al. USENIX ATC'16, AWS EFA best practices, and the KVDirect paper.

## Overview

Three major optimizations have been implemented:

1. **FI_INJECT for Small Messages** - Inline send optimization
2. **Doorbell Batching** - Batch RDMA operations to reduce PCIe overhead
3. **Transfer Coalescing** - Adaptive rail selection for efficient transfers

## 1. FI_INJECT for Small Messages

### Branch
`perf/fi-inject-small-messages`

### Docker Image
```
public.ecr.aws/v9l4g5s4/nixl-efa-dev:fi-inject-2026-01-16
```

### Description
Uses `fi_injectdata()` for small messages that fit within the provider's inject size limit. This optimization:
- Eliminates completion tracking overhead for small sends
- Reduces memory registration requirements
- Provides immediate send completion

### Files Modified
- `src/utils/libfabric/libfabric_rail.cpp` - Added inject path in `postSend()`
- `src/utils/libfabric/libfabric_rail.h` - Added `inject_size_` member

### How It Works
```cpp
// During rail initialization
inject_size_ = info->tx_attr->inject_size;

// In postSend()
if (req->buffer_size <= inject_size_) {
    ret = fi_injectdata(endpoint, req->buffer, req->buffer_size, immediate_data, dest_addr);
    // No completion tracking needed - operation completes inline
}
```

### Expected Benefits
- **Latency**: 10-20% reduction for small messages (< 256 bytes)
- **CPU**: Reduced completion queue polling overhead
- **Best for**: Control messages, metadata exchanges, notifications

---

## 2. Doorbell Batching

### Branch
`perf/doorbell-batching`

### Docker Image
```
public.ecr.aws/v9l4g5s4/nixl-efa-dev:doorbell-batch-2026-01-16
```

### Description
Uses `FI_MORE` flag to batch multiple RDMA operations before ringing the doorbell. Per Kalia et al. USENIX ATC'16, doorbell batching can improve throughput by up to 2x by reducing PCIe round-trips to the NIC.

### Files Modified
- `src/utils/libfabric/libfabric_rail.cpp` - Modified `postWrite()` and `postRead()` to accept `more` parameter
- `src/utils/libfabric/libfabric_rail.h` - Updated function signatures
- `src/utils/libfabric/libfabric_rail_manager.cpp` - Modified striping loop to use `more=true` for non-final operations

### How It Works
```cpp
// In striping loop
bool is_last_operation = (i == num_rails - 1);
bool use_more = !is_last_operation && (num_rails > 1);

// In postWrite()
if (more) {
    struct fi_msg_rma msg = { ... };
    ret = fi_writemsg(endpoint, &msg, FI_MORE | FI_REMOTE_CQ_DATA);
    // Doorbell deferred until next operation without FI_MORE
} else {
    ret = fi_writedata(endpoint, ...);  // Rings doorbell immediately
}
```

### Expected Benefits
- **Throughput**: Up to 2x improvement for multi-rail striping
- **CPU**: Reduced PCIe round-trips per batch
- **Best for**: Large transfers striped across multiple rails

---

## 3. Transfer Coalescing (Adaptive Rail Selection)

### Branch
`perf/transfer-coalescing`

### Docker Image
```
public.ecr.aws/v9l4g5s4/nixl-efa-dev:transfer-coalesce-2026-01-16
```

### Description
Dynamically reduces the number of rails used when striping would result in chunks smaller than the minimum efficient size (16KB). This trades parallelism for larger, more efficient operations.

### Files Modified
- `src/utils/libfabric/libfabric_common.h` - Added `NIXL_LIBFABRIC_MIN_CHUNK_SIZE` constant
- `src/utils/libfabric/libfabric_rail_manager.cpp` - Added adaptive rail selection logic

### How It Works
```cpp
#define NIXL_LIBFABRIC_MIN_CHUNK_SIZE (16 * 1024)  // 16KB

// In prepareAndSubmitTransfer()
size_t ideal_chunk_size = transfer_size / num_rails;
if (ideal_chunk_size < NIXL_LIBFABRIC_MIN_CHUNK_SIZE && num_rails > 1) {
    size_t optimal_rails = transfer_size / NIXL_LIBFABRIC_MIN_CHUNK_SIZE;
    if (optimal_rails == 0) optimal_rails = 1;
    if (optimal_rails < num_rails) {
        num_rails = optimal_rails;  // Reduce rails to maintain larger chunks
    }
}
```

### Example
| Transfer Size | Original Rails | Chunk Size | Optimized Rails | New Chunk Size |
|--------------|----------------|------------|-----------------|----------------|
| 64KB         | 8              | 8KB        | 4               | 16KB           |
| 32KB         | 4              | 8KB        | 2               | 16KB           |
| 16KB         | 4              | 4KB        | 1               | 16KB           |

### Expected Benefits
- **Efficiency**: Reduced per-operation overhead
- **Throughput**: Better utilization of NIC bandwidth
- **Best for**: Mid-size transfers (16KB - 128KB)

---

## Combined Optimization Image

For production testing, all three optimizations can be combined into a single image:

```dockerfile
FROM public.ecr.aws/v9l4g5s4/nixl-efa-dev:mrrc-2026-01-15

# Copy all optimized source files
COPY src/utils/libfabric/libfabric_common.h /workspace/nixl/src/utils/libfabric/
COPY src/utils/libfabric/libfabric_rail.cpp /workspace/nixl/src/utils/libfabric/
COPY src/utils/libfabric/libfabric_rail.h /workspace/nixl/src/utils/libfabric/
COPY src/utils/libfabric/libfabric_rail_manager.cpp /workspace/nixl/src/utils/libfabric/
COPY src/utils/libfabric/libfabric_rail_manager.h /workspace/nixl/src/utils/libfabric/

WORKDIR /workspace/nixl/build
RUN ninja -j$(nproc) && ninja install && ldconfig

LABEL patches.performance="fi-inject,doorbell-batching,transfer-coalescing"
```

---

## References

1. **Kalia et al. USENIX ATC'16** - "Design Guidelines for High Performance RDMA Systems"
   - Doorbell batching, unsignaled completions, inlining

2. **AWS EFA Best Practices**
   - Minimum efficient transfer sizes, multi-rail striping

3. **KVDirect Paper**
   - Transfer coalescing for fragmented KV cache pages

4. **libfabric EFA Provider Documentation**
   - FI_MORE flag, fi_injectdata, scatter-gather support
