# Legacy GET kernel call site

Inspection-only C++ reference, not a compilation target or supported current-main
API. The fenced text below preserves `owner_consumer_reproducer.cu` byte-for-byte, including its
original copyright and author attribution. SDK/private-view dependencies remain
unresolved; publishing it as documentation does not make it runnable.

```cpp
// SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Author: Yunxiao Ning
//
// Prototype kernel call-site only. The owner/consumer setup and limits are in
// README.md. This is not a current-main build target: the included legacy GET
// reference depends on DOCA 3.1/internal helpers absent from the public branch.

#include "legacy_doca31_get_reference.cuh"

#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>

#include <cstdint>

struct GetResult {
    nixl_status_t submission;
    nixl_status_t terminal;
    uint32_t polls;
};

__global__ void
submitOneGet(nixlMemViewH local_destination,
             nixlMemViewH remote_source,
             nixlMemViewH local_dump_scratch,
             size_t bytes,
             size_t destination_offset,
             size_t source_offset,
             size_t scratch_offset,
             uint64_t timeout_cycles,
             GetResult *result) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    nixl::gpu::xferStatusH status{};
    auto rc = nixl::gpu::experimental::get<nixl::gpu::level_t::THREAD>(
        {local_destination, 0, destination_offset},
        {remote_source, 0, source_offset},
        bytes,
        {local_dump_scratch, 0, scratch_offset},
        &status);
    result->submission = rc;
    result->terminal = rc;
    result->polls = 0;
    if (rc != NIXL_IN_PROG) {
        return;
    }

    const uint64_t deadline = clock64() + timeout_cycles;
    do {
        rc = nixl::gpu::getXferStatus<nixl::gpu::level_t::THREAD>(status);
        ++result->polls;
        if (rc != NIXL_IN_PROG) {
            result->terminal = rc;
            return;
        }
    } while (clock64() < deadline);
    // An accepted, unretired request is not cancellation. The host reproducer
    // must retain views/MRs/buffers and report timeout instead of freeing them.
    result->terminal = NIXL_IN_PROG;
}
```
