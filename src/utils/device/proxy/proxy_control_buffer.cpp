/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "proxy_control_buffer.h"

#include <algorithm>
#include <limits>

#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#endif

#include "device/device_allocator.h"
#include "nixl_log.h"

#ifdef HAVE_GDRCOPY
struct nixlProxyControlBuffer::GdrMapping {
    /** Padded HBM slab; the slots are the page-aligned view into it. */
    nixlDeviceMem allocation;
    size_t mapping_size = 0;
    gdr_t gdr = nullptr;
    gdr_mh_t handle{};
    bool pinned = false;
    void *cpu_ptr = nullptr;

    ~GdrMapping() {
        if (cpu_ptr != nullptr) {
            gdr_unmap(gdr, handle, cpu_ptr, mapping_size);
        }
        if (pinned) {
            gdr_unpin_buffer(gdr, handle);
        }
        if (gdr != nullptr) {
            gdr_close(gdr);
        }
    }
};
#else
struct nixlProxyControlBuffer::GdrMapping {};
#endif

nixlProxyControlBuffer::nixlProxyControlBuffer() = default;

nixlProxyControlBuffer::~nixlProxyControlBuffer() {
    deallocate();
}

nixl_status_t
nixlProxyControlBuffer::allocate(nixlDeviceAllocator &allocator, size_t count) {
    size_t max_size = std::numeric_limits<size_t>::max();
#ifdef HAVE_GDRCOPY
    // Leave room for page rounding and alignment padding in the HBM allocation.
    max_size -= 2 * (static_cast<size_t>(GPU_PAGE_SIZE) - 1);
#endif
    if (count == 0 || count > max_size / sizeof(uint64_t) || allocated()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixl_status_t gdr_status = allocateGdrCopy(allocator, count);
    if (gdr_status == NIXL_SUCCESS) {
        count_ = count;
        return NIXL_SUCCESS;
    }
    // Partial GDRCopy state is dropped before the fallback claims the members.
    deallocate();
#ifdef HAVE_GDRCOPY
    NIXL_INFO << "GDRCopy unavailable for the proxy control buffer (" << gdr_status
              << "); falling back to pinned host memory";
#endif

    const nixl_status_t status = allocateMappedHost(allocator, count);
    if (status != NIXL_SUCCESS) {
        deallocate();
        return status;
    }
    count_ = count;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyControlBuffer::allocateMappedHost(nixlDeviceAllocator &allocator, size_t count) {
    if (allocator.allocMappedHostMem(sizeof(uint64_t) * count, control_mem_) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to allocate host-mapped proxy control buffer";
        return NIXL_ERR_BACKEND;
    }
    cpu_write_ptr_ = control_mem_.asHost<uint64_t>();
    slots_dev_ = control_mem_.asDev<uint64_t>();
    std::fill_n(cpu_write_ptr_, count, uint64_t{0});
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyControlBuffer::allocateGdrCopy(nixlDeviceAllocator &allocator, size_t count) {
#ifdef HAVE_GDRCOPY
    auto mapping = std::make_unique<GdrMapping>();
    const size_t data_size = sizeof(uint64_t) * count;
    mapping->mapping_size = (data_size + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
    const size_t allocation_size = mapping->mapping_size + GPU_PAGE_SIZE - 1;
    if (allocator.allocDeviceMem(allocation_size, mapping->allocation) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to allocate HBM proxy control buffer";
        return NIXL_ERR_BACKEND;
    }

    const uintptr_t allocation_addr = reinterpret_cast<uintptr_t>(mapping->allocation.get());
    const uintptr_t aligned_addr =
        (allocation_addr + GPU_PAGE_SIZE - 1) & ~(static_cast<uintptr_t>(GPU_PAGE_SIZE) - 1);
    auto *slots = reinterpret_cast<uint64_t *>(aligned_addr);
    if (allocator.memsetDeviceMem(slots, 0, data_size) != NIXL_SUCCESS ||
        allocator.synchronize() != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to initialize HBM proxy control buffer";
        return NIXL_ERR_BACKEND;
    }

    mapping->gdr = gdr_open();
    if (mapping->gdr == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (gdr_pin_buffer(mapping->gdr,
                       reinterpret_cast<unsigned long>(slots),
                       mapping->mapping_size,
                       0,
                       0,
                       &mapping->handle) != 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    mapping->pinned = true;

    void *cpu_ptr = nullptr;
    if (gdr_map(mapping->gdr, mapping->handle, &cpu_ptr, mapping->mapping_size) != 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    mapping->cpu_ptr = cpu_ptr;

    slots_dev_ = slots;
    cpu_write_ptr_ = static_cast<uint64_t *>(cpu_ptr);
    gdr_ = std::move(mapping);
    return NIXL_SUCCESS;
#else
    static_cast<void>(allocator);
    static_cast<void>(count);
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

void
nixlProxyControlBuffer::deallocate() noexcept {
    gdr_.reset();
    control_mem_.reset();
    cpu_write_ptr_ = nullptr;
    slots_dev_ = nullptr;
    count_ = 0;
}

uint64_t *
nixlProxyControlBuffer::devicePtr(size_t index) const noexcept {
    return index < count_ ? slots_dev_ + index : nullptr;
}

nixl_status_t
nixlProxyControlBuffer::writeSlot(size_t index, uint64_t value) noexcept {
    if (index >= count_ || cpu_write_ptr_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
#ifdef HAVE_GDRCOPY
    if (gdr_) {
        if (gdr_copy_to_mapping(gdr_->handle, cpu_write_ptr_ + index, &value, sizeof(value)) != 0) {
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }
#endif
    __atomic_store_n(cpu_write_ptr_ + index, value, __ATOMIC_RELAXED);
    return NIXL_SUCCESS;
}
