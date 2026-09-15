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

#include "device/device_allocator.h"
#include "nixl_log.h"

nixlProxyControlBuffer::~nixlProxyControlBuffer() {
    deallocate();
}

nixl_status_t
nixlProxyControlBuffer::allocate(nixlDeviceAllocator &allocator, size_t count) {
    if (count == 0 || allocated()) {
        return NIXL_ERR_INVALID_PARAM;
    }

#ifdef HAVE_GDRCOPY
    const nixl_status_t gdr_status = allocateGdrCopy(allocator, count);
    if (gdr_status == NIXL_SUCCESS) {
        count_ = count;
        return NIXL_SUCCESS;
    }
    // Partial GDRCopy state is dropped before the fallback claims the members.
    deallocate();
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

#ifdef HAVE_GDRCOPY
nixl_status_t
nixlProxyControlBuffer::allocateGdrCopy(nixlDeviceAllocator &allocator, size_t count) {
    const size_t data_size = sizeof(uint64_t) * count;
    mapping_size_ = (data_size + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
    const size_t allocation_size = mapping_size_ + GPU_PAGE_SIZE - 1;
    if (allocator.allocDeviceMem(allocation_size, allocation_mem_) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to allocate HBM proxy control buffer";
        return NIXL_ERR_BACKEND;
    }

    const uintptr_t allocation_addr = reinterpret_cast<uintptr_t>(allocation_mem_.get());
    const uintptr_t aligned_addr =
        (allocation_addr + GPU_PAGE_SIZE - 1) & ~(static_cast<uintptr_t>(GPU_PAGE_SIZE) - 1);
    slots_dev_ = reinterpret_cast<uint64_t *>(aligned_addr);
    if (allocator.memsetDeviceMem(slots_dev_, 0, data_size) != NIXL_SUCCESS ||
        allocator.synchronize() != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to initialize HBM proxy control buffer";
        return NIXL_ERR_BACKEND;
    }

    gdr_ = gdr_open();
    if (gdr_ == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    gdr_mh_t mapping_handle{};
    if (gdr_pin_buffer(gdr_,
                       reinterpret_cast<unsigned long>(slots_dev_),
                       mapping_size_,
                       0,
                       0,
                       &mapping_handle) != 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    mapping_handle_ = mapping_handle;

    void *cpu_write_ptr = nullptr;
    if (gdr_map(gdr_, *mapping_handle_, &cpu_write_ptr, mapping_size_) != 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    cpu_write_ptr_ = static_cast<uint64_t *>(cpu_write_ptr);
    return NIXL_SUCCESS;
}
#endif

void
nixlProxyControlBuffer::deallocate() noexcept {
#ifdef HAVE_GDRCOPY
    if (mapping_handle_ && cpu_write_ptr_ != nullptr) {
        gdr_unmap(gdr_, *mapping_handle_, cpu_write_ptr_, mapping_size_);
        cpu_write_ptr_ = nullptr;
    }
    if (mapping_handle_) {
        gdr_unpin_buffer(gdr_, *mapping_handle_);
        mapping_handle_.reset();
    }
    if (gdr_ != nullptr) {
        gdr_close(gdr_);
        gdr_ = nullptr;
    }
    allocation_mem_.reset();
    mapping_size_ = 0;
#endif
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
    if (mapping_handle_) {
        if (gdr_copy_to_mapping(*mapping_handle_, cpu_write_ptr_ + index, &value, sizeof(value)) !=
            0) {
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }
#endif
    __atomic_store_n(cpu_write_ptr_ + index, value, __ATOMIC_RELAXED);
    return NIXL_SUCCESS;
}
