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
#include "nixl_gdr_buffer.h"

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>

#include "nixl_log.h"

nixlGdrBuffer::~nixlGdrBuffer() {
    deallocate();
}

nixl_status_t
nixlGdrBuffer::allocate(size_t count) {
    if (count == 0 || slots_dev_ != nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t data_size = sizeof(uint64_t) * count;
#ifdef HAVE_GDRCOPY
    if (cudaGetDevice(&device_id_) != cudaSuccess) {
        NIXL_ERROR << "Failed to query CUDA device for GDRCopy buffer";
        return NIXL_ERR_BACKEND;
    }

    mapping_size_ = (data_size + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
    allocation_size_ = mapping_size_ + GPU_PAGE_SIZE - 1;
    if (cudaMalloc(reinterpret_cast<void **>(&allocation_dev_), allocation_size_) != cudaSuccess) {
        NIXL_ERROR << "Failed to allocate HBM GDRCopy buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    const uintptr_t allocation_addr = reinterpret_cast<uintptr_t>(allocation_dev_);
    const uintptr_t aligned_addr =
        (allocation_addr + GPU_PAGE_SIZE - 1) & ~(static_cast<uintptr_t>(GPU_PAGE_SIZE) - 1);
    slots_dev_ = reinterpret_cast<uint64_t *>(aligned_addr);
    if (cudaMemset(slots_dev_, 0, data_size) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        NIXL_ERROR << "Failed to initialize HBM GDRCopy buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    gdr_ = gdr_open();
    if (gdr_ == nullptr) {
        NIXL_ERROR << "Failed to open GDRCopy; ensure the gdrdrv module is loaded";
        deallocate();
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (gdr_pin_buffer(gdr_,
                       reinterpret_cast<unsigned long>(slots_dev_),
                       mapping_size_,
                       0,
                       0,
                       &mapping_handle_) != 0) {
        NIXL_ERROR << "Failed to pin HBM buffer with GDRCopy";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    pinned_ = true;

    if (gdr_map(gdr_, mapping_handle_, &mapping_base_, mapping_size_) != 0) {
        NIXL_ERROR << "Failed to map HBM buffer with GDRCopy";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    mapped_ = true;

    gdr_info_t info{};
    if (gdr_get_info(gdr_, mapping_handle_, &info) != 0) {
        NIXL_ERROR << "Failed to query GDRCopy buffer mapping";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    const uintptr_t slots_addr = reinterpret_cast<uintptr_t>(slots_dev_);
    if (slots_addr < info.va || slots_addr - info.va + data_size > info.mapped_size) {
        NIXL_ERROR << "Invalid GDRCopy buffer mapping bounds";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    slots_map_ =
        reinterpret_cast<uint64_t *>(static_cast<char *>(mapping_base_) + (slots_addr - info.va));
#else
    // cudaHostAllocMapped guarantees cudaHostGetDevicePointer works (vs. relying on UVA).
    if (cudaHostAlloc(reinterpret_cast<void **>(&slots_map_), data_size, cudaHostAllocMapped) !=
        cudaSuccess) {
        NIXL_ERROR << "Failed to allocate host-mapped GDR buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    void *device_ptr = nullptr;
    if (cudaHostGetDevicePointer(&device_ptr, slots_map_, 0) != cudaSuccess) {
        NIXL_ERROR << "Failed to get device pointer for host-mapped GDR buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    slots_dev_ = static_cast<uint64_t *>(device_ptr);
    std::fill_n(slots_map_, count, uint64_t{0});
#endif

    count_ = count;
    return NIXL_SUCCESS;
}

void
nixlGdrBuffer::deallocate() noexcept {
#ifdef HAVE_GDRCOPY
    if (mapped_) {
        gdr_unmap(gdr_, mapping_handle_, mapping_base_, mapping_size_);
        mapped_ = false;
    }
    mapping_base_ = nullptr;
    slots_map_ = nullptr;
    if (pinned_) {
        gdr_unpin_buffer(gdr_, mapping_handle_);
        pinned_ = false;
    }
    if (gdr_ != nullptr) {
        gdr_close(gdr_);
        gdr_ = nullptr;
    }
    if (allocation_dev_ != nullptr) {
        cudaSetDevice(device_id_);
        cudaFree(allocation_dev_);
        allocation_dev_ = nullptr;
    }
#else
    if (slots_map_ != nullptr) {
        cudaFreeHost(slots_map_);
        slots_map_ = nullptr;
    }
#endif
    slots_dev_ = nullptr;
    allocation_size_ = 0;
    mapping_size_ = 0;
    count_ = 0;
}

uint64_t *
nixlGdrBuffer::devicePtr(size_t index) const noexcept {
    return index < count_ ? slots_dev_ + index : nullptr;
}

uint64_t *
nixlGdrBuffer::hostPtr(size_t index) const noexcept {
    return index < count_ ? slots_map_ + index : nullptr;
}

nixl_status_t
nixlGdrBuffer::publish(size_t index, uint64_t value) noexcept {
    if (index >= count_ || slots_map_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
#ifdef HAVE_GDRCOPY
    // Orders preceding host stores; gdr_copy_to_mapping does not guarantee ordering/
    // integrity vs. a running kernel. Safe here: each slot is an independent, monotonic
    // word consumed by a polling reader, so eventual visibility suffices.
    std::atomic_thread_fence(std::memory_order_release);
    if (gdr_copy_to_mapping(mapping_handle_, slots_map_ + index, &value, sizeof(value)) != 0) {
        return NIXL_ERR_BACKEND;
    }
#else
    __atomic_store_n(slots_map_ + index, value, __ATOMIC_RELEASE);
#endif
    return NIXL_SUCCESS;
}
