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
#include <cuda_runtime.h>

#include "nixl_log.h"

nixlProxyControlBuffer::~nixlProxyControlBuffer() {
    deallocate();
}

nixl_status_t
nixlProxyControlBuffer::allocate(size_t count) {
    if (count == 0 || allocated()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t data_size = sizeof(uint64_t) * count;
#ifdef HAVE_GDRCOPY
    if (cudaGetDevice(&device_id_) != cudaSuccess) {
        NIXL_ERROR << "Failed to query CUDA device for proxy control buffer";
        return NIXL_ERR_BACKEND;
    }

    mapping_size_ = (data_size + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
    const size_t allocation_size = mapping_size_ + GPU_PAGE_SIZE - 1;
    if (cudaMalloc(reinterpret_cast<void **>(&allocation_dev_), allocation_size) != cudaSuccess) {
        NIXL_ERROR << "Failed to allocate HBM proxy control buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    const uintptr_t allocation_addr = reinterpret_cast<uintptr_t>(allocation_dev_);
    const uintptr_t aligned_addr =
        (allocation_addr + GPU_PAGE_SIZE - 1) & ~(static_cast<uintptr_t>(GPU_PAGE_SIZE) - 1);
    slots_dev_ = reinterpret_cast<uint64_t *>(aligned_addr);
    if (cudaMemset(slots_dev_, 0, data_size) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        NIXL_ERROR << "Failed to initialize HBM proxy control buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    gdr_ = gdr_open();
    if (gdr_ == nullptr) {
        NIXL_ERROR << "Failed to open GDRCopy; ensure the gdrdrv module is loaded";
        deallocate();
        return NIXL_ERR_NOT_SUPPORTED;
    }
    gdr_mh_t mapping_handle{};
    if (gdr_pin_buffer(gdr_,
                       reinterpret_cast<unsigned long>(slots_dev_),
                       mapping_size_,
                       0,
                       0,
                       &mapping_handle) != 0) {
        NIXL_ERROR << "Failed to pin proxy control buffer with GDRCopy";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    mapping_handle_ = mapping_handle;

    void *cpu_write_ptr = nullptr;
    if (gdr_map(gdr_, *mapping_handle_, &cpu_write_ptr, mapping_size_) != 0) {
        NIXL_ERROR << "Failed to map proxy control buffer with GDRCopy";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    cpu_write_ptr_ = static_cast<uint64_t *>(cpu_write_ptr);
#else
    // cudaHostAllocMapped guarantees cudaHostGetDevicePointer works (vs. relying on UVA).
    if (cudaHostAlloc(reinterpret_cast<void **>(&cpu_write_ptr_), data_size, cudaHostAllocMapped) !=
        cudaSuccess) {
        NIXL_ERROR << "Failed to allocate host-mapped proxy control buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    void *device_ptr = nullptr;
    if (cudaHostGetDevicePointer(&device_ptr, cpu_write_ptr_, 0) != cudaSuccess) {
        NIXL_ERROR << "Failed to get device pointer for host-mapped proxy control buffer";
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    slots_dev_ = static_cast<uint64_t *>(device_ptr);
    std::fill_n(cpu_write_ptr_, count, uint64_t{0});
#endif

    count_ = count;
    return NIXL_SUCCESS;
}

void
nixlProxyControlBuffer::deallocate() noexcept {
#ifdef HAVE_GDRCOPY
    if (cpu_write_ptr_ != nullptr) {
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
    if (allocation_dev_ != nullptr) {
        cudaSetDevice(device_id_);
        cudaFree(allocation_dev_);
        allocation_dev_ = nullptr;
    }
    mapping_size_ = 0;
#else
    if (cpu_write_ptr_ != nullptr) {
        cudaFreeHost(cpu_write_ptr_);
        cpu_write_ptr_ = nullptr;
    }
#endif
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
    if (gdr_copy_to_mapping(*mapping_handle_, cpu_write_ptr_ + index, &value, sizeof(value)) != 0) {
        return NIXL_ERR_BACKEND;
    }
#else
    __atomic_store_n(cpu_write_ptr_ + index, value, __ATOMIC_RELAXED);
#endif
    return NIXL_SUCCESS;
}
