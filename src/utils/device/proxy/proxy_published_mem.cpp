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
#include "proxy_published_mem.h"

#include <cstring>
#include <limits>
#include <utility>

#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#endif

#include "device/device_ops.h"
#include "nixl_log.h"

namespace nixl {

namespace {
    /** GDRCopy maps whole GPU pages; the size limit leaves room for that with either backing. */
    constexpr size_t kGpuPageSize = size_t{1} << 16;
} // namespace

#ifdef HAVE_GDRCOPY
static_assert(GPU_PAGE_SIZE == kGpuPageSize, "GDRCopy GPU page size changed");

struct hostPublishedDeviceMem::GdrMapping {
    /** Padded device-memory range; the published bytes are the page-aligned view into it. */
    deviceMem allocation;
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
struct hostPublishedDeviceMem::GdrMapping {};
#endif

hostPublishedDeviceMem::hostPublishedDeviceMem() = default;

hostPublishedDeviceMem::~hostPublishedDeviceMem() = default;

nixl_status_t
hostPublishedDeviceMem::create(deviceOps &ops,
                               size_t bytes,
                               std::unique_ptr<hostPublishedDeviceMem> &out) {
    // Leave room for page rounding and alignment padding in a GDRCopy device allocation.
    const size_t max_bytes = std::numeric_limits<size_t>::max() - 2 * (kGpuPageSize - 1);
    if (bytes == 0 || bytes > max_bytes) {
        NIXL_ERROR << "Invalid host-published memory size: " << bytes << " byte(s)";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::unique_ptr<hostPublishedDeviceMem> mem(new hostPublishedDeviceMem());
    const nixl_status_t gdr_status = mem->allocateGdrCopy(ops, bytes);
    if (gdr_status != NIXL_SUCCESS) {
#ifdef HAVE_GDRCOPY
        NIXL_INFO << "GDRCopy unavailable for the proxy control buffer (" << gdr_status
                  << "); falling back to mapped host memory";
#endif
        // A fresh object, so no partial GDRCopy state survives into the fallback.
        mem.reset(new hostPublishedDeviceMem());
        const nixl_status_t status = mem->allocateMappedHost(ops, bytes);
        if (status != NIXL_SUCCESS) {
            return status;
        }
    }
    mem->size_ = bytes;
    out = std::move(mem);
    return NIXL_SUCCESS;
}

nixl_status_t
hostPublishedDeviceMem::allocateMappedHost(deviceOps &ops, size_t bytes) {
    if (ops.allocMappedHostMem(bytes, host_mem_) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to allocate " << bytes
                   << " bytes of mapped host memory for the proxy control buffer";
        return NIXL_ERR_BACKEND;
    }
    cpu_write_ptr_ = host_mem_.hostPointer<uint8_t>();
    device_ptr_ = host_mem_.devicePointer();
    std::memset(cpu_write_ptr_, 0, bytes);
    return NIXL_SUCCESS;
}

nixl_status_t
hostPublishedDeviceMem::allocateGdrCopy(deviceOps &ops, size_t bytes) {
#ifdef HAVE_GDRCOPY
    auto mapping = std::make_unique<GdrMapping>();
    mapping->mapping_size = (bytes + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
    const size_t allocation_size = mapping->mapping_size + GPU_PAGE_SIZE - 1;
    if (ops.allocDeviceMem(allocation_size, mapping->allocation) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to allocate " << allocation_size
                   << " bytes of device memory for the proxy control buffer";
        return NIXL_ERR_BACKEND;
    }

    const uintptr_t allocation_addr = reinterpret_cast<uintptr_t>(mapping->allocation.get());
    const uintptr_t aligned_addr =
        (allocation_addr + GPU_PAGE_SIZE - 1) & ~(static_cast<uintptr_t>(GPU_PAGE_SIZE) - 1);
    void *published = reinterpret_cast<void *>(aligned_addr);
    if (ops.memsetDeviceMem(published, 0, bytes) != NIXL_SUCCESS ||
        ops.synchronize() != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to zero " << bytes
                   << " bytes of device memory for the proxy control buffer";
        return NIXL_ERR_BACKEND;
    }

    mapping->gdr = gdr_open();
    if (mapping->gdr == nullptr) {
        NIXL_DEBUG << "Proxy control buffer: gdr_open failed";
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (gdr_pin_buffer(mapping->gdr,
                       reinterpret_cast<unsigned long>(published),
                       mapping->mapping_size,
                       0,
                       0,
                       &mapping->handle) != 0) {
        NIXL_DEBUG << "Proxy control buffer: gdr_pin_buffer failed for " << mapping->mapping_size
                   << " bytes";
        return NIXL_ERR_NOT_SUPPORTED;
    }
    mapping->pinned = true;

    void *cpu_ptr = nullptr;
    if (gdr_map(mapping->gdr, mapping->handle, &cpu_ptr, mapping->mapping_size) != 0) {
        NIXL_DEBUG << "Proxy control buffer: gdr_map failed for " << mapping->mapping_size
                   << " bytes";
        return NIXL_ERR_NOT_SUPPORTED;
    }
    mapping->cpu_ptr = cpu_ptr;

    device_ptr_ = published;
    cpu_write_ptr_ = static_cast<uint8_t *>(cpu_ptr);
    gdr_ = std::move(mapping);
    return NIXL_SUCCESS;
#else
    static_cast<void>(ops);
    static_cast<void>(bytes);
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

void *
hostPublishedDeviceMem::devicePointer() const noexcept {
    return device_ptr_;
}

nixl_status_t
hostPublishedDeviceMem::publish(size_t offset, uint64_t value) noexcept {
    if (offset % sizeof(uint64_t) != 0 || offset > size_ || size_ - offset < sizeof(uint64_t)) {
        return NIXL_ERR_INVALID_PARAM;
    }
#ifdef HAVE_GDRCOPY
    if (gdr_) {
        if (gdr_copy_to_mapping(gdr_->handle, cpu_write_ptr_ + offset, &value, sizeof(value)) !=
            0) {
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }
#endif
    __atomic_store_n(
        reinterpret_cast<uint64_t *>(cpu_write_ptr_ + offset), value, __ATOMIC_RELAXED);
    return NIXL_SUCCESS;
}

hostPublishedDeviceMem::backing
hostPublishedDeviceMem::backingKind() const noexcept {
    return gdr_ ? backing::gdrcopy : backing::mapped_host;
}

size_t
hostPublishedDeviceMem::size() const noexcept {
    return size_;
}

} // namespace nixl
