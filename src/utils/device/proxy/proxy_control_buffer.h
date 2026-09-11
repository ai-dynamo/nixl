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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONTROL_BUFFER_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONTROL_BUFFER_H

#include <cstddef>
#include <cstdint>

#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#include <optional>
#endif

#include "device/device_allocator.h"
#include "nixl_types.h"

/**
 * GPU-visible words the CPU publishes cheaply: the shutdown state and one
 * consumer index per (channel, peer) ring.
 *
 * Preferred backing is GDRCopy-mapped HBM, so a store lands in device memory
 * without a kernel launch or a copy engine. Where GDRCopy is unavailable -
 * no gdrdrv, or an allocator whose "device" memory cannot be pinned - the
 * buffer falls back to pinned host memory the GPU reads over PCIe.
 */
class nixlProxyControlBuffer {
public:
    nixlProxyControlBuffer() = default;
    ~nixlProxyControlBuffer();

    nixlProxyControlBuffer(const nixlProxyControlBuffer &) = delete;
    nixlProxyControlBuffer &
    operator=(const nixlProxyControlBuffer &) = delete;

    nixl_status_t
    allocate(nixlDeviceAllocator &allocator, size_t count);

    void
    deallocate() noexcept;

    [[nodiscard]] bool
    allocated() const noexcept {
        return cpu_write_ptr_ != nullptr;
    }

    [[nodiscard]] uint64_t *
    devicePtr(size_t index = 0) const noexcept;

    nixl_status_t
    writeSlot(size_t index, uint64_t value) noexcept;

private:
    nixl_status_t
    allocateMappedHost(nixlDeviceAllocator &allocator, size_t count);

#ifdef HAVE_GDRCOPY
    /** Returns NIXL_ERR_NOT_SUPPORTED when GDRCopy cannot back this buffer. */
    nixl_status_t
    allocateGdrCopy(nixlDeviceAllocator &allocator, size_t count);
#endif

    uint64_t *slots_dev_ = nullptr;
    uint64_t *cpu_write_ptr_ = nullptr;
    size_t count_ = 0;
    /** Owns the mapped host slab when GDRCopy is not in use. */
    nixlMappedHostMem control_mem_;
#ifdef HAVE_GDRCOPY
    /** Owns the padded HBM slab; slots_dev_ is the page-aligned view into it. */
    nixlDeviceMem allocation_mem_;
    size_t mapping_size_ = 0;
    gdr_t gdr_ = nullptr;
    std::optional<gdr_mh_t> mapping_handle_;
#endif
};

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONTROL_BUFFER_H
