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
#include <memory>

#include "device/device_ops.h"
#include "nixl_types.h"

namespace nixl {

/**
 * GPU-visible words the CPU publishes cheaply: the shutdown state and one
 * consumer index per (channel, peer) ring.
 *
 * Preferred backing is GDRCopy-mapped device memory, so a CPU store lands in
 * device memory without a kernel launch or a copy engine. Where GDRCopy is
 * unavailable - not built in, no gdrdrv, or device memory that cannot be
 * pinned - the buffer falls back to mapped host memory the GPU reads over PCIe.
 */
class proxyControlBuffer {
public:
    /**
     * @brief Create a zeroed buffer of `count` words.
     * @param[out] out Owning handle; unchanged on failure.
     * @retval NIXL_ERR_INVALID_PARAM count is zero or too large.
     * @retval NIXL_ERR_BACKEND Neither backing could be allocated.
     * @note `ops` must outlive the buffer.
     */
    [[nodiscard]] static nixl_status_t
    create(deviceOps &ops, size_t count, std::unique_ptr<proxyControlBuffer> &out);

    ~proxyControlBuffer();

    proxyControlBuffer(const proxyControlBuffer &) = delete;
    proxyControlBuffer &
    operator=(const proxyControlBuffer &) = delete;

    /** Device address of word `index`, or nullptr when out of range. */
    [[nodiscard]] uint64_t *
    devicePointer(size_t index = 0) const noexcept;

    [[nodiscard]] nixl_status_t
    writeSlot(size_t index, uint64_t value) noexcept;

private:
    /** GDRCopy state, complete only in the implementation so this header needs no gdrapi.h. */
    struct GdrMapping;

    proxyControlBuffer();

    [[nodiscard]] nixl_status_t
    allocateMappedHost(deviceOps &ops, size_t count);

    /** Returns NIXL_ERR_NOT_SUPPORTED when GDRCopy cannot back this buffer. */
    [[nodiscard]] nixl_status_t
    allocateGdrCopy(deviceOps &ops, size_t count);

    uint64_t *slots_dev_ = nullptr;
    uint64_t *cpu_write_ptr_ = nullptr;
    size_t count_ = 0;
    /** Owns the mapped host slab when GDRCopy is not in use. */
    mappedHostMem control_mem_;
    /** Owns the device-memory slab and its mapping when GDRCopy is in use. */
    std::unique_ptr<GdrMapping> gdr_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONTROL_BUFFER_H
