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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_PUBLISHED_MEM_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_PUBLISHED_MEM_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include "device/device_ops.h"
#include "nixl_types.h"

namespace nixl {

/**
 * Memory the CPU writes cheaply and the GPU reads.
 *
 * Preferred backing is GDRCopy-mapped device memory, so a CPU store lands in
 * device memory without a kernel launch or a copy engine. Where GDRCopy is
 * unavailable - not built in, no gdrdrv, or device memory that cannot be
 * pinned - it falls back to mapped host memory the GPU reads over PCIe.
 */
class hostPublishedDeviceMem {
public:
    enum class backing { gdrcopy, mapped_host };

    /**
     * @brief Create a zeroed range of `bytes` bytes.
     * @param[out] out Owning handle; unchanged on failure.
     * @retval NIXL_ERR_INVALID_PARAM bytes is zero or too large.
     * @retval NIXL_ERR_BACKEND Neither backing could be allocated.
     * @note `ops` must outlive the memory.
     */
    [[nodiscard]] static nixl_status_t
    create(deviceOps &ops, size_t bytes, std::unique_ptr<hostPublishedDeviceMem> &out);

    ~hostPublishedDeviceMem();

    hostPublishedDeviceMem(const hostPublishedDeviceMem &) = delete;
    hostPublishedDeviceMem &
    operator=(const hostPublishedDeviceMem &) = delete;

    /** The address the GPU reads. */
    [[nodiscard]] void *
    devicePointer() const noexcept;

    /**
     * @brief Publish one 8-byte word at `offset`.
     * @retval NIXL_ERR_INVALID_PARAM offset is not word-aligned or the word does not fit.
     * @retval NIXL_ERR_BACKEND The GDRCopy write failed.
     */
    [[nodiscard]] nixl_status_t
    publish(size_t offset, uint64_t value) noexcept;

    [[nodiscard]] backing
    backingKind() const noexcept;

    [[nodiscard]] size_t
    size() const noexcept;

private:
    /** GDRCopy state, complete only in the implementation so this header needs no gdrapi.h. */
    struct GdrMapping;

    hostPublishedDeviceMem();

    [[nodiscard]] nixl_status_t
    allocateMappedHost(deviceOps &ops, size_t bytes);

    /** Returns NIXL_ERR_NOT_SUPPORTED when GDRCopy cannot back this memory. */
    [[nodiscard]] nixl_status_t
    allocateGdrCopy(deviceOps &ops, size_t bytes);

    void *device_ptr_ = nullptr;
    uint8_t *cpu_write_ptr_ = nullptr;
    size_t size_ = 0;
    /** Owns the mapped host range when GDRCopy is not in use. */
    mappedHostMem host_mem_;
    /** Owns the device-memory range and its mapping when GDRCopy is in use. */
    std::unique_ptr<GdrMapping> gdr_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_PUBLISHED_MEM_H
