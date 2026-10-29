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
#ifndef NIXL_SRC_UTILS_DEVICE_HOST_PUBLISHED_MEM_H
#define NIXL_SRC_UTILS_DEVICE_HOST_PUBLISHED_MEM_H

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

namespace nixl {

/**
 * @brief Zeroed memory the CPU publishes 64-bit words into and the GPU reads.
 *
 * One CPU thread writes each word, and the GPU reads it with system-scope acquire loads. Host
 * writes made before publish() are visible to a GPU load that observes the published word.
 *
 * @note The deviceOps that allocated this memory must outlive it.
 */
class hostPublishedDeviceMem {
public:
    /** device: device memory the CPU writes directly; mapped_host: the portable fallback. */
    enum class backing { device, mapped_host };

    virtual ~hostPublishedDeviceMem() = default;

    hostPublishedDeviceMem(const hostPublishedDeviceMem &) = delete;
    hostPublishedDeviceMem &
    operator=(const hostPublishedDeviceMem &) = delete;

    /** The address the GPU reads. */
    [[nodiscard]] void *
    devicePointer() const noexcept {
        return devPtr_;
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return size_;
    }

    [[nodiscard]] backing
    backingKind() const noexcept {
        return backing_;
    }

    /**
     * @brief Publish one 8-byte word at `offset`.
     * @retval NIXL_ERR_INVALID_PARAM offset is not word-aligned or the word does not fit.
     * @retval NIXL_ERR_BACKEND The write failed.
     */
    [[nodiscard]] nixl_status_t
    publish(size_t offset, uint64_t value) noexcept {
        if (offset % sizeof(uint64_t) != 0 || offset > size_ || size_ - offset < sizeof(uint64_t)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        return doPublish(offset, value);
    }

protected:
    hostPublishedDeviceMem(void *dev_ptr, size_t size, backing kind) noexcept
        : devPtr_(dev_ptr),
          size_(size),
          backing_(kind) {}

    /** @brief Release-store `value` at a validated `offset`. */
    [[nodiscard]] virtual nixl_status_t
    doPublish(size_t offset, uint64_t value) noexcept = 0;

private:
    void *devPtr_;
    size_t size_;
    backing backing_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_HOST_PUBLISHED_MEM_H
