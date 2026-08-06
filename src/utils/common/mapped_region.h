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
#ifndef NIXL_SRC_UTILS_COMMON_MAPPED_REGION_H
#define NIXL_SRC_UTILS_COMMON_MAPPED_REGION_H

#include <sys/mman.h>

#include <cstddef>
#include <utility>

namespace nixl {

/**
 * @class mappedRegion
 * @brief Move-only owner of an mmap()ed region, unmapped on destruction.
 *
 * Held as a member, it also unmaps when a constructor throws after mapping. An
 * empty owner (no mapping) is a valid state, so a failed mapping can be returned
 * rather than thrown on.
 */
class mappedRegion {
public:
    mappedRegion() = default;

    /**
     * @brief Maps @p size bytes from the start of @p fd with protection @p prot.
     * @param flags mmap() flags, e.g. MAP_SHARED.
     *
     * Leaves the region invalid(), with errno set, when the mapping fails.
     */
    mappedRegion(int fd, std::size_t size, int prot, int flags) noexcept
        : addr_(::mmap(nullptr, size, prot, flags, fd, 0)),
          size_(size) {
        if (addr_ == MAP_FAILED) {
            addr_ = nullptr;
            size_ = 0;
        }
    }

    ~mappedRegion() {
        reset();
    }

    mappedRegion(const mappedRegion &) = delete;
    mappedRegion &
    operator=(const mappedRegion &) = delete;

    mappedRegion(mappedRegion &&other) noexcept
        : addr_(std::exchange(other.addr_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    mappedRegion &
    operator=(mappedRegion &&other) noexcept {
        if (this != &other) {
            reset();
            addr_ = std::exchange(other.addr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    [[nodiscard]] void *
    get() const noexcept {
        return addr_;
    }

    [[nodiscard]] std::size_t
    size() const noexcept {
        return size_;
    }

    [[nodiscard]] bool
    valid() const noexcept {
        return addr_ != nullptr;
    }

    void
    reset() noexcept {
        if (addr_ != nullptr) {
            ::munmap(addr_, size_);
            addr_ = nullptr;
            size_ = 0;
        }
    }

private:
    void *addr_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_MAPPED_REGION_H
