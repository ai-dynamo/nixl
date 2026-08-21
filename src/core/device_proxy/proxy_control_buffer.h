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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_CONTROL_BUFFER_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_CONTROL_BUFFER_H

#include <cstddef>
#include <cstdint>

#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#include <optional>
#endif

#include "nixl_types.h"

class nixlProxyControlBuffer {
public:
    nixlProxyControlBuffer() = default;
    ~nixlProxyControlBuffer();

    nixlProxyControlBuffer(const nixlProxyControlBuffer &) = delete;
    nixlProxyControlBuffer &
    operator=(const nixlProxyControlBuffer &) = delete;

    nixl_status_t
    allocate(size_t count);

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
    uint64_t *slots_dev_ = nullptr;
    uint64_t *cpu_write_ptr_ = nullptr;
    size_t count_ = 0;
#ifdef HAVE_GDRCOPY
    uint64_t *allocation_dev_ = nullptr;
    size_t mapping_size_ = 0;
    int device_id_ = 0;
    gdr_t gdr_ = nullptr;
    std::optional<gdr_mh_t> mapping_handle_;
#endif
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_CONTROL_BUFFER_H
