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
#ifndef NIXL_SRC_UTILS_GDRCOPY_NIXL_GDR_BUFFER_H
#define NIXL_SRC_UTILS_GDRCOPY_NIXL_GDR_BUFFER_H

#include <cstddef>
#include <cstdint>

#ifdef HAVE_GDRCOPY
#include <gdrapi.h>
#endif

#include "nixl_types.h"

/**
 * CPU-published uint64_t slab visible to the GPU.
 *
 * With HAVE_GDRCOPY: page-aligned HBM allocation pinned/mapped via GDRCopy.
 * Without HAVE_GDRCOPY: cudaHostAllocMapped UVA with cudaHostGetDevicePointer.
 *
 * publish() is the only supported CPU write path so both backends keep a
 * consistent release store into the mapped view.
 */
class nixlGdrBuffer {
public:
    nixlGdrBuffer() = default;
    ~nixlGdrBuffer();

    nixlGdrBuffer(const nixlGdrBuffer &) = delete;
    nixlGdrBuffer &
    operator=(const nixlGdrBuffer &) = delete;

    nixl_status_t
    allocate(size_t count);

    void
    deallocate() noexcept;

    [[nodiscard]] bool
    empty() const noexcept {
        return count_ == 0;
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return count_;
    }

    [[nodiscard]] uint64_t *
    devicePtr(size_t index = 0) const noexcept;

    [[nodiscard]] uint64_t *
    hostPtr(size_t index = 0) const noexcept;

    nixl_status_t
    publish(size_t index, uint64_t value) noexcept;

private:
    uint64_t *allocation_dev_ = nullptr;
    uint64_t *slots_dev_ = nullptr;
    uint64_t *slots_map_ = nullptr;
    void *mapping_base_ = nullptr;
    size_t allocation_size_ = 0;
    size_t mapping_size_ = 0;
    size_t count_ = 0;
    int device_id_ = 0;
#ifdef HAVE_GDRCOPY
    gdr_t gdr_ = nullptr;
    gdr_mh_t mapping_handle_{};
    bool pinned_ = false;
    bool mapped_ = false;
#endif
};

#endif // NIXL_SRC_UTILS_GDRCOPY_NIXL_GDR_BUFFER_H
