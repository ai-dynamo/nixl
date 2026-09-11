/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COALESCING_H
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COALESCING_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace nixl::gpunetio {

/** @brief Address, length, and keys for one prepared READ segment. */
struct ReadSegment {
    uintptr_t local_addr;
    uintptr_t remote_addr;
    size_t size;
    uint32_t local_key;
    uint32_t remote_key;
};

/**
 * @brief Test whether two READ segments form one safe contiguous WQE.
 * @param current Existing prepared segment.
 * @param next Input segment considered for merging.
 * @param max_size Maximum merged byte count.
 * @return true only when ranges and keys match without size or address overflow.
 */
inline bool
canCoalesceRead(const ReadSegment &current, const ReadSegment &next, size_t max_size) {
    if (current.size > max_size || next.size > max_size - current.size) {
        return false;
    }
    if (current.local_addr > std::numeric_limits<uintptr_t>::max() - current.size) {
        return false;
    }
    if (current.remote_addr > std::numeric_limits<uintptr_t>::max() - current.size) {
        return false;
    }

    return current.local_addr + current.size == next.local_addr &&
        current.remote_addr + current.size == next.remote_addr &&
        current.local_key == next.local_key && current.remote_key == next.remote_key;
}

} // namespace nixl::gpunetio

#endif /* NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COALESCING_H */
