/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace nixlbench {

// A key bank maps to one of depth disjoint local batches. The scheduler must
// retain ownership of that batch until its request has completed.
inline size_t
memosLocalBufferOffset(size_t key_index,
                       size_t block_size,
                       size_t batch,
                       size_t depth,
                       size_t allocation_size) {
    if (batch == 0 || depth == 0 || depth > std::numeric_limits<size_t>::max() / batch) {
        throw std::invalid_argument("invalid MEMOS local buffer slot count");
    }
    const size_t slots = batch * depth;
    if (block_size == 0 || slots > allocation_size / block_size) {
        throw std::invalid_argument("MEMOS local buffer is smaller than batch * depth values");
    }
    // Extra allocation bytes (including hugepage padding) are not extra slots.
    // Wrap at the logical batch * depth ring, independent of allocation size.
    return (key_index % slots) * block_size;
}

} // namespace nixlbench
