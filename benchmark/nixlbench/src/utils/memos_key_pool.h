/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <charconv>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace nixlbench {

// Zero means the environment variable is absent: retain the original benchmark.
inline size_t
parseMemosKeyPoolSize(const char *value) {
    if (value == nullptr) {
        return 0;
    }
    const std::string_view text(value);
    size_t count = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), count);
    if (text.empty() || result.ec != std::errc() || result.ptr != text.data() + text.size() ||
        count == 0 || count > 1000000) {
        throw std::invalid_argument("expected an integer in [1, 1000000]");
    }
    return count;
}

inline bool
parseMemosReuseLocal(const char *value) {
    if (value == nullptr || std::string_view(value) == "0") {
        return false;
    }
    if (std::string_view(value) == "1") {
        return true;
    }
    throw std::invalid_argument("NIXLBENCH_REUSE_LOCAL must be 0 or 1");
}

inline void
validateMemosKeyPoolSize(size_t keys,
                         size_t batch,
                         size_t max_block_size,
                         size_t total_buffer_size,
                         size_t threads,
                         int pipeline_depth = 1,
                         bool reuse_local = false) {
    if (batch == 0 || keys < batch || keys % batch != 0) {
        throw std::invalid_argument("key count must be a positive multiple of batch size");
    }
    if (pipeline_depth <= 0 || static_cast<size_t>(pipeline_depth) > keys / batch) {
        throw std::invalid_argument("pipeline depth must fit in the number of key-pool banks");
    }
    // Division avoids overflowing either the working-set size or descriptor IDs.
    if (threads == 0 || keys > static_cast<size_t>(std::numeric_limits<int>::max()) / threads) {
        throw std::invalid_argument("total key count exceeds the descriptor ID range");
    }
    if (reuse_local && (keys / batch) % static_cast<size_t>(pipeline_depth) != 0) {
        throw std::invalid_argument("reuse requires key-pool banks divisible by pipeline depth");
    }
    // Depth is bounded by keys / batch above, so this multiplication cannot overflow.
    const size_t local_values = reuse_local ? batch * static_cast<size_t>(pipeline_depth) : keys;
    if (max_block_size == 0 || local_values > (total_buffer_size / threads) / max_block_size) {
        throw std::invalid_argument("local values exceed the per-thread buffer at max_block_size");
    }
}

} // namespace nixlbench
