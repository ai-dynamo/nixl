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
#ifndef NIXL_SRC_CORE_TRACING_RANDOM_BUFFER_H
#define NIXL_SRC_CORE_TRACING_RANDOM_BUFFER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "tracing/trace.h"

namespace nixl::trace {

/**
 * @brief Serves random bytes from a block that @p refill fills whole, so most
 *        draws are a copy; clear() discards the rest of the block.
 */
class RandomBuffer {
public:
    explicit constexpr RandomBuffer(random_bytes_fn_t refill) noexcept : refill_(refill) {}

    void
    fill(std::span<std::uint8_t> output) {
        while (!output.empty()) {
            if (used_ == blockSize) {
                refill_(block_);
                used_ = 0;
            }
            const std::size_t chunk = std::min(output.size(), blockSize - used_);
            std::copy_n(block_.data() + used_, chunk, output.data());
            used_ += chunk;
            output = output.subspan(chunk);
        }
    }

    void
    clear() noexcept {
        used_ = blockSize;
    }

private:
    static constexpr std::size_t blockSize = 256;

    const random_bytes_fn_t refill_;
    std::size_t used_ = blockSize;
    std::array<std::uint8_t, blockSize> block_{};
};

} // namespace nixl::trace

#endif
