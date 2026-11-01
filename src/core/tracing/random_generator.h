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
#ifndef NIXL_SRC_CORE_TRACING_RANDOM_GENERATOR_H
#define NIXL_SRC_CORE_TRACING_RANDOM_GENERATOR_H

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "tracing/trace.h"

namespace nixl::trace {

class RandomGenerator {
public:
    static constexpr std::size_t reseedBytes = std::size_t{1} << 20;

    explicit constexpr RandomGenerator(random_bytes_fn_t seed) noexcept : seed_(seed) {}

    void
    fill(std::span<std::uint8_t> output) {
        while (!output.empty()) {
            if (remaining_ == 0) {
                reseed();
            }
            const std::uint64_t word = next();
            const std::size_t chunk = std::min(output.size(), sizeof(word));
            std::memcpy(output.data(), &word, chunk);
            output = output.subspan(chunk);
            remaining_ -= std::min(remaining_, sizeof(word));
        }
    }

    void
    clear() noexcept {
        remaining_ = 0;
    }

private:
    void
    reseed() {
        std::array<std::uint64_t, 4> state;
        do {
            seed_({reinterpret_cast<std::uint8_t *>(state.data()), sizeof(state)});
        } while (std::ranges::all_of(state, [](std::uint64_t word) { return word == 0; }));
        state_ = state;
        remaining_ = reseedBytes;
    }

    [[nodiscard]] std::uint64_t
    next() noexcept {
        const std::uint64_t result = std::rotl(state_[1] * 5, 7) * 9;
        const std::uint64_t shifted = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= shifted;
        state_[3] = std::rotl(state_[3], 45);
        return result;
    }

    const random_bytes_fn_t seed_;
    std::array<std::uint64_t, 4> state_{};
    std::size_t remaining_ = 0;
};

} // namespace nixl::trace

#endif
