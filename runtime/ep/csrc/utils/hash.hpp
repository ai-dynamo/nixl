/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include <cstdint>

namespace nixl_ep_v2 {

static uint64_t fnv1a(const std::vector<char>& data, const uint64_t& seed) {
    uint64_t h = seed;
    constexpr uint64_t prime = 0x100000001b3ull;
    for (const char& c: data) {
        h ^= static_cast<uint8_t>(c);
        h *= prime;
    }
    return h;
}

static std::string get_hex_digest(const std::vector<char>& data) {
    const auto state_0 = fnv1a(data, 0xc6a4a7935bd1e995ull);
    const auto state_1 = fnv1a(data, 0x9e3779b97f4a7c15ull);

    // Split-mix 64
    const auto split_mix = [](uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << split_mix(state_0)
        << std::setw(16) << split_mix(state_1);
    return oss.str();
}

static std::string get_hex_digest(const std::string& data) {
    return get_hex_digest(std::vector<char>{data.begin(), data.end()});
}

} // namespace nixl_ep_v2
