/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_MEMOS_KEY_H
#define NIXLBENCH_MEMOS_KEY_H

#include <cstdint>
#include <random>
#include <string>

namespace nixlbench {

// Encode 128 random bits as hex so DOCA_MEMOS decodes metaInfo into a
// full-width binary key. Do not encode the small descriptor ID as the key:
// placement may depend on bytes that are constant in small integer keys.
template<typename Generator>
std::string
makeMemosKey(Generator &generator) {
    constexpr char hex[] = "0123456789abcdef";
    std::uniform_int_distribution<uint32_t> distribution;
    std::string key;
    key.reserve(32);
    for (int word = 0; word < 4; ++word) {
        const uint32_t value = distribution(generator);
        for (int shift = 28; shift >= 0; shift -= 4) {
            key.push_back(hex[(value >> shift) & 0xf]);
        }
    }
    return key;
}

} // namespace nixlbench

#endif // NIXLBENCH_MEMOS_KEY_H
