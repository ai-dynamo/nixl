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
#include "uuid_v7.h"

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

namespace nixl {

namespace {

    uint64_t
    unix_ms_now() {
        using clock = std::chrono::system_clock;
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(clock::now().time_since_epoch())
                .count());
    }

    // 64 bits of entropy is enough for the v7 random regions; we splice it
    // across rand_a (12 bits) and rand_b (62 bits) below.
    uint64_t
    random_u64() {
        static thread_local std::mt19937_64 gen{std::random_device{}()};
        static thread_local std::uniform_int_distribution<uint64_t> dist;
        return dist(gen);
    }

} // namespace

UUIDv7::UUIDv7() {
    const uint64_t unix_ms = unix_ms_now() & 0x0000FFFFFFFFFFFFULL;
    const uint64_t r1 = random_u64();
    const uint64_t r2 = random_u64();

    data_[0] = static_cast<uint8_t>((unix_ms >> 40) & 0xFF);
    data_[1] = static_cast<uint8_t>((unix_ms >> 32) & 0xFF);
    data_[2] = static_cast<uint8_t>((unix_ms >> 24) & 0xFF);
    data_[3] = static_cast<uint8_t>((unix_ms >> 16) & 0xFF);
    data_[4] = static_cast<uint8_t>((unix_ms >> 8) & 0xFF);
    data_[5] = static_cast<uint8_t>(unix_ms & 0xFF);

    // version (4 bits) | rand_a (12 bits)
    data_[6] = static_cast<uint8_t>(0x70 | ((r1 >> 8) & 0x0F));
    data_[7] = static_cast<uint8_t>(r1 & 0xFF);

    // variant (2 bits) | rand_b (62 bits)
    data_[8] = static_cast<uint8_t>(0x80 | ((r2 >> 56) & 0x3F));
    data_[9] = static_cast<uint8_t>((r2 >> 48) & 0xFF);
    data_[10] = static_cast<uint8_t>((r2 >> 40) & 0xFF);
    data_[11] = static_cast<uint8_t>((r2 >> 32) & 0xFF);
    data_[12] = static_cast<uint8_t>((r2 >> 24) & 0xFF);
    data_[13] = static_cast<uint8_t>((r2 >> 16) & 0xFF);
    data_[14] = static_cast<uint8_t>((r2 >> 8) & 0xFF);
    data_[15] = static_cast<uint8_t>(r2 & 0xFF);
}

std::string
UUIDv7::toString() const {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            oss << '-';
        }
        oss << std::setw(2) << static_cast<int>(data_[i]);
    }
    return oss.str();
}

} // namespace nixl
