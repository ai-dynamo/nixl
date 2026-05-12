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
#ifndef NIXL_SRC_UTILS_COMMON_UUID_V7_H
#define NIXL_SRC_UTILS_COMMON_UUID_V7_H

#include <array>
#include <cstdint>
#include <string>

namespace nixl {

/**
 * @class UUIDv7
 * @brief RFC 9562 UUID version 7 (Unix-time-ordered + random).
 *
 * Layout per RFC 9562 §5.7:
 *   - 48 bits: Unix milliseconds (big-endian)
 *   - 4 bits:  version = 0x7
 *   - 12 bits: rand_a
 *   - 2 bits:  variant = 0b10
 *   - 62 bits: rand_b
 *
 * Compared to v4, the time-ordered prefix makes lexicographic UUID order
 * align with creation order. We use it to identify nixl agents so newer
 * agents sort after older ones in ETCD/TCPStore prefix scans.
 */
class UUIDv7 {
public:
    UUIDv7();
    ~UUIDv7() = default;

    /**
     * @brief 8-4-4-4-12 hex string (e.g. "018f...-...").
     */
    [[nodiscard]] std::string
    toString() const;

    [[nodiscard]] const std::array<uint8_t, 16> &
    getData() const noexcept {
        return data_;
    }

private:
    std::array<uint8_t, 16> data_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_UUID_V7_H
