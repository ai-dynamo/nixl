/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "uuid_v4.h"

#include <iomanip>
#include <sstream>
#include <random>

namespace nixl {

UUIDv4::UUIDv4() {
    generate_random_bytes(data.data(), data.size());
    // Set version 4 bits (version 4 = 0100 in binary)
    data[6] = (data[6] & 0x0F) | 0x40;
    // Set variant bits (RFC 9562 variant = 10 in binary)
    data[8] = (data[8] & 0x3F) | 0x80;
}

std::string
UUIDv4::to_string() const {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    // Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (RFC 9562 UUID version 4)
    for (size_t i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            oss << '-';
        }
        oss << std::setw(2) << static_cast<int>(data[i]);
    }

    return oss.str();
}

void
UUIDv4::generate_random_bytes(uint8_t *output, size_t size) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);

    for (size_t i = 0; i < size; ++i) {
        output[i] = dis(gen);
    }
}

} // namespace nixl
