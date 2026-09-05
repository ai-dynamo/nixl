/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <system_error>

#include <cerrno>
#include <sys/random.h>

namespace nixl {

void
generateRandomBytes(std::uint8_t *output, std::size_t size) {
    constexpr std::size_t max_entropy_request = 256;

    for (std::size_t filled = 0; filled < size;) {
        const std::size_t chunk = std::min(size - filled, max_entropy_request);
        if (getentropy(output + filled, chunk) == 0) {
            filled += chunk;
            continue;
        }
        if (errno == EINTR) {
            continue;
        }
        throw std::system_error(errno, std::generic_category(), "getentropy failed");
    }
}

UUIDv4::UUIDv4() {
    generateRandomBytes(data.data(), data.size());
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

} // namespace nixl
