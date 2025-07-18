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
#ifndef UUID_V4_H
#define UUID_V4_H

#include <array>
#include <string>
#include <random>

namespace nixl {

/**
 * @brief A class that generates RFC 9562 UUID version 4 identifiers
 *
 * This class generates cryptographically random 16-byte values and converts them
 * to the standard UUID version 4 format (8-4-4-4-12) following RFC 9562 specification.
 */
class UUIDv4 {
public:
    /**
     * @brief Default constructor that generates a new random UUID version 4
     */
    UUIDv4();
    ~UUIDv4() = default;

    /**
     * @brief Converts the 16-byte random value to a UUID version 4 string format
     *
     * The UUID format follows the RFC 9562 standard 8-4-4-4-12 pattern:
     * xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
     * where x is a hexadecimal digit, 4 indicates version 4, and y is the variant.
     *
     * @return String representation in UUID version 4 format
     */
    std::string
    to_string() const;

    /**
     * @brief Gets the raw 16-byte data
     * @return Const reference to the internal byte array
     */
    const std::array<uint8_t, 16> &
    get_data() const {
        return data;
    }

private:
    std::array<uint8_t, 16> data;

    /**
     * @brief Generates cryptographically random bytes for UUID version 4
     * @param output Pointer to the output buffer
     * @param size Number of bytes to generate
     */
    static void
    generate_random_bytes(uint8_t *output, size_t size);
};

} // namespace nixl

#endif /* UUID_V4_H */
