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
#ifndef RANDOM_ID_H
#define RANDOM_ID_H

#include <array>
#include <string>
#include <random>

namespace nixl {

/**
 * @brief A class that generates 16-byte random values and converts them to UUID format
 *
 * This class generates random 16-byte values and provides
 * a method to convert them to the UUID string format (8-4-4-4-12).
 */
class RandomID {
public:
    RandomID();
    ~RandomID() = default;

    /**
     * @brief Converts the 16-byte random value to a UUID string format
     *
     * The UUID format follows the standard 8-4-4-4-12 pattern:
     * xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
     *
     * @return String representation in UUID format
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
     * @brief Generates random bytes using a random number generator
     * @param output Pointer to the output buffer
     * @param size Number of bytes to generate
     */
    static void
    generate_random_bytes(uint8_t *output, size_t size);
};

} // namespace nixl

#endif /* RANDOM_ID_H */