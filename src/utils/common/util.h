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
#ifndef UTIL_H
#define UTIL_H

#define CONCAT(a, b) CONCAT_0(a, b)
#define CONCAT_0(a, b) a ## b
#define UNIQUE_NAME(name) CONCAT(name, __COUNTER__)

template <typename T>
constexpr T nextPowerOf2(T n) {
    static_assert(std::is_unsigned<T>::value, "nextPowerOf2 requires an unsigned integer type");

    if (n == 0) return 1;

    --n;
    for (std::size_t i = 1; i < sizeof(T) * 8; i <<= 1) {
        n |= n >> i;
    }
    return n + 1;
}

#endif /* UTIL_H */
