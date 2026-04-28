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
#ifndef NIXL_SRC_UTILS_COMMON_BACKEND_H
#define NIXL_SRC_UTILS_COMMON_BACKEND_H

#include "nixl_types.h"

#include <charconv>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

namespace nixl {

template<typename... Ts>
constexpr inline bool dependent_false = false;

template<typename T>
[[nodiscard]] T
getBackendParam(const nixl_b_params_t &params, const std::string &key, const T default_) {
    const auto it = params.find(key);

    if (it == params.end()) {
        return default_;
    }

    if constexpr (std::is_same_v<T, std::string>) {
        return it->second;
    } else if constexpr (std::is_integral_v<T>) {
        T result;
        {
            const std::string &value = it->second;
            const auto status = std::from_chars(value.data(), value.data() + value.size(), result, 10);
            switch (status.ec) {
            case std::errc::invalid_argument:
                // throw std::runtime_error("Invalid integer string " + value);
                return default_;
            case std::errc::result_out_of_range:
                // throw std::runtime_error("Integer string out of range " + value);
                return default_;
            default:
                if (status.ptr != value.data() + value.size()) {
                    // throw std::runtime_error("Trailing garbage in integer string " + value);
                    return default_;
                }
                break;
            }
        }
        return result;
    } else {
        static_assert(dependent_false<T>, "No conversion implemented for type");
    }
}

template<typename T>
[[nodiscard]] T
getBackendParam(const nixl_b_params_t *params, const std::string &key, const T default_) {
    return (params != nullptr) ? getBackendParam<T>(*params, key, default_) : default_;
}

} // namespace nixl

#endif
