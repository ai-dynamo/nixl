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

#ifndef NIXL_SRC_UTILS_COMMON_CONFIG_TRAITS_H
#define NIXL_SRC_UTILS_COMMON_CONFIG_TRAITS_H

#include "exception.h"

#include <absl/strings/str_join.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <string>
#include <strings.h>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace nixl::config {

template<typename, typename = void> struct configTraits;

template<> struct configTraits<bool> {
    [[nodiscard]] static bool
    convert(const std::string &value) {
        static const std::vector<std::string> positive = {"y", "yes", "on", "1", "true", "enable"};

        static const std::vector<std::string> negative = {
            "n", "no", "off", "0", "false", "disable"};

        if (match(value, positive)) {
            return true;
        }

        if (match(value, negative)) {
            return false;
        }

        throwRuntimeError("Conversion to bool failed for string '",
                          value,
                          "' known are ",
                          absl::StrJoin(positive, ", "),
                          " as positive and ",
                          absl::StrJoin(negative, ", "),
                          " as negative (case insensitive)");
    }

private:
    [[nodiscard]] static bool
    match(const std::string &value, const std::vector<std::string> &haystack) noexcept {
        const auto pred = [&](const std::string &ref) {
            return strcasecmp(ref.c_str(), value.c_str()) == 0;
        };
        return std::find_if(haystack.begin(), haystack.end(), pred) != haystack.end();
    }
};

template<> struct configTraits<std::string> {
    [[nodiscard]] static std::string
    convert(const std::string &value) {
        return value;
    }
};

template<> struct configTraits<std::filesystem::path> {
    [[nodiscard]] static std::filesystem::path
    convert(const std::string &value) {
        return std::filesystem::path(value);
    }
};

template<typename integer> struct integralStringTraits {
    [[nodiscard]] static integer
    convert(const std::string &value) {
        integer result;
        const auto status =
            std::from_chars(start(value), value.data() + value.size(), result, base(value));
        switch (status.ec) {
        case std::errc::invalid_argument:
            throwRuntimeError(
                "Invalid integer string '", value, "' for type ", typeid(integer).name());
        case std::errc::result_out_of_range:
            throwRuntimeError(
                "Integer string '", value, "' out of range for type ", typeid(integer).name());
        default:
            if (status.ptr != value.data() + value.size()) {
                throwRuntimeError("Trailing garbage in integer string '", value, "'");
            }
            break;
        }
        return result;
    }

private:
    [[nodiscard]] static bool
    isHex(const std::string &value) noexcept {
        return std::is_unsigned_v<integer> && (value.size() > 2) && (value[0] == '0') &&
            ((value[1] == 'x') || (value[1] == 'X'));
    }

    [[nodiscard]] static int
    base(const std::string &value) noexcept {
        return isHex(value) ? 16 : 10;
    }

    [[nodiscard]] static const char *
    start(const std::string &value) noexcept {
        return value.data() + (isHex(value) ? 2 : 0);
    }
};

// Error out for now, in case plain char will be used for strings of length 1.
// Please use the integer types signed char or unsigned char for 8-bit integers.
template<> struct configTraits<char> {};

template<typename integer>
struct configTraits<integer, std::enable_if_t<std::is_integral_v<integer>>>
    : integralStringTraits<integer> {};

template<> struct configTraits<std::chrono::milliseconds> {
    [[nodiscard]] static std::chrono::milliseconds
    convert(const std::string &value) {
        return std::chrono::milliseconds(configTraits<uint64_t>::convert(value));
    }
};

} // namespace nixl::config

#endif
