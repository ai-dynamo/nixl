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

#ifndef NIXL_SRC_UTILS_COMMON_CONFIGURATION_H
#define NIXL_SRC_UTILS_COMMON_CONFIGURATION_H

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <strings.h>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <absl/strings/str_join.h>

#include "nixl_log.h"
#include "nixl_types.h"

namespace nixl::config {

[[nodiscard]] inline std::optional<std::string>
getenvOptional(const std::string &name) {
    if (const char *value = std::getenv(name.c_str())) {
        NIXL_DEBUG << "Obtained environment variable " << name << "=" << value;
        return std::string(value);
    }
    NIXL_DEBUG << "Missing environment variable " << name;
    return std::nullopt;
}

[[nodiscard]] inline std::string
getenvDefaulted(const std::string &name, const std::string &fallback) {
    if (const char *value = std::getenv(name.c_str())) {
        NIXL_DEBUG << "Obtained environment variable " << name << "=" << value;
        return std::string(value);
    }
    NIXL_DEBUG << "Using default '" << fallback << "' for missing environment variable " << name;
    return fallback;
}

template<typename, typename = void> struct convertTraits;

template<> struct convertTraits<bool> {
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

        const std::string msg = "Conversion to bool failed for string '" + value + "' known are " +
            absl::StrJoin(positive, ", ") + " as positive and " + absl::StrJoin(negative, ", ") +
            " as negative (case insensitive)";
        throw std::runtime_error(msg);
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

template<> struct convertTraits<std::string> {
    [[nodiscard]] static std::string
    convert(const std::string &value) {
        return value;
    }
};

template<typename integer> struct integralTraits {
    [[nodiscard]] static integer
    convert(const std::string &value) {
        integer result;
        const auto status =
            std::from_chars(start(value), value.data() + value.size(), result, base(value));
        switch (status.ec) {
        case std::errc::invalid_argument:
            throw std::runtime_error("Invalid integer string '" + value + "' for type " +
                                     typeid(integer).name());
        case std::errc::result_out_of_range:
            throw std::runtime_error("Integer string '" + value + "' out of range for type " +
                                     typeid(integer).name());
        default:
            if (status.ptr != value.data() + value.size()) {
                throw std::runtime_error("Trailing garbage in integer string '" + value + "'");
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

template<typename integer>
struct convertTraits<integer, std::enable_if_t<std::is_integral_v<integer>>>
    : integralTraits<integer> {};

template<typename type, template<typename...> class traits = convertTraits>
[[nodiscard]] nixl_status_t
getValueWithStatus(type &result, const std::string &env) {
    if (const auto opt = getenvOptional(env)) {
        try {
            result = traits<std::decay_t<type>>::convert(*opt);
            return NIXL_SUCCESS;
        }
        catch (const std::exception &e) {
            NIXL_DEBUG << "Unable to convert value '" << *opt << "' from environment variable '"
                       << env << "' to target type " << typeid(type).name();
            return NIXL_ERR_MISMATCH;
        }
    }
    return NIXL_ERR_NOT_FOUND;
}

template<typename type, template<typename...> class traits = convertTraits>
[[nodiscard]] type
getValue(const std::string &env) {
    if (const auto opt = getenvOptional(env)) {
        return traits<type>::convert(*opt);
    }
    throw std::runtime_error("Missing environment variable '" + env + "'");
}

template<typename type, template<typename...> class traits = convertTraits>
[[nodiscard]] std::optional<type>
getValueOptional(const std::string &env) {
    if (const auto opt = getenvOptional(env)) {
        return traits<type>::convert(*opt);
    }
    return std::nullopt;
}

template<typename type, template<typename...> class traits = convertTraits>
[[nodiscard]] type
getValueDefaulted(const std::string &env, const type &fallback) {
    return getValueOptional<type, traits>(env).value_or(fallback);
}

} // namespace nixl::config

#endif
