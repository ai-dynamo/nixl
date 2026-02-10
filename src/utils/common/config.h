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

#ifndef NIXL_SRC_UTILS_COMMON_CONFIG_H
#define NIXL_SRC_UTILS_COMMON_CONFIG_H

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <strings.h> // strcasecmp
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "nixl_types.h"

#include "common/nixl_log.h"
#include "common/str_tools.h"

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
    return getenvOptional(name).value_or(fallback);
 }

template<typename, typename = void> struct convertTraits;

template<>
struct convertTraits<bool> {
    [[nodiscard]] static bool
    convert(const std::string &value) {
        static const std::vector<std::string> positive = {
            "y", "yes", "on", "1", "true", "enable"
        };

        static const std::vector<std::string> negative = {
            "n", "no", "off", "0", "false", "disable"
        };

        if (match(value, positive)) {
            return true;
        }

        if (match(value, negative)) {
            return false;
        }

        NIXL_ERROR << "Unknown value for bool '"
                   << value
                   << "' known are "
                   << strJoin(positive)
                   << " as positive and "
                   << strJoin(negative)
                   << " as negative (case insensitive)";
        throw std::runtime_error("Conversion to bool failed");
    }

private:
    [[nodiscard]] static bool
    match(const std::string &value, const std::vector<std::string> &haystack) noexcept {
        static const auto pred = [&](const std::string &ref) {
            return strcasecmp(ref.c_str(), value.c_str()) == 0;
        };
        return std::find_if(haystack.begin(), haystack.end(), pred) != haystack.end();
    }   
};

template<>
struct convertTraits<std::string> {
    [[nodiscard]] static std::string
    convert(const std::string &value) {
        return value;
    }
};

template<typename integer, typename longLong>
struct integerTraits {
    static_assert(std::is_signed_v<integer> == std::is_signed_v<longLong>);
    static_assert(std::is_unsigned_v<integer> == std::is_unsigned_v<longLong>);

    [[nodiscard]] static integer
    convert(const std::string &value) {
        size_t pos = 0;
        const longLong ll = std::is_unsigned_v<integer> ? std::stoull(value, &pos) : std::stoll(value, &pos);
        if (pos != value.size()) {
            throw std::runtime_error("Invalid integer");
        }
        if constexpr (sizeof(integer) < sizeof(longLong)) {
            if (longLong(integer(ll)) != ll) {
                throw std::runtime_error("Integer overflow");
            }
        }
        return integer(ll);
    }
};

template<typename integer>
struct convertTraits<integer, std::enable_if_t<std::is_signed_v<integer>>>
    : integerTraits<integer, long long> {};

template<typename integer>
struct convertTraits<integer, std::enable_if_t<std::is_unsigned_v<integer>>>
    : integerTraits<integer, unsigned long long> {};

template<typename type, template<typename...> class traits = convertTraits>
[[nodiscard]] nixl_status_t
getNothrow(type &result, const std::string &env) {
    if (const auto opt = getenvOptional(env)) {
        try {
            result = traits<std::decay_t<type>>::convert(*opt);
            return NIXL_SUCCESS;
        }
        catch(const std::exception &e) {
            NIXL_DEBUG << "Unable to convert value '"
                       << *opt
                       << "' from environment variable '"
                       << env
                       << "' to target type "
                       << typeid(type).name();
            // TODO: Demangle? Manual name?
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
getOptional(const std::string &env) {
    if (const auto opt = getenvOptional(env)) {
        return traits<type>::convert(*opt);
    }
    return std::nullopt;
}

template<typename type, template<typename...> class traits = convertTraits>
[[nodiscard]] type
getDefaulted(const std::string &env, const type &fallback) {
    return getOptional<type, traits>(env).value_or(fallback);
}
 
} // namespace nixl::config

#endif
