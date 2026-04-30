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
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <strings.h>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <absl/strings/str_join.h>

#include <toml++/toml.hpp>

#include "nixl_log.h"
#include "nixl_types.h"

// General design guideline for configuration handling:
// - When something does not exist and there is a fallback, use the fallback.
// - When something does exist and there is an error using it, propagate the error.
// Using the fallback in case of errors only hides the error and can lead to
// unintended effects when mistakes are hidden or ignored instead of being fixed.

namespace nixl::config {

namespace internal {

    template<typename... Ts>
    [[noreturn]] void
    throwRuntimeError(Ts &&...ts) {
        std::ostringstream oss;
        (void)(oss << ... << ts);
        throw std::runtime_error(std::move(oss).str());
    }

    [[nodiscard]] std::optional<std::string>
    getenvOptional(const std::string &name);

    [[nodiscard]] std::string
    getenvDefaulted(const std::string &name, const std::string &fallback);

    [[nodiscard]] toml::node_view<const toml::node>
    findTomlNode(const toml::path &path);

    [[nodiscard]] toml::node_view<const toml::node>
    findTomlNode(const std::string &path);

    void
    warnIgnoreToml(const std::string &path);

    template<typename, typename = void> struct convertTraits;

    template<> struct convertTraits<bool> {
        [[nodiscard]] static bool
        convert(const std::string &value) {
            static const std::vector<std::string> positive = {
                "y", "yes", "on", "1", "true", "enable"};

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

        [[nodiscard]] static bool
        convert(const toml::node_view<const toml::node> &view) {
            if (const auto *node = view.as_boolean()) {
                return node->get();
            }
            throwRuntimeError("Invalid TOML type '", view.type(), "' for Boolean");
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

        [[nodiscard]] static std::string
        convert(const toml::node_view<const toml::node> &view) {
            if (const auto *node = view.as_string()) {
                return node->get();
            }
            throwRuntimeError("Invalid TOML type '", view.type(), "' for string");
        }
    };

    template<> struct convertTraits<std::filesystem::path> {
        [[nodiscard]] static std::filesystem::path
        convert(const std::string &value) {
            return std::filesystem::path(value);
        }

        [[nodiscard]] static std::filesystem::path
        convert(const toml::node_view<const toml::node> &view) {
            return std::filesystem::path(convertTraits<std::string>::convert(view));
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

        [[nodiscard]] static integer
        convert(const toml::node_view<const toml::node> &view) {
            if (const auto *node = view.as_integer()) {
                const auto value = node->get();
                if (in_range(value)) {
                    return integer(value);
                }
                throwRuntimeError(
                    "Integer value '", value, "' out of range for type ", typeid(integer).name());
            }
            throwRuntimeError("Invalid TOML type '", view.type(), "' for integer");
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

        template<typename T>
        [[nodiscard]] static bool
        in_range(const T value) noexcept {
            static_assert(std::is_signed_v<T>);
            if constexpr (std::is_signed_v<integer>) {
                return value >= std::numeric_limits<integer>::min() &&
                    value <= std::numeric_limits<integer>::max();
            } else {
                return value >= 0 &&
                    static_cast<uint64_t>(value) <= std::numeric_limits<integer>::max();
            }
        }
    };

    // Error out for now, in case plain char will be used for strings of length 1.
    // Please use the integer types signed char or unsigned char for 8-bit integers.
    template<> struct convertTraits<char> {};

    template<typename integer>
    struct convertTraits<integer, std::enable_if_t<std::is_integral_v<integer>>>
        : integralTraits<integer> {};

    template<> struct convertTraits<std::chrono::milliseconds> {
        [[nodiscard]] static std::chrono::milliseconds
        convert(const std::string &value) {
            return std::chrono::milliseconds(convertTraits<uint64_t>::convert(value));
        }

        [[nodiscard]] static std::chrono::milliseconds
        convert(const toml::node_view<const toml::node> &view) {
            if (const auto *node = view.as_time()) {
                const auto &time = node->get();
                return std::chrono::milliseconds((time.hour * 3600000) + (time.minute * 60000) +
                                                 (time.second * 1000) +
                                                 (time.nanosecond / 1000000));
            }
            if (const auto *node = view.as_integer()) {
                return std::chrono::milliseconds(node->get());
            }
            throwRuntimeError("Invalid TOML type '", view.type(), "' for milliseconds");
        }
    };

} // namespace internal

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] nixl_status_t
getValueWithStatus(type &result, const std::string &env) {
    if (const auto opt = internal::getenvOptional(env)) {
        try {
            result = traits<std::decay_t<type>>::convert(*opt);
        }
        catch (const std::exception &e) {
            NIXL_DEBUG << "Unable to convert environment variable '" << env << "' to target type "
                       << typeid(type).name();
            return NIXL_ERR_MISMATCH;
        }
        internal::warnIgnoreToml(env);
        return NIXL_SUCCESS;
    }

    if (const auto view = internal::findTomlNode(env)) {
        try {
            result = traits<std::decay_t<type>>::convert(view);
        }
        catch (const std::exception &e) {
            NIXL_DEBUG << "Unable to convert config value '" << env << "' to target type "
                       << typeid(type).name();
            return NIXL_ERR_MISMATCH;
        }
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_NOT_FOUND;
}

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] type
getValue(const std::string &env) {
    if (const auto opt = internal::getenvOptional(env)) {
        auto result = traits<type>::convert(*opt);
        internal::warnIgnoreToml(env);
        return result;
    }

    if (const auto view = internal::findTomlNode(env)) {
        return traits<type>::convert(view);
    }
    internal::throwRuntimeError("Missing config entry '", env, "'");
}

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] std::optional<type>
getValueOptional(const std::string &env) {
    if (const auto opt = internal::getenvOptional(env)) {
        auto result = traits<type>::convert(*opt);
        internal::warnIgnoreToml(env);
        return result;
    }

    if (const auto view = internal::findTomlNode(env)) {
        return traits<type>::convert(view);
    }
    return std::nullopt;
}

template<typename type, template<typename...> class traits = internal::convertTraits>
[[nodiscard]] type
getValueDefaulted(const std::string &env, const type &fallback) {
    return getValueOptional<type, traits>(env).value_or(fallback);
}

[[nodiscard]] inline std::string
getNonEmptyString(const std::string &env) {
    const std::string result = getValue<std::string>(env);

    if (result.empty()) {
        internal::throwRuntimeError("Config parameter '", env, "' needs non-empty value");
    }
    return result;
}

[[nodiscard]] inline bool
checkExistence(const std::string &env) {
    return (std::getenv(env.c_str()) != nullptr) || bool(internal::findTomlNode(env));
}

} // namespace nixl::config

#endif
