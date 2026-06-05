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

#ifndef NIXL_SRC_UTILS_COMMON_TOML_TRAITS_H
#define NIXL_SRC_UTILS_COMMON_TOML_TRAITS_H

#include "exception.h"

#include <toml++/toml.hpp>

#include <chrono>
#include <filesystem>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace nixl::config {

template<typename, typename = void> struct tomlTraits;

template<> struct tomlTraits<bool> {
    [[nodiscard]] static bool
    convert(const toml::node_view<const toml::node> &view) {
        if (const auto *node = view.as_boolean()) {
            return node->get();
        }
        throwRuntimeError("Invalid TOML type '", view.type(), "' for Boolean");
    }
};

template<> struct tomlTraits<std::string> {
    [[nodiscard]] static std::string
    convert(const toml::node_view<const toml::node> &view) {
        if (const auto *node = view.as_string()) {
            return node->get();
        }
        throwRuntimeError("Invalid TOML type '", view.type(), "' for string");
    }
};

template<> struct tomlTraits<std::filesystem::path> {
    [[nodiscard]] static std::filesystem::path
    convert(const toml::node_view<const toml::node> &view) {
        return std::filesystem::path(tomlTraits<std::string>::convert(view));
    }
};

template<typename integer> struct integralTomlTraits {
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
template<> struct tomlTraits<char> {};

template<typename integer>
struct tomlTraits<integer, std::enable_if_t<std::is_integral_v<integer>>>
    : integralTomlTraits<integer> {};

template<> struct tomlTraits<std::chrono::milliseconds> {
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

} // namespace nixl::config

#endif

