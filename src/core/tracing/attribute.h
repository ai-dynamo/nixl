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
#ifndef NIXL_SRC_CORE_TRACING_ATTRIBUTE_H
#define NIXL_SRC_CORE_TRACING_ATTRIBUTE_H

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

#include "tracing/attribute_key.h"

namespace nixl::trace {

/**
 * @brief An attribute value as a record holds it: integers and enums as
 *        std::int64_t, floating-point values as double, strings as views.
 */
template<typename T>
    requires std::is_integral_v<T> || std::is_enum_v<T>
[[nodiscard]] constexpr std::int64_t
attributeValue(T value) noexcept {
    return static_cast<std::int64_t>(value);
}

template<typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]] constexpr double
attributeValue(T value) noexcept {
    return static_cast<double>(value);
}

[[nodiscard]] constexpr std::string_view
attributeValue(std::string_view value) noexcept {
    return value;
}

/**
 * @brief Typed attribute of an emitted span or event. The key is a string
 *        literal; a string value is a view, valid only for the call.
 */
struct Attribute {
    template<typename Value>
    constexpr Attribute(AttributeKey attr_key, const Value &attr_value) noexcept
        : key(attr_key),
          value(attributeValue(attr_value)) {}

    AttributeKey key;
    std::variant<std::string_view, std::int64_t, double> value;
};

} // namespace nixl::trace

#endif
