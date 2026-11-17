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
#ifndef NIXL_SRC_CORE_TRACING_API_DETAIL_SPAN_BACKEND_H
#define NIXL_SRC_CORE_TRACING_API_DETAIL_SPAN_BACKEND_H

#include <cstdint>
#include <span>
#include <string_view>
#include <variant>

#include "tracing/api/attribute.h"
#include "tracing/api/attribute_key.h"

namespace nixl::trace {

/**
 * @brief A single active span within one backend. Its destructor ends the
 *        range / fixes the duration for that backend.
 */
class SpanBackend {
public:
    virtual ~SpanBackend() = default;

    virtual void
    addAttribute(AttributeKey key, std::string_view value) = 0;
    virtual void
    addAttribute(AttributeKey key, std::int64_t value) = 0;
    virtual void
    addAttribute(AttributeKey key, double value) = 0;

    virtual void
    addAttributes(std::span<const Attribute> attrs) {
        for (const Attribute &attr : attrs) {
            std::visit([&](const auto value) { addAttribute(attr.key, value); }, attr.value);
        }
    }
};

} // namespace nixl::trace

#endif
