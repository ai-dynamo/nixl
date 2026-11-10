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
#ifndef NIXL_SRC_CORE_TRACING_API_ATTRIBUTE_KEY_H
#define NIXL_SRC_CORE_TRACING_API_ATTRIBUTE_KEY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nixl::trace {

inline constexpr std::array<std::string_view, 14> attributeKeys{
    "bytes",
    "desc_count",
    "mem_type",
    "nixl.backend",
    "nixl.end_us",
    "nixl.phase.label",
    "nixl.phase.timestamp_us",
    "nixl.remote_agent",
    "nixl.span.label",
    "nixl.start_us",
    "nixl.status",
    "nixl.xfer_execution",
    "nixl.xfer_request_id",
    "remote_agent",
};

void
attributeKeyIsNotInAttributeKeys();

class AttributeKey {
public:
    template<std::size_t Size>
    consteval AttributeKey(const char (&key)[Size]) noexcept : index_(indexOf({key, Size - 1})) {}

    [[nodiscard]] constexpr const char *
    c_str() const noexcept {
        return attributeKeys[index_].data();
    }

    [[nodiscard]] constexpr std::string_view
    view() const noexcept {
        return attributeKeys[index_];
    }

    [[nodiscard]] constexpr std::size_t
    index() const noexcept {
        return index_;
    }

private:
    [[nodiscard]] static consteval std::uint8_t
    indexOf(std::string_view key) noexcept {
        for (std::size_t i = 0; i < attributeKeys.size(); ++i) {
            if (attributeKeys[i] == key) {
                return static_cast<std::uint8_t>(i);
            }
        }
        attributeKeyIsNotInAttributeKeys();
        return 0;
    }

    std::uint8_t index_;
};

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_API_ATTRIBUTE_KEY_H
