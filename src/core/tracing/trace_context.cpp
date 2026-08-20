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
#include "trace_context.h"

#include <algorithm>

#include "common/uuid_v4.h"

template<std::size_t Size>
[[nodiscard]] static bool
allZero(const std::array<std::uint8_t, Size> &bytes) noexcept {
    return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t byte) { return byte == 0; });
}

[[nodiscard]] static int
hexValue(char value) noexcept {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

[[nodiscard]] static bool
parseByte(std::string_view value, std::size_t offset, std::uint8_t &result) noexcept {
    const int high = hexValue(value[offset]);
    const int low = hexValue(value[offset + 1]);
    if (high < 0 || low < 0) {
        return false;
    }
    result = static_cast<std::uint8_t>((high << 4) | low);
    return true;
}

template<std::size_t Size>
[[nodiscard]] static bool
parseBytes(std::string_view value,
           std::size_t offset,
           std::array<std::uint8_t, Size> &result) noexcept {
    for (std::size_t index = 0; index < Size; ++index) {
        if (!parseByte(value, offset + index * 2, result[index])) {
            return false;
        }
    }
    return true;
}

static void
appendByte(std::string &result, std::uint8_t value) {
    constexpr std::array<char, 16> hex{
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    result.push_back(hex[value >> 4]);
    result.push_back(hex[value & 0x0f]);
}

bool
nixl::trace::TraceContext::valid() const noexcept {
    return !allZero(traceId) && !allZero(spanId);
}

bool
nixl::trace::TraceContext::sampled() const noexcept {
    return (flags & 0x01) != 0;
}

std::uint64_t
nixl::trace::TraceContext::correlationId64() const noexcept {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < sizeof(result); ++index) {
        result = (result << 8) | traceId[index];
    }
    return result;
}

std::optional<nixl::trace::TraceContext>
nixl::trace::parseTraceparent(std::string_view value) {
    if (value.size() != 55 || value[0] != '0' || value[1] != '0' || value[2] != '-' ||
        value[35] != '-' || value[52] != '-') {
        return std::nullopt;
    }

    nixl::trace::TraceContext context;
    if (!parseBytes(value, 3, context.traceId) || !parseBytes(value, 36, context.spanId) ||
        !parseByte(value, 53, context.flags) || !context.valid()) {
        return std::nullopt;
    }
    return context;
}

std::string
nixl::trace::formatTraceparent(const nixl::trace::TraceContext &context) {
    if (!context.valid()) {
        return {};
    }

    std::string result;
    result.reserve(55);
    result.append("00-");
    for (const auto byte : context.traceId) {
        appendByte(result, byte);
    }
    result.push_back('-');
    for (const auto byte : context.spanId) {
        appendByte(result, byte);
    }
    result.push_back('-');
    appendByte(result, context.flags);
    return result;
}

nixl::trace::TraceContext
nixl::trace::generateTraceContext() {
    nixl::trace::TraceContext context;
    do {
        context.traceId = nixl::UUIDv4{}.get_data();
    } while (allZero(context.traceId));

    do {
        const auto span_source = nixl::UUIDv4{}.get_data();
        std::copy_n(span_source.begin(), context.spanId.size(), context.spanId.begin());
    } while (allZero(context.spanId));
    return context;
}
