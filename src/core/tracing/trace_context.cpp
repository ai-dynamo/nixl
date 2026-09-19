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
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <absl/strings/numbers.h>

#include "common/configuration.h"
#include "common/nixl_log.h"
#include "common/uuid_v4.h"
#include "trace.h"
#include "trace_context.h"

namespace {
constexpr std::uint8_t supported_trace_flags = 0x03;
constexpr std::size_t wire_flags_offset = 1;
constexpr std::size_t wire_trace_id_offset = 2;
constexpr std::size_t wire_span_id_offset = 18;
constexpr std::size_t trace_id_randomness_bytes = 7;
constexpr double two_pow_56 = 0x1p56;

template<std::size_t Size>
[[nodiscard]] bool
isAllZero(const std::array<std::uint8_t, Size> &bytes) noexcept {
    return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t byte) { return byte == 0; });
}

[[nodiscard]] int
hexValue(char value) noexcept {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return -1;
}

[[nodiscard]] bool
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
[[nodiscard]] bool
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

[[nodiscard]] std::optional<double>
parseSampleRatio(const std::string &value) {
    double ratio = 0.0;
    if (!absl::SimpleAtod(value, &ratio) || !std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        return std::nullopt;
    }
    return ratio;
}

[[nodiscard]] std::uint64_t
traceIdRandomness(const nixl::trace::TraceContext &context) noexcept {
    std::uint64_t result = 0;
    for (auto index = context.traceId.size() - trace_id_randomness_bytes;
         index < context.traceId.size();
         ++index) {
        result = (result << 8) | context.traceId[index];
    }
    return result;
}

void
generateInto(nixl::trace::TraceContext &context, double sample_ratio) {
    do {
        nixl::generateRandomBytes(context.traceId.data(), context.traceId.size());
    } while (isAllZero(context.traceId));

    do {
        nixl::generateRandomBytes(context.spanId.data(), context.spanId.size());
    } while (isAllZero(context.spanId));

    context.flags = 0x02;
    if (nixl::trace::sampledByRatio(context, sample_ratio)) {
        context.flags |= 0x01;
    }
}

void
appendByte(std::string &result, std::uint8_t value) {
    constexpr std::array<char, 16> hex{
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    result.push_back(hex[value >> 4]);
    result.push_back(hex[value & 0x0f]);
}
} // namespace

nixl::trace::TraceContext::TraceContext(const nixl::trace::Tracer *tracer) {
    if (tracer != nullptr) {
        generateInto(*this, tracer->sampleRatio());
    }
}

bool
nixl::trace::TraceContext::valid() const noexcept {
    return !isAllZero(traceId) && !isAllZero(spanId);
}

bool
nixl::trace::TraceContext::sampled() const noexcept {
    return (flags & 0x01) != 0;
}

std::uint64_t
nixl::trace::TraceContext::correlationId64() const noexcept {
    if (!valid()) {
        return 0;
    }

    std::uint64_t result = 0;
    for (const auto byte : spanId) {
        result = (result << 8) | byte;
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
    appendByte(result, context.flags & supported_trace_flags);
    return result;
}

std::optional<std::size_t>
nixl::trace::encodeTraceContext(const nixl::trace::TraceContext &context,
                                std::span<std::uint8_t> buffer) {
    if (!context.valid() || buffer.size() < nixl::trace::traceContextWireSize) {
        return std::nullopt;
    }

    buffer[0] = nixl::trace::traceContextWireVersion;
    buffer[wire_flags_offset] = context.flags & supported_trace_flags;
    std::copy(context.traceId.begin(),
              context.traceId.end(),
              buffer.begin() + static_cast<std::ptrdiff_t>(wire_trace_id_offset));
    std::copy(context.spanId.begin(),
              context.spanId.end(),
              buffer.begin() + static_cast<std::ptrdiff_t>(wire_span_id_offset));
    return nixl::trace::traceContextWireSize;
}

nixl::trace::WireDecodeResult
nixl::trace::decodeTraceContext(std::span<const std::uint8_t> buffer,
                                nixl::trace::TraceContext &context) {
    if (buffer.empty()) {
        return nixl::trace::WireDecodeResult::Malformed;
    }
    if (buffer[0] != nixl::trace::traceContextWireVersion) {
        return nixl::trace::WireDecodeResult::UnknownVersion;
    }
    if (buffer.size() != nixl::trace::traceContextWireSize) {
        return nixl::trace::WireDecodeResult::Malformed;
    }

    context.flags = buffer[wire_flags_offset] & supported_trace_flags;
    std::copy_n(buffer.begin() + static_cast<std::ptrdiff_t>(wire_trace_id_offset),
                context.traceId.size(),
                context.traceId.begin());
    std::copy_n(buffer.begin() + static_cast<std::ptrdiff_t>(wire_span_id_offset),
                context.spanId.size(),
                context.spanId.begin());
    if (!context.valid()) {
        return nixl::trace::WireDecodeResult::Malformed;
    }

    return nixl::trace::WireDecodeResult::Ok;
}

bool
nixl::trace::sampledByRatio(const nixl::trace::TraceContext &context, double ratio) noexcept {
    if (!context.valid() || !(ratio > 0.0)) {
        return false;
    }

    const double kept = ratio * two_pow_56;
    if (kept >= two_pow_56) {
        return true;
    }
    // Subtract in integers: (1 - ratio) * two_pow_56 rounds to two_pow_56 for
    // every ratio below 2^-53, which would turn small probabilities into never.
    const auto threshold =
        static_cast<std::uint64_t>(two_pow_56) - static_cast<std::uint64_t>(kept);
    return traceIdRandomness(context) >= threshold;
}

double
nixl::trace::resolveTraceSampleRatio() {
    const auto spec =
        nixl::config::getValueOptional<std::string>(std::string(nixl::trace::traceSampleRatioVar));
    if (spec) {
        if (spec->empty()) {
            return 0.0;
        }

        const auto ratio = parseSampleRatio(*spec);
        if (!ratio) {
            throw std::invalid_argument(std::string(nixl::trace::traceSampleRatioVar) + "='" +
                                        *spec + "' must be a number between 0 and 1");
        }
        return *ratio;
    }

    const auto otel_spec = nixl::config::getValueOptional<std::string>(
        std::string(nixl::trace::otelTracesSampleRatioVar));
    if (!otel_spec || otel_spec->empty()) {
        return 0.0;
    }

    const auto ratio = parseSampleRatio(*otel_spec);
    if (!ratio) {
        NIXL_WARN << "nixl::trace: ignoring " << nixl::trace::otelTracesSampleRatioVar << "='"
                  << *otel_spec << "': not a number between 0 and 1";
        return 0.0;
    }
    return *ratio;
}

nixl::trace::TraceContext
nixl::trace::generateTraceContext(double sample_ratio) {
    nixl::trace::TraceContext context;
    generateInto(context, sample_ratio);
    return context;
}
