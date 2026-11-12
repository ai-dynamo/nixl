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
#ifndef NIXL_SRC_PLUGINS_TRACING_CORE_TRACE_CONTEXT_H
#define NIXL_SRC_PLUGINS_TRACING_CORE_TRACE_CONTEXT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "tracing/api/trace_identity.h"

class nixlSerDes;

namespace nixl::trace {

/**
 * @brief Wire encoding of a trace context, fixed at 26 bytes:
 *        byte 0 version, byte 1 flags, bytes 2-17 trace id, bytes 18-25 span id.
 *        The two ids are byte arrays, copied verbatim. Version first so a peer
 *        can tell a record it cannot interpret from a corrupt one without
 *        parsing the rest; the record carries no length of its own, so the
 *        carrier's framing is what delimits it. This is the only binary form of
 *        a context; the struct below is the in-memory value, not an encoding.
 */
inline constexpr std::uint8_t traceContextWireVersion = 0x01;
inline constexpr std::size_t traceContextWireSize = 26;

/**
 * @brief Outcome of decoding a wire record. UnknownVersion is a record written
 *        by a peer speaking a later version and must be ignored rather than
 *        treated as corruption; Malformed means known version but invalid
 *        encoding for that version.
 */
enum class WireDecodeResult : std::uint8_t {
    Ok,
    UnknownVersion,
    Malformed,
};

/**
 * @brief Encode @p context into the first traceContextWireSize bytes of
 *        @p buffer; any trailing bytes are left untouched so a carrier can
 *        append the record to a larger message.
 * @return Number of bytes written, so a carrier can frame or advance without
 *         naming a version's size; std::nullopt, writing nothing, when the
 *         context is invalid or the buffer is too small. Flags are masked to the
 *         bits this version defines.
 */
[[nodiscard]] std::optional<std::size_t>
encodeTraceContext(const TraceContext &context, std::span<std::uint8_t> buffer);

/**
 * @brief Decode one wire record from @p buffer.
 * @return Ok only for a version-1 record of exactly traceContextWireSize bytes
 *         carrying non-zero ids. UnknownVersion leaves @p context untouched, so
 *         a later peer's record can be skipped; Malformed leaves it unspecified.
 *         Flags are masked to the bits this version defines, mirroring the
 *         encoder, so a peer's reserved bits never reach a stored context.
 */
[[nodiscard]] WireDecodeResult
decodeTraceContext(std::span<const std::uint8_t> buffer, TraceContext &context);

/**
 * @brief Head-based sampling decision, derived from the context's own trace id
 *        rather than from a random draw, so it needs no shared generator state
 *        and every peer inspecting the same context agrees with it. Only the
 *        trailing 7 bytes are read: those are what W3C Trace Context Level 2's
 *        random flag covers, while the leading bytes may carry a non-random
 *        prefix a peer put there. Sampled iff that 56-bit value is >=
 *        2^56 - round(ratio * 2^56), the OpenTelemetry consistent-probability
 *        form down to its rounding, so a downstream sampler at a lower ratio
 *        keeps a subset of what this one keeps rather than a disjoint set.
 *        Ratios below 2^-56 round to nothing kept: 56 bits cannot express a
 *        smaller probability, which is why that is also the smallest
 *        resolveTraceSampleRatio accepts.
 */
[[nodiscard]] bool
sampledByRatio(const TraceContext &context, double ratio) noexcept;

[[nodiscard]] std::optional<TraceContext>
parseTraceparent(std::string_view value);

[[nodiscard]] std::string
formatTraceparent(const TraceContext &context);

void
encodeTraceTags(nixlSerDes &ser_des, const TraceIdentity &identity);

[[nodiscard]] std::optional<TraceIdentity>
decodeTraceTags(nixlSerDes &ser_des);

class TraceparentText {
public:
    explicit TraceparentText(const TraceContext &context) noexcept;

    [[nodiscard]] std::string_view
    view() const noexcept {
        return {text_.data(), size_};
    }

private:
    std::array<char, 55> text_;
    std::uint8_t size_;
};

} // namespace nixl::trace

#endif
