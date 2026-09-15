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
#ifndef NIXL_SRC_CORE_TRACING_TRACE_CONTEXT_H
#define NIXL_SRC_CORE_TRACING_TRACE_CONTEXT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace nixl::trace {

class Tracer;

/**
 * @brief Wire encoding of a trace context, fixed at 26 bytes:
 *        byte 0 version, byte 1 flags, bytes 2-17 trace id, bytes 18-25 span id.
 *        Version first so a peer that does not know a future record can skip it
 *        instead of misreading it. This is the only binary form of a context;
 *        the struct below is the in-memory value, not an encoding.
 */
inline constexpr std::uint8_t traceContextWireVersion = 0x01;
inline constexpr std::size_t traceContextWireSize = 26;

/**
 * @brief Outcome of decoding a wire record. UnknownVersion is a record written
 *        by a peer speaking a later version and must be ignored rather than
 *        treated as corruption; Malformed is not a version-1 record at all.
 */
enum class WireDecodeResult : std::uint8_t {
    Ok,
    UnknownVersion,
    Malformed,
};

struct TraceContext {
    TraceContext() = default;
    explicit TraceContext(const Tracer *tracer);

    std::array<std::uint8_t, 16> traceId{};
    std::array<std::uint8_t, 8> spanId{};
    std::uint8_t flags{};

    [[nodiscard]] bool
    valid() const noexcept;

    [[nodiscard]] bool
    sampled() const noexcept;

    [[nodiscard]] std::uint64_t
    correlationId64() const noexcept;
};

[[nodiscard]] std::optional<TraceContext>
parseTraceparent(std::string_view value);

[[nodiscard]] std::string
formatTraceparent(const TraceContext &context);

/**
 * @brief Encode @p context into the first traceContextWireSize bytes of
 *        @p buffer; any trailing bytes are left untouched so a carrier can
 *        append the record to a larger message.
 * @return False, writing nothing, when the context is invalid or the buffer is
 *         too small. Flags are masked to the bits this version defines.
 */
[[nodiscard]] bool
encodeTraceContext(const TraceContext &context, std::span<std::uint8_t> buffer);

/**
 * @brief Decode one wire record from @p buffer.
 * @return Ok only for a version-1 record of exactly traceContextWireSize bytes
 *         carrying non-zero ids; @p context is assigned in that case only, so a
 *         skipped or rejected record leaves the caller's value intact. Flags
 *         are masked to the bits this version defines, mirroring the encoder,
 *         so a peer's reserved bits never reach a stored context.
 */
[[nodiscard]] WireDecodeResult
decodeTraceContext(std::span<const std::uint8_t> buffer, TraceContext &context);

[[nodiscard]] TraceContext
generateTraceContext();

} // namespace nixl::trace

#endif
