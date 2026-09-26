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
 *        The two ids are byte arrays, copied verbatim. Version first so a peer
 *        can tell a record it cannot interpret from a corrupt one without
 *        parsing the rest; the record carries no length of its own, so the
 *        carrier's framing is what delimits it. This is the only binary form of
 *        a context; the struct below is the in-memory value, not an encoding.
 */
inline constexpr std::uint8_t traceContextWireVersion = 0x01;
inline constexpr std::size_t traceContextWireSize = 26;

inline constexpr std::string_view traceSampleRatioVar = "NIXL_TRACE_SAMPLE_RATIO";
inline constexpr std::string_view otelTracesSampleRatioVar = "OTEL_TRACES_SAMPLE_RATIO";

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
 *        (1 - ratio) * 2^56, the OpenTelemetry consistent-probability form, so
 *        a downstream sampler at a lower ratio keeps a subset of what this one
 *        keeps rather than a disjoint set.
 */
[[nodiscard]] bool
sampledByRatio(const TraceContext &context, double ratio) noexcept;

/**
 * @brief Resolve the sampling ratio: traceSampleRatioVar when set, else
 *        otelTracesSampleRatioVar, which is Dynamo's convention rather than an
 *        OpenTelemetry one (the SDK spec defines OTEL_TRACES_SAMPLER and
 *        OTEL_TRACES_SAMPLER_ARG) but carries the same [0, 1] head-sampling
 *        meaning, so a NIXL running inside Dynamo inherits it. Unset diverges:
 *        Dynamo samples everything, NIXL samples nothing. Setting
 *        traceSampleRatioVar empty is an explicit "off" that beats the
 *        fallback, matching how NIXL_TRACE_BACKENDS is treated.
 * @return The resolved ratio, or 0 (sample nothing) when neither is usable.
 * @throws std::invalid_argument when traceSampleRatioVar holds anything other
 *         than a number in [0, 1], so a caller who asked for sampling is not
 *         silently given none. A bad value in the shared OpenTelemetry variable
 *         only warns: NIXL does not own it and must not fail construction on it.
 */
[[nodiscard]] double
resolveTraceSampleRatio();

/**
 * @brief Generate a context with fresh random ids, marked sampled when
 *        @p sample_ratio selects it by the rule sampledByRatio documents.
 * @param sample_ratio Probability in [0, 1]; the default samples nothing.
 */
[[nodiscard]] TraceContext
generateTraceContext(double sample_ratio = 0.0);

} // namespace nixl::trace

#endif
