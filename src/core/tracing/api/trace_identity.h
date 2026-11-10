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
#ifndef NIXL_SRC_CORE_TRACING_API_TRACE_IDENTITY_H
#define NIXL_SRC_CORE_TRACING_API_TRACE_IDENTITY_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "nixl_types.h"

namespace nixl::trace {

class Tracer;

inline constinit const nixl_blob_t noSuppliedContext{};

struct TraceContext {
    std::array<std::uint8_t, 16> traceId{};
    std::array<std::uint8_t, 8> spanId{};
    std::uint8_t flags{};
    std::array<std::uint8_t, 8> parentSpanId{};

    [[nodiscard]] bool
    valid() const noexcept;

    [[nodiscard]] bool
    sampled() const noexcept;

    [[nodiscard]] bool
    operator==(const TraceContext &) const noexcept = default;
};

struct TraceIdentity {
    TraceContext context;
    std::uint64_t requestId = 0;
    std::uint64_t execution = 0;
};

[[nodiscard]] std::optional<TraceContext>
parseTraceparent(std::string_view value);

[[nodiscard]] std::string
formatTraceparent(const TraceContext &context);

/**
 * @brief The context an application supplied as @p value, a `traceparent` or a
 *        trace context record, with only the flags this version defines;
 *        std::nullopt when @p value is empty, @p tracer is null, or @p value is
 *        neither form, which @p tracer reports once.
 */
[[nodiscard]] std::optional<TraceContext>
parseParentContext(const Tracer *tracer, const nixl_blob_t &value);

[[nodiscard]] TraceIdentity
generateRequestIdentity(const Tracer *tracer, const nixl_blob_t &parent = noSuppliedContext);

/**
 * @brief Draw a context for one post of @p request: its trace id and flags with
 *        a span id from @p tracer's random source, again while it is zero, and
 *        its span id as the parent; @p request itself when @p tracer is null or
 *        its source fails.
 */
[[nodiscard]] TraceContext
generatePostContext(const TraceContext &request, const Tracer *tracer);

} // namespace nixl::trace

#endif
