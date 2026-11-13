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
#ifndef NIXL_SRC_CORE_TRACING_TRACE_IDENTITY_H
#define NIXL_SRC_CORE_TRACING_TRACE_IDENTITY_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace nixl::trace {

class Tracer;

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
 * @brief Draw a request id from @p tracer's random source, again while it is
 *        zero; 0, which means none, when @p tracer is null or its source fails.
 */
[[nodiscard]] std::uint64_t
generateRequestId(const Tracer *tracer);

/**
 * @brief Draw a context for one post of @p request: its trace id and flags with
 *        a span id from @p tracer's random source, again while it is zero;
 *        @p request itself when @p tracer is null or its source fails.
 */
[[nodiscard]] TraceContext
generatePostContext(const TraceContext &request, const Tracer *tracer);

} // namespace nixl::trace

#endif
