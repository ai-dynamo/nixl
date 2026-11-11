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

#include <algorithm>
#include <array>
#include <cstdint>

#include "nixl_types.h"

namespace nixl::trace {

class Tracer;

inline constinit const nixl_blob_t noSuppliedContext{};

inline constexpr std::uint8_t sampledTraceFlag = 0x01;

struct TraceContext {
    std::array<std::uint8_t, 16> traceId{};
    std::array<std::uint8_t, 8> spanId{};
    std::uint8_t flags{};
    std::array<std::uint8_t, 8> parentSpanId{};

    [[nodiscard]] constexpr bool
    valid() const noexcept {
        const auto nonzero = [](std::uint8_t byte) { return byte != 0; };
        return std::ranges::any_of(traceId, nonzero) && std::ranges::any_of(spanId, nonzero);
    }

    [[nodiscard]] constexpr bool
    sampled() const noexcept {
        return (flags & sampledTraceFlag) != 0;
    }

    [[nodiscard]] bool
    operator==(const TraceContext &) const noexcept = default;
};

struct TraceIdentity {
    TraceContext context;
    std::uint64_t requestId = 0;
    std::uint64_t execution = 0;
};

#if !defined(NIXL_TRACE_ENABLED)

[[nodiscard]] inline TraceIdentity
generateRequestIdentity(const Tracer *, const nixl_blob_t & = noSuppliedContext) noexcept {
    return {};
}

#endif

} // namespace nixl::trace

#endif
