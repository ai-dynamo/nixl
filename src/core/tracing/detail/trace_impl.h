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
#ifndef NIXL_SRC_CORE_TRACING_DETAIL_TRACE_IMPL_H
#define NIXL_SRC_CORE_TRACING_DETAIL_TRACE_IMPL_H

#include <cstdint>
#include <span>
#include <string_view>

#include "common/nixl_time.h"
#include "tracing/trace_identity.h"

namespace nixl::trace {

enum class Kind : std::uint8_t;
struct Attribute;
struct SpanId;
class Span;
class Tracer;

/**
 * @brief A single active span within one backend. Its destructor ends the
 *        range / fixes the duration for that backend.
 */
class SpanBackend {
public:
    virtual ~SpanBackend() = default;

    virtual void
    addAttribute(std::string_view key, std::string_view value) = 0;
    virtual void
    addAttribute(std::string_view key, std::int64_t value) = 0;
    virtual void
    addAttribute(std::string_view key, double value) = 0;

    virtual void
    addCtrlDep(SpanId parent) = 0;
    virtual void
    addDataDep(SpanId parent) = 0;

    [[nodiscard]] virtual SpanId
    id() const noexcept = 0;
};

} // namespace nixl::trace

namespace nixl::trace::detail {

[[nodiscard]] Span
beginSpan(Tracer &tracer, std::string_view name, Kind kind);

[[nodiscard]] Span
beginContextSpan(Tracer &tracer,
                 std::string_view name,
                 Kind kind,
                 const TraceContext &context,
                 std::uint64_t request_id);

void
mark(Tracer &tracer, std::string_view name, Kind kind);

void
pushContext(Tracer &tracer, const TraceContext &context, std::uint64_t request_id);

void
popContext(Tracer &tracer);

void
emitEvent(Tracer &tracer,
          std::string_view name,
          Kind kind,
          const TraceContext &context,
          std::uint64_t request_id,
          nixlTime::us_t at,
          std::span<const Attribute> attrs);

} // namespace nixl::trace::detail

#endif
