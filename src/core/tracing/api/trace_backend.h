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
#ifndef NIXL_SRC_CORE_TRACING_API_TRACE_BACKEND_H
#define NIXL_SRC_CORE_TRACING_API_TRACE_BACKEND_H

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "common/nixl_time.h"
#include "tracing/api/attribute.h"
#include "tracing/api/detail/span.h"
#include "tracing/api/detail/trace_impl.h"
#include "tracing/api/record.h"
#include "tracing/api/trace_identity.h"

namespace nixl::trace {

/**
 * @brief One enabled backend type (NVTX, Chakra, ...). Each record names the
 *        span id of its context, pushed or passed, as its parent, except the
 *        span beginContextSpan() begins or emitContextSpan() records, which
 *        has that id as its own and the context's parentSpanId, when set, as
 *        its parent.
 */
class TraceBackend {
public:
    virtual ~TraceBackend() = default;

    [[nodiscard]] virtual std::unique_ptr<SpanBackend>
    beginSpan(Name name, Kind kind) = 0;

    virtual void
    pushContext(const TraceContext &context, std::uint64_t request_id) = 0;
    virtual void
    popContext() = 0;

    [[nodiscard]] virtual std::string_view
    name() const noexcept = 0;

    [[nodiscard]] virtual bool
    ownsTimestamps() const noexcept {
        return false;
    }

    [[nodiscard]] virtual bool
    recording() const noexcept {
        return true;
    }

    /**
     * @brief Record a span from @p start to @p end, or an event at @p at, that
     *        already happened. Times are `nixlTime` microseconds on
     *        `std::chrono::steady_clock`; @p context is the record's own, not
     *        the pushed one, and an invalid context means none. @p request_id
     *        is likewise the record's own, and 0 means none.
     */
    virtual void
    emitSpan(Name name,
             Kind kind,
             const TraceContext &context,
             std::uint64_t request_id,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const Attribute> attrs) = 0;
    virtual void
    emitEvent(Name name,
              Kind kind,
              const TraceContext &context,
              std::uint64_t request_id,
              nixlTime::us_t at,
              std::span<const Attribute> attrs) = 0;

    /**
     * @brief Begin the span that @p context identifies, rather than a span
     *        inside the pushed context. @p context and @p request_id are the
     *        span's own, as in emitSpan().
     */
    [[nodiscard]] virtual std::unique_ptr<SpanBackend>
    beginContextSpan(Name name,
                     Kind kind,
                     const TraceContext &context,
                     std::uint64_t request_id) = 0;

    /**
     * @brief Record the span that @p context identifies, as beginContextSpan()
     *        would begin it, from @p start to @p end, that already happened.
     *        Times are as in emitSpan().
     */
    virtual void
    emitContextSpan(Name name,
                    Kind kind,
                    const TraceContext &context,
                    std::uint64_t request_id,
                    nixlTime::us_t start,
                    nixlTime::us_t end,
                    std::span<const Attribute> attrs) = 0;
};

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_API_TRACE_BACKEND_H
