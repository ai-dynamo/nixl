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
#ifndef NIXL_SRC_CORE_TRACING_API_TRACER_H
#define NIXL_SRC_CORE_TRACING_API_TRACER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "common/nixl_time.h"
#include "nixl_types.h"
#include "tracing/api/attribute.h"
#include "tracing/api/backend_trace.h"
#include "tracing/api/detail/span.h"
#include "tracing/api/record.h"
#include "tracing/api/trace_identity.h"

namespace nixl::trace {

class Tracer {
public:
    virtual ~Tracer() = default;

    [[nodiscard]] virtual Span
    beginSpan(Name name, Kind kind = Kind::Generic) = 0;

    [[nodiscard]] virtual Span
    beginContextSpan(Name name,
                     Kind kind,
                     const TraceContext &context,
                     std::uint64_t request_id) = 0;

    virtual void
    pushContext(const TraceContext &context, std::uint64_t request_id) = 0;

    virtual void
    popContext() = 0;

    virtual void
    emitEvent(Name name,
              Kind kind,
              const TraceContext &context,
              std::uint64_t request_id,
              std::span<const Attribute> attrs) = 0;

    virtual void
    emitSpan(Name name, Kind kind, nixlTime::us_t start) = 0;

    virtual void
    emitContextSpan(Name name,
                    Kind kind,
                    const TraceContext &context,
                    std::uint64_t request_id,
                    nixlTime::us_t start,
                    nixlTime::us_t end,
                    std::span<const Attribute> attrs) = 0;

    [[nodiscard]] virtual TraceIdentity
    generateRequestIdentity(const nixl_blob_t &parent) const = 0;

    [[nodiscard]] virtual TraceContext
    generatePostContext(const TraceContext &request) const = 0;

    [[nodiscard]] virtual std::optional<TraceContext>
    parseParentContext(const nixl_blob_t &value) const = 0;

    [[nodiscard]] virtual std::string
    traceparent(const TraceContext &context) const = 0;

    [[nodiscard]] virtual std::unique_ptr<nixlBackendTraceSink>
    makeSink(std::string backend, std::function<void(std::uint64_t)> on_dropped) = 0;
};

using TracerPtr = std::unique_ptr<Tracer>;

using BackendTraceSinkPtr = std::unique_ptr<nixlBackendTraceSink>;

[[nodiscard]] inline TraceIdentity
generateRequestIdentity(const Tracer *tracer, const nixl_blob_t &parent = noSuppliedContext) {
    return (tracer != nullptr) ? tracer->generateRequestIdentity(parent) : TraceIdentity{};
}

/**
 * @brief Draw a context for one post of @p request: its trace id and flags with
 *        a span id from @p tracer's random source, again while it is zero, and
 *        its span id as the parent; @p request itself when @p tracer is null, and
 *        @p request unsampled when its source fails, so the post records nothing
 *        rather than reuse a span id.
 */
[[nodiscard]] inline TraceContext
generatePostContext(const TraceContext &request, const Tracer *tracer) {
    return (tracer != nullptr) ? tracer->generatePostContext(request) : request;
}

/**
 * @brief The context an application supplied as @p value, a `traceparent` or a
 *        trace context record, with only the flags this version defines;
 *        std::nullopt when @p value is empty, @p tracer is null, or @p value is
 *        neither form, which @p tracer reports once.
 */
[[nodiscard]] inline std::optional<TraceContext>
parseParentContext(const Tracer *tracer, const nixl_blob_t &value) {
    if ((tracer == nullptr) || value.empty()) {
        return std::nullopt;
    }
    return tracer->parseParentContext(value);
}

} // namespace nixl::trace

#endif
