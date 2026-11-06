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
#ifndef NIXL_SRC_CORE_TRACING_TRACING_H
#define NIXL_SRC_CORE_TRACING_TRACING_H

/*
 * nixl::trace -- internal NIXL core tracing API.
 *
 * A single set of call sites fans out to every enabled tracing backend at
 * runtime (NVTX today; Chakra and others later). This is an internal core API
 * (compiled into libnixl, not installed as a public header). It is kept
 * deliberately self-contained -- depending only on standard headers and on
 * attribute.h, attribute_key.h, log_once.h, record.h, trace_identity.h and the
 * tracing/detail/ headers -- so it stays easy to reason about and could be
 * promoted to a public header later without churn if an external consumer
 * appears.
 *
 * A module uses only the NIXL_TRACE_* macros and Traceable, with Record,
 * Attribute and AttributeKey as their arguments; logOnce(); what
 * trace_identity.h declares; and the tracer and the backends' trace sinks as
 * opaque handles from makeAgentTracer() and makeBackendTraceSink().
 * tracing/detail/ holds only what the macros and Traceable expand into.
 */

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "tracing/attribute.h"
#include "tracing/attribute_key.h"
#include "tracing/detail/span.h"
#include "tracing/log_once.h"
#include "tracing/record.h"
#include "tracing/trace_identity.h"

class nixlBackendTraceSink;
class nixlTelemetry;

namespace nixl::trace {

struct TracerDeleter {
    void
    operator()(Tracer *tracer) const noexcept;
};

using TracerPtr = std::unique_ptr<Tracer, TracerDeleter>;

struct BackendTraceSinkDeleter {
    void
    operator()(nixlBackendTraceSink *sink) const noexcept;
};

using BackendTraceSinkPtr = std::unique_ptr<nixlBackendTraceSink, BackendTraceSinkDeleter>;

[[nodiscard]] TracerPtr
makeAgentTracer(const std::string &name);

[[nodiscard]] BackendTraceSinkPtr
makeBackendTraceSink(Tracer &tracer, std::string backend, nixlTelemetry *telemetry);

/**
 * @brief A traced object's trace state: the context and request id it was
 *        created with, and the context and execution number of its latest post.
 */
class Traceable {
public:
    explicit Traceable(Tracer *tracer);

    Traceable(Tracer *tracer, const TraceIdentity &identity) noexcept;

    [[nodiscard]] TraceIdentity
    identity() const noexcept;

    [[nodiscard]] TraceContext
    beginPost();

    TraceIdentity
    acceptPost(const TraceContext &post_context) noexcept;

    void
    record(Record record) const;

    void
    record(Record record, const Attribute &attr) const;

private:
    friend struct detail::TraceableAccess;

    [[nodiscard]] std::uint64_t
    execution() const noexcept;

    Tracer *const tracer_;
    const TraceContext context_;
    TraceContext postContext_;
    bool spanIdUsed_ = false;
    const std::uint64_t requestId_;
    std::atomic<std::uint64_t> execution_ = 0;
};

} // namespace nixl::trace

/*
 * One-line call-site macros for nixl::trace.
 *
 * When tracing is compiled out (NIXL_TRACE_ENABLED undefined) every macro
 * expands to nothing, and NIXL_TRACE_FINISH to its status, so call sites cost
 * nothing. When compiled in, the span macros declare a caller-named RAII Span
 * and take a predictable null-check branch on the tracer.
 *
 * Usage: NIXL_TRACE_SCOPE(span, tracer, record, {key, value}...) declares
 * the local `span` with its attributes; NIXL_TRACE_ATTR(span, key, value) adds
 * one known later, and `return NIXL_TRACE_FINISH(span, status);` records the
 * call's status as it returns it.
 * The explicit name avoids any hidden coupling/shadowing and allows more than
 * one span per scope. Attribute arguments are only evaluated when a backend is
 * actually active.
 */

#define NIXL_TRACE_SCOPE(span, tracer_ptr, trace_record, ...) \
    NIXL_TRACE_DETAIL_SCOPE(span, tracer_ptr, trace_record __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_SAMPLED_SCOPE(span, tracer_ptr, trace_record, context, ...) \
    NIXL_TRACE_DETAIL_SAMPLED_SCOPE(                                           \
        span, tracer_ptr, trace_record, context __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_CORRELATED_SPAN(span, request_trace, trace_record, context, ...) \
    NIXL_TRACE_DETAIL_CORRELATED_SPAN(                                              \
        span, request_trace, trace_record, context __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_CORRELATED_SCOPE(span, request_trace, trace_record, ...) \
    NIXL_TRACE_DETAIL_CORRELATED_SCOPE(span, request_trace, trace_record __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_ATTR(span, attr_key, attr_value) \
    NIXL_TRACE_DETAIL_ATTR(span, attr_key, attr_value)

#define NIXL_TRACE_FINISH(span, status) NIXL_TRACE_DETAIL_FINISH(span, status)

#define NIXL_TRACE_NOW(tracer_ptr) NIXL_TRACE_DETAIL_NOW(tracer_ptr)

#define NIXL_TRACE_EMIT_SPAN(tracer_ptr, trace_record, start_us) \
    NIXL_TRACE_DETAIL_EMIT_SPAN(tracer_ptr, trace_record, start_us)

#include "tracing/detail/tracing_impl.h"

#endif // NIXL_SRC_CORE_TRACING_TRACING_H
