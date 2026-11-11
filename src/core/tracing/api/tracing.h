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
#ifndef NIXL_SRC_CORE_TRACING_API_TRACING_H
#define NIXL_SRC_CORE_TRACING_API_TRACING_H

/*
 * nixl::trace -- internal NIXL core tracing API.
 *
 * A single set of call sites fans out to every enabled tracing backend at
 * runtime (NVTX today; Chakra and others later). This is an internal core API
 * (compiled into libnixl, not installed as a public header). It is kept
 * deliberately self-contained -- depending only on standard headers and on
 * attribute.h, attribute_key.h, log_once.h, record.h, trace_identity.h and the
 * tracing/api/detail/ headers -- so it stays easy to reason about and could be
 * promoted to a public header later without churn if an external consumer
 * appears.
 *
 * A module uses only the NIXL_TRACE_* macros, Traceable and Scope, with Record,
 * Attribute and AttributeKey as their arguments; logOnce(); what
 * trace_identity.h declares; and the tracer and the backends' trace sinks as
 * opaque handles from makeAgentTracer() and makeBackendTraceSink().
 * tracing/api/detail/ holds only what the macros and Traceable expand into.
 */

#include <cstdint>
#include <memory>
#include <string>

#include "common/nixl_time.h"
#include "tracing/api/attribute.h"
#include "tracing/api/attribute_key.h"
#include "tracing/api/log_once.h"
#include "tracing/api/record.h"
#include "tracing/api/trace_identity.h"

#if defined(NIXL_TRACE_ENABLED)
#include "tracing/api/detail/span.h"
#endif

class nixlBackendTraceSink;
class nixlTelemetry;

namespace nixl::trace {

#if defined(NIXL_TRACE_ENABLED)

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
 *        created with, and the context and execution number of its latest post,
 *        whose parent is the span supplied to that post or else the request's
 *        span.
 */
class Traceable {
public:
    explicit Traceable(Tracer *tracer);

    Traceable(Tracer *tracer, const TraceIdentity &identity) noexcept;

    ~Traceable();

    Traceable(const Traceable &) = delete;
    Traceable &
    operator=(const Traceable &) = delete;

    [[nodiscard]] Tracer *
    tracer() const noexcept;

    [[nodiscard]] const TraceIdentity &
    postIdentity() const noexcept;

    [[nodiscard]] TraceIdentity
    requestIdentity() const noexcept;

    [[nodiscard]] const TraceIdentity &
    post(const nixl_blob_t &parent = noSuppliedContext);

    void
    trace(Record record) const;

    void
    trace(Record record, const Attribute &attr) const;

private:
    Tracer *const tracer_;
    const TraceContext requestContext_;
    TraceIdentity postIdentity_;
    const nixlTime::us_t createdAt_;
};

/**
 * @brief Makes @p identity and the tracer that records it current on the
 *        calling thread until it ends, so a backend called meanwhile reads the
 *        identity through currentIdentity(). Scopes nest. A scope
 *        whose context is valid and sampled also pushes it to the tracer's
 *        backends for its lifetime, to correlate records across threads.
 *        @p sink, when given, is the trace sink a backend constructed meanwhile
 *        takes as its own.
 */
class Scope {
public:
    [[nodiscard]] Scope(Tracer *tracer,
                        const TraceIdentity &identity,
                        nixlBackendTraceSink *sink = nullptr);
    ~Scope();

    Scope(const Scope &) = delete;
    Scope &
    operator=(const Scope &) = delete;

    [[nodiscard]] static const Scope *
    current() noexcept;

    [[nodiscard]] Tracer *
    tracer() const noexcept {
        return tracer_;
    }

    [[nodiscard]] const TraceIdentity &
    identity() const noexcept {
        return identity_;
    }

    [[nodiscard]] nixlBackendTraceSink *
    sink() const noexcept {
        return sink_;
    }

private:
    static void
    makeCurrent(const Scope *scope) noexcept;

    const Scope *const previous_;
    Tracer *const tracer_;
    const TraceIdentity identity_;
    nixlBackendTraceSink *const sink_;
    const bool pushed_;
};

#else

struct TracerPtr {
    [[nodiscard]] constexpr Tracer *
    get() const noexcept {
        return nullptr;
    }
};

struct BackendTraceSinkPtr {
    [[nodiscard]] constexpr nixlBackendTraceSink *
    get() const noexcept {
        return nullptr;
    }
};

[[nodiscard]] inline TracerPtr
makeAgentTracer(const std::string &) noexcept {
    return {};
}

class Traceable {
public:
    Traceable(Tracer *, const TraceIdentity &) noexcept {}

    [[nodiscard]] TraceIdentity
    postIdentity() const noexcept {
        return {};
    }

    [[nodiscard]] TraceIdentity
    requestIdentity() const noexcept {
        return {};
    }

    void
    trace(Record) const noexcept {}

    void
    trace(Record, const Attribute &) const noexcept {}
};

#endif

} // namespace nixl::trace

/*
 * One-line call-site macros for nixl::trace.
 *
 * When tracing is compiled out (NIXL_TRACE_ENABLED undefined) every macro
 * expands to nothing, NIXL_TRACE_FINISH and NIXL_TRACE_END to their status and
 * NIXL_TRACE_NOW to 0, so call sites cost nothing. When compiled in, the span
 * macros declare a caller-named RAII Span and take a predictable null-check
 * branch on the tracer.
 *
 * Usage: NIXL_TRACE_SCOPE(span, tracer, record, {key, value}...) declares
 * the local `span` with its attributes; NIXL_TRACE_ATTR(span, key, value) adds
 * one known later, and `return NIXL_TRACE_FINISH(span, status);` records the
 * call's status as it returns it. NIXL_TRACE_REQUEST_SPAN(span, traceable,
 * record, {key, value}...) does the same for a call on a request, inside the
 * request's scope, and NIXL_TRACE_POST_SPAN(span, traceable, parent, record, ...)
 * for a post, inside the scope of the post it begins under `parent`, the context
 * the application supplied, if any; both add the request id and execution number.
 * NIXL_TRACE_SCOPE_UNDER(span, tracer, parent, record, ...) is NIXL_TRACE_SCOPE for
 * a call the application may give a context: under a sampled `parent` its span is
 * a child of it, inside a scope of its own; under an unsampled one there is no span.
 * NIXL_TRACE_END(span, status) records the status as NIXL_TRACE_FINISH does and
 * ends the span at once, for a call whose last step must follow its span.
 * The explicit name avoids any hidden coupling/shadowing and allows more than
 * one span per scope. Attribute arguments are only evaluated when a backend is
 * actually active.
 */

#define NIXL_TRACE_SCOPE(span, tracer_ptr, trace_record, ...) \
    NIXL_TRACE_DETAIL_SCOPE(span, tracer_ptr, trace_record __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_SCOPE_UNDER(span, tracer_ptr, parent, trace_record, ...) \
    NIXL_TRACE_DETAIL_SCOPE_UNDER(span, tracer_ptr, parent, trace_record __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_REQUEST_SPAN(span, traceable, trace_record, ...) \
    NIXL_TRACE_DETAIL_REQUEST_SPAN(span, traceable, trace_record __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_POST_SPAN(span, traceable, parent, trace_record, ...) \
    NIXL_TRACE_DETAIL_POST_SPAN(span, traceable, parent, trace_record __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_ATTR(span, attr_key, attr_value) \
    NIXL_TRACE_DETAIL_ATTR(span, attr_key, attr_value)

#define NIXL_TRACE_FINISH(span, status) NIXL_TRACE_DETAIL_FINISH(span, status)

#define NIXL_TRACE_END(span, status) NIXL_TRACE_DETAIL_END(span, status)

#define NIXL_TRACE_NOW(tracer_ptr) NIXL_TRACE_DETAIL_NOW(tracer_ptr)

#define NIXL_TRACE_EMIT_SPAN(tracer_ptr, trace_record, start_us) \
    NIXL_TRACE_DETAIL_EMIT_SPAN(tracer_ptr, trace_record, start_us)

#if defined(NIXL_TRACE_ENABLED)

#include "tracing/api/detail/tracing_impl.h"

#else

#define NIXL_TRACE_DETAIL_SCOPE(span, tracer_ptr, trace_record, ...) \
    do {                                                             \
    } while (0)
#define NIXL_TRACE_DETAIL_SCOPE_UNDER(span, tracer_ptr, parent, trace_record, ...) \
    do {                                                                           \
    } while (0)
#define NIXL_TRACE_DETAIL_REQUEST_SPAN(span, traceable, trace_record, ...) \
    do {                                                                   \
    } while (0)
#define NIXL_TRACE_DETAIL_POST_SPAN(span, traceable, parent, trace_record, ...) \
    do {                                                                        \
    } while (0)
#define NIXL_TRACE_DETAIL_FINISH(span, status) (status)
#define NIXL_TRACE_DETAIL_END(span, status) (status)
#define NIXL_TRACE_DETAIL_ATTR(span, attr_key, attr_value) \
    do {                                                   \
    } while (0)
#define NIXL_TRACE_DETAIL_NOW(tracer_ptr) (::nixlTime::us_t{0})
#define NIXL_TRACE_DETAIL_EMIT_SPAN(tracer_ptr, trace_record, start_us) \
    do {                                                                \
    } while (0)

#endif

#endif // NIXL_SRC_CORE_TRACING_API_TRACING_H
