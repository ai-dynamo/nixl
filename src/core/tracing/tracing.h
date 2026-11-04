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
 * log_once.h, trace_identity.h and nixl_time.h -- so it stays easy to reason
 * about and could be promoted to a public header later without churn if an
 * external consumer appears.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "common/nixl_time.h"
#include "tracing/log_once.h"
#include "tracing/trace_identity.h"

class nixlBackendTraceSink;
class nixlTelemetry;

namespace nixl::trace {

/**
 * @brief Operation kind. Aligns 1:1 with the Chakra NodeType vocabulary and is
 *        used as a color/label hint by NVTX.
 */
enum class Kind : std::uint8_t {
    Generic = 0,
    Compute,
    MemoryR,
    MemoryW,
    CommSend,
    CommRecv,
    CommColl,
    Metadata,
};

/**
 * @brief Opaque span identifier. Meaningful on backends that build a DAG
 *        (Chakra Node.id); returned as {0} by backends that do not (NVTX).
 */
struct SpanId {
    std::uint64_t value{0};
};

/**
 * @brief Typed attribute of an emitted span or event. The key and a string
 *        value are views, valid only for the call.
 */
struct Attribute {
    std::string_view key;
    std::variant<std::string_view, std::int64_t, double> value;
};

class Tracer;
class Span;

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

namespace nixl::trace {

/**
 * @brief Public span handle. Fans out attribute/dependency calls to every
 *        backend span it owns and ends them all on destruction. Move-only.
 */
class Span {
public:
    Span() = default;
    Span(Span &&) noexcept = default;
    Span &
    operator=(Span &&) noexcept = default;
    Span(const Span &) = delete;
    Span &
    operator=(const Span &) = delete;
    ~Span() = default;

    /** @brief True when at least one backend produced a span (cheap gate). */
    [[nodiscard]] bool
    active() const noexcept {
        return !backends_.empty();
    }

    void
    addAttribute(std::string_view key, std::string_view value);
    void
    addAttribute(std::string_view key, std::int64_t value);
    void
    addAttribute(std::string_view key, double value);

    void
    addCtrlDep(SpanId parent);
    void
    addDataDep(SpanId parent);

    /** @brief Id of the first backend that provides a non-zero id, or {0}
     *         when none do (e.g. NVTX-only). Order-independent across backends. */
    [[nodiscard]] SpanId
    id() const noexcept;

private:
    friend Span
    detail::beginSpan(Tracer &tracer, std::string_view name, Kind kind);
    friend Span
    detail::beginContextSpan(Tracer &tracer,
                             std::string_view name,
                             Kind kind,
                             const TraceContext &context,
                             std::uint64_t request_id);

    explicit Span(std::vector<std::unique_ptr<SpanBackend>> backends) noexcept
        : backends_(std::move(backends)) {}

    std::vector<std::unique_ptr<SpanBackend>> backends_;
};

[[nodiscard]] TracerPtr
makeAgentTracer(const std::string &name);

[[nodiscard]] BackendTraceSinkPtr
makeBackendTraceSink(Tracer &tracer, std::string backend, nixlTelemetry *telemetry);

/**
 * @brief RAII guard that keeps a trace context active on the tracer for its
 *        lifetime, so every span/mark created while it is in scope carries
 *        that context. Null-tracer-safe; an invalid context with no request id
 *        pushes nothing. Used to link spans emitted on different threads (e.g.
 *        postXferReq on the caller thread vs. the completion on the polling
 *        thread) to the same logical request.
 */
class CorrelationScope {
public:
    [[nodiscard]] CorrelationScope(Tracer *tracer,
                                   const TraceContext &context,
                                   std::uint64_t request_id)
        : tracer_(((tracer != nullptr) && (context.valid() || (request_id != 0))) ? tracer :
                                                                                    nullptr) {
        if (tracer_ != nullptr) {
            detail::pushContext(*tracer_, context, request_id);
        }
    }

    ~CorrelationScope() {
        if (tracer_ != nullptr) {
            detail::popContext(*tracer_);
        }
    }

    CorrelationScope(const CorrelationScope &) = delete;
    CorrelationScope &
    operator=(const CorrelationScope &) = delete;
    CorrelationScope(CorrelationScope &&) = delete;
    CorrelationScope &
    operator=(CorrelationScope &&) = delete;

private:
    Tracer *tracer_;
};

/**
 * @brief A traced object's trace state: the context and request id it was
 *        created with, and the context and execution number of its latest post.
 */
class Traceable {
public:
    Traceable(Tracer *tracer, const TraceIdentity &identity) noexcept
        : tracer_(tracer),
          context_(identity.context),
          postContext_(identity.context),
          requestId_(identity.requestId) {}

    [[nodiscard]] Tracer *
    tracer() const noexcept {
        return tracer_;
    }

    [[nodiscard]] const TraceContext &
    context() const noexcept {
        return context_;
    }

    /** @brief Trace context of the latest post handed to the backend, which that
     *         post's nixl::xfer.complete carries; context() before any post. */
    [[nodiscard]] const TraceContext &
    postContext() const noexcept {
        return postContext_;
    }

    [[nodiscard]] std::uint64_t
    requestId() const noexcept {
        return requestId_;
    }

    [[nodiscard]] std::uint64_t
    execution() const noexcept {
        return execution_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] TraceIdentity
    identity() const noexcept {
        return {postContext_, requestId_, execution()};
    }

    [[nodiscard]] TraceContext
    beginPost() {
        const TraceContext post_context =
            (context_.sampled() && spanIdUsed_) ? generatePostContext(context_, tracer_) : context_;
        spanIdUsed_ = true;
        return post_context;
    }

    TraceIdentity
    acceptPost(const TraceContext &post_context) noexcept {
        postContext_ = post_context;
        if (requestId_ != 0) {
            execution_.fetch_add(1, std::memory_order_relaxed);
        }
        return identity();
    }

private:
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
 * expands to `do {} while (0)` so call sites cost nothing. When compiled in,
 * NIXL_TRACE_SCOPE and NIXL_TRACE_CONTEXT_SPAN declare a caller-named RAII Span
 * and take a predictable null-check branch on the tracer.
 *
 * Usage: NIXL_TRACE_SCOPE(span, tracer, name, kind) or
 * NIXL_TRACE_CONTEXT_SPAN(span, tracer, name, kind, context, request_id)
 * declares the local `span`; pass that same `span` to
 * NIXL_TRACE_ATTR(span, key, value) to annotate it.
 * The explicit name avoids any hidden coupling/shadowing and allows more than
 * one span per scope. Attribute arguments are only evaluated when a backend is
 * actually active.
 */

#if defined(NIXL_TRACE_ENABLED)

#define NIXL_TRACE_SCOPE(span, tracer_ptr, span_name, span_kind)                              \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                 \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                              \
        return nixl_trace_tracer_ ?                                                           \
            ::nixl::trace::detail::beginSpan(*nixl_trace_tracer_, (span_name), (span_kind)) : \
            ::nixl::trace::Span{};                                                            \
    }()

// For a sampled `context`, begins the span that `context` identifies, with `request_id` as
// its own; otherwise a span inside the active context, as NIXL_TRACE_SCOPE does.
#define NIXL_TRACE_CONTEXT_SPAN(span, tracer_ptr, span_name, span_kind, context, request_id) \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                             \
        if (!nixl_trace_tracer_) {                                                           \
            return ::nixl::trace::Span{};                                                    \
        }                                                                                    \
        const ::nixl::trace::TraceContext &nixl_trace_context_ = (context);                  \
        return nixl_trace_context_.sampled() ?                                               \
            ::nixl::trace::detail::beginContextSpan(*nixl_trace_tracer_,                     \
                                                    (span_name),                             \
                                                    (span_kind),                             \
                                                    nixl_trace_context_,                     \
                                                    (request_id)) :                          \
            ::nixl::trace::detail::beginSpan(*nixl_trace_tracer_, (span_name), (span_kind)); \
    }()

#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind)                               \
    do {                                                                                \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                  \
            ::nixl::trace::detail::mark(*nixl_trace_tracer_, (mark_name), (mark_kind)); \
        }                                                                               \
    } while (0)

#define NIXL_TRACE_EVENT(tracer_ptr, event_name, event_kind, event_context, event_request_id, ...) \
    do {                                                                                           \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                             \
            ::nixl::trace::detail::emitEvent(                                                      \
                *nixl_trace_tracer_,                                                               \
                (event_name),                                                                      \
                (event_kind),                                                                      \
                (event_context),                                                                   \
                (event_request_id),                                                                \
                ::nixlTime::getUs(),                                                               \
                std::initializer_list<::nixl::trace::Attribute>{__VA_ARGS__});                     \
        }                                                                                          \
    } while (0)

// Declare before NIXL_TRACE_SCOPE or NIXL_TRACE_CONTEXT_SPAN so the span is created while
// the context is active. One per scope.
#define NIXL_TRACE_CORRELATION_SCOPE(tracer_ptr, context, request_id) \
    ::nixl::trace::CorrelationScope nixl_trace_correlation_scope_(    \
        (tracer_ptr), (context), (request_id))

#define NIXL_TRACE_ATTR(span, attr_key, attr_value)        \
    do {                                                   \
        if ((span).active()) {                             \
            (span).addAttribute((attr_key), (attr_value)); \
        }                                                  \
    } while (0)

#else // !NIXL_TRACE_ENABLED

#define NIXL_TRACE_SCOPE(span, tracer_ptr, span_name, span_kind) \
    do {                                                         \
    } while (0)
#define NIXL_TRACE_CONTEXT_SPAN(span, tracer_ptr, span_name, span_kind, context, request_id) \
    do {                                                                                     \
    } while (0)
#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind) \
    do {                                                  \
    } while (0)
#define NIXL_TRACE_EVENT(tracer_ptr, event_name, event_kind, event_context, event_request_id, ...) \
    do {                                                                                           \
    } while (0)
#define NIXL_TRACE_ATTR(span, attr_key, attr_value) \
    do {                                            \
    } while (0)
#define NIXL_TRACE_CORRELATION_SCOPE(tracer_ptr, context, request_id) \
    do {                                                              \
    } while (0)

#endif // NIXL_TRACE_ENABLED

#endif // NIXL_SRC_CORE_TRACING_TRACING_H
