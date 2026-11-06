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
 * attribute.h, attribute_key.h, detail/trace_impl.h, log_once.h, record.h,
 * trace_identity.h and nixl_time.h -- so it stays easy to reason about and
 * could be promoted to a public header later without churn if an external
 * consumer appears.
 *
 * A module uses only the NIXL_TRACE_* macros, Span, CorrelationScope and
 * Traceable, with Record, Attribute and AttributeKey as their arguments;
 * logOnce(); what trace_identity.h declares; and the tracer and the backends'
 * trace sinks as opaque handles from makeAgentTracer() and
 * makeBackendTraceSink(). nixl::trace::detail, declared in
 * tracing/detail/trace_impl.h, is only what the macros expand into.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "common/nixl_time.h"
#include "tracing/attribute.h"
#include "tracing/attribute_key.h"
#include "tracing/detail/trace_impl.h"
#include "tracing/log_once.h"
#include "tracing/record.h"
#include "tracing/trace_identity.h"

class nixlBackendTraceSink;
class nixlTelemetry;

namespace nixl::trace {

/**
 * @brief Opaque span identifier. Meaningful on backends that build a DAG
 *        (Chakra Node.id); returned as {0} by backends that do not (NVTX).
 */
struct SpanId {
    std::uint64_t value{0};
};

class Name {
public:
    constexpr Name(Record record) noexcept
        : Name(toStringView(record),
               (static_cast<std::size_t>(record) < recordCount) ?
                   static_cast<std::uint8_t>(record) :
                   noRecord) {}

    template<std::size_t Size>
    consteval Name(const char (&text)[Size]) noexcept : Name(std::string_view(text, Size - 1)) {}

    explicit constexpr Name(std::string_view text) noexcept : Name(text, noRecord) {}

    [[nodiscard]] constexpr std::string_view
    view() const noexcept {
        return {data_, size_};
    }

    [[nodiscard]] constexpr std::optional<Record>
    record() const noexcept {
        if (record_ == noRecord) {
            return std::nullopt;
        }
        return static_cast<Record>(record_);
    }

private:
    static constexpr std::uint8_t noRecord = 0xff;

    constexpr Name(std::string_view text, std::uint8_t record) noexcept
        : data_(text.data()),
          size_(static_cast<std::uint32_t>(text.size())),
          record_(record) {}

    const char *data_;
    std::uint32_t size_;
    std::uint8_t record_;
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

inline constexpr std::size_t maxTraceBackends = 4;

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
        return backends_[0] != nullptr;
    }

    void
    addAttribute(AttributeKey key, std::string_view value);
    void
    addAttribute(AttributeKey key, std::int64_t value);
    void
    addAttribute(AttributeKey key, double value);

    void
    addAttributes(std::span<const Attribute> attrs);

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
    detail::beginSpan(Tracer &tracer, Name name, Kind kind);
    friend Span
    detail::beginContextSpan(Tracer &tracer,
                             Name name,
                             Kind kind,
                             const TraceContext &context,
                             std::uint64_t request_id);

    template<typename Call>
    void
    forEachBackend(const Call &call) const;

    std::array<std::unique_ptr<SpanBackend>, maxTraceBackends> backends_;
};

[[nodiscard]] TracerPtr
makeAgentTracer(const std::string &name);

[[nodiscard]] BackendTraceSinkPtr
makeBackendTraceSink(Tracer &tracer, std::string backend, nixlTelemetry *telemetry);

/**
 * @brief RAII guard that keeps a trace context active on the tracer for its
 *        lifetime, so every span/mark created while it is in scope carries
 *        that context. Null-tracer-safe; an unsampled context pushes nothing.
 *        Used to link spans emitted on different threads (e.g.
 *        postXferReq on the caller thread vs. the completion on the polling
 *        thread) to the same logical request.
 */
class CorrelationScope {
public:
    [[nodiscard]] CorrelationScope(Tracer *tracer,
                                   const TraceContext &context,
                                   std::uint64_t request_id)
        : tracer_(((tracer != nullptr) && context.valid() && context.sampled()) ? tracer :
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

    void
    record([[maybe_unused]] Record record) const {
#if defined(NIXL_TRACE_ENABLED)
        if ((tracer_ != nullptr) && postContext_.sampled()) {
            const Attribute attrs[] = {{"nixl.xfer_request_id", requestId_},
                                       {"nixl.xfer_execution", execution()}};
            detail::emitEvent(*tracer_, record, kindOf(record), postContext_, requestId_, attrs);
        }
#endif
    }

    void
    record([[maybe_unused]] Record record, [[maybe_unused]] const Attribute &attr) const {
#if defined(NIXL_TRACE_ENABLED)
        if ((tracer_ != nullptr) && postContext_.sampled()) {
            const Attribute attrs[] = {
                {"nixl.xfer_request_id", requestId_}, {"nixl.xfer_execution", execution()}, attr};
            detail::emitEvent(*tracer_, record, kindOf(record), postContext_, requestId_, attrs);
        }
#endif
    }

private:
    friend struct detail::TraceableAccess;

    [[nodiscard]] std::uint64_t
    execution() const noexcept {
        return execution_.load(std::memory_order_relaxed);
    }

    Tracer *const tracer_;
    const TraceContext context_;
    TraceContext postContext_;
    bool spanIdUsed_ = false;
    const std::uint64_t requestId_;
    std::atomic<std::uint64_t> execution_ = 0;
};

struct detail::TraceableAccess {
    [[nodiscard]] static Tracer *
    tracer(const Traceable &traceable) noexcept {
        return traceable.tracer_;
    }

    [[nodiscard]] static const TraceContext &
    postContext(const Traceable &traceable) noexcept {
        return traceable.postContext_;
    }

    [[nodiscard]] static std::uint64_t
    requestId(const Traceable &traceable) noexcept {
        return traceable.requestId_;
    }
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

#if defined(NIXL_TRACE_ENABLED)

namespace nixl::trace::detail {

inline void
addAttributes(Span &span, std::initializer_list<Attribute> attributes) {
    if (attributes.size() != 0) {
        span.addAttributes({attributes.begin(), attributes.size()});
    }
}

template<typename Status>
[[nodiscard]] Status
finish(Span &span, Status status) {
    if (span.active()) {
        span.addAttribute("nixl.status", attributeValue(status));
    }
    return status;
}

} // namespace nixl::trace::detail

#define NIXL_TRACE_DETAIL_ANNOTATED_(begin, ...)                               \
    ::nixl::trace::Span nixl_trace_span_ = (begin);                            \
    if (nixl_trace_span_.active()) {                                           \
        ::nixl::trace::detail::addAttributes(nixl_trace_span_, {__VA_ARGS__}); \
    }                                                                          \
    return nixl_trace_span_

#define NIXL_TRACE_SCOPE(span, tracer_ptr, trace_record, ...)                                   \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                   \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                                \
        if (nixl_trace_tracer_ == nullptr) {                                                    \
            return ::nixl::trace::Span{};                                                       \
        }                                                                                       \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                        \
        NIXL_TRACE_DETAIL_ANNOTATED_(::nixl::trace::detail::beginSpan(                          \
            *nixl_trace_tracer_, nixl_trace_record_, ::nixl::trace::kindOf(nixl_trace_record_)) \
                                         __VA_OPT__(, ) __VA_ARGS__);                           \
    }()

#define NIXL_TRACE_SAMPLED_SCOPE(span, tracer_ptr, trace_record, context, ...)                  \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                   \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                                \
        if ((nixl_trace_tracer_ == nullptr) || !(context).sampled()) {                          \
            return ::nixl::trace::Span{};                                                       \
        }                                                                                       \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                        \
        NIXL_TRACE_DETAIL_ANNOTATED_(::nixl::trace::detail::beginSpan(                          \
            *nixl_trace_tracer_, nixl_trace_record_, ::nixl::trace::kindOf(nixl_trace_record_)) \
                                         __VA_OPT__(, ) __VA_ARGS__);                           \
    }()

// For a sampled `context`, begins the span that `context` identifies, with `request_id` as
// its own; otherwise no span.
#define NIXL_TRACE_CONTEXT_SPAN(span, tracer_ptr, trace_record, context, request_id, ...)      \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                  \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                               \
        const ::nixl::trace::TraceContext &nixl_trace_context_ = (context);                    \
        if ((nixl_trace_tracer_ == nullptr) || !nixl_trace_context_.sampled()) {               \
            return ::nixl::trace::Span{};                                                      \
        }                                                                                      \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                       \
        NIXL_TRACE_DETAIL_ANNOTATED_(                                                          \
            ::nixl::trace::detail::beginContextSpan(*nixl_trace_tracer_,                       \
                                                    nixl_trace_record_,                        \
                                                    ::nixl::trace::kindOf(nixl_trace_record_), \
                                                    nixl_trace_context_,                       \
                                                    (request_id)) __VA_OPT__(, ) __VA_ARGS__); \
    }()

#define NIXL_TRACE_CORRELATED_SPAN(span, request_trace, trace_record, context, ...)          \
    const ::nixl::trace::Traceable &span##_trace_ = (request_trace);                         \
    ::nixl::trace::CorrelationScope span##_correlation_(                                     \
        ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_),                       \
        (context),                                                                           \
        ::nixl::trace::detail::TraceableAccess::requestId(span##_trace_));                   \
    NIXL_TRACE_CONTEXT_SPAN(span,                                                            \
                            ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_),   \
                            trace_record,                                                    \
                            context,                                                         \
                            ::nixl::trace::detail::TraceableAccess::requestId(span##_trace_) \
                                __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_CORRELATED_SCOPE(span, request_trace, trace_record, ...)                     \
    const ::nixl::trace::Traceable &span##_trace_ = (request_trace);                            \
    ::nixl::trace::CorrelationScope span##_correlation_(                                        \
        ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_),                          \
        ::nixl::trace::detail::TraceableAccess::postContext(span##_trace_),                     \
        ::nixl::trace::detail::TraceableAccess::requestId(span##_trace_));                      \
    NIXL_TRACE_SAMPLED_SCOPE(span,                                                              \
                             ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_),     \
                             trace_record,                                                      \
                             ::nixl::trace::detail::TraceableAccess::postContext(span##_trace_) \
                                 __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_FINISH(span, status) (::nixl::trace::detail::finish((span), (status)))

#define NIXL_TRACE_MARK(tracer_ptr, trace_record)                                   \
    do {                                                                            \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                              \
            const ::nixl::trace::Record nixl_trace_record_ = (trace_record);        \
            ::nixl::trace::detail::mark(*nixl_trace_tracer_,                        \
                                        nixl_trace_record_,                         \
                                        ::nixl::trace::kindOf(nixl_trace_record_)); \
        }                                                                           \
    } while (0)

#define NIXL_TRACE_NOW(tracer_ptr) ((tracer_ptr) ? ::nixlTime::getUs() : ::nixlTime::us_t{0})

#define NIXL_TRACE_EMIT_SPAN(tracer_ptr, trace_record, start_us)                       \
    do {                                                                               \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                 \
            const ::nixl::trace::Record nixl_trace_record_ = (trace_record);           \
            ::nixl::trace::detail::emitSpan(*nixl_trace_tracer_,                       \
                                            nixl_trace_record_,                        \
                                            ::nixl::trace::kindOf(nixl_trace_record_), \
                                            (start_us));                               \
        }                                                                              \
    } while (0)

#define NIXL_TRACE_ATTR(span, attr_key, attr_value)                                     \
    do {                                                                                \
        if ((span).active()) {                                                          \
            (span).addAttribute((attr_key), ::nixl::trace::attributeValue(attr_value)); \
        }                                                                               \
    } while (0)

#else // !NIXL_TRACE_ENABLED

#define NIXL_TRACE_SCOPE(span, tracer_ptr, trace_record, ...) \
    do {                                                      \
    } while (0)
#define NIXL_TRACE_SAMPLED_SCOPE(span, tracer_ptr, trace_record, context, ...) \
    do {                                                                       \
    } while (0)
#define NIXL_TRACE_CONTEXT_SPAN(span, tracer_ptr, trace_record, context, request_id, ...) \
    do {                                                                                  \
    } while (0)
#define NIXL_TRACE_CORRELATED_SPAN(span, request_trace, trace_record, context, ...) \
    do {                                                                            \
    } while (0)
#define NIXL_TRACE_CORRELATED_SCOPE(span, request_trace, trace_record, ...) \
    do {                                                                    \
    } while (0)
#define NIXL_TRACE_FINISH(span, status) (status)
#define NIXL_TRACE_MARK(tracer_ptr, trace_record) \
    do {                                          \
    } while (0)
#define NIXL_TRACE_ATTR(span, attr_key, attr_value) \
    do {                                            \
    } while (0)
#define NIXL_TRACE_NOW(tracer_ptr) (::nixlTime::us_t{0})
#define NIXL_TRACE_EMIT_SPAN(tracer_ptr, trace_record, start_us) \
    do {                                                         \
    } while (0)

#endif // NIXL_TRACE_ENABLED

#endif // NIXL_SRC_CORE_TRACING_TRACING_H
