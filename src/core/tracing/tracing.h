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
 * attribute_key.h, log_once.h, trace_identity.h, traceable.h and nixl_time.h
 * -- so it stays easy to reason about and could be promoted to a public header
 * later without churn if an external consumer appears.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

#include "common/nixl_time.h"
#include "tracing/attribute_key.h"
#include "tracing/log_once.h"
#include "tracing/trace_identity.h"
#include "tracing/traceable.h"

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
 * @brief An attribute value as a record holds it: integers and enums as
 *        std::int64_t, floating-point values as double, strings as views.
 */
template<typename T>
    requires std::is_integral_v<T> || std::is_enum_v<T>
[[nodiscard]] constexpr std::int64_t
attributeValue(T value) noexcept {
    return static_cast<std::int64_t>(value);
}

template<typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]] constexpr double
attributeValue(T value) noexcept {
    return static_cast<double>(value);
}

[[nodiscard]] constexpr std::string_view
attributeValue(std::string_view value) noexcept {
    return value;
}

/**
 * @brief Typed attribute of an emitted span or event. The key is a string
 *        literal; a string value is a view, valid only for the call.
 */
struct Attribute {
    template<typename Value>
    constexpr Attribute(AttributeKey attr_key, const Value &attr_value) noexcept
        : key(attr_key),
          value(attributeValue(attr_value)) {}

    AttributeKey key;
    std::variant<std::string_view, std::int64_t, double> value;
};

class Tracer;
class SpanBackend;

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

struct SpanBackendDeleter {
    void
    operator()(SpanBackend *span) const noexcept;
};

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
    friend class Tracer;

    template<typename Call>
    void
    forEachBackend(const Call &call) const;

    std::array<std::unique_ptr<SpanBackend, SpanBackendDeleter>, maxTraceBackends> backends_;
};

[[nodiscard]] TracerPtr
makeAgentTracer(const std::string &name);

[[nodiscard]] BackendTraceSinkPtr
makeBackendTraceSink(Tracer &tracer, std::string backend, nixlTelemetry *telemetry);

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
          std::span<const Attribute> attrs);

void
emitSpan(Tracer &tracer, std::string_view name, Kind kind, nixlTime::us_t start);

} // namespace nixl::trace::detail

namespace nixl::trace {

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

} // namespace nixl::trace

/*
 * One-line call-site macros for nixl::trace.
 *
 * When tracing is compiled out (NIXL_TRACE_ENABLED undefined) every macro
 * expands to nothing, and NIXL_TRACE_FINISH to its status, so call sites cost
 * nothing. When compiled in, the span macros declare a caller-named RAII Span
 * and take a predictable null-check branch on the tracer.
 *
 * Usage: NIXL_TRACE_SCOPE(span, tracer, name, kind, {key, value}...) declares
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

#define NIXL_TRACE_SCOPE(span, tracer_ptr, span_name, span_kind, ...)                   \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                           \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                        \
        if (nixl_trace_tracer_ == nullptr) {                                            \
            return ::nixl::trace::Span{};                                               \
        }                                                                               \
        NIXL_TRACE_DETAIL_ANNOTATED_(::nixl::trace::detail::beginSpan(                  \
            *nixl_trace_tracer_, (span_name), (span_kind)) __VA_OPT__(, ) __VA_ARGS__); \
    }()

#define NIXL_TRACE_SAMPLED_SCOPE(span, tracer_ptr, span_name, span_kind, context, ...)  \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                           \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                        \
        if ((nixl_trace_tracer_ == nullptr) || !(context).sampled()) {                  \
            return ::nixl::trace::Span{};                                               \
        }                                                                               \
        NIXL_TRACE_DETAIL_ANNOTATED_(::nixl::trace::detail::beginSpan(                  \
            *nixl_trace_tracer_, (span_name), (span_kind)) __VA_OPT__(, ) __VA_ARGS__); \
    }()

// For a sampled `context`, begins the span that `context` identifies, with `request_id` as
// its own; otherwise no span.
#define NIXL_TRACE_CONTEXT_SPAN(span, tracer_ptr, span_name, span_kind, context, request_id, ...) \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                     \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                                  \
        const ::nixl::trace::TraceContext &nixl_trace_context_ = (context);                       \
        if ((nixl_trace_tracer_ == nullptr) || !nixl_trace_context_.sampled()) {                  \
            return ::nixl::trace::Span{};                                                         \
        }                                                                                         \
        NIXL_TRACE_DETAIL_ANNOTATED_(::nixl::trace::detail::beginContextSpan(                     \
            *nixl_trace_tracer_, (span_name), (span_kind), nixl_trace_context_, (request_id))     \
                                         __VA_OPT__(, ) __VA_ARGS__);                             \
    }()

#define NIXL_TRACE_CORRELATED_SPAN(                                                             \
    span, tracer_ptr, span_name, span_kind, context, request_id, ...)                           \
    ::nixl::trace::CorrelationScope span##_correlation_((tracer_ptr), (context), (request_id)); \
    NIXL_TRACE_CONTEXT_SPAN(                                                                    \
        span, tracer_ptr, span_name, span_kind, context, request_id __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_CORRELATED_SCOPE(                                                            \
    span, tracer_ptr, span_name, span_kind, context, request_id, ...)                           \
    ::nixl::trace::CorrelationScope span##_correlation_((tracer_ptr), (context), (request_id)); \
    NIXL_TRACE_SAMPLED_SCOPE(                                                                   \
        span, tracer_ptr, span_name, span_kind, context __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_FINISH(span, status) (::nixl::trace::detail::finish((span), (status)))

#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind)                               \
    do {                                                                                \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                  \
            ::nixl::trace::detail::mark(*nixl_trace_tracer_, (mark_name), (mark_kind)); \
        }                                                                               \
    } while (0)

#define NIXL_TRACE_EVENT(tracer_ptr, event_name, event_kind, event_context, event_request_id, ...) \
    do {                                                                                           \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                                   \
        const ::nixl::trace::TraceContext &nixl_trace_context_ = (event_context);                  \
        if (nixl_trace_tracer_ && nixl_trace_context_.sampled()) {                                 \
            ::nixl::trace::detail::emitEvent(                                                      \
                *nixl_trace_tracer_,                                                               \
                (event_name),                                                                      \
                (event_kind),                                                                      \
                nixl_trace_context_,                                                               \
                (event_request_id),                                                                \
                std::initializer_list<::nixl::trace::Attribute>{__VA_ARGS__});                     \
        }                                                                                          \
    } while (0)

#define NIXL_TRACE_NOW(tracer_ptr) ((tracer_ptr) ? ::nixlTime::getUs() : ::nixlTime::us_t{0})

#define NIXL_TRACE_EMIT_SPAN(tracer_ptr, span_name, span_kind, start_us)    \
    do {                                                                    \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                      \
            ::nixl::trace::detail::emitSpan(                                \
                *nixl_trace_tracer_, (span_name), (span_kind), (start_us)); \
        }                                                                   \
    } while (0)

#define NIXL_TRACE_ATTR(span, attr_key, attr_value)                                     \
    do {                                                                                \
        if ((span).active()) {                                                          \
            (span).addAttribute((attr_key), ::nixl::trace::attributeValue(attr_value)); \
        }                                                                               \
    } while (0)

#else // !NIXL_TRACE_ENABLED

#define NIXL_TRACE_SCOPE(span, tracer_ptr, span_name, span_kind, ...) \
    do {                                                              \
    } while (0)
#define NIXL_TRACE_SAMPLED_SCOPE(span, tracer_ptr, span_name, span_kind, context, ...) \
    do {                                                                               \
    } while (0)
#define NIXL_TRACE_CONTEXT_SPAN(span, tracer_ptr, span_name, span_kind, context, request_id, ...) \
    do {                                                                                          \
    } while (0)
#define NIXL_TRACE_CORRELATED_SPAN(                                   \
    span, tracer_ptr, span_name, span_kind, context, request_id, ...) \
    do {                                                              \
    } while (0)
#define NIXL_TRACE_CORRELATED_SCOPE(                                  \
    span, tracer_ptr, span_name, span_kind, context, request_id, ...) \
    do {                                                              \
    } while (0)
#define NIXL_TRACE_FINISH(span, status) (status)
#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind) \
    do {                                                  \
    } while (0)
#define NIXL_TRACE_EVENT(tracer_ptr, event_name, event_kind, event_context, event_request_id, ...) \
    do {                                                                                           \
    } while (0)
#define NIXL_TRACE_ATTR(span, attr_key, attr_value) \
    do {                                            \
    } while (0)
#define NIXL_TRACE_NOW(tracer_ptr) (::nixlTime::us_t{0})
#define NIXL_TRACE_EMIT_SPAN(tracer_ptr, span_name, span_kind, start_us) \
    do {                                                                 \
    } while (0)

#endif // NIXL_TRACE_ENABLED

#endif // NIXL_SRC_CORE_TRACING_TRACING_H
