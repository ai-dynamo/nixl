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
#ifndef NIXL_SRC_CORE_TRACING_TRACE_MACROS_H
#define NIXL_SRC_CORE_TRACING_TRACE_MACROS_H

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

#include <initializer_list>
#include <string_view>
#include <variant>

#include "tracing/trace.h"

#if defined(NIXL_TRACE_ENABLED)

namespace nixl::trace::detail {

struct SpanAttribute {
    template<typename Value>
    SpanAttribute(AttributeKey attr_key, const Value &attr_value) noexcept
        : key(attr_key),
          value(attributeValue(attr_value)) {}

    AttributeKey key;
    std::variant<std::string_view, std::int64_t, double> value;
};

inline void
addAttributes(Span &span, std::initializer_list<SpanAttribute> attributes) {
    for (const SpanAttribute &attribute : attributes) {
        std::visit([&](const auto value) { span.addAttribute(attribute.key, value); },
                   attribute.value);
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

#define NIXL_TRACE_SCOPE(span, tracer_ptr, span_name, span_kind, ...)                        \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                             \
        if (nixl_trace_tracer_ == nullptr) {                                                 \
            return ::nixl::trace::Span{};                                                    \
        }                                                                                    \
        NIXL_TRACE_DETAIL_ANNOTATED_(nixl_trace_tracer_->beginSpan((span_name), (span_kind)) \
                                         __VA_OPT__(, ) __VA_ARGS__);                        \
    }()

#define NIXL_TRACE_SAMPLED_SCOPE(span, tracer_ptr, span_name, span_kind, context, ...)       \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                             \
        if ((nixl_trace_tracer_ == nullptr) || !(context).sampled()) {                       \
            return ::nixl::trace::Span{};                                                    \
        }                                                                                    \
        NIXL_TRACE_DETAIL_ANNOTATED_(nixl_trace_tracer_->beginSpan((span_name), (span_kind)) \
                                         __VA_OPT__(, ) __VA_ARGS__);                        \
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
        NIXL_TRACE_DETAIL_ANNOTATED_(nixl_trace_tracer_->beginContextSpan(                        \
            (span_name), (span_kind), nixl_trace_context_, (request_id)) __VA_OPT__(, )           \
                                         __VA_ARGS__);                                            \
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

#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind)       \
    do {                                                        \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {          \
            nixl_trace_tracer_->mark((mark_name), (mark_kind)); \
        }                                                       \
    } while (0)

#define NIXL_TRACE_EVENT(tracer_ptr, event_name, event_kind, event_context, event_request_id, ...) \
    do {                                                                                           \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                                   \
        const ::nixl::trace::TraceContext &nixl_trace_context_ = (event_context);                  \
        if (nixl_trace_tracer_ && nixl_trace_context_.sampled()) {                                 \
            nixl_trace_tracer_->emitEvent(                                                         \
                (event_name),                                                                      \
                (event_kind),                                                                      \
                nixl_trace_context_,                                                               \
                (event_request_id),                                                                \
                nixl_trace_tracer_->needsTimestamps() ? ::nixlTime::getUs() : ::nixlTime::us_t{0}, \
                std::initializer_list<::nixl::trace::Attribute>{__VA_ARGS__});                     \
        }                                                                                          \
    } while (0)

#define NIXL_TRACE_NOW(tracer_ptr) ((tracer_ptr) ? ::nixlTime::getUs() : ::nixlTime::us_t{0})

#define NIXL_TRACE_EMIT_SPAN(tracer_ptr, span_name, span_kind, start_us)               \
    do {                                                                               \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                 \
            nixl_trace_tracer_->emitSpan((span_name),                                  \
                                         (span_kind),                                  \
                                         ::nixl::trace::TraceContext{},                \
                                         0,                                            \
                                         (start_us),                                   \
                                         ::nixlTime::getUs(),                          \
                                         std::span<const ::nixl::trace::Attribute>{}); \
        }                                                                              \
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

#endif // NIXL_SRC_CORE_TRACING_TRACE_MACROS_H
