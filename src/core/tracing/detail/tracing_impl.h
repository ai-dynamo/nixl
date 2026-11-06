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
#ifndef NIXL_SRC_CORE_TRACING_DETAIL_TRACING_IMPL_H
#define NIXL_SRC_CORE_TRACING_DETAIL_TRACING_IMPL_H

#include <atomic>
#include <cstdint>
#include <initializer_list>

#include "common/nixl_time.h"
#include "tracing/attribute.h"
#include "tracing/detail/span.h"
#include "tracing/detail/trace_impl.h"
#include "tracing/record.h"
#include "tracing/trace_identity.h"

namespace nixl::trace {

inline Traceable::Traceable(Tracer *tracer) : Traceable(tracer, generateRequestIdentity(tracer)) {}

inline Traceable::Traceable(Tracer *tracer, const TraceIdentity &identity) noexcept
    : tracer_(tracer),
      context_(identity.context),
      postContext_(identity.context),
      requestId_(identity.requestId) {}

inline TraceIdentity
Traceable::identity() const noexcept {
    return {postContext_, requestId_, execution()};
}

inline TraceContext
Traceable::beginPost() {
    const TraceContext post_context =
        (context_.sampled() && spanIdUsed_) ? generatePostContext(context_, tracer_) : context_;
    spanIdUsed_ = true;
    return post_context;
}

inline TraceIdentity
Traceable::acceptPost(const TraceContext &post_context) noexcept {
    postContext_ = post_context;
    if (requestId_ != 0) {
        execution_.fetch_add(1, std::memory_order_relaxed);
    }
    return identity();
}

inline void
Traceable::record([[maybe_unused]] Record record) const {
#if defined(NIXL_TRACE_ENABLED)
    if ((tracer_ != nullptr) && postContext_.sampled()) {
        const Attribute attrs[] = {{"nixl.xfer_request_id", requestId_},
                                   {"nixl.xfer_execution", execution()}};
        detail::emitEvent(*tracer_, record, kindOf(record), postContext_, requestId_, attrs);
    }
#endif
}

inline void
Traceable::record([[maybe_unused]] Record record, [[maybe_unused]] const Attribute &attr) const {
#if defined(NIXL_TRACE_ENABLED)
    if ((tracer_ != nullptr) && postContext_.sampled()) {
        const Attribute attrs[] = {
            {"nixl.xfer_request_id", requestId_}, {"nixl.xfer_execution", execution()}, attr};
        detail::emitEvent(*tracer_, record, kindOf(record), postContext_, requestId_, attrs);
    }
#endif
}

inline std::uint64_t
Traceable::execution() const noexcept {
    return execution_.load(std::memory_order_relaxed);
}

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

#define NIXL_TRACE_DETAIL_SCOPE(span, tracer_ptr, trace_record, ...)                            \
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

#define NIXL_TRACE_DETAIL_SAMPLED_SCOPE(span, tracer_ptr, trace_record, context, ...)           \
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
#define NIXL_TRACE_DETAIL_CONTEXT_SPAN(span, tracer_ptr, trace_record, context, request_id, ...) \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                    \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                                 \
        const ::nixl::trace::TraceContext &nixl_trace_context_ = (context);                      \
        if ((nixl_trace_tracer_ == nullptr) || !nixl_trace_context_.sampled()) {                 \
            return ::nixl::trace::Span{};                                                        \
        }                                                                                        \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                         \
        NIXL_TRACE_DETAIL_ANNOTATED_(                                                            \
            ::nixl::trace::detail::beginContextSpan(*nixl_trace_tracer_,                         \
                                                    nixl_trace_record_,                          \
                                                    ::nixl::trace::kindOf(nixl_trace_record_),   \
                                                    nixl_trace_context_,                         \
                                                    (request_id)) __VA_OPT__(, ) __VA_ARGS__);   \
    }()

#define NIXL_TRACE_DETAIL_CORRELATED_SPAN(span, request_trace, trace_record, context, ...)        \
    const ::nixl::trace::Traceable &span##_trace_ = (request_trace);                              \
    ::nixl::trace::CorrelationScope span##_correlation_(                                          \
        ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_),                            \
        (context),                                                                                \
        ::nixl::trace::detail::TraceableAccess::requestId(span##_trace_));                        \
    NIXL_TRACE_DETAIL_CONTEXT_SPAN(span,                                                          \
                                   ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_), \
                                   trace_record,                                                  \
                                   context,                                                       \
                                   ::nixl::trace::detail::TraceableAccess::requestId(             \
                                       span##_trace_) __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_DETAIL_CORRELATED_SCOPE(span, request_trace, trace_record, ...)                 \
    const ::nixl::trace::Traceable &span##_trace_ = (request_trace);                               \
    ::nixl::trace::CorrelationScope span##_correlation_(                                           \
        ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_),                             \
        ::nixl::trace::detail::TraceableAccess::postContext(span##_trace_),                        \
        ::nixl::trace::detail::TraceableAccess::requestId(span##_trace_));                         \
    NIXL_TRACE_DETAIL_SAMPLED_SCOPE(span,                                                          \
                                    ::nixl::trace::detail::TraceableAccess::tracer(span##_trace_), \
                                    trace_record,                                                  \
                                    ::nixl::trace::detail::TraceableAccess::postContext(           \
                                        span##_trace_) __VA_OPT__(, ) __VA_ARGS__)

#define NIXL_TRACE_DETAIL_FINISH(span, status) (::nixl::trace::detail::finish((span), (status)))

#define NIXL_TRACE_DETAIL_NOW(tracer_ptr) ((tracer_ptr) ? ::nixlTime::getUs() : ::nixlTime::us_t{0})

#define NIXL_TRACE_DETAIL_EMIT_SPAN(tracer_ptr, trace_record, start_us)                \
    do {                                                                               \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                 \
            const ::nixl::trace::Record nixl_trace_record_ = (trace_record);           \
            ::nixl::trace::detail::emitSpan(*nixl_trace_tracer_,                       \
                                            nixl_trace_record_,                        \
                                            ::nixl::trace::kindOf(nixl_trace_record_), \
                                            (start_us));                               \
        }                                                                              \
    } while (0)

#define NIXL_TRACE_DETAIL_ATTR(span, attr_key, attr_value)                              \
    do {                                                                                \
        if ((span).active()) {                                                          \
            (span).addAttribute((attr_key), ::nixl::trace::attributeValue(attr_value)); \
        }                                                                               \
    } while (0)

#else // !NIXL_TRACE_ENABLED

#define NIXL_TRACE_DETAIL_SCOPE(span, tracer_ptr, trace_record, ...) \
    do {                                                             \
    } while (0)
#define NIXL_TRACE_DETAIL_SAMPLED_SCOPE(span, tracer_ptr, trace_record, context, ...) \
    do {                                                                              \
    } while (0)
#define NIXL_TRACE_DETAIL_CORRELATED_SPAN(span, request_trace, trace_record, context, ...) \
    do {                                                                                   \
    } while (0)
#define NIXL_TRACE_DETAIL_CORRELATED_SCOPE(span, request_trace, trace_record, ...) \
    do {                                                                           \
    } while (0)
#define NIXL_TRACE_DETAIL_FINISH(span, status) (status)
#define NIXL_TRACE_DETAIL_ATTR(span, attr_key, attr_value) \
    do {                                                   \
    } while (0)
#define NIXL_TRACE_DETAIL_NOW(tracer_ptr) (::nixlTime::us_t{0})
#define NIXL_TRACE_DETAIL_EMIT_SPAN(tracer_ptr, trace_record, start_us) \
    do {                                                                \
    } while (0)

#endif // NIXL_TRACE_ENABLED

#endif
