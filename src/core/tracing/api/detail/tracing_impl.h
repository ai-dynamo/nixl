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
#ifndef NIXL_SRC_CORE_TRACING_API_DETAIL_TRACING_IMPL_H
#define NIXL_SRC_CORE_TRACING_API_DETAIL_TRACING_IMPL_H

#include <cstdint>
#include <initializer_list>

#include "common/nixl_time.h"
#include "tracing/api/attribute.h"
#include "tracing/api/detail/span.h"
#include "tracing/api/detail/trace_impl.h"
#include "tracing/api/record.h"
#include "tracing/api/trace_identity.h"

namespace nixl::trace {

inline Traceable::Traceable(Tracer *tracer) : Traceable(tracer, generateRequestIdentity(tracer)) {}

inline Traceable::Traceable(Tracer *tracer, const TraceIdentity &identity) noexcept
    : tracer_(tracer),
      requestContext_(identity.context),
      postIdentity_{identity.context, identity.requestId, 0},
      createdAt_(detail::requestStart(tracer, identity.context)) {}

inline Traceable::~Traceable() {
#if defined(NIXL_TRACE_ENABLED)
    if ((tracer_ != nullptr) && requestContext_.sampled()) {
        try {
            const Attribute attrs[] = {{"nixl.xfer_request_id", postIdentity_.requestId},
                                       {"nixl.xfer_execution", postIdentity_.execution}};
            detail::emitContextSpan(*tracer_,
                                    Record::XferReq,
                                    kindOf(Record::XferReq),
                                    requestContext_,
                                    postIdentity_.requestId,
                                    createdAt_,
                                    nixlTime::getUs(),
                                    attrs);
        }
        catch (...) {
        }
    }
#endif
}

inline Tracer *
Traceable::tracer() const noexcept {
    return tracer_;
}

inline const TraceIdentity &
Traceable::postIdentity() const noexcept {
    return postIdentity_;
}

inline TraceIdentity
Traceable::requestIdentity() const noexcept {
    return {requestContext_, postIdentity_.requestId, postIdentity_.execution};
}

inline const TraceIdentity &
Traceable::post(const nixl_blob_t &parent) {
    if (postIdentity_.requestId != 0) {
        const std::optional<TraceContext> supplied =
            parent.empty() ? std::nullopt : parseParentContext(tracer_, parent);
        const TraceContext &under = supplied ? *supplied : requestContext_;
        postIdentity_.context = under.sampled() ? generatePostContext(under, tracer_) : under;
        ++postIdentity_.execution;
    }
    return postIdentity_;
}

inline void
Traceable::trace([[maybe_unused]] Record record) const {
#if defined(NIXL_TRACE_ENABLED)
    if ((tracer_ != nullptr) && postIdentity_.context.sampled()) {
        const Attribute attrs[] = {{"nixl.xfer_request_id", postIdentity_.requestId},
                                   {"nixl.xfer_execution", postIdentity_.execution}};
        detail::emitEvent(*tracer_,
                          record,
                          kindOf(record),
                          postIdentity_.context,
                          postIdentity_.requestId,
                          attrs);
    }
#endif
}

inline void
Traceable::trace([[maybe_unused]] Record record, [[maybe_unused]] const Attribute &attr) const {
#if defined(NIXL_TRACE_ENABLED)
    if ((tracer_ != nullptr) && postIdentity_.context.sampled()) {
        const Attribute attrs[] = {{"nixl.xfer_request_id", postIdentity_.requestId},
                                   {"nixl.xfer_execution", postIdentity_.execution},
                                   attr};
        detail::emitEvent(*tracer_,
                          record,
                          kindOf(record),
                          postIdentity_.context,
                          postIdentity_.requestId,
                          attrs);
    }
#endif
}

inline TraceIdentity
detail::suppliedIdentity(const Tracer *tracer, const nixl_blob_t &parent) {
    const std::optional<TraceContext> supplied = parseParentContext(tracer, parent);
    if (!supplied) {
        return {};
    }
    return {supplied->sampled() ? generatePostContext(*supplied, tracer) : *supplied, 0, 0};
}

inline nixlTime::us_t
detail::requestStart([[maybe_unused]] const Tracer *tracer,
                     [[maybe_unused]] const TraceContext &context) noexcept {
#if defined(NIXL_TRACE_ENABLED)
    if ((tracer != nullptr) && context.sampled()) {
        return nixlTime::getUs();
    }
#endif
    return 0;
}

} // namespace nixl::trace

#define NIXL_TRACE_DETAIL_REQUEST_SCOPE_(span, tracer_ptr, identity)      \
    std::optional<::nixl::trace::Scope> span##_scope;                     \
    if (::nixl::trace::Tracer *nixl_trace_scope_tracer_ = (tracer_ptr)) { \
        span##_scope.emplace(nixl_trace_scope_tracer_, (identity));       \
    }

#define NIXL_TRACE_DETAIL_SUPPLIED_SCOPE_(span, tracer_ptr, parent)                            \
    std::optional<::nixl::trace::Scope> span##_scope;                                          \
    if (((tracer_ptr) != nullptr) && !(parent).empty()) {                                      \
        span##_scope.emplace((tracer_ptr),                                                     \
                             ::nixl::trace::detail::suppliedIdentity((tracer_ptr), (parent))); \
    }

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

template<typename Status>
[[nodiscard]] Status
end(Span &span, Status status) {
    const Status finished = finish(span, status);
    span = Span{};
    return finished;
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

#define NIXL_TRACE_DETAIL_SCOPE_UNDER(span, tracer_ptr, parent, trace_record, ...)             \
    NIXL_TRACE_DETAIL_SUPPLIED_SCOPE_(span, tracer_ptr, parent)                                \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                  \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                               \
        if (nixl_trace_tracer_ == nullptr) {                                                   \
            return ::nixl::trace::Span{};                                                      \
        }                                                                                      \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                       \
        if (!span##_scope || !span##_scope->identity().context.valid()) {                      \
            NIXL_TRACE_DETAIL_ANNOTATED_(::nixl::trace::detail::beginSpan(                     \
                *nixl_trace_tracer_,                                                           \
                nixl_trace_record_,                                                            \
                ::nixl::trace::kindOf(nixl_trace_record_)) __VA_OPT__(, ) __VA_ARGS__);        \
        }                                                                                      \
        if (!span##_scope->identity().context.sampled()) {                                     \
            return ::nixl::trace::Span{};                                                      \
        }                                                                                      \
        NIXL_TRACE_DETAIL_ANNOTATED_(                                                          \
            ::nixl::trace::detail::beginContextSpan(*nixl_trace_tracer_,                       \
                                                    nixl_trace_record_,                        \
                                                    ::nixl::trace::kindOf(nixl_trace_record_), \
                                                    span##_scope->identity().context,          \
                                                    0) __VA_OPT__(, ) __VA_ARGS__);            \
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

#define NIXL_TRACE_DETAIL_REQUEST_SPAN(span, traceable, trace_record, ...)                      \
    NIXL_TRACE_DETAIL_REQUEST_SCOPE_(span, (traceable).tracer(), (traceable).requestIdentity()) \
    NIXL_TRACE_DETAIL_SAMPLED_SCOPE(                                                            \
        span,                                                                                   \
        (traceable).tracer(),                                                                   \
        trace_record,                                                                           \
        span##_scope->identity().context,                                                       \
        __VA_ARGS__ __VA_OPT__(, ){"nixl.xfer_request_id", span##_scope->identity().requestId}, \
        {"nixl.xfer_execution", span##_scope->identity().execution})

#define NIXL_TRACE_DETAIL_POST_SPAN(span, traceable, parent, trace_record, ...)        \
    const ::nixl::trace::TraceIdentity &span##_identity = (traceable).post(parent);    \
    NIXL_TRACE_DETAIL_REQUEST_SCOPE_(span, (traceable).tracer(), span##_identity)      \
    NIXL_TRACE_DETAIL_CONTEXT_SPAN(                                                    \
        span,                                                                          \
        (traceable).tracer(),                                                          \
        trace_record,                                                                  \
        span##_identity.context,                                                       \
        span##_identity.requestId,                                                     \
        __VA_ARGS__ __VA_OPT__(, ){"nixl.xfer_request_id", span##_identity.requestId}, \
        {"nixl.xfer_execution", span##_identity.execution})

#define NIXL_TRACE_DETAIL_FINISH(span, status) (::nixl::trace::detail::finish((span), (status)))

#define NIXL_TRACE_DETAIL_END(span, status) (::nixl::trace::detail::end((span), (status)))

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
#define NIXL_TRACE_DETAIL_SCOPE_UNDER(span, tracer_ptr, parent, trace_record, ...) \
    NIXL_TRACE_DETAIL_SUPPLIED_SCOPE_(span, tracer_ptr, parent)                    \
    do {                                                                           \
    } while (0)
#define NIXL_TRACE_DETAIL_SAMPLED_SCOPE(span, tracer_ptr, trace_record, context, ...) \
    do {                                                                              \
    } while (0)
#define NIXL_TRACE_DETAIL_REQUEST_SPAN(span, traceable, trace_record, ...) \
    NIXL_TRACE_DETAIL_REQUEST_SCOPE_(span, (traceable).tracer(), (traceable).requestIdentity())
#define NIXL_TRACE_DETAIL_POST_SPAN(span, traceable, parent, trace_record, ...)     \
    const ::nixl::trace::TraceIdentity &span##_identity = (traceable).post(parent); \
    NIXL_TRACE_DETAIL_REQUEST_SCOPE_(span, (traceable).tracer(), span##_identity)
#define NIXL_TRACE_DETAIL_FINISH(span, status) (status)
#define NIXL_TRACE_DETAIL_END(span, status) (status)
#define NIXL_TRACE_DETAIL_ATTR(span, attr_key, attr_value) \
    do {                                                   \
    } while (0)
#define NIXL_TRACE_DETAIL_NOW(tracer_ptr) (::nixlTime::us_t{0})
#define NIXL_TRACE_DETAIL_EMIT_SPAN(tracer_ptr, trace_record, start_us) \
    do {                                                                \
    } while (0)

#endif // NIXL_TRACE_ENABLED

#endif
