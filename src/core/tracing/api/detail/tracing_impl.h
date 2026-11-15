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

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <string_view>

#include "common/nixl_time.h"
#include "tracing/api/attribute.h"
#include "tracing/api/detail/call_scope.h"
#include "tracing/api/detail/span.h"
#include "tracing/api/record.h"
#include "tracing/api/trace_identity.h"
#include "tracing/api/tracer.h"

namespace nixl::trace::detail {

[[nodiscard]] inline Call
currentCall() noexcept;

[[nodiscard]] inline nixlTime::us_t
requestStart(const Tracer *tracer, const TraceContext &context) noexcept;

[[nodiscard]] inline TraceIdentity
suppliedIdentity(const Tracer *tracer, const nixl_blob_t &parent);

inline void
addAttributes(Span &span, std::initializer_list<Attribute> attributes);

inline void
addBaggage(Span &span, std::string_view baggage);

template<typename Status>
[[nodiscard]] Status
finish(Span &span, Status status);

template<typename Status>
[[nodiscard]] Status
end(Span &span, Status status);

} // namespace nixl::trace::detail

namespace nixl::trace {

inline Traceable::Traceable() : Traceable(detail::currentCall()) {}

inline Traceable::Traceable(const detail::Call &call)
    : Traceable(call.tracer, generateRequestIdentity(call.tracer, suppliedContext(call.options))) {
    if (requestContext_.sampled()) {
        requestBaggage_ = parseBaggage(tracer_, suppliedBaggage(call.options));
    }
}

inline Traceable::Traceable(Tracer *tracer, const TraceIdentity &identity) noexcept
    : tracer_(tracer),
      requestContext_(identity.context),
      postIdentity_{identity.context, identity.requestId, 0},
      createdAt_(detail::requestStart(tracer, identity.context)) {}

inline Traceable::~Traceable() {
    if ((tracer_ != nullptr) && requestContext_.sampled()) {
        try {
            const Attribute attrs[] = {{AttributeKey::NixlXferRequestId, postIdentity_.requestId},
                                       {AttributeKey::NixlXferExecution, postIdentity_.execution}};
            tracer_->emitContextSpan(Record::XferReq,
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

inline std::string_view
Traceable::requestBaggage() const noexcept {
    return requestBaggage_;
}

inline std::string_view
Traceable::postBaggage() const noexcept {
    return postBaggage_;
}

inline const TraceIdentity &
Traceable::post(const nixl_blob_t &parent, const nixl_blob_t &baggage) {
    if (postIdentity_.requestId != 0) {
        const std::optional<TraceContext> supplied =
            parent.empty() ? std::nullopt : parseParentContext(tracer_, parent);
        const TraceContext &under = supplied ? *supplied : requestContext_;
        postIdentity_.context = under.sampled() ? generatePostContext(under, tracer_) : under;
        ++postIdentity_.execution;
        if (postIdentity_.context.sampled()) {
            const std::string_view own = parseBaggage(tracer_, baggage);
            postBaggage_.assign(own.empty() ? std::string_view{requestBaggage_} : own);
        } else {
            postBaggage_.clear();
        }
    }
    return postIdentity_;
}

inline void
Traceable::trace(Record record) const {
    if ((tracer_ != nullptr) && postIdentity_.context.sampled()) {
        const Attribute attrs[] = {{AttributeKey::NixlXferRequestId, postIdentity_.requestId},
                                   {AttributeKey::NixlXferExecution, postIdentity_.execution}};
        tracer_->emitEvent(
            record, kindOf(record), postIdentity_.context, postIdentity_.requestId, attrs);
    }
}

inline void
Traceable::trace(Record record, const Attribute &attr) const {
    if ((tracer_ != nullptr) && postIdentity_.context.sampled()) {
        const Attribute attrs[] = {{AttributeKey::NixlXferRequestId, postIdentity_.requestId},
                                   {AttributeKey::NixlXferExecution, postIdentity_.execution},
                                   attr};
        tracer_->emitEvent(
            record, kindOf(record), postIdentity_.context, postIdentity_.requestId, attrs);
    }
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
detail::requestStart(const Tracer *tracer, const TraceContext &context) noexcept {
    if ((tracer != nullptr) && context.sampled()) {
        return nixlTime::getUs();
    }
    return 0;
}

} // namespace nixl::trace

#define NIXL_TRACE_DETAIL_REQUEST_SCOPE_(span, tracer_ptr, identity, baggage)  \
    std::optional<::nixl::trace::Scope> span##_scope;                          \
    if (::nixl::trace::Tracer *nixl_trace_scope_tracer_ = (tracer_ptr)) {      \
        span##_scope.emplace(nixl_trace_scope_tracer_, (identity), (baggage)); \
    }

#define NIXL_TRACE_DETAIL_SUPPLIED_SCOPE_(span, tracer_ptr, options)                   \
    std::optional<::nixl::trace::Scope> span##_scope;                                  \
    std::string_view span##_baggage;                                                   \
    if (::nixl::trace::Tracer *nixl_trace_scope_tracer_ = (tracer_ptr)) {              \
        const nixl_opt_args_t *const nixl_trace_scope_options_ = (options);            \
        if (const nixl_blob_t &nixl_trace_scope_parent_ =                              \
                ::nixl::trace::suppliedContext(nixl_trace_scope_options_);             \
            !nixl_trace_scope_parent_.empty()) {                                       \
            const ::nixl::trace::TraceIdentity nixl_trace_scope_identity_ =            \
                ::nixl::trace::detail::suppliedIdentity(nixl_trace_scope_tracer_,      \
                                                        nixl_trace_scope_parent_);     \
            if (nixl_trace_scope_identity_.context.sampled()) {                        \
                span##_baggage = ::nixl::trace::parseBaggage(                          \
                    nixl_trace_scope_tracer_,                                          \
                    ::nixl::trace::suppliedBaggage(nixl_trace_scope_options_));        \
            }                                                                          \
            span##_scope.emplace(                                                      \
                nixl_trace_scope_tracer_, nixl_trace_scope_identity_, span##_baggage); \
        }                                                                              \
    }

namespace nixl::trace::detail {

extern std::atomic<bool> callScopeOpened;

[[nodiscard]] Call
currentCallOnThread() noexcept;

inline Call
currentCall() noexcept {
    return callScopeOpened.load(std::memory_order_relaxed) ? currentCallOnThread() : Call{};
}

inline void
addAttributes(Span &span, std::initializer_list<Attribute> attributes) {
    if (attributes.size() != 0) {
        span.addAttributes({attributes.begin(), attributes.size()});
    }
}

inline void
addBaggage(Span &span, std::string_view baggage) {
    if (span.active() && !baggage.empty()) {
        span.addAttribute(AttributeKey::NixlBaggage, baggage);
    }
}

template<typename Status>
[[nodiscard]] Status
finish(Span &span, Status status) {
    if (span.active()) {
        span.addAttribute(AttributeKey::NixlStatus, attributeValue(status));
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

#define NIXL_TRACE_DETAIL_SCOPE(span, tracer_ptr, trace_record, ...)                      \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                             \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                          \
        if (nixl_trace_tracer_ == nullptr) {                                              \
            return ::nixl::trace::Span{};                                                 \
        }                                                                                 \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                  \
        NIXL_TRACE_DETAIL_ANNOTATED_(nixl_trace_tracer_->beginSpan(                       \
            nixl_trace_record_, ::nixl::trace::kindOf(nixl_trace_record_)) __VA_OPT__(, ) \
                                         __VA_ARGS__);                                    \
    }()

#define NIXL_TRACE_DETAIL_SCOPE_UNDER(span, tracer_ptr, options, trace_record, ...)           \
    ::nixl::trace::Tracer *const span##_tracer = (tracer_ptr);                                \
    NIXL_TRACE_DETAIL_SUPPLIED_SCOPE_(span, span##_tracer, options)                           \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                 \
        auto *nixl_trace_tracer_ = span##_tracer;                                             \
        if (nixl_trace_tracer_ == nullptr) {                                                  \
            return ::nixl::trace::Span{};                                                     \
        }                                                                                     \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                      \
        if (!span##_scope || !span##_scope->identity().context.valid()) {                     \
            NIXL_TRACE_DETAIL_ANNOTATED_(nixl_trace_tracer_->beginSpan(                       \
                nixl_trace_record_, ::nixl::trace::kindOf(nixl_trace_record_)) __VA_OPT__(, ) \
                                             __VA_ARGS__);                                    \
        }                                                                                     \
        if (!span##_scope->identity().context.sampled()) {                                    \
            return ::nixl::trace::Span{};                                                     \
        }                                                                                     \
        NIXL_TRACE_DETAIL_ANNOTATED_(                                                         \
            nixl_trace_tracer_->beginContextSpan(nixl_trace_record_,                          \
                                                 ::nixl::trace::kindOf(nixl_trace_record_),   \
                                                 span##_scope->identity().context,            \
                                                 0) __VA_OPT__(, ) __VA_ARGS__);              \
    }();                                                                                      \
    ::nixl::trace::detail::addBaggage(span, span##_baggage)

#define NIXL_TRACE_DETAIL_SAMPLED_SCOPE(span, tracer_ptr, trace_record, context, ...)     \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                             \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                          \
        if ((nixl_trace_tracer_ == nullptr) || !(context).sampled()) {                    \
            return ::nixl::trace::Span{};                                                 \
        }                                                                                 \
        const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                  \
        NIXL_TRACE_DETAIL_ANNOTATED_(nixl_trace_tracer_->beginSpan(                       \
            nixl_trace_record_, ::nixl::trace::kindOf(nixl_trace_record_)) __VA_OPT__(, ) \
                                         __VA_ARGS__);                                    \
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
            nixl_trace_tracer_->beginContextSpan(nixl_trace_record_,                             \
                                                 ::nixl::trace::kindOf(nixl_trace_record_),      \
                                                 nixl_trace_context_,                            \
                                                 (request_id)) __VA_OPT__(, ) __VA_ARGS__);      \
    }()

#define NIXL_TRACE_DETAIL_REQUEST_SPAN(span, traceable, trace_record, ...)                       \
    NIXL_TRACE_DETAIL_REQUEST_SCOPE_(                                                            \
        span, (traceable).tracer(), (traceable).requestIdentity(), (traceable).requestBaggage()) \
    NIXL_TRACE_DETAIL_SAMPLED_SCOPE(                                                             \
        span,                                                                                    \
        (traceable).tracer(),                                                                    \
        trace_record,                                                                            \
        span##_scope->identity().context,                                                        \
        __VA_ARGS__ __VA_OPT__(, ){::nixl::trace::AttributeKey::NixlXferRequestId,               \
                                   span##_scope->identity().requestId},                          \
        {::nixl::trace::AttributeKey::NixlXferExecution, span##_scope->identity().execution});   \
    ::nixl::trace::detail::addBaggage(span, (traceable).requestBaggage())

#define NIXL_TRACE_DETAIL_POST_SPAN(span, traceable, options, trace_record, ...)      \
    const nixl_opt_args_t *const span##_options = (options);                          \
    const ::nixl::trace::TraceIdentity &span##_identity =                             \
        (traceable).post(::nixl::trace::suppliedContext(span##_options),              \
                         ::nixl::trace::suppliedBaggage(span##_options));             \
    NIXL_TRACE_DETAIL_REQUEST_SCOPE_(                                                 \
        span, (traceable).tracer(), span##_identity, (traceable).postBaggage())       \
    NIXL_TRACE_DETAIL_CONTEXT_SPAN(                                                   \
        span,                                                                         \
        (traceable).tracer(),                                                         \
        trace_record,                                                                 \
        span##_identity.context,                                                      \
        span##_identity.requestId,                                                    \
        __VA_ARGS__ __VA_OPT__(, ){::nixl::trace::AttributeKey::NixlXferRequestId,    \
                                   span##_identity.requestId},                        \
        {::nixl::trace::AttributeKey::NixlXferExecution, span##_identity.execution}); \
    ::nixl::trace::detail::addBaggage(span, (traceable).postBaggage())

#define NIXL_TRACE_DETAIL_FINISH(span, status) (::nixl::trace::detail::finish((span), (status)))

#define NIXL_TRACE_DETAIL_END(span, status) (::nixl::trace::detail::end((span), (status)))

#define NIXL_TRACE_DETAIL_NOW(tracer_ptr) ((tracer_ptr) ? ::nixlTime::getUs() : ::nixlTime::us_t{0})

#define NIXL_TRACE_DETAIL_EMIT_SPAN(tracer_ptr, trace_record, start_us)                     \
    do {                                                                                    \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                      \
            const ::nixl::trace::Record nixl_trace_record_ = (trace_record);                \
            nixl_trace_tracer_->emitSpan(                                                   \
                nixl_trace_record_, ::nixl::trace::kindOf(nixl_trace_record_), (start_us)); \
        }                                                                                   \
    } while (0)

#define NIXL_TRACE_DETAIL_ATTR(span, attr_key, attr_value)                              \
    do {                                                                                \
        if ((span).active()) {                                                          \
            (span).addAttribute((attr_key), ::nixl::trace::attributeValue(attr_value)); \
        }                                                                               \
    } while (0)

#endif
