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

#include <initializer_list>

#include "tracing/trace.h"

#if defined(NIXL_TRACE_ENABLED)

#define NIXL_TRACE_SCOPE(span, tracer_ptr, span_name, span_kind)                              \
    ::nixl::trace::Span span = [&]() -> ::nixl::trace::Span {                                 \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                              \
        return nixl_trace_tracer_ ? nixl_trace_tracer_->beginSpan((span_name), (span_kind)) : \
                                    ::nixl::trace::Span{};                                    \
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
            nixl_trace_tracer_->beginContextSpan(                                            \
                (span_name), (span_kind), nixl_trace_context_, (request_id)) :               \
            nixl_trace_tracer_->beginSpan((span_name), (span_kind));                         \
    }()

#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind)       \
    do {                                                        \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {          \
            nixl_trace_tracer_->mark((mark_name), (mark_kind)); \
        }                                                       \
    } while (0)

#define NIXL_TRACE_EVENT(tracer_ptr, event_name, event_kind, event_context, event_request_id, ...) \
    do {                                                                                           \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {                                             \
            nixl_trace_tracer_->emitEvent(                                                         \
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

#endif // NIXL_SRC_CORE_TRACING_TRACE_MACROS_H
