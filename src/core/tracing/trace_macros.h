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
 * the scope macros bind the returned Span to a named variable (satisfying its
 * [[nodiscard]]) and take a predictable null-check branch on the tracer.
 *
 * Usage contract: at most one NIXL_TRACE_SCOPE per lexical scope (the span is
 * bound to a fixed variable name). Use NIXL_TRACE_ATTR afterwards to annotate
 * it; attribute arguments are only evaluated when a backend is actually active.
 */

#include "tracing/trace.h"

#if defined(NIXL_TRACE_ENABLED)

#define NIXL_TRACE_SCOPE(tracer_ptr, span_name, span_kind)                                    \
    ::nixl::trace::Span nixl_trace_span_ = [&]() -> ::nixl::trace::Span {                     \
        auto *nixl_trace_tracer_ = (tracer_ptr);                                              \
        return nixl_trace_tracer_ ? nixl_trace_tracer_->beginSpan((span_name), (span_kind)) : \
                                    ::nixl::trace::Span{};                                    \
    }()

#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind)       \
    do {                                                        \
        if (auto *nixl_trace_tracer_ = (tracer_ptr)) {          \
            nixl_trace_tracer_->mark((mark_name), (mark_kind)); \
        }                                                       \
    } while (0)

#define NIXL_TRACE_ATTR(attr_key, attr_value)                        \
    do {                                                             \
        if (nixl_trace_span_.active()) {                             \
            nixl_trace_span_.addAttribute((attr_key), (attr_value)); \
        }                                                            \
    } while (0)

#else // !NIXL_TRACE_ENABLED

#define NIXL_TRACE_SCOPE(tracer_ptr, span_name, span_kind) \
    do {                                                   \
    } while (0)
#define NIXL_TRACE_MARK(tracer_ptr, mark_name, mark_kind) \
    do {                                                  \
    } while (0)
#define NIXL_TRACE_ATTR(attr_key, attr_value) \
    do {                                      \
    } while (0)

#endif // NIXL_TRACE_ENABLED

#endif // NIXL_SRC_CORE_TRACING_TRACE_MACROS_H
