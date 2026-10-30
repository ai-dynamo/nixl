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
#ifndef NIXL_SRC_CORE_TRACING_TRACEABLE_H
#define NIXL_SRC_CORE_TRACING_TRACEABLE_H

#include <atomic>
#include <cstdint>
#include <string_view>

#include "tracing/trace_context.h"

namespace nixl::trace {

/**
 * @brief A traced object's trace state: the context and request id it was
 *        created with, and the context and execution number of its latest post.
 */
class Traceable {
public:
    Traceable(const TraceContext &context, std::uint64_t request_id) noexcept
        : context_(context),
          postContext_(context),
          requestIdText_(request_id),
          requestId_(request_id) {}

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

    /** @brief requestId() as nixl.xfer_request_id records it. */
    [[nodiscard]] std::string_view
    requestIdText() const noexcept {
        return requestIdText_.view();
    }

    [[nodiscard]] std::uint64_t
    execution() const noexcept {
        return execution_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] TraceContext
    nextPostContext(const Tracer *tracer) const {
        return (context_.sampled() && spanIdUsed_) ? generatePostContext(context_, tracer) :
                                                     context_;
    }

    void
    markSpanIdUsed() noexcept {
        spanIdUsed_ = true;
    }

    void
    acceptPost(const TraceContext &post_context) noexcept {
        postContext_ = post_context;
        if (requestId_ != 0) {
            execution_.fetch_add(1, std::memory_order_relaxed);
        }
    }

private:
    const TraceContext context_;
    TraceContext postContext_;
    bool spanIdUsed_ = false;
    const RequestIdText requestIdText_;
    const std::uint64_t requestId_;
    std::atomic<std::uint64_t> execution_ = 0;
};

} // namespace nixl::trace

#endif
