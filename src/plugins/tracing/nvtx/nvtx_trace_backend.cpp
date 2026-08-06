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

#include "nvtx_trace_backend.h"

#include <cstdint>
#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "nvtx_events.h"
#include "nvtx_span.h"

namespace nixl::trace::nvtx_internal {
namespace {

    using correlation_stack_t = std::stack<std::uint64_t, std::vector<std::uint64_t>>;

    // static thread_local is shared across backend instances on a thread, which
    // is safe here: push/pop are balanced within a single agent call, so the
    // stack is empty between calls.
    [[nodiscard]] correlation_stack_t &
    correlationStack() noexcept {
        static thread_local correlation_stack_t stack;
        return stack;
    }

    void
    applyCorrelation(nvtxEventAttributes_t &ev) noexcept {
        const correlation_stack_t &stack = correlationStack();
        if (!stack.empty()) {
            ev.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            ev.payload.ullValue = stack.top();
        }
    }

} // namespace

NvtxTraceBackend::NvtxTraceBackend(const std::string_view domain)
    : domainName_(domain),
      domain_(nvtxDomainCreateA(domainName_.c_str())),
      schemaIds_(registerPayloadSchemas(domain_)),
      registeredHandles_(registerSpanNames(domain_)) {}

NvtxTraceBackend::~NvtxTraceBackend() {
    nvtxDomainDestroy(domain_);
}

std::unique_ptr<SpanBackend>
NvtxTraceBackend::beginSpan(const std::string_view name, const Kind kind) {
    nvtxEventAttributes_t ev = eventForName(name, kind, domain_, registeredHandles_);
    applyCorrelation(ev);
    auto span = std::make_unique<NvtxSpan>(domain_, schemaIds_);
    nvtxDomainRangePushEx(domain_, &ev);
    return span;
}

void
NvtxTraceBackend::mark(const std::string_view name, const Kind kind) {
    nvtxEventAttributes_t ev = eventForName(name, kind, domain_, registeredHandles_);
    applyCorrelation(ev);
    nvtxDomainMarkEx(domain_, &ev);
}

void
NvtxTraceBackend::pushCorrelationId(const std::uint64_t id) {
    correlationStack().push(id);
}

void
NvtxTraceBackend::popCorrelationId() {
    correlation_stack_t &stack = correlationStack();
    if (!stack.empty()) {
        stack.pop();
    }
}

} // namespace nixl::trace::nvtx_internal
