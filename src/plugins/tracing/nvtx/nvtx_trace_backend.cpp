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

#include <memory>
#include <string>
#include <utility>

#include "nvtx_events.h"
#include "nvtx_span.h"

namespace nixl::trace::nvtx_internal {

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
    std::string fallback;
    const nvtxEventAttributes_t ev = eventForName(name, kind, registeredHandles_, fallback);
    auto span = std::make_unique<NvtxSpan>(domain_, schemaIds_);
    nvtxDomainRangePushEx(domain_, &ev);
    return span;
}

void
NvtxTraceBackend::mark(const std::string_view name, const Kind kind) {
    std::string fallback;
    const nvtxEventAttributes_t ev = eventForName(name, kind, registeredHandles_, fallback);
    nvtxDomainMarkEx(domain_, &ev);
}

} // namespace nixl::trace::nvtx_internal
