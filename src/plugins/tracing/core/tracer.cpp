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

#include <algorithm>

#include "tracing/core/trace.h"

#include "common/nixl_log.h"
#include "tracing/core/trace_sink.h"

namespace {

void
emitSpanTo(std::span<const std::unique_ptr<nixl::trace::TraceBackend>> backends,
           nixl::trace::Name name,
           nixl::trace::Kind kind,
           const nixl::trace::TraceIdentity &identity,
           nixlTime::us_t start,
           nixlTime::us_t end,
           std::span<const nixl::trace::Attribute> attrs) {
    for (const auto &backend : backends) {
        backend->emitSpan(name, kind, identity, start, end, attrs);
    }
}

void
emitEventTo(std::span<const std::unique_ptr<nixl::trace::TraceBackend>> backends,
            nixl::trace::Name name,
            nixl::trace::Kind kind,
            const nixl::trace::TraceIdentity &identity,
            nixlTime::us_t at,
            std::span<const nixl::trace::Attribute> attrs) {
    for (const auto &backend : backends) {
        backend->emitEvent(name, kind, identity, at, attrs);
    }
}

[[nodiscard]] std::vector<std::unique_ptr<nixl::trace::TraceBackend>>
capBackends(std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends) noexcept {
    if (backends.size() > nixl::trace::maxTraceBackends) {
        try {
            NIXL_WARN << "nixl::trace: only the first " << nixl::trace::maxTraceBackends << " of "
                      << backends.size() << " backends are active";
        }
        catch (...) {
        }
        backends.resize(nixl::trace::maxTraceBackends);
    }
    return backends;
}

} // namespace

namespace nixl::trace {

CompositeTracer::CompositeTracer(std::vector<std::unique_ptr<TraceBackend>> backends,
                                 double sample_ratio,
                                 random_bytes_fn_t random_bytes) noexcept
    : backends_(capBackends(std::move(backends))),
      sampleRatio_(sample_ratio),
      randomBytes_(random_bytes),
      needsTimestamps_(std::ranges::any_of(backends_, [](const auto &backend) {
          return !backend->ownsTimestamps();
      })) {}

CompositeTracer::~CompositeTracer() = default;

Span
CompositeTracer::beginSpan(Name name, Kind kind) {
    std::array<std::unique_ptr<SpanBackend>, maxTraceBackends> spans;
    std::size_t count = 0;
    for (const auto &backend : backends_) {
        // Drop null backend spans so an all-null Span is not reported active()
        // (which would null-deref in addAttribute()/id()).
        if (count == spans.size()) {
            break;
        }
        if (auto backend_span = backend->beginSpan(name, kind)) {
            spans[count++] = std::move(backend_span);
        }
    }
    return Span(std::move(spans));
}

Span
CompositeTracer::beginContextSpan(Name name, Kind kind, const TraceIdentity &identity) {
    std::array<std::unique_ptr<SpanBackend>, maxTraceBackends> spans;
    std::size_t count = 0;
    for (const auto &backend : backends_) {
        if (count == spans.size()) {
            break;
        }
        if (auto backend_span = backend->beginContextSpan(name, kind, identity)) {
            spans[count++] = std::move(backend_span);
        }
    }
    return Span(std::move(spans));
}

void
CompositeTracer::pushContext(const TraceIdentity &identity) {
    for (const auto &backend : backends_) {
        backend->pushContext(identity);
    }
}

void
CompositeTracer::popContext() {
    for (const auto &backend : backends_) {
        backend->popContext();
    }
}

void
CompositeTracer::emitSpan(Name name, Kind kind, nixlTime::us_t start) {
    emitSpanTo(backends_, name, kind, TraceIdentity{}, start, nixlTime::getUs(), {});
}

void
CompositeTracer::emitSpan(Name name,
                          Kind kind,
                          const TraceIdentity &identity,
                          nixlTime::us_t start,
                          nixlTime::us_t end,
                          std::span<const Attribute> attrs) {
    emitSpanTo(backends_, name, kind, identity, start, end, attrs);
}

void
CompositeTracer::emitEvent(Name name,
                           Kind kind,
                           const TraceIdentity &identity,
                           std::span<const Attribute> attrs) {
    emitEventTo(backends_,
                name,
                kind,
                identity,
                needsTimestamps_ ? nixlTime::getUs() : nixlTime::us_t{0},
                attrs);
}

void
CompositeTracer::emitEvent(Name name,
                           Kind kind,
                           const TraceIdentity &identity,
                           nixlTime::us_t at,
                           std::span<const Attribute> attrs) {
    emitEventTo(backends_, name, kind, identity, at, attrs);
}

void
CompositeTracer::emitContextSpan(Name name,
                                 Kind kind,
                                 const TraceIdentity &identity,
                                 nixlTime::us_t start,
                                 nixlTime::us_t end,
                                 std::span<const Attribute> attrs) {
    for (const auto &backend : backends_) {
        backend->emitContextSpan(name, kind, identity, start, end, attrs);
    }
}

std::string
CompositeTracer::traceparent(const TraceContext &context) const {
    return formatTraceparent(context);
}

std::unique_ptr<nixlBackendTraceSink>
CompositeTracer::makeSink(std::string backend, std::function<void(std::uint64_t)> on_dropped) {
    return std::make_unique<TracerPhaseSink>(*this, std::move(backend), std::move(on_dropped));
}

} // namespace nixl::trace
