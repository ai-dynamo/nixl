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
#include <unordered_set>

#include "tracing/core/trace.h"

#include "common/nixl_log.h"
#include "plugin_manager.h"
#include "tracing/api/trace_plugin.h"
#include "tracing/core/trace_sink.h"

namespace {

void
emitSpanTo(std::span<const std::unique_ptr<nixl::trace::TraceBackend>> backends,
           nixl::trace::Name name,
           nixl::trace::Kind kind,
           const nixl::trace::TraceContext &context,
           std::uint64_t request_id,
           nixlTime::us_t start,
           nixlTime::us_t end,
           std::span<const nixl::trace::Attribute> attrs) {
    for (const auto &backend : backends) {
        backend->emitSpan(name, kind, context, request_id, start, end, attrs);
    }
}

void
emitEventTo(std::span<const std::unique_ptr<nixl::trace::TraceBackend>> backends,
            nixl::trace::Name name,
            nixl::trace::Kind kind,
            const nixl::trace::TraceContext &context,
            std::uint64_t request_id,
            nixlTime::us_t at,
            std::span<const nixl::trace::Attribute> attrs) {
    for (const auto &backend : backends) {
        backend->emitEvent(name, kind, context, request_id, at, attrs);
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
CompositeTracer::beginContextSpan(Name name,
                                  Kind kind,
                                  const TraceContext &context,
                                  std::uint64_t request_id) {
    std::array<std::unique_ptr<SpanBackend>, maxTraceBackends> spans;
    std::size_t count = 0;
    for (const auto &backend : backends_) {
        if (count == spans.size()) {
            break;
        }
        if (auto backend_span = backend->beginContextSpan(name, kind, context, request_id)) {
            spans[count++] = std::move(backend_span);
        }
    }
    return Span(std::move(spans));
}

void
CompositeTracer::pushContext(const TraceContext &context, std::uint64_t request_id) {
    for (const auto &backend : backends_) {
        backend->pushContext(context, request_id);
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
    emitSpanTo(backends_, name, kind, TraceContext{}, 0, start, nixlTime::getUs(), {});
}

void
CompositeTracer::emitSpan(Name name,
                          Kind kind,
                          const TraceContext &context,
                          std::uint64_t request_id,
                          nixlTime::us_t start,
                          nixlTime::us_t end,
                          std::span<const Attribute> attrs) {
    emitSpanTo(backends_, name, kind, context, request_id, start, end, attrs);
}

void
CompositeTracer::emitEvent(Name name,
                           Kind kind,
                           const TraceContext &context,
                           std::uint64_t request_id,
                           std::span<const Attribute> attrs) {
    emitEventTo(backends_,
                name,
                kind,
                context,
                request_id,
                needsTimestamps_ ? nixlTime::getUs() : nixlTime::us_t{0},
                attrs);
}

void
CompositeTracer::emitEvent(Name name,
                           Kind kind,
                           const TraceContext &context,
                           std::uint64_t request_id,
                           nixlTime::us_t at,
                           std::span<const Attribute> attrs) {
    emitEventTo(backends_, name, kind, context, request_id, at, attrs);
}

void
CompositeTracer::emitContextSpan(Name name,
                                 Kind kind,
                                 const TraceContext &context,
                                 std::uint64_t request_id,
                                 nixlTime::us_t start,
                                 nixlTime::us_t end,
                                 std::span<const Attribute> attrs) {
    for (const auto &backend : backends_) {
        backend->emitContextSpan(name, kind, context, request_id, start, end, attrs);
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

/*** Factory ***/

std::unique_ptr<CompositeTracer>
makeTracer(const TracerConfig &config) {
    std::vector<std::unique_ptr<TraceBackend>> backends;
    std::unordered_set<std::string> seen;

    auto &plugin_manager = nixlPluginManager::getInstance();
    const nixlTraceBackendInitParams init_params{config.agentName};

    for (const auto &requested : config.backends) {
        // Skip blanks and duplicates so e.g. "nvtx,nvtx" activates one backend.
        if (requested.empty() || !seen.insert(requested).second) {
            continue;
        }

        if (backends.size() == maxTraceBackends) {
            NIXL_WARN << "nixl::trace: backend '" << requested << "' not activated: at most "
                      << maxTraceBackends << " backends can be active";
            continue;
        }

        // Each backend is an on-demand .so plugin (libtrace_backend_<name>.so).
        // A missing plugin is not fatal: warn and skip so the rest still apply.
        auto handle = plugin_manager.loadTracePlugin(requested);
        if (!handle) {
            NIXL_WARN << "nixl::trace: backend '" << requested
                      << "' requested but plugin libtrace_backend_" << requested
                      << ".so was not found";
            continue;
        }

        auto backend = handle->createBackend(init_params);
        if (!backend) {
            NIXL_WARN << "nixl::trace: backend '" << requested << "' failed to initialize";
            continue;
        }
        if (!backend->recording()) {
            NIXL_INFO << "nixl::trace: backend '" << requested
                      << "' has nothing to record to and is not activated";
            continue;
        }

        NIXL_DEBUG << "nixl::trace: activated '" << requested << "' backend for agent '"
                   << config.agentName << "'";
        backends.push_back(std::move(backend));
    }

    if (backends.empty()) {
        return nullptr;
    }
    return std::make_unique<CompositeTracer>(std::move(backends), config.sampleRatio);
}

} // namespace nixl::trace
