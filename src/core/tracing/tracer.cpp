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

#include "tracing/trace.h"

#include "common/nixl_log.h"
#include "plugin_manager.h"
#include "tracing/trace_plugin.h"

namespace {

void
emitSpanTo(std::span<const std::unique_ptr<nixl::trace::TraceBackend>> backends,
           std::string_view name,
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
            std::string_view name,
            nixl::trace::Kind kind,
            const nixl::trace::TraceContext &context,
            std::uint64_t request_id,
            nixlTime::us_t at,
            std::span<const nixl::trace::Attribute> attrs) {
    for (const auto &backend : backends) {
        backend->emitEvent(name, kind, context, request_id, at, attrs);
    }
}

} // namespace

namespace nixl::trace {

/*** Span ***/

template<typename Call>
void
Span::forEachBackend(const Call &call) const {
    for (const auto &backend : backends_) {
        if (backend == nullptr) {
            return;
        }
        call(*backend);
    }
}

void
Span::addAttribute(AttributeKey key, std::string_view value) {
    forEachBackend([&](SpanBackend &backend) { backend.addAttribute(key, value); });
}

void
Span::addAttribute(AttributeKey key, std::int64_t value) {
    forEachBackend([&](SpanBackend &backend) { backend.addAttribute(key, value); });
}

void
Span::addAttribute(AttributeKey key, double value) {
    forEachBackend([&](SpanBackend &backend) { backend.addAttribute(key, value); });
}

void
Span::addAttributes(std::span<const Attribute> attrs) {
    forEachBackend([&](SpanBackend &backend) { backend.addAttributes(attrs); });
}

void
Span::addCtrlDep(SpanId parent) {
    forEachBackend([&](SpanBackend &backend) { backend.addCtrlDep(parent); });
}

void
Span::addDataDep(SpanId parent) {
    forEachBackend([&](SpanBackend &backend) { backend.addDataDep(parent); });
}

SpanId
Span::id() const noexcept {
    // Return the first non-zero backend id so the SpanId stays meaningful
    // regardless of backend ordering (NVTX has no id and returns {0}).
    for (const auto &backend : backends_) {
        if (backend == nullptr) {
            break;
        }
        const SpanId backend_id = backend->id();
        if (backend_id.value != 0) {
            return backend_id;
        }
    }
    return SpanId{};
}

/*** Tracer ***/

Tracer::Tracer(std::vector<std::unique_ptr<TraceBackend>> backends,
               double sample_ratio,
               random_bytes_fn_t random_bytes) noexcept
    : backends_(std::move(backends)),
      sampleRatio_(sample_ratio),
      randomBytes_(random_bytes),
      needsTimestamps_(std::ranges::any_of(backends_, [](const auto &backend) {
          return !backend->ownsTimestamps();
      })) {}

Tracer::~Tracer() = default;

void
Tracer::emitSpan(std::string_view name,
                 Kind kind,
                 const TraceContext &context,
                 std::uint64_t request_id,
                 nixlTime::us_t start,
                 nixlTime::us_t end,
                 std::span<const Attribute> attrs) {
    emitSpanTo(backends_, name, kind, context, request_id, start, end, attrs);
}

void
Tracer::emitEvent(std::string_view name,
                  Kind kind,
                  const TraceContext &context,
                  std::uint64_t request_id,
                  nixlTime::us_t at,
                  std::span<const Attribute> attrs) {
    emitEventTo(backends_, name, kind, context, request_id, at, attrs);
}

/*** Call sites ***/

Span
detail::beginSpan(Tracer &tracer, std::string_view name, Kind kind) {
    Span span;
    std::size_t count = 0;
    for (const auto &backend : tracer.backends()) {
        // Drop null backend spans so an all-null Span is not reported active()
        // (which would null-deref in addAttribute()/id()).
        if (count == span.backends_.size()) {
            break;
        }
        if (auto backend_span = backend->beginSpan(name, kind)) {
            span.backends_[count++] = std::move(backend_span);
        }
    }
    return span;
}

Span
detail::beginContextSpan(Tracer &tracer,
                         std::string_view name,
                         Kind kind,
                         const TraceContext &context,
                         std::uint64_t request_id) {
    Span span;
    std::size_t count = 0;
    for (const auto &backend : tracer.backends()) {
        if (count == span.backends_.size()) {
            break;
        }
        if (auto backend_span = backend->beginContextSpan(name, kind, context, request_id)) {
            span.backends_[count++] = std::move(backend_span);
        }
    }
    return span;
}

void
detail::mark(Tracer &tracer, std::string_view name, Kind kind) {
    for (const auto &backend : tracer.backends()) {
        backend->mark(name, kind);
    }
}

void
detail::pushContext(Tracer &tracer, const TraceContext &context, std::uint64_t request_id) {
    for (const auto &backend : tracer.backends()) {
        backend->pushContext(context, request_id);
    }
}

void
detail::popContext(Tracer &tracer) {
    for (const auto &backend : tracer.backends()) {
        backend->popContext();
    }
}

void
detail::emitEvent(Tracer &tracer,
                  std::string_view name,
                  Kind kind,
                  const TraceContext &context,
                  std::uint64_t request_id,
                  std::span<const Attribute> attrs) {
    emitEventTo(tracer.backends(),
                name,
                kind,
                context,
                request_id,
                tracer.needsTimestamps() ? nixlTime::getUs() : nixlTime::us_t{0},
                attrs);
}

void
detail::emitSpan(Tracer &tracer, std::string_view name, Kind kind, nixlTime::us_t start) {
    emitSpanTo(tracer.backends(), name, kind, TraceContext{}, 0, start, nixlTime::getUs(), {});
}

/*** Factory ***/

std::unique_ptr<Tracer>
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
    return std::make_unique<Tracer>(std::move(backends), config.sampleRatio);
}

} // namespace nixl::trace
