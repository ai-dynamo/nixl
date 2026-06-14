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

#include <unordered_set>

#include "tracing/trace.h"

#include "common/nixl_log.h"

#ifdef NIXL_TRACE_BACKEND_NVTX
#include "nvtx_backend.h"
#endif

namespace nixl::trace {

/*** Span ***/

void
Span::addAttribute(std::string_view key, std::string_view value) {
    for (auto &backend : backends_) {
        backend->addAttribute(key, value);
    }
}

void
Span::addAttribute(std::string_view key, std::int64_t value) {
    for (auto &backend : backends_) {
        backend->addAttribute(key, value);
    }
}

void
Span::addAttribute(std::string_view key, double value) {
    for (auto &backend : backends_) {
        backend->addAttribute(key, value);
    }
}

void
Span::addCtrlDep(SpanId parent) {
    for (auto &backend : backends_) {
        backend->addCtrlDep(parent);
    }
}

void
Span::addDataDep(SpanId parent) {
    for (auto &backend : backends_) {
        backend->addDataDep(parent);
    }
}

SpanId
Span::id() const noexcept {
    // Return the first non-zero backend id so the SpanId stays meaningful
    // regardless of backend ordering (NVTX has no id and returns {0}).
    for (const auto &backend : backends_) {
        const SpanId backend_id = backend->id();
        if (backend_id.value != 0) {
            return backend_id;
        }
    }
    return SpanId{};
}

/*** Tracer ***/

Tracer::Tracer(std::vector<std::unique_ptr<TraceBackend>> backends) noexcept
    : backends_(std::move(backends)) {}

Tracer::~Tracer() = default;

Span
Tracer::beginSpan(std::string_view name, Kind kind, Color color) {
    std::vector<std::unique_ptr<SpanBackend>> spans;
    spans.reserve(backends_.size());
    for (auto &backend : backends_) {
        // Drop null backend spans so an all-null Span is not reported active()
        // (which would null-deref in addAttribute()/id()).
        if (auto span = backend->beginSpan(name, kind, color)) {
            spans.push_back(std::move(span));
        }
    }
    return Span{std::move(spans)};
}

void
Tracer::mark(std::string_view name, Kind kind) {
    for (auto &backend : backends_) {
        backend->mark(name, kind);
    }
}

void
Tracer::pushCorrelationId(std::uint64_t id) {
    for (auto &backend : backends_) {
        backend->pushCorrelationId(id);
    }
}

void
Tracer::popCorrelationId() {
    for (auto &backend : backends_) {
        backend->popCorrelationId();
    }
}

/*** Factory ***/

std::unique_ptr<Tracer>
makeTracer(const TracerConfig &config) {
    std::vector<std::unique_ptr<TraceBackend>> backends;
    std::unordered_set<std::string> seen;

    for (const auto &requested : config.backends) {
        // Skip blanks and duplicates so e.g. "nvtx,nvtx" activates one backend.
        if (requested.empty() || !seen.insert(requested).second) {
            continue;
        }
        if (requested == "nvtx") {
#ifdef NIXL_TRACE_BACKEND_NVTX
            backends.push_back(makeNvtxBackend(config.agentName));
            NIXL_DEBUG << "nixl::trace: activated NVTX backend for agent '" << config.agentName
                       << "'";
#else
            NIXL_WARN << "nixl::trace: backend 'nvtx' requested but not compiled in";
#endif
        } else {
            NIXL_WARN << "nixl::trace: unknown backend '" << requested << "' requested";
        }
    }

    if (backends.empty()) {
        return nullptr;
    }
    return std::make_unique<Tracer>(std::move(backends));
}

} // namespace nixl::trace
