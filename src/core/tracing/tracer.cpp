/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    return backends_.empty() ? SpanId{} : backends_.front()->id();
}

/*** Tracer ***/

Tracer::Tracer(std::vector<std::unique_ptr<iTraceBackend>> backends) noexcept
    : backends_(std::move(backends)) {}

Tracer::~Tracer() = default;

Span
Tracer::beginSpan(std::string_view name, Kind kind, Color color) {
    std::vector<std::unique_ptr<iSpanBackend>> spans;
    spans.reserve(backends_.size());
    for (auto &backend : backends_) {
        spans.push_back(backend->beginSpan(name, kind, color));
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
    std::vector<std::unique_ptr<iTraceBackend>> backends;

    for (const auto &requested : config.backends) {
        if (requested.empty()) {
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
