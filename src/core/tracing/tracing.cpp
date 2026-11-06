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

#include "tracing/tracing.h"

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/configuration.h"
#include "common/str_util.h"
#include "tracing/trace.h"
#include "tracing/trace_sink.h"

namespace nixl::trace {

// True when the process runs under Nsight Systems: nsys injects
// NVTX_INJECTION64_PATH into the profiled process's environment (its presence
// means "running under nsys", not merely that nsys is installed).
[[nodiscard]] bool
runningUnderNsys() {
    return nixl::config::checkExistence("NVTX_INJECTION64_PATH");
}

// Backend-selection policy for the agent-wiring layer (kept out of the
// backend-agnostic facade so nsys/NVTX specifics never reach makeTracer()).
// Running under nsys auto-enables NVTX *in addition to* any explicitly requested
// backends; a set-but-empty NIXL_TRACE_BACKENDS is a hard "off" that beats it.
[[nodiscard]] std::vector<std::string>
resolveTraceBackends(const std::optional<std::string> &explicit_spec, bool under_nsys) {
    std::set<std::string> backends;
    bool explicit_off = false;
    if (explicit_spec) {
        // Trim entries so a padded value like "chakra, nvtx" matches backend
        // names; the set dedups them.
        backends = nixl::str::splitStrippedSet(*explicit_spec);
        // Set-but-empty (or all-blank) is an explicit "off" that must beat the
        // nsys auto-enable below.
        explicit_off = backends.empty();
    }

    if (under_nsys && !explicit_off) {
        backends.emplace("nvtx");
    }
    return {backends.begin(), backends.end()};
}

void
TracerDeleter::operator()(Tracer *tracer) const noexcept {
    delete tracer;
}

void
BackendTraceSinkDeleter::operator()(nixlBackendTraceSink *sink) const noexcept {
    delete sink;
}

// Build the agent's composite tracer. Backend selection follows
// nixl::trace::resolveTraceBackends (explicit NIXL_TRACE_BACKENDS wins; unset +
// running under nsys auto-enables NVTX). Returns null when nothing resolves, so
// the agent can hold the result in a const member and call sites take the no-op
// branch.
TracerPtr
makeAgentTracer(const std::string &name) {
    const auto trace_env = nixl::config::getValueOptional<std::string>("NIXL_TRACE_BACKENDS");
    auto requested_backends = resolveTraceBackends(trace_env, runningUnderNsys());
    const auto sample_ratio = resolveTraceSampleRatio();
    if (requested_backends.empty()) {
        return nullptr;
    }
    return TracerPtr{
        makeTracer(TracerConfig{name, std::move(requested_backends), sample_ratio}).release()};
}

BackendTraceSinkPtr
makeBackendTraceSink(Tracer &tracer, std::string backend, nixlTelemetry *telemetry) {
    return BackendTraceSinkPtr{new TracerPhaseSink(tracer, std::move(backend), telemetry)};
}

} // namespace nixl::trace
