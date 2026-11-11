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
#include "tracing/api/agent_trace.h"

#include <cstdint>
#include <functional>

#include "backend/backend_engine.h"
#include "common/nixl_log.h"
#include "telemetry.h"

namespace nixl::trace {

BackendTraceSinkPtr
BackendTraces::makeSink(Tracer *tracer, const nixl_backend_t &type, nixlTelemetry *telemetry) {
    if (tracer == nullptr) {
        return nullptr;
    }
    std::function<void(std::uint64_t)> on_dropped;
    if (telemetry != nullptr) {
        on_dropped = [telemetry](std::uint64_t dropped) {
            telemetry->updateTracePhasesDropped(dropped);
        };
    }
    return tracer->makeSink(type, std::move(on_dropped));
}

void
BackendTraces::adopt(const nixl_backend_t &type,
                     BackendTraceSinkPtr sink,
                     const nixlBackendEngine &backend) {
    if (sink == nullptr) {
        return;
    }
    if (backend.supportsRemote() && !backend.supportsTraceContext()) {
        contextUnsupported_.try_emplace(type);
    }
    sinks_.insert_or_assign(type, std::move(sink));
}

void
BackendTraces::warnIfContextUnsupported(const nixlBackendEngine &backend,
                                        const Traceable &traceable) noexcept {
    if (!traceable.requestIdentity().context.sampled()) {
        return;
    }
    const auto it = contextUnsupported_.find(backend.getType());
    if (it == contextUnsupported_.end()) {
        return;
    }
    logOnce(it->second, [&backend] {
        NIXL_WARN << "nixl::trace: backend '" << backend.getType()
                  << "' cannot carry trace contexts, so sampled transfers through it reach "
                     "remote agents untraced; logged once per backend and agent";
    });
}

nixl_status_t
traceInfo(const Traceable &traceable, nixl_xfer_trace_info_t &info) {
    const TraceIdentity identity = traceable.postIdentity();
    if ((identity.requestId == 0) || (traceable.tracer() == nullptr)) {
        return NIXL_ERR_NOT_FOUND;
    }
    info.requestId = identity.requestId;
    info.execution = identity.execution;
    info.traceparent = traceable.tracer()->traceparent(identity.context);
    return NIXL_SUCCESS;
}

} // namespace nixl::trace
