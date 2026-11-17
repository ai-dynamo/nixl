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

#include <atomic>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <absl/strings/numbers.h>

#include "tracing/api/tracing.h"

#include "common/configuration.h"
#include "common/nixl_log.h"
#include "common/str_util.h"
#include "tracing/api/trace_core_plugin.h"
#include "tracing/api/trace_plugin.h"
#include "tracing/core/trace_config.h"
#include "tracing/core/trace_plugin_loader.h"

namespace {

constexpr double min_sample_ratio = 0x1p-56;

thread_local nixl::trace::detail::Call current_call;

[[nodiscard]] std::optional<double>
parseSampleRatio(const std::string &value) {
    double ratio = 0.0;
    if (!absl::SimpleAtod(value, &ratio) || !std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        return std::nullopt;
    }
    if (ratio > 0.0 && ratio < min_sample_ratio) {
        return std::nullopt;
    }
    return ratio;
}

} // namespace

namespace nixl::trace {

// True when the process runs under Nsight Systems: nsys injects
// NVTX_INJECTION64_PATH into the profiled process's environment (its presence
// means "running under nsys", not merely that nsys is installed).
[[nodiscard]] bool
runningUnderNsys() {
    return nixl::config::checkExistence("NVTX_INJECTION64_PATH");
}

// Backend-selection policy for the agent-wiring layer (kept out of the
// backend-agnostic tracing core so nsys/NVTX specifics never reach it).
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

double
resolveTraceSampleRatio() {
    const auto spec = nixl::config::getValueOptional<std::string>(std::string(traceSampleRatioVar));
    if (spec) {
        if (spec->empty()) {
            return 0.0;
        }

        const auto ratio = parseSampleRatio(*spec);
        if (!ratio) {
            throw std::invalid_argument(std::string(traceSampleRatioVar) + "='" + *spec +
                                        "' must be 0 or a number in [2^-56, 1]");
        }
        return *ratio;
    }

    const auto otel_spec =
        nixl::config::getValueOptional<std::string>(std::string(otelTracesSampleRatioVar));
    if (!otel_spec || otel_spec->empty()) {
        return defaultSampleRatio;
    }

    const auto ratio = parseSampleRatio(*otel_spec);
    if (!ratio) {
        NIXL_WARN << "nixl::trace: ignoring " << otelTracesSampleRatioVar << "='" << *otel_spec
                  << "': not 0 or a number in [2^-56, 1]; sampling every request";
        return defaultSampleRatio;
    }
    return *ratio;
}

std::vector<std::unique_ptr<TraceBackend>>
loadTraceBackends(const std::vector<std::string> &names, std::string_view agent_name) {
    std::vector<std::unique_ptr<TraceBackend>> backends;
    std::unordered_set<std::string> seen;

    const nixlTraceBackendInitParams init_params{agent_name};

    for (const auto &requested : names) {
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
        auto handle = loadTracePlugin(requested);
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
                   << agent_name << "'";
        backends.push_back(std::move(backend));
    }
    return backends;
}

// Build the agent's tracer through the tracing core plugin. Backend selection
// follows nixl::trace::resolveTraceBackends (explicit NIXL_TRACE_BACKENDS wins;
// unset + running under nsys auto-enables NVTX). Returns null when nothing
// resolves or the core plugin is missing, so the agent can hold the result in a
// const member and call sites take the no-op branch.
TracerPtr
makeAgentTracer(const std::string &name) {
    const auto trace_env = nixl::config::getValueOptional<std::string>("NIXL_TRACE_BACKENDS");
    const auto requested_backends = resolveTraceBackends(trace_env, runningUnderNsys());
    const auto sample_ratio = resolveTraceSampleRatio();
    if (requested_backends.empty()) {
        return nullptr;
    }
    const auto core = loadTraceCorePlugin();
    if (core == nullptr) {
        NIXL_WARN << "nixl::trace: tracing is requested, but the tracing core plugin "
                     "libtrace_core.so was not found or is stale; tracing is off";
        return nullptr;
    }
    auto backends = loadTraceBackends(requested_backends, name);
    if (backends.empty()) {
        return nullptr;
    }
    return core->createTracer({name, std::move(backends), sample_ratio});
}

std::atomic<bool> detail::callScopeOpened{false};

detail::Call
detail::currentCallOnThread() noexcept {
    return current_call;
}

CallScope::CallScope(Tracer *tracer, const nixl_opt_args_t *options) noexcept
    : active_(tracer != nullptr),
      previous_(active_ ? std::exchange(current_call, {tracer, options}) : detail::Call{}) {
    if (active_) {
        detail::callScopeOpened.store(true, std::memory_order_relaxed);
    }
}

CallScope::~CallScope() {
    if (active_) {
        current_call = previous_;
    }
}

void
Traceable::emitRequestSpan() const noexcept {
    try {
        const Attribute attrs[] = {{AttributeKey::NixlXferRequestId, postIdentity_.requestId},
                                   {AttributeKey::NixlXferExecution, postIdentity_.execution}};
        tracer_->emitContextSpan(Record::XferReq,
                                 kindOf(Record::XferReq),
                                 requestIdentity(),
                                 createdAt_,
                                 nixlTime::getUs(),
                                 attrs);
    }
    catch (...) {
    }
}

void
Traceable::advancePost(const nixl_blob_t &parent, const nixl_blob_t &baggage) {
    const std::optional<TraceContext> supplied =
        parent.empty() ? std::nullopt : parseParentContext(tracer_, parent);
    const TraceContext &under = supplied ? *supplied : requestContext_;
    postIdentity_.context = under.sampled() ? generatePostContext(under, tracer_) : under;
    ++postIdentity_.execution;
    if (postIdentity_.context.sampled()) {
        const std::string_view own = parseBaggage(tracer_, baggage);
        postBaggage_.assign(own.empty() ? std::string_view{requestBaggage_} : own);
    } else {
        postBaggage_.clear();
    }
}

void
Traceable::emitEvent(Record record) const {
    const Attribute attrs[] = {{AttributeKey::NixlXferRequestId, postIdentity_.requestId},
                               {AttributeKey::NixlXferExecution, postIdentity_.execution}};
    tracer_->emitEvent(record, kindOf(record), postIdentity_, attrs);
}

void
Traceable::emitEvent(Record record, const Attribute &attr) const {
    const Attribute attrs[] = {{AttributeKey::NixlXferRequestId, postIdentity_.requestId},
                               {AttributeKey::NixlXferExecution, postIdentity_.execution},
                               attr};
    tracer_->emitEvent(record, kindOf(record), postIdentity_, attrs);
}

void
Traceable::emitEvent(Record record, std::span<const Attribute> attrs) const {
    std::vector<Attribute> event_attrs{{AttributeKey::NixlXferRequestId, postIdentity_.requestId},
                                       {AttributeKey::NixlXferExecution, postIdentity_.execution}};
    event_attrs.insert(event_attrs.end(), attrs.begin(), attrs.end());
    tracer_->emitEvent(record, kindOf(record), postIdentity_, event_attrs);
}

TraceIdentity
detail::suppliedIdentity(const Tracer *tracer, const nixl_blob_t &parent) {
    const std::optional<TraceContext> supplied = parseParentContext(tracer, parent);
    if (!supplied) {
        return {};
    }
    return {supplied->sampled() ? generatePostContext(*supplied, tracer) : *supplied, 0, 0};
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

Scope::Scope(Tracer *tracer, const TraceIdentity &identity, std::string_view baggage)
    : tracer_(tracer),
      identity_(identity),
      previous_((tracer != nullptr) ? tracer->swapCurrentIdentity(&identity_) : nullptr),
      previousBaggage_((tracer != nullptr) ? tracer->swapCurrentBaggage(baggage) :
                                             std::string_view{}),
      pushed_((tracer != nullptr) && identity.context.valid() && identity.context.sampled()) {
    if (pushed_) {
        tracer_->pushContext(identity_);
    }
}

Scope::~Scope() {
    if (tracer_ != nullptr) {
        tracer_->swapCurrentIdentity(previous_);
        tracer_->swapCurrentBaggage(previousBaggage_);
    }
    if (pushed_) {
        tracer_->popContext();
    }
}

} // namespace nixl::trace
