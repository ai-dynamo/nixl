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
#include "plugin_manager.h"

namespace nixl::trace {

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

        NIXL_DEBUG << "nixl::trace: activated '" << requested << "' backend for agent '"
                   << config.agentName << "'";
        backends.push_back(std::move(backend));
    }

    if (backends.empty()) {
        return nullptr;
    }
    return std::make_unique<Tracer>(std::move(backends));
}

} // namespace nixl::trace
