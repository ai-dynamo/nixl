/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <exception>
#include <string>

#include "ais_mt_engine.h"
#include "backend/backend_plugin.h"
#include "common/backend.h"
#include "common/nixl_log.h"

namespace {
[[nodiscard]] nixl_b_params_t
getAisBackendOptions() {
    return {{"mode", "mt"}, {"thread_count", std::to_string(defaultAisMtThreadCount())}};
}

[[nodiscard]] nixlBackendEngine *
createAisEngine(const nixlBackendInitParams *init_params) {
    try {
        const std::string mode =
            nixl::getBackendParamDefaulted(init_params->customParams, "mode", std::string("mt"));
        if (mode == "mt") {
            return new nixlAisMtEngine(init_params);
        }

        // Batch support is coming soon; it will be selected with mode=batch.
        NIXL_ERROR << "AIS: invalid mode '" << mode << "'; expected 'mt'";
        return nullptr;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create AIS engine: " << e.what();
        return nullptr;
    }
}

void
destroyAisEngine(nixlBackendEngine *engine) {
    delete engine;
}

[[nodiscard]] const char *
getAisPluginName() {
    return "AIS";
}

[[nodiscard]] const char *
getAisPluginVersion() {
    return "0.1.0";
}

// DRAM_SEG is not advertised until hipFile supports host buffers.
[[nodiscard]] nixl_mem_list_t
getAisBackendMems() {
    return {VRAM_SEG, FILE_SEG};
}

[[nodiscard]] nixlBackendPlugin *
getAisPlugin() {
    static nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                                       createAisEngine,
                                       destroyAisEngine,
                                       getAisPluginName,
                                       getAisPluginVersion,
                                       getAisBackendOptions,
                                       getAisBackendMems};
    return &plugin;
}
} // namespace

#ifdef STATIC_PLUGIN_AIS
[[nodiscard]] nixlBackendPlugin *
createStaticAISPlugin() {
    return getAisPlugin();
}
#else
extern "C" [[nodiscard]] NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return getAisPlugin();
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
