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

#include <exception>
#include <string>

#include "backend/backend_plugin.h"
#include "common/backend.h"
#include "common/nixl_log.h"
#include "gds_batch_engine.h"
#include "gds_mt_engine.h"

namespace {
nixl_b_params_t
getGdsBackendOptions() {
    return {{"mode", "batch"},
            {"batch_pool_size", "16"},
            {"batch_limit", "128"},
            {"max_request_size", "16777216"},
            {"thread_count", std::to_string(defaultGdsMtThreadCount())}};
}

nixlBackendEngine *
createGdsEngine(const nixlBackendInitParams *init_params) {
    try {
        const std::string mode =
            nixl::getBackendParamDefaulted(init_params->customParams, "mode", std::string("batch"));
        if (mode == "batch") {
            return new nixlGdsBatchEngine(init_params);
        }
        if (mode == "mt") {
            return new nixlGdsMtEngine(init_params);
        }

        NIXL_ERROR << "GDS: invalid mode '" << mode << "'; expected 'batch' or 'mt'";
        return nullptr;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create GDS engine: " << e.what();
        return nullptr;
    }
}

void
destroyGdsEngine(nixlBackendEngine *engine) {
    delete engine;
}

const char *
getGdsPluginName() {
    return "GDS";
}

const char *
getGdsPluginVersion() {
    return "0.1.1";
}

nixl_mem_list_t
getGdsBackendMems() {
    return {DRAM_SEG, VRAM_SEG, FILE_SEG};
}

nixlBackendPlugin *
getGdsPlugin() {
    static nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                                       createGdsEngine,
                                       destroyGdsEngine,
                                       getGdsPluginName,
                                       getGdsPluginVersion,
                                       getGdsBackendOptions,
                                       getGdsBackendMems};
    return &plugin;
}
} // namespace

#ifdef STATIC_PLUGIN_GDS
nixlBackendPlugin *
createStaticGDSPlugin() {
    return getGdsPlugin();
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return getGdsPlugin();
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
