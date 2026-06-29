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

#include "backend/backend_plugin.h"
#include "common/nixl_log.h"
#include "gds_mt_engine.h"

// "GDS_MT" backend: multi-threaded TaskFlow engine sharing the same base as
// "GDS". Only this entry point (and its engine) pulls in TaskFlow. The plugin
// struct is hand-written so that, in static builds, GDS and GDS_MT remain
// distinct plugin instances (see gds_plugin.cpp).
namespace {

nixlBackendEngine *
createGdsMtEngine(const nixlBackendInitParams *init_params) {
    try {
        return new nixlGdsMtEngine(init_params);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create GDS_MT engine: " << e.what();
        return nullptr;
    }
}

void
destroyGdsMtEngine(nixlBackendEngine *engine) {
    delete engine;
}

nixlBackendPlugin gds_mt_plugin = {NIXL_PLUGIN_API_VERSION,
                                   createGdsMtEngine,
                                   destroyGdsMtEngine,
                                   []() { return "GDS_MT"; },
                                   []() { return "0.1.0"; },
                                   []() { return nixl_b_params_t{}; },
                                   []() {
                                       return nixl_mem_list_t{DRAM_SEG, VRAM_SEG, FILE_SEG};
                                   }};

} // namespace

#ifdef STATIC_PLUGIN_GDS_MT
nixlBackendPlugin *
createStaticGDS_MTPlugin() {
    return &gds_mt_plugin;
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return &gds_mt_plugin;
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
