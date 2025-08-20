/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nixl_types.h"
#include "obj_backend.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"

namespace {

[[nodiscard]] nixlBackendEngine *
create_obj_engine(const nixlBackendInitParams *init_params) {
    try {
        return new nixlObjEngine(init_params);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create obj engine: " << e.what();
        return nullptr;
    }
}

void
destroy_obj_engine(nixlBackendEngine *engine) {
    delete engine;
}

[[nodiscard]] const char *
get_plugin_name() {
    return "OBJ";
}

[[nodiscard]] const char *
get_plugin_version() {
    return "0.1.0";
}

[[nodiscard]] nixl_b_params_t
get_backend_options() {
    return nixl_b_params_t();
}

[[nodiscard]] nixl_mem_list_t
get_backend_mems() {
    return {DRAM_SEG, OBJ_SEG};
}

nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                            create_obj_engine,
                            destroy_obj_engine,
                            get_plugin_name,
                            get_plugin_version,
                            get_backend_options,
                            get_backend_mems};
} // namespace

#ifdef STATIC_PLUGIN_OBJ

nixlBackendPlugin *
createStaticObjPlugin() {
    return &plugin; // Return the static plugin instance
}

#else // !STATIC_PLUGIN_OBJ

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return &plugin;
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}

#endif // !STATIC_PLUGIN_OBJ
