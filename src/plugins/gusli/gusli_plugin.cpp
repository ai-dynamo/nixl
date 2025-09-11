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

#include "backend/backend_plugin.h"
#include "gusli_backend.h"

namespace {
[[nodiscard]] nixlBackendEngine *
create_gusli_engine(const nixlBackendInitParams *init_params) {
    return new nixlGusliEngine(init_params);
}
void
destroy_gusli_engine(nixlBackendEngine *engine) {
    delete engine;
}
[[nodiscard]] const char *
get_plugin_name() {
    return "GUSLI";
} // PLUGIN_NAME
[[nodiscard]] const char *
get_plugin_version() {
    return "0.1.0";
} // PLUGIN_VERSION
[[nodiscard]] nixl_b_params_t
get_backend_options() {
    nixl_b_params_t params;
    params["client_name"] = "Debug unique client name (optional)";
    params["max_num_simultaneous_requests"] = "Integer, typically ~256 (optional)";
    params["config_file"] = "string of block devices or path to config file (mandatory)";
    return params;
}

[[nodiscard]] nixl_mem_list_t
get_backend_mems() {
    return {BLK_SEG, DRAM_SEG};
}

nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                            create_gusli_engine,
                            destroy_gusli_engine,
                            get_plugin_name,
                            get_plugin_version,
                            get_backend_options,
                            get_backend_mems};
} // namespace
#ifdef STATIC_PLUGIN_GUSLI
nixlBackendPlugin *
createStaticGusliPlugin() {
    return &plugin;
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return &plugin;
}
extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
