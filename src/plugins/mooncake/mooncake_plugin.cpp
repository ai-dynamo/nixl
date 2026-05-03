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
#include "mooncake_backend.h"

namespace {
nixl_b_params_t
getMooncakeOptions() {
    nixl_b_params_t params;
    params["mooncake_devices"] = "";
    return params;
}

nixl_backend_option_list_t
buildMooncakeOptionSpecs() {
    return {{"mooncake_devices", "Mooncake device list", nixl_backend_option_type_t::STRING, false, ""}};
}

nixlBackendPluginCapabilities
buildMooncakeCapabilities() {
    return {false, true, false};
}
} // namespace

// Plugin type alias for convenience
using mooncake_plugin_t = nixlBackendPluginCreator<nixlMooncakeEngine>;

#ifdef STATIC_PLUGIN_MOONCAKE
nixlBackendPlugin *
createStaticMOONCAKEPlugin() {
    return mooncake_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "MOONCAKE",
                                     "0.1.0",
                                     getMooncakeOptions(),
                                     {DRAM_SEG, VRAM_SEG},
                                     buildMooncakeOptionSpecs(),
                                     buildMooncakeCapabilities());
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return mooncake_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "MOONCAKE",
                                     "0.1.0",
                                     getMooncakeOptions(),
                                     {DRAM_SEG, VRAM_SEG},
                                     buildMooncakeOptionSpecs(),
                                     buildMooncakeCapabilities());
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
