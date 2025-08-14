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
#include "gds_backend.h"

namespace {
nixl_b_params_t
get_gds_options() {
    nixl_b_params_t params;
    return params;
}


} // namespace

// Plugin type alias for convenience
using GdsPlugin = nixlBackendPluginTemplate<nixlGdsEngine>;

#ifdef STATIC_PLUGIN_GDS
// Function for static loading
extern "C" nixlBackendPlugin *
createStaticGDSPlugin() {
    return GdsPlugin::initialize_plugin(
        "GDS", "0.1.1", get_gds_options, {DRAM_SEG, VRAM_SEG, FILE_SEG});
}
#else
// Export functions for dynamic loading
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return GdsPlugin::initialize_plugin(
        "GDS", "0.1.1", get_gds_options, {DRAM_SEG, VRAM_SEG, FILE_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
