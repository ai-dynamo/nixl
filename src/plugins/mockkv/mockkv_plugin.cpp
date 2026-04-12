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

/**
 * @file mockkv_plugin.cpp
 * @brief MOCKKV Plugin Registration
 * 
 * This file registers the MOCKKV backend as a NIXL plugin.
 * It provides the plugin entry point that NIXL uses to discover and load the backend.
 * 
 * Plugin Registration:
 * - Static plugin: Built into libnixl.so (when STATIC_PLUGIN_MOCKKV is defined)
 * - Dynamic plugin: Built as shared library (libplugin_MOCKKV.so)
 * 
 * Usage:
 * - Static: Compile with -Dstatic_plugins=MOCKKV
 * - Dynamic: Load libplugin_MOCKKV.so at runtime
 */

#include "nixl_types.h"
#include "mockkv_backend.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"

// Plugin type alias for convenience
using mockkv_plugin_t = nixlBackendPluginCreator<nixlMockKVEngine>;

#ifdef STATIC_PLUGIN_MOCKKV
/**
 * @brief Create static MOCKKV plugin
 * 
 * This function is called during static plugin registration.
 * It's linked into libnixl.so when MOCKKV is built as a static plugin.
 * 
 * @return Pointer to plugin object
 */
nixlBackendPlugin *
createStaticMOCKKVPlugin() {
    return mockkv_plugin_t::create(
        NIXL_PLUGIN_API_VERSION,  // Plugin API version
        "MOCKKV",                  // Plugin name
        "0.1.0",                   // Plugin version
        {},                        // Required parameters (none)
        {DRAM_SEG}                // Supported memory types
    );
}
#else
/**
 * @brief Plugin initialization function (dynamic plugin)
 * 
 * This is called by NIXL when loading the plugin dynamically.
 * 
 * @return Pointer to plugin object
 */
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return mockkv_plugin_t::create(
        NIXL_PLUGIN_API_VERSION,
        "MOCKKV",
        "0.1.0",
        {},
        {DRAM_SEG}
    );
}

/**
 * @brief Plugin cleanup function (dynamic plugin)
 * 
 * Called when unloading the plugin. For MOCKKV, no cleanup needed.
 */
extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
