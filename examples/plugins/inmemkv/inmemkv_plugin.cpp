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
 * @file inmemkv_plugin.cpp
 * @brief INMEMKV Plugin Registration
 *
 * This file registers the INMEMKV backend as a dynamic NIXL example plugin.
 * It provides the plugin entry point that NIXL uses to discover and load the backend.
 *
 * Plugin Registration:
 * - Dynamic plugin only: built as shared library (libplugin_INMEMKV.so)
 *
 * Usage:
 * - Load libplugin_INMEMKV.so at runtime from the example plugin build directory.
 */

#include "nixl_types.h"
#include "inmemkv_backend.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"

// Plugin type alias for convenience
using inmemkv_plugin_t = nixlBackendPluginCreator<nixlInMemKVEngine>;

/**
 * @brief Plugin initialization function (dynamic plugin)
 *
 * This is called by NIXL when loading the plugin dynamically.
 *
 * @return Pointer to plugin object
 */
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return inmemkv_plugin_t::create(
        NIXL_PLUGIN_API_VERSION,
        "INMEMKV",
        "0.1.0",
        {},
        {DRAM_SEG}
    );
}

/**
 * @brief Plugin cleanup function (dynamic plugin)
 *
 * Called when unloading the plugin. For INMEMKV, no cleanup needed.
 */
extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
