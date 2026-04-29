/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DataDirect Networks, Inc.
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
#include "infinia_backend.h"

// Function to create a new Infinia backend engine instance
static nixlBackendEngine *
create_infinia_engine(const nixlBackendInitParams *init_params) {
    return new infinia_engine(init_params);
}

static void
destroy_infinia_engine(nixlBackendEngine *engine) {
    delete engine;
}

// Function to get the plugin name
static const char *
get_plugin_name() {
    return INFINIA_PLUGIN_NAME;
}

// Function to get the plugin version
static const char *
get_plugin_version() {
    return INFINIA_PLUGIN_VERSION;
}

// Function to get backend options
// Returns empty params - INFINIA backend uses environment variables
// and doesn't require default parameter values
static nixl_b_params_t
get_backend_options() {
    nixl_b_params_t params;
    // Return empty params - configuration is done via environment variables
    // or explicit parameters passed to createBackend()
    return params;
}

// Function to get supported backend mem types
static nixl_mem_list_t
get_backend_mems() {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    mems.push_back(OBJ_SEG);
    return mems;
}

// Static plugin structure
static nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                                   create_infinia_engine,
                                   destroy_infinia_engine,
                                   get_plugin_name,
                                   get_plugin_version,
                                   get_backend_options,
                                   get_backend_mems};

#ifdef STATIC_PLUGIN_INFINIA

nixlBackendPlugin *
createStaticInfiniaPlugin() {
    return &plugin; // Return the static plugin instance
}

#else

// Plugin initialization function
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return &plugin;
}

// Plugin cleanup function
extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {
    // Cleanup any resources if needed
}

#endif
