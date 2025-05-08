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

#include <memory>
#include "posix_backend.h"
#include "backend/backend_plugin.h"

// Function to create a new POSIX backend engine instance
static nixlBackendEngine* create_posix_engine(const nixlBackendInitParams* init_params) {
    return new nixlPosixEngine(init_params);
}

static void destroy_posix_engine(nixlBackendEngine *engine) {
    delete engine;
}

// Function to get the plugin name
static const char* get_plugin_name() {
    return "POSIX";
}

// Function to get the plugin version
static const char* get_plugin_version() {
    return "0.1.0";
}

// Function to get backend options
static nixl_b_params_t get_backend_options() {
    nixl_b_params_t params;
    return params;
}

// Function to get supported backend mem types
static nixl_mem_list_t get_backend_mems() {
    return {DRAM_SEG, FILE_SEG};
}

#ifdef STATIC_PLUGIN_POSIX

// Static plugin structure
static nixlBackendPlugin plugin = {
    NIXL_PLUGIN_API_VERSION,
    create_posix_engine,
    destroy_posix_engine,
    get_plugin_name,
    get_plugin_version,
    get_backend_options,
    get_backend_mems
};

nixlBackendPlugin* createStaticPosixPlugin() {
    return &plugin; // Return the static plugin instance
}

#else

// Plugin initialization function
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin* nixl_plugin_init() {
    try {
        std::unique_ptr<nixlBackendPlugin> plugin = std::make_unique<nixlBackendPlugin>();
        plugin->create_engine = create_posix_engine;
        plugin->destroy_engine = destroy_posix_engine;
        plugin->get_plugin_name = get_plugin_name;
        plugin->get_plugin_version = get_plugin_version;
        plugin->get_backend_options = get_backend_options;
        plugin->get_backend_mems = get_backend_mems;
        plugin->api_version = NIXL_PLUGIN_API_VERSION;  // Set the API version
        return plugin.release();
    } catch (const std::exception& e) {
        return nullptr;
    }
}

// Plugin cleanup function
extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {
}

#endif
