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

// Plugin version information
static const char* PLUGIN_NAME = "GDS";
static const char* PLUGIN_VERSION = "0.1.0";

// Function to create a new GDS backend engine instance
static nixlBackendEngine* create_gds_engine(const nixlBackendInitParams* init_params) {
    return new nixlGdsEngine(init_params);
}

static void destroy_gds_engine(nixlBackendEngine* engine) {
    delete engine;
}

// Function to get the plugin name
static const char* get_plugin_name() {
    return PLUGIN_NAME;
}

// Function to get the plugin version
static const char* get_plugin_version() {
    return PLUGIN_VERSION;
}

// Function to get backend options
static nixl_b_params_t get_backend_options() {
    nixl_b_params_t params;
    return params;
}

// Function to get supported backend mem types
static nixl_mem_list_t get_backend_mems() {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    mems.push_back(FILE_SEG);
    return mems;
}

// Static plugin structure
static nixlBackendPlugin plugin = {
    NIXL_PLUGIN_API_VERSION,
    create_gds_engine,
    destroy_gds_engine,
    get_plugin_name,
    get_plugin_version,
    get_backend_options,
    get_backend_mems
};

#ifdef STATIC_PLUGIN_GDS

nixlBackendPlugin* createStaticGdsPlugin() {
    return &plugin; // Return the static plugin instance
}

#else

// Plugin initialization function
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin* nixl_plugin_init() {
    return &plugin;
}

// Plugin cleanup function
extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {
    // Cleanup any resources if needed
}

#endif
