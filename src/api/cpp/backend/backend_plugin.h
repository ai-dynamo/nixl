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

#ifndef __BACKEND_PLUGIN_H
#define __BACKEND_PLUGIN_H

#include "backend/backend_engine.h"

// Define the plugin API version
#define NIXL_PLUGIN_API_VERSION 1

// Define the plugin interface class
class nixlBackendPlugin {
public:
    int api_version;

    // Function pointer for creating a new backend engine instance
    nixlBackendEngine* (*create_engine)(const nixlBackendInitParams* init_params);

    // Function pointer for destroying a backend engine instance
    void (*destroy_engine)(nixlBackendEngine* engine);

    // Function to get the plugin name
    const char* (*get_plugin_name)();

    // Function to get the plugin version
    const char* (*get_plugin_version)();

    // Function to get backend options
    nixl_b_params_t (*get_backend_options)();

    // Function to get supported backend mem types
    nixl_mem_list_t (*get_backend_mems)();
};

// Macro to define exported C functions for the plugin
#define NIXL_PLUGIN_EXPORT __attribute__((visibility("default")))

// Template for creating backend plugins with minimal boilerplate
template<typename EngineType> class nixlBackendPluginTemplate {
private:
    static nixlBackendPlugin plugin_instance;
    static bool initialized;

public:
    [[nodiscard]] static nixlBackendEngine *
    create_engine_impl(const nixlBackendInitParams *init_params) {
        try {
            return new EngineType(init_params);
        }
        catch (const std::exception &e) {
            return nullptr;
        }
    }

    static void
    destroy_engine_impl(nixlBackendEngine *engine) {
        delete engine;
    }

    static nixlBackendPlugin *
    initialize_plugin(const char *name,
                     const char *version,
                     nixl_b_params_t (*get_options)(),
                     nixl_mem_list_t (*get_mems)()) {

        if (!initialized) {
            static const char *plugin_name = name;
            static const char *plugin_version = version;

            plugin_instance = {NIXL_PLUGIN_API_VERSION,
                              create_engine_impl,
                              destroy_engine_impl,
                              []() { return plugin_name; },
                              []() { return plugin_version; },
                              get_options,
                              get_mems};
            initialized = true;
        }
        return &plugin_instance;
    }

    // Get the plugin instance (for use by exported functions)
    static nixlBackendPlugin *get_instance() {
        return &plugin_instance;
    }
};

// Static member definitions
template<typename EngineType>
nixlBackendPlugin nixlBackendPluginTemplate<EngineType>::plugin_instance = {};

template<typename EngineType>
bool nixlBackendPluginTemplate<EngineType>::initialized = false;



// Creator Function type for static plugins
typedef nixlBackendPlugin* (*nixlStaticPluginCreatorFunc)();

// Plugin must implement these functions for dynamic loading
extern "C" {
    // Initialize the plugin
    NIXL_PLUGIN_EXPORT nixlBackendPlugin* nixl_plugin_init();

    // Cleanup the plugin
    NIXL_PLUGIN_EXPORT void nixl_plugin_fini();
}

#endif // __BACKEND_PLUGIN_H
