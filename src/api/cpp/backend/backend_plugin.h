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

    [[nodiscard]] static nixlBackendPlugin
    create_plugin(const char *name,
                  const char *version,
                  nixl_b_params_t (*get_options)(),
                  nixl_mem_list_t (*get_mems)()) {

        static const char *plugin_name = name;
        static const char *plugin_version = version;

        return {NIXL_PLUGIN_API_VERSION,
                create_engine_impl,
                destroy_engine_impl,
                []() { return plugin_name; },
                []() { return plugin_version; },
                get_options,
                get_mems};
    }
};

#define NIXL_DEFINE_PLUGIN(PluginName, EngineType, PluginVersion, GetOptionsFunc, GetMemsFunc) \
    namespace {                                                                                \
        /* Core implementation functions */                                                    \
        nixlBackendEngine *                                                                    \
        create_engine_impl(const nixlBackendInitParams *init_params) {                         \
            try {                                                                              \
                return new EngineType(init_params);                                            \
            }                                                                                  \
            catch (const std::exception &e) {                                                  \
                return nullptr;                                                                \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        void                                                                                   \
        destroy_engine_impl(nixlBackendEngine *engine) {                                       \
            delete engine;                                                                     \
        }                                                                                      \
                                                                                               \
        /* [[nodiscard]] wrapper functions */                                                  \
        [[nodiscard]] nixlBackendEngine *                                                      \
        create_engine_wrapper(const nixlBackendInitParams *init_params) {                      \
            return create_engine_impl(init_params);                                            \
        }                                                                                      \
                                                                                               \
        [[nodiscard]] const char *                                                             \
        get_name_wrapper() {                                                                   \
            return #PluginName;                                                                \
        }                                                                                      \
                                                                                               \
        [[nodiscard]] const char *                                                             \
        get_version_wrapper() {                                                                \
            return PluginVersion;                                                              \
        }                                                                                      \
                                                                                               \
        [[nodiscard]] nixl_b_params_t                                                          \
        get_options_wrapper() {                                                                \
            return GetOptionsFunc();                                                          \
        }                                                                                      \
                                                                                               \
        [[nodiscard]] nixl_mem_list_t                                                          \
        get_mems_wrapper() {                                                                   \
            return GetMemsFunc();                                                             \
        }                                                                                      \
                                                                                               \
        /* Plugin structure using [[nodiscard]] wrapper functions */                           \
        nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,                                   \
                                    create_engine_wrapper,                                     \
                                    destroy_engine_impl,                                       \
                                    get_name_wrapper,                                          \
                                    get_version_wrapper,                                       \
                                    get_options_wrapper,                                       \
                                    get_mems_wrapper};                                         \
    }                                                                                          \
                                                                                               \
    extern "C" nixlBackendPlugin *createStatic##PluginName##Plugin() {                         \
        return &plugin;                                                                        \
    }                                                                                          \
                                                                                               \
    extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init() {                      \
        return &plugin;                                                                        \
    }                                                                                          \
                                                                                               \
    extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {}

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
