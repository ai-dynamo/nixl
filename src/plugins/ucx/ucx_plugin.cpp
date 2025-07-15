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
#include "ucx_backend.h"

#include "nixl_log.h"

namespace
{
   const char* ucx_plugin_name = "UCX";
   const char* ucx_plugin_version = "0.1.0";

   [[nodiscard]] nixlBackendEngine* create_ucx_engine(const nixlBackendInitParams* init_params) {
        try {
            return new nixlUcxEngine(init_params);
        } catch (const std::exception &e) {
            NIXL_ERROR << "Failed to create UCX engine: " << e.what();
            return nullptr;
        }
   }

   void destroy_ucx_engine(nixlBackendEngine *engine) {
       delete engine;
   }

   [[nodiscard]] const char* get_plugin_name() {
       return ucx_plugin_name;
   }

   [[nodiscard]] const char* get_plugin_version() {
       return ucx_plugin_version;
   }

   [[nodiscard]] nixl_b_params_t get_backend_options() {
       return get_ucx_backend_common_options();
   }

   [[nodiscard]] nixl_mem_list_t get_backend_mems() {
       return {
	 DRAM_SEG,
	 VRAM_SEG
       };
   }

   // Static plugin structure
   nixlBackendPlugin plugin = {
       NIXL_PLUGIN_API_VERSION,
       create_ucx_engine,
       destroy_ucx_engine,
       get_plugin_name,
       get_plugin_version,
       get_backend_options,
       get_backend_mems
   };

}  // namespace

#ifdef STATIC_PLUGIN_UCX

nixlBackendPlugin* createStaticUcxPlugin() {
    return &plugin;
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
