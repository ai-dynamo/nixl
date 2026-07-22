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
 * @brief Dynamic plugin registration for the INMEMKV example backend.
 *
 * Exports nixl_plugin_init() / nixl_plugin_fini() so NIXL can load
 * libplugin_INMEMKV.so from NIXL_PLUGIN_DIR at runtime.
 */

#include "backend/backend_plugin.h"
#include "common/nixl_log.h"
#include "inmemkv_backend.h"
#include "nixl_types.h"

using inmemkv_plugin_t = nixlBackendPluginCreator<nixlInMemKVEngine>;

/**
 * @brief Plugin initialization entry point for dynamic loading.
 * @return Pointer to the registered backend plugin object.
 */
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return inmemkv_plugin_t::create(NIXL_PLUGIN_API_VERSION, "INMEMKV", "0.1.0", {}, {DRAM_SEG});
}

/**
 * @brief Plugin cleanup entry point for dynamic unloading.
 */
extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
