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

#include "backend/backend_plugin.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include "wqskv_backend.h"

using wqskv_plugin_t = nixlBackendPluginCreator<nixlWQSKVEngine>;

#ifdef STATIC_PLUGIN_WQSKV
nixlBackendPlugin *
createStaticWQSKVPlugin() {
    return wqskv_plugin_t::create(NIXL_PLUGIN_API_VERSION, "WQSKV", "0.1.0", {}, {DRAM_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return wqskv_plugin_t::create(NIXL_PLUGIN_API_VERSION, "WQSKV", "0.1.0", {}, {DRAM_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
