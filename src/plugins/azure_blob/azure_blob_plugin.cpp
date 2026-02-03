/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Microsoft Corporation.
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

#include "nixl_types.h"
#include "azure_blob_backend.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"

// Plugin type alias for convenience
using azure_blob_plugin_t = nixlBackendPluginCreator<nixlAzureBlobEngine>;

#ifdef STATIC_PLUGIN_AZURE
nixlBackendPlugin *
createStaticAZUREPlugin() {
    return azure_blob_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "AZURE_BLOB", "0.1.0", {}, {DRAM_SEG, OBJ_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return azure_blob_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "AZURE_BLOB", "0.1.0", {}, {DRAM_SEG, OBJ_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
