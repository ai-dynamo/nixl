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

namespace {

nixl_backend_option_list_t
buildAzureBlobOptionSpecs() {
    return {{"account_url", "Azure Blob account URL", nixl_backend_option_type_t::STRING, false, ""},
            {"container_name",
             "Azure Blob container",
             nixl_backend_option_type_t::STRING,
             false,
             ""},
            {"connection_string",
             "Azure Blob connection string",
             nixl_backend_option_type_t::STRING,
             false,
             ""},
            {"ca_bundle", "Azure Blob CA bundle", nixl_backend_option_type_t::STRING, false, ""},
            {"num_threads",
             "Azure Blob worker thread count",
             nixl_backend_option_type_t::INT,
             false,
             "0"}};
}

nixlBackendPluginCapabilities
buildAzureBlobCapabilities() {
    return {true, false, false};
}

} // namespace

#ifdef STATIC_PLUGIN_AZURE
nixlBackendPlugin *
createStaticAZUREPlugin() {
    return azure_blob_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                       "AZURE_BLOB",
                                       "0.1.0",
                                       {},
                                       {DRAM_SEG, OBJ_SEG},
                                       buildAzureBlobOptionSpecs(),
                                       buildAzureBlobCapabilities());
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return azure_blob_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                       "AZURE_BLOB",
                                       "0.1.0",
                                       {},
                                       {DRAM_SEG, OBJ_SEG},
                                       buildAzureBlobOptionSpecs(),
                                       buildAzureBlobCapabilities());
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
