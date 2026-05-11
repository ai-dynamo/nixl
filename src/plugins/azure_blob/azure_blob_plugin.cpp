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

#include <algorithm>
#include <thread>

#include "azure_blob_backend.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
 
// Plugin type alias for convenience
using azure_blob_plugin_t = nixlBackendPluginCreator<nixlAzureBlobEngine>;

namespace {

nixl_b_params_t
getAzureBlobBackendOptions() {
    const auto num_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    return {{"account_url", ""},
            {"container_name", ""},
            {"connection_string", ""},
            {"ca_bundle", ""},
            {"num_threads", ""}};
}

} // namespace

nixlBackendPluginCapabilities
buildAzureBlobCapabilities() {
    return {true, false, false};
}

#ifdef STATIC_PLUGIN_AZURE
nixlBackendPlugin *
createStaticAZUREPlugin() {
    return azure_blob_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                       "AZURE_BLOB",
                                       "0.1.0",
                                       getAzureBlobBackendOptions(),
                                       {DRAM_SEG, OBJ_SEG},
                                       buildAzureBlobCapabilities());
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return azure_blob_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                       "AZURE_BLOB",
                                       "0.1.0",
                                       getAzureBlobBackendOptions(),
                                       {DRAM_SEG, OBJ_SEG},
                                       buildAzureBlobCapabilities());
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
 
