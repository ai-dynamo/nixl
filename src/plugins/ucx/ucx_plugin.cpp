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

// Plugin type alias for convenience
using ucx_plugin_t = nixlBackendPluginCreator<nixlUcxEngine>;

namespace {

nixl_backend_option_list_t
buildUcxOptionSpecs() {
    return {{"ucx_devices", "UCX device list", nixl_backend_option_type_t::STRING, false, ""},
            {"device_list", "UCX device list", nixl_backend_option_type_t::STRING, false, ""},
            {"num_workers", "UCX worker count", nixl_backend_option_type_t::INT, false, "1"},
            {"num_threads", "UCX progress thread count", nixl_backend_option_type_t::INT, false, "0"}};
}

nixlBackendPluginCapabilities
buildUcxCapabilities() {
    return {false, true};
}

} // namespace

#ifdef STATIC_PLUGIN_UCX
nixlBackendPlugin *
createStaticUCXPlugin() {
    return ucx_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "UCX",
                                "0.1.0",
                                get_ucx_backend_common_options(),
                                {DRAM_SEG, VRAM_SEG},
                                buildUcxOptionSpecs(),
                                buildUcxCapabilities());
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return ucx_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "UCX",
                                "0.1.0",
                                get_ucx_backend_common_options(),
                                {DRAM_SEG, VRAM_SEG},
                                buildUcxOptionSpecs(),
                                buildUcxCapabilities());
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
