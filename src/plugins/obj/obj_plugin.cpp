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

#include <algorithm>
#include <thread>

#include "backend/backend_plugin.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include "obj_backend.h"
 
// Plugin type alias for convenience
using obj_plugin_t = nixlBackendPluginCreator<nixlObjEngine>;

namespace {

nixl_b_params_t
getObjBackendOptions() {
    return {{"access_key", ""},
            {"secret_key", ""},
            {"session_token", ""},
            {"bucket", ""},
            {"scheme", ""},
            {"region", ""},
            {"use_virtual_addressing", "false"},
            {"endpoint_override", ""},
            {"req_checksum", ""},
            {"ca_bundle", ""},
            {"crtMinLimit", "0"},
            {"accelerated", "false"},
            {"type", ""},
            {"num_threads", ""}};
}

} // namespace
 
 static const nixl_mem_list_t supported_segments = {DRAM_SEG, OBJ_SEG};
 
nixlBackendPluginCapabilities
buildObjCapabilities() {
    return {false};
}


#ifdef STATIC_PLUGIN_OBJ
nixlBackendPlugin *
createStaticOBJPlugin() {
    return obj_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "OBJ",
                                "0.10.0",
                                getObjBackendOptions(),
                                supported_segments,
                                buildObjCapabilities());
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return obj_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "OBJ",
                                "0.10.0",
                                getObjBackendOptions(),
                                supported_segments,
                                buildObjCapabilities());
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
 
