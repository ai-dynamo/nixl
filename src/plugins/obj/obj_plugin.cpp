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

#include "nixl_types.h"
#include "obj_backend.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"

// Plugin type alias for convenience
using obj_plugin_t = nixlBackendPluginCreator<nixlObjEngine>;

static const nixl_mem_list_t supported_segments = {DRAM_SEG, OBJ_SEG};

namespace {

nixl_backend_option_list_t
buildObjOptionSpecs() {
    return {{"bucket", "Object bucket name", nixl_backend_option_type_t::STRING, false, ""},
            {"access_key", "S3 access key", nixl_backend_option_type_t::STRING, false, ""},
            {"secret_key", "S3 secret key", nixl_backend_option_type_t::STRING, false, ""},
            {"session_token", "S3 session token", nixl_backend_option_type_t::STRING, false, ""},
            {"scheme", "S3 scheme", nixl_backend_option_type_t::STRING, false, "http"},
            {"region", "S3 region", nixl_backend_option_type_t::STRING, false, "eu-central-1"},
            {"use_virtual_addressing",
             "Use S3 virtual addressing",
             nixl_backend_option_type_t::BOOL,
             false,
             "false"},
            {"endpoint_override",
             "S3 endpoint override",
             nixl_backend_option_type_t::STRING,
             false,
             ""},
            {"req_checksum",
             "S3 checksum mode",
             nixl_backend_option_type_t::STRING,
             false,
             "supported"},
            {"ca_bundle", "S3 CA bundle", nixl_backend_option_type_t::STRING, false, ""},
            {"crtMinLimit",
             "S3 CRT minimum object size",
             nixl_backend_option_type_t::UINT64,
             false,
             "0"},
            {"accelerated",
             "Enable S3 accelerated client",
             nixl_backend_option_type_t::BOOL,
             false,
             "false"},
            {"type", "S3 accelerated client type", nixl_backend_option_type_t::STRING, false, ""}};
}

nixlBackendPluginCapabilities
buildObjCapabilities() {
    return {true, false, false};
}

} // namespace

#ifdef STATIC_PLUGIN_OBJ
nixlBackendPlugin *
createStaticOBJPlugin() {
    return obj_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "OBJ",
                                "0.10.0",
                                {},
                                supported_segments,
                                buildObjOptionSpecs(),
                                buildObjCapabilities());
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return obj_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "OBJ",
                                "0.10.0",
                                {},
                                supported_segments,
                                buildObjOptionSpecs(),
                                buildObjCapabilities());
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
