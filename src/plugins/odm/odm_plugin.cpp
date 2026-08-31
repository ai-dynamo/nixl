/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/backend_plugin.h"
#include "odm_backend.h"

using odm_plugin_t = nixlBackendPluginCreator<nixlOdmEngine>;

namespace {

[[nodiscard]] nixl_b_params_t
getOdmBackendOptions() {
    nixl_b_params_t params;
    params["dmadev_param"] = "odm0";
    params["odm_qid"] = "0";
    params["odm_qid_start"] = "0";
    params["odm_qid_end"] = "0";
    params["dmabuf_cache_max"] = "512";
    return params;
}

} // namespace

#ifdef STATIC_PLUGIN_ODM
nixlBackendPlugin *
createStaticODMPlugin() {
    return odm_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "ODM", "0.1.0", getOdmBackendOptions(), odmSupportedMems());
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return odm_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "ODM", "0.1.0", getOdmBackendOptions(), odmSupportedMems());
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
