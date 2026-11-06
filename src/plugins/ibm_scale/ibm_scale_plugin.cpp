/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation. All rights reserved.
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
#include "ibm_scale_backend.h"

// Plugin type alias — same pattern as posix_plugin.cpp.
using ibm_scale_plugin_t = nixlBackendPluginCreator<nixlScaleEngine>;

namespace {
const nixl_mem_list_t supported_segments = {FILE_SEG, DRAM_SEG};
} // namespace

#ifdef STATIC_PLUGIN_IBM_SCALE
nixlBackendPlugin *
createStaticIBMScalePlugin() {
    return ibm_scale_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                      IBM_SCALE_PLUGIN_NAME,
                                      IBM_SCALE_PLUGIN_VERSION,
                                      {},
                                      supported_segments);
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return ibm_scale_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                      IBM_SCALE_PLUGIN_NAME,
                                      IBM_SCALE_PLUGIN_VERSION,
                                      {},
                                      supported_segments);
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
