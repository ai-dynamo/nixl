/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string>

#include "ais_mt_engine.h"
#include "backend/backend_plugin.h"

namespace {
nixl_b_params_t
getAisMtBackendOptions() {
    return {{"thread_count", std::to_string(defaultAisMtThreadCount())}};
}

using ais_mt_plugin_t = nixlBackendPluginCreator<nixlAisMtEngine>;
} // namespace

#ifdef STATIC_PLUGIN_AIS_MT
nixlBackendPlugin *
createStaticAIS_MTPlugin() {
    return ais_mt_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                   "AIS_MT",
                                   "0.1.0",
                                   getAisMtBackendOptions(),
                                   {DRAM_SEG, VRAM_SEG, FILE_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return ais_mt_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                   "AIS_MT",
                                   "0.1.0",
                                   getAisMtBackendOptions(),
                                   {DRAM_SEG, VRAM_SEG, FILE_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
