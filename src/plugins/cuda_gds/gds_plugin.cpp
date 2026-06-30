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
#include "gds_batch_engine.h"

namespace {
nixl_b_params_t
getGdsBackendOptions() {
    return {{"batch_pool_size", "16"}, {"batch_limit", "128"}, {"max_request_size", "16777216"}};
}

using gds_plugin_t = nixlBackendPluginCreator<nixlGdsBatchEngine>;
} // namespace

#ifdef STATIC_PLUGIN_GDS
nixlBackendPlugin *
createStaticGDSPlugin() {
    return gds_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "GDS",
                                "0.1.1",
                                getGdsBackendOptions(),
                                {DRAM_SEG, VRAM_SEG, FILE_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return gds_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                "GDS",
                                "0.1.1",
                                getGdsBackendOptions(),
                                {DRAM_SEG, VRAM_SEG, FILE_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
