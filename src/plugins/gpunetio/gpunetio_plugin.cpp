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
#include "gpunetio_backend.h"

namespace {
nixl_b_params_t
getGpunetioBackendOptions() {
    return {{"network_devices", ""},
            {"oob_interface", ""},
            {"oob_port", ""},
            {"gpu_devices", ""},
            {"cuda_streams", ""},
            {"gid_index", ""},
            {"data_qp_count", ""},
            {"write_stripe", "slots"},
            {"write_stripe_min_bytes", "4194304"}};
}

using gpunetio_plugin_t = nixlBackendPluginCreator<nixlDocaEngine>;
} // namespace

#ifdef STATIC_PLUGIN_GPUNETIO
nixlBackendPlugin *
createStaticGPUNETIOPlugin() {
    return gpunetio_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "GPUNETIO",
                                     "0.2.0",
                                     getGpunetioBackendOptions(),
                                     {DRAM_SEG, VRAM_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return gpunetio_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "GPUNETIO",
                                     "0.2.0",
                                     getGpunetioBackendOptions(),
                                     {DRAM_SEG, VRAM_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
