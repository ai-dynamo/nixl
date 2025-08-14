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
#include "hf3fs_backend.h"
#include <iostream>

namespace {
nixl_b_params_t
get_hf3fs_options() {
    nixl_b_params_t params;
    return params;
}

nixl_mem_list_t
get_hf3fs_mems() {
    nixl_mem_list_t mems;
    mems.push_back(FILE_SEG);
    mems.push_back(DRAM_SEG);
    return mems;
}
} // namespace

// Plugin type alias for convenience
using Hf3fsPlugin = nixlBackendPluginTemplate<nixlHf3fsEngine>;

#ifdef STATIC_PLUGIN_HF3FS
// Function for static loading
extern "C" nixlBackendPlugin *createStaticHF3FSPlugin() {
    return Hf3fsPlugin::initialize_plugin("HF3FS", "0.1.0", get_hf3fs_options, get_hf3fs_mems);
}
#else
// Export functions for dynamic loading
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init() {
    return Hf3fsPlugin::initialize_plugin("HF3FS", "0.1.0", get_hf3fs_options, get_hf3fs_mems);
}

extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {}
#endif
