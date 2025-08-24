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


// Plugin type alias for convenience
using hf3fs_plugin_t = nixlBackendPluginCreator<nixlHf3fsEngine>;

#ifdef STATIC_PLUGIN_HF3FS
nixlBackendPlugin *
createStaticHF3FSPlugin() {
    return hf3fs_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "HF3FS", "0.1.0", {}, {FILE_SEG, DRAM_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return hf3fs_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "HF3FS", "0.1.0", {}, {FILE_SEG, DRAM_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
