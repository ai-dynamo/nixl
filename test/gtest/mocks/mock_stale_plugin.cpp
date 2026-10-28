/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "backend/backend_engine.h"
#include "backend/backend_plugin.h"

namespace mocks::stale_plugin {

nixlBackendEngine *
createEngine(const nixlBackendInitParams *) {
    return nullptr;
}

void
destroyEngine(nixlBackendEngine *) {}

const char *
getName() {
    return "MOCK_STALE_BACKEND";
}

const char *
getVersion() {
    return "0.0.1";
}

nixl_b_params_t
getBackendOptions() {
    return nixl_b_params_t();
}

nixl_mem_list_t
getBackendMems() {
    return nixl_mem_list_t();
}

nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION - 1,
                            createEngine,
                            destroyEngine,
                            getName,
                            getVersion,
                            getBackendOptions,
                            getBackendMems};

} // namespace mocks::stale_plugin

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return &mocks::stale_plugin::plugin;
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
