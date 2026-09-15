/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backend/backend_plugin.h"
#include "daos_backend.h"

using daos_plugin_t = nixlBackendPluginCreator<nixlDaosEngine>;

#ifdef STATIC_PLUGIN_DAOS
nixlBackendPlugin *
createStaticDAOSPlugin() {
    return daos_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                 "DAOS",
                                 "0.1.0",
                                 nixlDaosEngine::getPluginParams(),
                                 {DRAM_SEG, OBJ_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return daos_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                 "DAOS",
                                 "0.1.0",
                                 nixlDaosEngine::getPluginParams(),
                                 {DRAM_SEG, OBJ_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
