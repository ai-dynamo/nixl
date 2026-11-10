// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "backend/backend_plugin.h"
#include "wqs_backend.h"

namespace {
nixlBackendPlugin *
createPlugin() {
    const nixlWqsConfig defaults;
    return nixlBackendPluginCreator<nixlWqsEngine>::create(
        NIXL_PLUGIN_API_VERSION,
        wqs_plugin_name,
        wqs_plugin_version,
        {{"unix_path", defaults.unixPath},
         {"staging_pool_bytes", std::to_string(defaults.stagingPoolBytes)},
         {"max_inflight", std::to_string(defaults.maxInflight)},
         {"max_batch_entries", std::to_string(defaults.maxBatchEntries)},
         {"payload_bytes", std::to_string(defaults.payloadBytes)},
         {"key_prefix", defaults.keyPrefix}},
        {DRAM_SEG, OBJ_SEG});
}
} // namespace

#ifdef STATIC_PLUGIN_WQS
nixlBackendPlugin *
createStaticWQSPlugin() {
    return createPlugin();
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return createPlugin();
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
