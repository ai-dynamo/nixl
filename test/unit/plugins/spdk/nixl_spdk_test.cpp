/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <string>
#include <vector>

#include "backend/backend_plugin.h"
#include "nixl.h"
#include "nixl_params.h"

// The plugin's factory entry point differs by build mode: baked-in plugins
// export createStaticSPDKPlugin() (and have no fini), while dynamic plugins
// export nixl_plugin_init()/nixl_plugin_fini().
#ifdef STATIC_PLUGIN_SPDK
extern nixlBackendPlugin *
createStaticSPDKPlugin();
#else
extern "C" nixlBackendPlugin *
nixl_plugin_init();
extern "C" void
nixl_plugin_fini();
#endif

namespace {

void
checkSupportedMems(const std::vector<nixl_mem_t> &mems) {
    bool hasDram = false;
    bool hasBlk = false;

    for (auto mem : mems) {
        hasDram = hasDram || mem == DRAM_SEG;
        hasBlk = hasBlk || mem == BLK_SEG;
    }

    assert(hasDram);
    assert(hasBlk);
}

void
checkAdvertisedOptions(const nixl_b_params_t &options) {
    // The plugin must advertise its configurable knobs so getPluginParams()
    // surfaces them to users. Both JSON config inputs must be discoverable.
    assert(options.count("json_config") == 1);
    assert(options.count("json_config_file") == 1);
    assert(options.count("msg_mempool_size") == 1);
    // Convenience single-bdev params must also be discoverable.
    assert(options.count("bdev_type") == 1);
    assert(options.count("bdev_name") == 1);
    // Knobs that only mattered to the DPDK env must not be advertised.
    assert(options.count("no_pci") == 0);
    assert(options.count("no_huge") == 0);
    assert(options.count("mem_size") == 0);
    assert(options.count("shm_id") == 0);
    assert(options.count("iova_mode") == 0);
}

void
checkMissingConfigFails(nixlBackendPlugin *plugin, nixl_thread_sync_t syncMode) {
    nixl_b_params_t customParams;
    nixlBackendInitParams params;
    params.localAgent = "spdk-test";
    params.type = "SPDK";
    params.customParams = &customParams;
    params.enableProgTh = false;
    params.syncMode = syncMode;
    params.pthrDelay = 0;
    params.enableTelemetry_ = false;

    nixlBackendEngine *engine = plugin->create_engine(&params);
    assert(engine != nullptr);
    assert(engine->getInitErr());
    plugin->destroy_engine(engine);
}

} // namespace

int
main() {
#ifdef STATIC_PLUGIN_SPDK
    nixlBackendPlugin *plugin = createStaticSPDKPlugin();
#else
    nixlBackendPlugin *plugin = nixl_plugin_init();
#endif
    assert(plugin != nullptr);
    assert(std::string(plugin->get_plugin_name()) == "SPDK");
    checkSupportedMems(plugin->get_backend_mems());
    checkAdvertisedOptions(plugin->get_backend_options());

    checkMissingConfigFails(plugin, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    checkMissingConfigFails(plugin, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);

#ifndef STATIC_PLUGIN_SPDK
    nixl_plugin_fini();
#endif
    return 0;
}
