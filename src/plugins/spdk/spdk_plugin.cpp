/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backend/backend_plugin.h"
#include "spdk_backend.h"

using spdk_plugin_t = nixlBackendPluginCreator<nixlSpdkEngine>;

// Advertised so getPluginParams() exposes the configurable knobs and their
// defaults. Exactly one of json_config / json_config_file must be set.
[[nodiscard]] static nixl_b_params_t
get_spdk_backend_options() {
    nixl_b_params_t params;
    // Bdev configuration. Provide JSON directly, or via the convenience params
    // below for the common single-bdev case (explicit JSON takes precedence).
    params["json_config"] = ""; // inline SPDK bdev subsystem JSON (preferred)
    params["json_config_file"] = ""; // path to the same JSON (alternative)
    // Convenience single-bdev config (used when no JSON is given):
    params["bdev_type"] = ""; // malloc | aio | nvme
    params["bdev_name"] = ""; // bdev name (NVMe exposes namespaces as <name>n1)
    params["bdev_num_blocks"] = ""; // malloc: number of blocks
    params["bdev_block_size"] = ""; // malloc/aio: block size in bytes (default 512)
    params["bdev_filename"] = ""; // aio: backing file or device path
    params["bdev_traddr"] = ""; // nvme: PCIe transport address
    params["spdk_name"] = "nixl_spdk"; // names the SPDK thread in logs
    params["core_mask"] = ""; // SPDK core mask (empty = SPDK default)
    params["msg_mempool_size"] = "0"; // SPDK thread message-pool size (0 = default)
    return params;
}

#ifdef STATIC_PLUGIN_SPDK
nixlBackendPlugin *
createStaticSPDKPlugin() {
    return spdk_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "SPDK", "0.1.0", get_spdk_backend_options(), {DRAM_SEG, BLK_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return spdk_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "SPDK", "0.1.0", get_spdk_backend_options(), {DRAM_SEG, BLK_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
