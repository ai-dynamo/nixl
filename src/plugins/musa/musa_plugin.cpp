// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "musa_policy.h"
#include "../ucx/ucx_backend.h"
#include "../ucx/ucx_binding.h"
#include "backend/backend_plugin.h"
#include "common/nixl_log.h"

#include <exception>

namespace {
nixlBackendEngine *
createEngine(const nixlBackendInitParams *params) {
    if (!params) {
        return nullptr;
    }
    try {
        auto policy = std::make_shared<nixl::musa::MemoryPolicy>(nixl::musa::makeRuntime());
        return nixlUcxEngine::create(*params, std::move(policy)).release();
    }
    catch (const std::exception &error) {
        NIXL_ERROR << "MUSA_UCX engine creation failed: " << error.what();
        return nullptr;
    }
}

void
destroyEngine(nixlBackendEngine *engine) {
    delete engine;
}

const char *
getName() {
    return "MUSA_UCX";
}

const char *
getVersion() {
    return "0.1.0";
}

nixl_b_params_t
getOptions() {
    return {{"device_list", ""},
            {"num_threads", "0"},
            {"num_workers", "1"},
            {"ucx_error_handling_mode", "peer"}};
}

nixl_mem_list_t
getMems() {
    return {DRAM_SEG, VRAM_SEG};
}

nixlBackendPlugin plugin{NIXL_PLUGIN_API_VERSION,
                         createEngine,
                         destroyEngine,
                         getName,
                         getVersion,
                         getOptions,
                         getMems};
} // namespace

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return nixl::ucx::validateBinding() ? &plugin : nullptr;
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
