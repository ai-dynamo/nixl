// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "backend/backend_plugin.h"

#include <dlfcn.h>
#include <string>

int
main(int argc, char **argv) {
    NIXL_TEST_CHECK(argc == 2 || argc == 3);
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::cerr << dlerror() << std::endl;
        return 1;
    }
    auto init = reinterpret_cast<nixlBackendPlugin *(*)()>(dlsym(library, "nixl_plugin_init"));
    auto fini = reinterpret_cast<void (*)()>(dlsym(library, "nixl_plugin_fini"));
    NIXL_TEST_CHECK(init && fini);
    auto *plugin = init();
    NIXL_TEST_CHECK(plugin && plugin->api_version == NIXL_PLUGIN_API_VERSION);
    NIXL_TEST_CHECK(std::string(plugin->get_plugin_name()) == "MUSA_UCX");
    NIXL_TEST_CHECK(plugin->get_backend_mems() == nixl_mem_list_t({DRAM_SEG, VRAM_SEG}));
    const auto options = plugin->get_backend_options();
    NIXL_TEST_CHECK(options.contains("device_list") && !options.contains("ucx_devices"));
    NIXL_TEST_CHECK(!options.contains("ucx_num_device_channels"));
    NIXL_TEST_CHECK(plugin->create_engine(nullptr) == nullptr);
    if (argc == 3) {
        NIXL_TEST_CHECK(std::string(argv[2]) == "--expect-no-provider");
        nixlBackendInitParams params;
        nixl_b_params_t custom;
        params.localAgent = "musa-no-provider-test";
        params.type = "MUSA_UCX";
        params.customParams = &custom;
        auto *engine = plugin->create_engine(&params);
        if (engine) {
            plugin->destroy_engine(engine);
        }
        NIXL_TEST_CHECK(engine == nullptr);
    }
    fini();
    NIXL_TEST_CHECK(dlclose(library) == 0);
    std::cout << "MUSA plugin ABI/options checks passed\n";
}
