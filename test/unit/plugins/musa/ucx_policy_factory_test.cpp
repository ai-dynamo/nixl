// SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "ucx/ucx_backend.h"
#include "ucx/ucx_thread_pool_engine.h"

#include <array>
#include <memory>
#include <string>
#include <type_traits>

static_assert(std::is_same_v<decltype(&nixlUcxEngine::create),
                             std::unique_ptr<nixlUcxEngine> (*)(const nixlBackendInitParams &)>);

int
main(int argc, char **) {
#ifdef HAVE_UCX_SGL_API
    const bool reject_sgl = argc > 1;
#else
    const bool reject_sgl = false;
    (void)argc;
#endif
    const nixl::ucx::MemoryRegistrationPolicy policy{"nixl_test_unavailable_device", {}};
    for (int mode = 0; mode < 3; ++mode) {
        nixlBackendInitParams init{};
        nixl_b_params_t params{{"num_threads", mode == 2 ? "2" : "0"}};
        init.enableProgTh = mode == 1;
        init.pthrDelay = 100;
        init.localAgent = "policy-test-" + std::to_string(mode);
        init.customParams = &params;
        init.type = "UCX";
        checkThrows([&] {
            (void)nixlUcxEngine::createWithMemoryPolicy(
                init, {"", [](const nixlBlobDesc &, nixl_mem_t) { return NIXL_SUCCESS; }});
        });
        bool rejected = false;
        try {
            auto engine = nixlUcxEngine::createWithMemoryPolicy(init, policy);
        }
        catch (const std::exception &error) {
            const std::string message = error.what();
            NIXL_TEST_CHECK(message.find(reject_sgl ?
                                             "do not support UCX SGL" :
                                             "does not advertise required device memory type") !=
                            std::string::npos);
            rejected = true;
        }
        NIXL_TEST_CHECK(rejected);
        if (reject_sgl) {
            continue;
        }
        auto engine = nixlUcxEngine::create(init);
        NIXL_TEST_CHECK(!engine->getInitErr());
        NIXL_TEST_CHECK(engine->supportsLocal() && engine->supportsRemote() &&
                        engine->supportsNotif());
        NIXL_TEST_CHECK((dynamic_cast<nixlUcxThreadPoolEngine *>(engine.get()) != nullptr) ==
                        (mode == 2));
        NIXL_TEST_CHECK((dynamic_cast<nixlUcxThreadEngine *>(engine.get()) != nullptr) ==
                        (mode != 0));
        std::array<char, 4096> buffer{};
        nixlBlobDesc desc;
        desc.addr = reinterpret_cast<uintptr_t>(buffer.data());
        desc.len = buffer.size();
        desc.devId = 0;
        nixlBackendMD *metadata = nullptr;
        NIXL_TEST_CHECK(engine->registerMem(desc, DRAM_SEG, metadata) == NIXL_SUCCESS);
        NIXL_TEST_CHECK(metadata != nullptr);
        nixlBackendMD *remote_metadata = metadata;
        desc.metaInfo = "not-a-real-rkey";
        NIXL_TEST_CHECK(engine->loadRemoteMD(desc, FILE_SEG, "not-connected", remote_metadata) ==
                        NIXL_ERR_NOT_FOUND);
        NIXL_TEST_CHECK(remote_metadata == nullptr);
        desc.metaInfo.clear();
        desc.addr = 0;
        NIXL_TEST_CHECK(engine->loadRemoteMD(desc, DRAM_SEG, "not-connected", remote_metadata) ==
                        NIXL_ERR_INVALID_PARAM);
        NIXL_TEST_CHECK(remote_metadata == nullptr);
        desc.addr = reinterpret_cast<uintptr_t>(buffer.data());
        std::string connection;
        NIXL_TEST_CHECK(engine->getConnInfo(connection) == NIXL_SUCCESS);
        NIXL_TEST_CHECK(engine->loadRemoteConnInfo(init.localAgent, connection) == NIXL_SUCCESS);
        NIXL_TEST_CHECK(engine->loadRemoteMD(desc, DRAM_SEG, init.localAgent, remote_metadata) ==
                        NIXL_ERR_INVALID_PARAM);
        NIXL_TEST_CHECK(remote_metadata == nullptr); // Empty packed key must not reach UCP.
        NIXL_TEST_CHECK(engine->deregisterMem(metadata) == NIXL_SUCCESS);
    }
    nixlBackendInitParams proxy_init{};
    nixl_b_params_t proxy_params{{"device_proxy", "true"}};
    proxy_init.customParams = &proxy_params;
    proxy_init.localAgent = "policy-proxy-test";
    proxy_init.type = "UCX";
    bool proxy_rejected = false;
    try {
        (void)nixlUcxEngine::createWithMemoryPolicy(proxy_init, policy);
    }
    catch (const std::exception &error) {
        NIXL_TEST_CHECK(std::string(error.what())
                            .find("A memory policy is not supported with device_proxy=true") !=
                        std::string::npos);
        proxy_rejected = true;
    }
    NIXL_TEST_CHECK(proxy_rejected);
    std::cout << "real UCX factory modes and policy rejection passed\n";
}
