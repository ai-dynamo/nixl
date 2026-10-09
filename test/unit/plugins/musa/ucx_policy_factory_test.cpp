// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "test_support.h"
#include "ucx/ucx_backend.h"
#include "ucx/ucx_thread_pool_engine.h"

#include <array>
#include <string>

class UnavailablePolicy final : public nixl::ucx::MemoryRegistrationPolicy {
public:
    std::string_view
    deviceMemoryType() const noexcept override {
        return "nixl_test_unavailable_device";
    }

    nixl_status_t
    validateBeforeMap(const nixlBlobDesc &, nixl_mem_t) const override {
        return NIXL_SUCCESS;
    }
};

int
main(int argc, char **) {
    const bool reject_sgl = argc > 1;
    auto policy = std::make_shared<UnavailablePolicy>();
    for (int mode = 0; mode < 3; ++mode) {
        nixlBackendInitParams init;
        nixl_b_params_t params{{"num_threads", mode == 2 ? "2" : "0"}};
        init.enableProgTh = mode == 1;
        init.pthrDelay = 100;
        init.localAgent = "policy-test-" + std::to_string(mode);
        init.customParams = &params;
        init.type = "UCX";
        bool rejected = false;
        try {
            auto engine = nixlUcxEngine::create(init, policy);
        }
        catch (const std::exception &error) {
            const std::string message = error.what();
            NIXL_TEST_CHECK(message.find(reject_sgl ?
                                             "do not support UCX SGL" :
                                             "does not advertise required device memory type") !=
                            std::string::npos);
            rejected = true;
        }
        NIXL_TEST_CHECK(rejected && policy.use_count() == 1);
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
    std::cout << "real UCX factory modes and policy rejection passed\n";
}
