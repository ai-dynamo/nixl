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
#include <gtest/gtest.h>
#include "nixl.h"
#include "plugin_manager.h"
#include <thread>
#include <filesystem>

namespace gtest {
namespace multi_threading {

class MultiThreadingTestFixture : public testing::Test {
protected:
    uintptr_t addr = 0;
    size_t len = 1024;
    uint64_t dev_id = 0;

    nixlAgent createAgent() {
        nixlAgentConfig cfg(false, false, 0, 0, 100000, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
        return nixlAgent("test_agent", cfg);
    }

    nixl_opt_args_t createExtraParams(nixlBackendH* backend) {
        nixl_opt_args_t extra_params;
        extra_params.backends = {backend};
        return extra_params;
    }

    nixlBackendH* verifyMockDramBackendCreation(nixlAgent& agent) {
        nixlBackendH* backend_handle = nullptr;
        nixl_b_params_t params;
        nixl_status_t status = agent.createBackend("MOCK_DRAM", params, backend_handle);
        EXPECT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(backend_handle, nullptr);
        return backend_handle;
    }

    void verifyMemoryRegistration(nixlAgent& agent, const nixl_opt_args_t& extra_params) {
        nixlBlobDesc blob(addr, len, dev_id, "");
        nixlDescList<nixlBlobDesc> desc_list(DRAM_SEG);
        desc_list.addDesc(blob);

        nixl_status_t status = agent.registerMem(desc_list, &extra_params);
        EXPECT_EQ(status, NIXL_SUCCESS);
    }

    void verifyTransfer(nixlAgent& agent, const nixl_opt_args_t& extra_params) {
        nixlXferReqH* xfer_req = nullptr;
        nixlDescList<nixlBasicDesc> src_list(DRAM_SEG);
        nixlDescList<nixlBasicDesc> dst_list(DRAM_SEG);

        nixlBasicDesc basic_desc(addr, len, dev_id);
        src_list.addDesc(basic_desc);
        dst_list.addDesc(basic_desc);

        nixl_status_t status = agent.createXferReq(NIXL_WRITE, src_list, dst_list, "test_agent", xfer_req, &extra_params);
        EXPECT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(xfer_req, nullptr);

        status = agent.postXferReq(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        status = agent.getXferStatus(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        status = agent.releaseXferReq(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);
    }
};

TEST_F(MultiThreadingTestFixture, ConcurrentTransfersWithPerThreadAgent) {
    auto transfer_sequence = [&]() {
        nixlAgent agent = createAgent();
        nixlBackendH* backend = verifyMockDramBackendCreation(agent);
        nixl_opt_args_t extra_params = createExtraParams(backend);

        verifyMemoryRegistration(agent, extra_params);
        verifyTransfer(agent, extra_params);
    };

    std::thread t1(transfer_sequence);
    std::thread t2(transfer_sequence);

    t1.join();
    t2.join();
}

TEST_F(MultiThreadingTestFixture, ConcurrentAddPlugingDirWithPerThreadAgent) {
    auto transfer_sequence = [&]() {
        nixlAgent agent = createAgent();
        nixlBackendH* backend = verifyMockDramBackendCreation(agent);
        nixl_opt_args_t extra_params = createExtraParams(backend);

        using namespace std::filesystem;
        path dir_path = temp_directory_path() / "nixl_mt_test_plugin_dir";
        create_directory(dir_path);
        nixlPluginManager::getInstance().addPluginDirectory(dir_path);

        verifyMemoryRegistration(agent, extra_params);
        verifyTransfer(agent, extra_params);
    };

    std::thread t1(transfer_sequence);
    std::thread t2(transfer_sequence);

    t1.join();
    t2.join();
}

TEST_F(MultiThreadingTestFixture, ConcurrentTransfersWithPerThreadMemory) {
    nixlAgent agent = createAgent();
    nixlBackendH* backend = verifyMockDramBackendCreation(agent);
    nixl_opt_args_t extra_params = createExtraParams(backend);

    auto transfer_sequence = [&]() {
        verifyMemoryRegistration(agent, extra_params);
        verifyTransfer(agent, extra_params);
    };

    std::thread t1(transfer_sequence);
    std::thread t2(transfer_sequence);

    t1.join();
    t2.join();
}

TEST_F(MultiThreadingTestFixture, ConcurrentTransfers) {
    nixlAgent agent = createAgent();
    nixlBackendH* backend = verifyMockDramBackendCreation(agent);
    nixl_opt_args_t extra_params = createExtraParams(backend);

    verifyMemoryRegistration(agent, extra_params);

    auto transfer_sequence = [&]() {
        verifyTransfer(agent, extra_params);
    };

    std::thread t1(transfer_sequence);
    std::thread t2(transfer_sequence);

    t1.join();
    t2.join();
}

TEST_F(MultiThreadingTestFixture, RegisterMemWithMockDram) {
    nixlAgent agent = createAgent();
    nixlBackendH* backend = verifyMockDramBackendCreation(agent);
    nixl_opt_args_t extra_params = createExtraParams(backend);

    verifyMemoryRegistration(agent, extra_params);
    verifyTransfer(agent, extra_params);
}

} // namespace mt
} // namespace gtest
