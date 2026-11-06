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

// Server-free tests for the VALKEY backend.

#include <gtest/gtest.h>

#include "valkey_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gtest::valkey {

TEST(valkeyConfigTest, UsesDefaultsWithoutParameters) {
    const auto config = nixlValkeyConfig::fromBackendParams(nullptr);
    const nixlValkeyConfig defaults;
    EXPECT_EQ(config.host, defaults.host);
    EXPECT_EQ(config.port, 6379);
    EXPECT_EQ(config.tls, "off");
    EXPECT_FALSE(config.clusterMode);
    EXPECT_EQ(config.rdmaProvider, "efa-direct");
    EXPECT_TRUE(config.username.empty());
    EXPECT_TRUE(config.password.empty());
    EXPECT_TRUE(config.rdmaInterface.empty());
    EXPECT_TRUE(config.keyPrefix.empty());
    EXPECT_EQ(config.timeoutMs, 30000);
    EXPECT_EQ(config.transferTimeoutMs, 0);
    EXPECT_EQ(config.numWorkers, 8);
    EXPECT_FALSE(config.writeChecksum);
}

TEST(valkeyConfigTest, BackendParametersOverrideDefaults) {
    const nixl_b_params_t params = {{"host", "10.0.0.1"},
                                    {"port", "7000"},
                                    {"tls", "insecure"},
                                    {"cluster_mode", "true"},
                                    {"username", "user"},
                                    {"password", "secret"},
                                    {"rdma_provider", "tcp"},
                                    {"rdma_interface", "lo"},
                                    {"key_prefix", "kv:"},
                                    {"timeout_ms", "500"},
                                    {"transfer_timeout_ms", "2000"},
                                    {"num_workers", "16"},
                                    {"write_checksum", "true"}};
    const auto config = nixlValkeyConfig::fromBackendParams(&params);
    EXPECT_EQ(config.host, "10.0.0.1");
    EXPECT_EQ(config.port, 7000);
    EXPECT_EQ(config.tls, "insecure");
    EXPECT_TRUE(config.clusterMode);
    EXPECT_EQ(config.username, "user");
    EXPECT_EQ(config.password, "secret");
    EXPECT_EQ(config.rdmaProvider, "tcp");
    EXPECT_EQ(config.rdmaInterface, "lo");
    EXPECT_EQ(config.keyPrefix, "kv:");
    EXPECT_EQ(config.timeoutMs, 500);
    EXPECT_EQ(config.transferTimeoutMs, 2000);
    EXPECT_EQ(config.numWorkers, 16);
    EXPECT_TRUE(config.writeChecksum);
}

TEST(valkeyConfigTest, RejectsInvalidValues) {
    const std::vector<nixl_b_params_t> invalid = {
        {{"port", "0"}},
        {{"port", "65536"}},
        {{"port", "abc"}},
        {{"port", "6379x"}},
        {{"tls", "yes"}},
        {{"cluster_mode", "maybe"}},
        {{"rdma_provider", "verbs"}},
        {{"rdma_provider", "none"}},
        {{"username", "user"}},
        {{"timeout_ms", "0"}},
        {{"timeout_ms", "-1"}},
        {{"transfer_timeout_ms", "-1"}},
        {{"num_workers", "0"}},
        {{"write_checksum", "sometimes"}},
    };
    for (const auto &params : invalid) {
        EXPECT_THROW(nixlValkeyConfig::fromBackendParams(&params), std::runtime_error)
            << params.begin()->first << "=" << params.begin()->second;
    }
}

// An engine whose server is unreachable: initialization fails, so no worker
// runs, but registration and descriptor validation need no server.
class valkeyEngineTest : public testing::Test {
protected:
    void
    SetUp() override {
        customParams_ = {{"host", "127.0.0.1"},
                         {"port", "1"},
                         {"rdma_provider", "tcp"},
                         {"timeout_ms", "200"},
                         {"key_prefix", "test:"}};
        initParams_.localAgent = "valkey-test-agent";
        initParams_.type = "VALKEY";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        engine_ = std::make_unique<nixlValkeyEngine>(&initParams_);
    }

    nixlBackendMD *
    registerMem(const nixlBlobDesc &desc, nixl_mem_t mem) {
        nixlBackendMD *md = nullptr;
        EXPECT_EQ(engine_->registerMem(desc, mem, md), NIXL_SUCCESS);
        EXPECT_NE(md, nullptr);
        return md;
    }

    nixl_status_t
    prepXfer(nixl_xfer_op_t op,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             nixlBackendReqH *&handle) {
        return engine_->prepXfer(op, local, remote, initParams_.localAgent, handle, nullptr);
    }

    nixl_status_t
    postXfer(nixl_xfer_op_t op,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             nixlBackendReqH *&handle) {
        return engine_->postXfer(op, local, remote, initParams_.localAgent, handle, nullptr);
    }

    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;
    std::unique_ptr<nixlValkeyEngine> engine_;
};

TEST_F(valkeyEngineTest, FailsInitializationWithoutServer) {
    EXPECT_TRUE(engine_->getInitErr());
}

TEST_F(valkeyEngineTest, ReportsCapabilities) {
    EXPECT_FALSE(engine_->supportsRemote());
    EXPECT_TRUE(engine_->supportsLocal());
    EXPECT_FALSE(engine_->supportsNotif());
    EXPECT_EQ(engine_->getSupportedMems(), (nixl_mem_list_t{DRAM_SEG, OBJ_SEG}));
}

TEST_F(valkeyEngineTest, RegistersAndDeregistersMemory) {
    std::vector<char> buffer(4096);
    auto *dram = registerMem(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(), 0, ""), DRAM_SEG);
    auto *named = registerMem(nixlBlobDesc(0, buffer.size(), 1, "named"), OBJ_SEG);
    auto *by_dev_id = registerMem(nixlBlobDesc(0, buffer.size(), 2, ""), OBJ_SEG);

    EXPECT_EQ(engine_->deregisterMem(dram), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(named), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(by_dev_id), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(nullptr), NIXL_SUCCESS);


    nixlBackendMD *unsupported = reinterpret_cast<nixlBackendMD *>(1);
    EXPECT_EQ(engine_->registerMem(nixlBlobDesc(), FILE_SEG, unsupported), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(unsupported, nullptr);
}

TEST_F(valkeyEngineTest, PrepXferValidatesRequests) {
    std::vector<char> buffer(16);
    nixl_meta_dlist_t local(DRAM_SEG);
    local.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(), 0, nullptr));
    nixl_meta_dlist_t remote(OBJ_SEG);
    remote.addDesc(nixlMetaDesc(0, buffer.size(), 1, nullptr));

    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(prepXfer(static_cast<nixl_xfer_op_t>(99), local, remote, handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    nixl_meta_dlist_t wrong_local(OBJ_SEG);
    wrong_local.addDesc(nixlMetaDesc(0, buffer.size(), 0, nullptr));
    EXPECT_EQ(prepXfer(NIXL_WRITE, wrong_local, remote, handle), NIXL_ERR_INVALID_PARAM);

    nixl_meta_dlist_t wrong_remote(DRAM_SEG);
    wrong_remote.addDesc(nixlMetaDesc(0, buffer.size(), 1, nullptr));
    EXPECT_EQ(prepXfer(NIXL_WRITE, local, wrong_remote, handle), NIXL_ERR_INVALID_PARAM);

    nixl_meta_dlist_t two_remotes(OBJ_SEG);
    two_remotes.addDesc(nixlMetaDesc(0, buffer.size(), 1, nullptr));
    two_remotes.addDesc(nixlMetaDesc(0, buffer.size(), 2, nullptr));
    EXPECT_EQ(prepXfer(NIXL_WRITE, local, two_remotes, handle), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    // Another agent's name only warns: the objects are this agent's.
    EXPECT_EQ(engine_->prepXfer(NIXL_READ, local, remote, "other-agent", handle, nullptr),
              NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
}

TEST_F(valkeyEngineTest, PostXferRequiresWholeRegisteredObjects) {
    std::vector<char> buffer(64);
    const auto base = reinterpret_cast<uintptr_t>(buffer.data());
    auto *region = registerMem(nixlBlobDesc(base, buffer.size(), 0, ""), DRAM_SEG);
    auto *object = registerMem(nixlBlobDesc(0, 32, 1, "object"), OBJ_SEG);

    nixl_meta_dlist_t local(DRAM_SEG);
    local.addDesc(nixlMetaDesc(base, 32, 0, region));
    nixl_meta_dlist_t whole(OBJ_SEG);
    whole.addDesc(nixlMetaDesc(0, 32, 1, object));

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(prepXfer(NIXL_WRITE, local, whole, handle), NIXL_SUCCESS);

    // BLOB has no range commands: an object offset or a shorter buffer is refused.
    nixl_meta_dlist_t offset(OBJ_SEG);
    offset.addDesc(nixlMetaDesc(16, 32, 1, object));
    EXPECT_EQ(postXfer(NIXL_WRITE, local, offset, handle), NIXL_ERR_NOT_SUPPORTED);

    nixl_meta_dlist_t short_local(DRAM_SEG);
    short_local.addDesc(nixlMetaDesc(base, 16, 0, region));
    EXPECT_EQ(postXfer(NIXL_READ, short_local, whole, handle), NIXL_ERR_NOT_SUPPORTED);

    // Unregistered memory on either side.
    nixl_meta_dlist_t unregistered_obj(OBJ_SEG);
    unregistered_obj.addDesc(nixlMetaDesc(0, 32, 1, nullptr));
    EXPECT_EQ(postXfer(NIXL_WRITE, local, unregistered_obj, handle), NIXL_ERR_INVALID_PARAM);

    nixl_meta_dlist_t unregistered_local(DRAM_SEG);
    unregistered_local.addDesc(nixlMetaDesc(base, 32, 0, nullptr));
    EXPECT_EQ(postXfer(NIXL_WRITE, unregistered_local, whole, handle), NIXL_ERR_INVALID_PARAM);

    nixlBackendReqH *no_handle = nullptr;
    EXPECT_EQ(postXfer(NIXL_WRITE, local, whole, no_handle), NIXL_ERR_INVALID_PARAM);

    // Nothing was queued, so the request is idle.
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(object), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(region), NIXL_SUCCESS);
}

} // namespace gtest::valkey
