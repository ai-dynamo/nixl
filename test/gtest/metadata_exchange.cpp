/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <thread>
#include <random>
#include "nixl.h"
#include "common.h"

// Used to avoid failures when etcd is not available
#if HAVE_ETCD
#define VERIFY_ETCD_MODE()                                                                  \
    do {                                                                                    \
        const char *etcd_endpoints = std::getenv("NIXL_ETCD_ENDPOINTS");                    \
        if (!etcd_endpoints || !*etcd_endpoints) {                                          \
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is empty or not set, skipping etcd tests"; \
            return;                                                                         \
        }                                                                                   \
    } while (0)
#else
#define VERIFY_ETCD_MODE()                                                         \
    do {                                                                           \
        GTEST_SKIP() << "NIXL compiled without etcd support, skipping etcd tests"; \
        return;                                                                    \
    } while (0)
#endif

namespace gtest {
namespace metadata_exchange {
class MemBuffer {
public:
    explicit MemBuffer(size_t size) : vec_(size) {}

    operator uintptr_t() const
    {
        return reinterpret_cast<uintptr_t>(vec_.data());
    }

    nixlBasicDesc getBasicDesc() const
    {
        return nixlBasicDesc(static_cast<uintptr_t>(*this), vec_.size(), dev_id_);
    }

    nixlBlobDesc getBlobDesc() const
    {
        return nixlBlobDesc(getBasicDesc(), "");
    }

    size_t getSize() const
    {
        return vec_.size();
    }

private:
    std::vector<std::byte> vec_;
    constexpr static uint64_t dev_id_ = 0;
};

class MetadataExchangeTestFixture : public testing::Test {

    struct AgentContext {
        static constexpr size_t BUFF_COUNT_ = 5;
        static constexpr size_t BUFF_SIZE_ = 1024;

        std::unique_ptr<nixlAgent> agent;
        const std::string name;
        const std::string ip = "127.0.0.1";
        const int port;
        nixlBackendH *backend_handle = nullptr;
        std::vector<MemBuffer> buffers;

        AgentContext(std::unique_ptr<nixlAgent> agent, std::string name, int port) :
            agent(std::move(agent)), name(std::move(name)), port(port)
        {
        }

        void createAgentBackend()
        {
            ASSERT_EQ(agent->createBackend("UCX", {}, backend_handle), NIXL_SUCCESS);
            ASSERT_NE(backend_handle, nullptr);
        }

        void initAndRegisterBuffers(size_t count, size_t size)
        {
            for (size_t i = 0; i < count; i++) {
                buffers.emplace_back(size);
            }

            nixl_reg_dlist_t dlist(DRAM_SEG);
            for (const auto &buffer : buffers) {
                dlist.addDesc(buffer.getBlobDesc());
            }

            ASSERT_EQ(agent->registerMem(dlist), NIXL_SUCCESS);
        }

        void initDefault() {
            createAgentBackend();
            initAndRegisterBuffers(BUFF_COUNT_, BUFF_SIZE_);
        }
    };

protected:

    void SetUp() override
    {
        // Create two agents
        for (int i = 0; i < AGENT_COUNT_; i++) {
            const auto port = PortAllocator::next_tcp_port();
            std::string name = "agent_" + std::to_string(i);
            nixlAgentConfig cfg(false, true, port, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);

            auto agent = std::make_unique<nixlAgent>(name, cfg);

            agents_.emplace_back(std::move(agent), std::move(name), port);
        }
    }

    void TearDown() override
    {
        for (auto &agent : agents_) {
            if (agent.agent) {
                agent.agent->invalidateLocalMD(nullptr);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        agents_.clear();
    }

    void initAgentsDefault()
    {
        for (auto &agent : agents_) {
            agent.initDefault();
        }
    }

    static constexpr int AGENT_COUNT_ = 2;

    std::vector<AgentContext> agents_;
};

TEST_F(MetadataExchangeTestFixture, GetLocalAndLoadRemote) {
    initAgentsDefault();

    nixl_xfer_dlist_t dlist(DRAM_SEG);
    for (const auto &buffer : agents_[1].buffers) {
        dlist.addDesc(buffer.getBasicDesc());
    }

    std::string remote_name;
    nixl_blob_t md;

    auto &src = agents_[1];
    auto &dst = agents_[0];

    ASSERT_EQ(src.agent->getLocalMD(md), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, dlist), NIXL_SUCCESS);

    // Invalidate
    ASSERT_EQ(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, dlist), NIXL_ERR_NOT_FOUND);

    // Remote does not exist so cannot invalidate
    ASSERT_NE(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, LoadRemoteWithErrors) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    src.initDefault();

    std::string remote_name;
    nixl_blob_t md;

    ASSERT_EQ(src.agent->getLocalMD(md), NIXL_SUCCESS);

    // No backend on dst agent
    {
        const LogIgnoreGuard lig("loadRemoteMD: no common backend found");

        ASSERT_NE(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);
        ASSERT_NE(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }

    dst.initDefault();

    // Invalid metadata
    {
        const LogIgnoreGuard lig1("Deserialization failed, missing nixlSerDes tag");
        const LogIgnoreGuard lig2("loadRemoteMD: failed to deserialize remote metadata");

        ASSERT_NE(dst.agent->loadRemoteMD("invalid", remote_name), NIXL_SUCCESS);

        EXPECT_EQ(lig1.getIgnoredCount(), 1);
        EXPECT_EQ(lig2.getIgnoredCount(), 1);
    }

    // Remote does not exist so cannot invalidate
    ASSERT_NE(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, GetLocalPartialAndLoadRemote) {
    initAgentsDefault();

    auto &src = agents_[0];
    auto &dst = agents_[1];

    std::string remote_name;
    nixl_blob_t md;

    // Step 1: Get and load connection info

    ASSERT_EQ(src.agent->getLocalPartialMD({DRAM_SEG}, md, nullptr), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    // Step 2: Get partial metadata for agent 0 buffers except the last one

    nixl_reg_dlist_t valid_descs(DRAM_SEG);
    for (size_t i = 0; i < src.buffers.size() - 1; i++) {
        valid_descs.addDesc(src.buffers[i].getBlobDesc());
    }
    nixl_reg_dlist_t invalid_descs(DRAM_SEG);
    invalid_descs.addDesc(src.buffers.back().getBlobDesc());

    ASSERT_EQ(src.agent->getLocalPartialMD(valid_descs, md, nullptr), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, invalid_descs.trim()), NIXL_ERR_NOT_FOUND);

    ASSERT_EQ(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);

    // Step 3: Get and load again but with extra params

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(src.backend_handle);
    extra_params.includeConnInfo = true;

    ASSERT_EQ(src.agent->getLocalPartialMD(valid_descs, md, &extra_params), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, invalid_descs.trim()), NIXL_ERR_NOT_FOUND);

    // Step 4: add the last buffer to the valid descriptors

    valid_descs.addDesc(src.buffers.back().getBlobDesc());

    ASSERT_EQ(src.agent->getLocalPartialMD(valid_descs, md, nullptr), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, GetLocalPartialWithErrors) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    src.initDefault();

    std::string remote_name;
    nixl_blob_t md;

    // Case 1: Use unregistered descriptors
    MemBuffer unregistered_buffer(1024);
    nixl_reg_dlist_t unregistered_descs(DRAM_SEG);
    unregistered_descs.addDesc(unregistered_buffer.getBlobDesc());

    {
        const LogIgnoreGuard lig("getLocalPartialMD: serialization failed");

        ASSERT_NE(src.agent->getLocalPartialMD(unregistered_descs, md, nullptr), NIXL_SUCCESS);

        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }

    // Case 2: Attempt to load connection info on agent without backend
    ASSERT_EQ(src.agent->getLocalPartialMD({DRAM_SEG}, md, nullptr), NIXL_SUCCESS);

    // Agent 1 has no backend
    {
        const LogIgnoreGuard lig("loadRemoteMD: no common backend found");

        ASSERT_NE(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);

        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }

    // Case 3: Attempt to load metadata without connection info

    dst.initDefault();

    nixl_reg_dlist_t valid_descs(DRAM_SEG);
    for (const auto& buffer : src.buffers) {
        valid_descs.addDesc(buffer.getBlobDesc());
    }

    ASSERT_EQ(src.agent->getLocalPartialMD(valid_descs, md, nullptr), NIXL_SUCCESS);

    // Agent 1 has no connection info of agent 0
    {
        const LogIgnoreGuard lig("loadRemoteMD: error loading remote metadata for agent 'agent_0' "
                                 "with status NIXL_ERR_NOT_FOUND");

        ASSERT_NE(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);

        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }

    // Case 4: Attempt to reload connection info with changed metadata

    md.clear();
    ASSERT_EQ(src.agent->getLocalPartialMD({DRAM_SEG}, md, nullptr), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    // Change the metadata before loading
    {
        const LogIgnoreGuard lig("loadRemoteMD: error loading connection info for backend 'UCX' "
                                 "with status NIXL_ERR_NOT_ALLOWED");

        md[100] += 1;
        ASSERT_NE(dst.agent->loadRemoteMD(md, remote_name), NIXL_SUCCESS);

        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }
}

TEST_F(MetadataExchangeTestFixture, SocketSendLocalAndInvalidateLocal) {
    initAgentsDefault();

    auto &src = agents_[0];
    auto &dst = agents_[1];

    nixl_blob_t md;

    nixl_opt_args_t send_args;
    send_args.ipAddr = dst.ip;
    send_args.port = dst.port;

    ASSERT_EQ(src.agent->sendLocalMD(&send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    ASSERT_EQ(src.agent->invalidateLocalMD(&send_args), NIXL_SUCCESS);

    // Send to invalid IP address, should not block the test
    const std::string ip_str = "10.10.10.10";
    const uint16_t port = 1234;
    const std::string port_str = std::to_string(port);
    send_args.ipAddr = ip_str;
    send_args.port = port;
    {
        const LogIgnoreGuard lig1("poll timed out for ip_addr: " + ip_str +
                                  " and port: " + port_str);
        const LogIgnoreGuard lig2("Listener thread could not connect to IP " + ip_str +
                                  " and port " + port_str);
        const LogIgnoreGuard lig3("getsockopt gave error for ip_addr: " + ip_str +
                                  " and port: " + port_str + ": No route to host");

        ASSERT_EQ(src.agent->sendLocalMD(&send_args), NIXL_SUCCESS);

        std::this_thread::sleep_for(std::chrono::seconds(3)); // Must exceed timeout to catch logs.

        const size_t ignored =
            lig1.getIgnoredCount() + lig2.getIgnoredCount() + lig3.getIgnoredCount();
        EXPECT_GE(ignored, 1);
    }

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_ERR_NOT_FOUND);
}

TEST_F(MetadataExchangeTestFixture, SocketFetchRemoteAndInvalidateLocal) {
    initAgentsDefault();

    auto &src = agents_[0];
    auto &dst = agents_[1];

    auto sleep_time = std::chrono::seconds(1);
    nixl_blob_t md;

    nixl_opt_args_t fetch_args;
    fetch_args.ipAddr = src.ip;
    fetch_args.port = src.port;

    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &fetch_args), NIXL_SUCCESS);
    std::this_thread::sleep_for(sleep_time);
    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    nixl_opt_args_t invalidate_args;
    invalidate_args.ipAddr = dst.ip;
    invalidate_args.port = dst.port;

    ASSERT_EQ(src.agent->invalidateLocalMD(&invalidate_args), NIXL_SUCCESS);
    std::this_thread::sleep_for(sleep_time);
    ASSERT_NE(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, SocketSendPartialLocal) {
    initAgentsDefault();

    auto &src = agents_[0];
    auto &dst = agents_[1];

    auto sleep_time = std::chrono::seconds(1);
    nixl_blob_t md;

    nixl_opt_args_t send_args;
    send_args.ipAddr = dst.ip;
    send_args.port = dst.port;

    // Step 1: Get and load connection info

    ASSERT_EQ(src.agent->sendLocalPartialMD({DRAM_SEG}, &send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    // Step 2: Get partial metadata for agent 0 buffers except the last one

    nixl_reg_dlist_t valid_descs(DRAM_SEG);
    for (size_t i = 0; i < src.buffers.size() - 1; i++) {
        valid_descs.addDesc(src.buffers[i].getBlobDesc());
    }
    nixl_reg_dlist_t invalid_descs(DRAM_SEG);
    invalid_descs.addDesc(src.buffers.back().getBlobDesc());

    ASSERT_EQ(src.agent->sendLocalPartialMD(valid_descs, &send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, invalid_descs.trim()), NIXL_ERR_NOT_FOUND);

    ASSERT_EQ(src.agent->invalidateLocalMD(&send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    // Step 3: Get and load again but with additional extra params

    send_args.backends.push_back(src.backend_handle);
    send_args.includeConnInfo = true;

    ASSERT_EQ(src.agent->sendLocalPartialMD(valid_descs, &send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, invalid_descs.trim()), NIXL_ERR_NOT_FOUND);
}

TEST_F(MetadataExchangeTestFixture, SocketSendLocalPartialWithErrors) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    src.initDefault();

    nixl_blob_t md;

    nixl_opt_args_t send_args;
    send_args.ipAddr = dst.ip;
    send_args.port = dst.port;

    // Case 1: Use unregistered descriptors
    MemBuffer unregistered_buffer(1024);
    nixl_reg_dlist_t unregistered_descs(DRAM_SEG);
    unregistered_descs.addDesc(unregistered_buffer.getBlobDesc());

    {
        const LogIgnoreGuard lig1("getLocalPartialMD: serialization failed");
        const LogIgnoreGuard lig2("sendLocalPartialMD: error getting local partial metadata with "
                                  "status NIXL_ERR_NOT_FOUND");

        ASSERT_NE(src.agent->sendLocalPartialMD(unregistered_descs, &send_args), NIXL_SUCCESS);

        EXPECT_EQ(lig1.getIgnoredCount(), 1);
        EXPECT_EQ(lig2.getIgnoredCount(), 1);
    }

    // Case 2: Attempt to load connection info on agent without backend
    {
        const LogIgnoreGuard lig1("loadRemoteMD: no common backend found");
        const LogIgnoreGuard lig2(std::regex("loadRemoteMD in listener thread failed for md from "
                                             "peer 127.0.0.1:[0-9]+ with error NIXL_ERR_BACKEND"));

        ASSERT_EQ(src.agent->sendLocalPartialMD({DRAM_SEG}, &send_args), NIXL_SUCCESS);

        std::this_thread::sleep_for(std::chrono::seconds(1));

        EXPECT_EQ(lig1.getIgnoredCount(), 1);
        EXPECT_EQ(lig2.getIgnoredCount(), 1);
    }

    // Agent 1 has no backend

    ASSERT_NE(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, LocalNonLocalMDExchange) {
    auto &src = agents_[0];
    auto &dst = agents_[1];

    nixlBackendH *backend;
    nixl_status_t status;
    std::string backend_name;
    for (const auto& name : std::set<std::string>{"GDS", "POSIX"}) {
        const LogIgnoreGuard lig1("Error initializing GPU Direct Storage driver");
        const LogIgnoreGuard lig2("createBackend: backend initialization error for 'GDS'");
        status = src.agent->createBackend(name, {}, backend);
        if (status == NIXL_SUCCESS) {
            backend_name = name;
            break;
        }
    }

    if (status != NIXL_SUCCESS) {
        GTEST_SKIP() << "No local-only backend found";
    }

    ASSERT_EQ(NIXL_SUCCESS, dst.agent->createBackend(backend_name, {}, backend));

    initAgentsDefault();

    std::string meta, remote_name;
    ASSERT_EQ(NIXL_SUCCESS, src.agent->getLocalMD(meta));
    ASSERT_EQ(NIXL_SUCCESS, dst.agent->loadRemoteMD(meta, remote_name));
    ASSERT_EQ("agent_0", remote_name);
}

TEST_F(MetadataExchangeTestFixture, EtcdSendLocalAndFetchRemote) {
    VERIFY_ETCD_MODE();
    initAgentsDefault();

    auto &src = agents_[0];
    auto &dst = agents_[1];

    auto sleep_time = std::chrono::seconds(10);
    nixl_blob_t md;

    {
        // Expected due to failure of checkRemoteMd() below?
        const LogIgnoreGuard lig1(
            std::regex("Watch timed out for key: /nixl/cpp_ci/[0-9]+/agent_0/metadata"));
        const LogIgnoreGuard lig2("Failed to fetch metadata from etcd: NIXL_ERR_BACKEND");

        ASSERT_EQ(dst.agent->fetchRemoteMD(src.name), NIXL_SUCCESS);

        std::this_thread::sleep_for(sleep_time);

        EXPECT_EQ(lig1.getIgnoredCount(), 1);
        EXPECT_EQ(lig2.getIgnoredCount(), 1);
    }
    ASSERT_NE(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    ASSERT_EQ(src.agent->sendLocalMD(), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    ASSERT_EQ(src.agent->invalidateLocalMD(), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_ERR_NOT_FOUND);

    ASSERT_NE(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);

    // Fetch invalid agent name. This should not block the commWorker thread forever
    {
        const LogIgnoreGuard lig1(
            std::regex("Watch timed out for key: /nixl/cpp_ci/[0-9]+/invalid_agent_name/metadata"));
        const LogIgnoreGuard lig2("Failed to fetch metadata from etcd: NIXL_ERR_BACKEND");

        ASSERT_EQ(dst.agent->fetchRemoteMD("invalid_agent_name"), NIXL_SUCCESS);

        // Sleep to make sure commWorker started handling the fetch before exiting
        std::this_thread::sleep_for(sleep_time);

        EXPECT_EQ(lig1.getIgnoredCount(), 1);
        EXPECT_EQ(lig2.getIgnoredCount(), 1);
    }

    // Prevent invalidateLocalMD() from begin called again in TearDown()
    // (which would generate more undesired warning/error log messages).
    src.agent.reset();
}

TEST_F(MetadataExchangeTestFixture, EtcdSendLocalPartialAndFetchRemote) {
    VERIFY_ETCD_MODE();
    initAgentsDefault();

    auto &src = agents_[0];
    auto &dst = agents_[1];

    auto sleep_time = std::chrono::seconds(10);
    nixl_blob_t md;

    nixl_opt_args_t send_args;
    send_args.metadataLabel = "conn_info";

    // Step 1: Get and load connection info

    ASSERT_EQ(src.agent->sendLocalPartialMD({DRAM_SEG}, &send_args), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_SUCCESS);

    // Step 2: Send partial metadata for agent 0 buffers except the last one

    send_args.metadataLabel = "first_partial";

    nixl_reg_dlist_t valid_descs(DRAM_SEG);
    for (size_t i = 0; i < src.buffers.size() - 1; i++) {
        valid_descs.addDesc(src.buffers[i].getBlobDesc());
    }
    nixl_reg_dlist_t invalid_descs(DRAM_SEG);
    invalid_descs.addDesc(src.buffers.back().getBlobDesc());

    ASSERT_EQ(src.agent->sendLocalPartialMD(valid_descs, &send_args), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, invalid_descs.trim()), NIXL_ERR_NOT_FOUND);

    ASSERT_EQ(dst.agent->invalidateRemoteMD(src.name), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_ERR_NOT_FOUND);

    // Step 3: Send and fetch again but with additional extra params

    send_args.metadataLabel = "second_partial";
    send_args.backends.push_back(src.backend_handle);
    send_args.includeConnInfo = true;

    ASSERT_EQ(src.agent->sendLocalPartialMD(valid_descs, &send_args), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, invalid_descs.trim()), NIXL_ERR_NOT_FOUND);

    // Step 4: add the last buffer to the valid descriptors

    send_args.metadataLabel = "third_partial";
    send_args.backends.clear();
    send_args.includeConnInfo = false;

    valid_descs.addDesc(src.buffers.back().getBlobDesc());

    ASSERT_EQ(src.agent->sendLocalPartialMD(valid_descs, &send_args), NIXL_SUCCESS);

    ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &send_args), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_SUCCESS);

    // Step 5: Invalidate local metadata

    ASSERT_EQ(src.agent->invalidateLocalMD(nullptr), NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, valid_descs.trim()), NIXL_ERR_NOT_FOUND);

    // Prevent invalidateLocalMD() from begin called again in TearDown()
    // (which would generate more undesired warning/error log messages).
    src.agent.reset();
}

TEST_F(MetadataExchangeTestFixture, EtcdSendLocalPartialAndFetchRemoteWithErrors) {
    VERIFY_ETCD_MODE();

    auto &src = agents_[0];
    auto &dst = agents_[1];

    src.initDefault();

    auto sleep_time = std::chrono::seconds(10);
    nixl_blob_t md;
    nixl_opt_args_t send_args;

    // Case 1: Send without label
    {
        const LogIgnoreGuard lig("sendLocalPartialMD: metadata label is required for etcd send of "
                                 "local partial metadata");

        ASSERT_NE(src.agent->sendLocalPartialMD({DRAM_SEG}, nullptr), NIXL_SUCCESS);

        ASSERT_NE(src.agent->sendLocalPartialMD({DRAM_SEG}, &send_args), NIXL_SUCCESS);

        EXPECT_EQ(lig.getIgnoredCount(), 2);
    }

    // Case 2: Fetch without backend (currently only prints error)
    send_args.metadataLabel = "conn_info";

    ASSERT_EQ(src.agent->sendLocalPartialMD({DRAM_SEG}, &send_args), NIXL_SUCCESS);

    {
        const LogIgnoreGuard lig1(
            std::regex("Watch timed out for key: /nixl/cpp_ci/[0-9]+/agent_0/metadata"));
        const LogIgnoreGuard lig2("Failed to fetch metadata from etcd: NIXL_ERR_BACKEND");

        nixl_opt_args_t fetch_args;
        ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &fetch_args), NIXL_SUCCESS);

        std::this_thread::sleep_for(sleep_time);

        EXPECT_EQ(lig1.getIgnoredCount(), 1);
        EXPECT_EQ(lig2.getIgnoredCount(), 1);
    }

    ASSERT_EQ(dst.agent->checkRemoteMD(src.name, {DRAM_SEG}), NIXL_ERR_NOT_FOUND);

    // Case 3: Fetch with invalid label (should not block the test)
    {
        const LogIgnoreGuard lig1(
            std::regex("Watch timed out for key: /nixl/cpp_ci/[0-9]+/agent_0/invalid_label"));
        const LogIgnoreGuard lig2("Failed to fetch metadata from etcd: NIXL_ERR_BACKEND");

        nixl_opt_args_t fetch_args;
        fetch_args.metadataLabel = "invalid_label";
        ASSERT_EQ(dst.agent->fetchRemoteMD(src.name, &fetch_args), NIXL_SUCCESS);

        std::this_thread::sleep_for(sleep_time);

        EXPECT_EQ(lig1.getIgnoredCount(), 1);
        EXPECT_EQ(lig2.getIgnoredCount(), 1);
    }
}

} // namespace metadata_exchange
} // namespace gtest
