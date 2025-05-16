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
#include <thread>
#include "nixl.h"
#include "common.h"

namespace gtest {
namespace metadata_exchange {

class MemBuffer : std::shared_ptr<void> {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type = DRAM_SEG) :
        std::shared_ptr<void>(allocate(size, mem_type),
                              [mem_type](void *ptr) {
                                  release(ptr, mem_type);
                              }),
        size(size),
        dev_id(0),
        mem_type(mem_type)
    {
    }

    operator uintptr_t() const
    {
        return reinterpret_cast<uintptr_t>(get());
    }

    nixlBasicDesc getBasicDesc() const
    {
        return nixlBasicDesc(static_cast<uintptr_t>(*this), size, dev_id);
    }

    nixlBlobDesc getBlobDesc() const
    {
        return nixlBlobDesc(getBasicDesc(), "");
    }

    size_t getSize() const
    {
        return size;
    }

private:
    static void *allocate(size_t size, nixl_mem_t mem_type)
    {
        switch (mem_type) {
        case DRAM_SEG:
            return malloc(size);
        default:
            return nullptr; // TODO
        }
    }

    static void release(void *ptr, nixl_mem_t mem_type)
    {
        switch (mem_type) {
        case DRAM_SEG:
            free(ptr);
            break;
        default:
            return; // TODO
        }
    }

    const size_t size;
    const uint64_t dev_id;
    const nixl_mem_t mem_type;
};

class MetadataExchangeTestFixture : public testing::Test {

    struct AgentContext {
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
            nixl_status_t status = agent->createBackend("UCX", {}, backend_handle);
            ASSERT_EQ(status, NIXL_SUCCESS);
            ASSERT_NE(backend_handle, nullptr);
        }

        void initAndRegisterBuffers(size_t count, size_t size)
        {
            constexpr nixl_mem_t mem_type = DRAM_SEG;

            for (size_t i = 0; i < count; i++) {
                buffers.emplace_back(size, mem_type);
            }

            nixl_reg_dlist_t dlist(mem_type);
            for (const auto &buffer : buffers) {
                dlist.addDesc(buffer.getBlobDesc());
            }

            nixl_status_t status = agent->registerMem(dlist);
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
    };

protected:

    void SetUp() override
    {
        // Get random port between 10000 and 65535
        int port_base = 10000 + (std::rand() % (65535 - 10000 + 1));

        // Create two agents
        for (int i = 0; i < 2; i++) {
            int port = port_base + i;
            std::string name = "agent_" + std::to_string(i);
            nixlAgentConfig cfg(false, true, port, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);

            auto agent = std::make_unique<nixlAgent>(name, cfg);

            agents.emplace_back(std::move(agent), std::move(name), port);
        }
    }

    void TearDown() override
    {
        agents.clear();
    }

    std::vector<AgentContext> agents;
};

TEST_F(MetadataExchangeTestFixture, GetLocalAndLoadRemote)
{
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].createAgentBackend();

    size_t count = (std::rand() % 10) + 2;
    size_t size = (std::rand() % 1024) + 1;
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].initAndRegisterBuffers(count, size);

    nixl_xfer_dlist_t dlist(DRAM_SEG);
    for (const auto &buffer : agents[1].buffers) {
        dlist.addDesc(buffer.getBasicDesc());
    }

    std::string remote_name;
    nixl_status_t status;
    nixl_blob_t md;

    auto &src = agents[1];
    auto &dst = agents[0];

    status = src.agent->getLocalMD(md);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->loadRemoteMD(md, remote_name);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    status = dst.agent->checkRemoteMD(src.name, dlist);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Invalidate
    status = dst.agent->invalidateRemoteMD(src.name);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->checkRemoteMD(src.name, dlist);
    ASSERT_EQ(status, NIXL_ERR_NOT_FOUND);

    // Remote does not exist so cannot invalidate
    status = dst.agent->invalidateRemoteMD(src.name);
    ASSERT_NE(status, NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, LoadRemoteWithErrors)
{
    auto &src = agents[0];
    auto &dst = agents[1];

    src.createAgentBackend();
    src.initAndRegisterBuffers(3, 1024);

    std::string remote_name;
    nixl_status_t status;
    nixl_blob_t md;

    status = src.agent->getLocalMD(md);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // No backend on dst agent
    status = dst.agent->loadRemoteMD(md, remote_name);
    ASSERT_NE(status, NIXL_SUCCESS);

    status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    ASSERT_NE(status, NIXL_SUCCESS);

    dst.createAgentBackend();

    // Invalid metadata
    status = dst.agent->loadRemoteMD("invalid", remote_name);
    ASSERT_NE(status, NIXL_SUCCESS);

    // Remote does not exist so cannot invalidate
    status = dst.agent->invalidateRemoteMD(src.name);
    ASSERT_NE(status, NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, GetLocalPartialAndLoadRemote)
{
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].createAgentBackend();

    size_t count = (std::rand() % 10) + 2;
    size_t size = (std::rand() % 1024) + 1;
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].initAndRegisterBuffers(count, size);

    auto &src = agents[0];
    auto &dst = agents[1];

    std::string remote_name;
    nixl_status_t status;
    nixl_blob_t md;

    // Step 1: Get and load connection info

    status = src.agent->getLocalPartialMD({DRAM_SEG}, md, nullptr);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->loadRemoteMD(md, remote_name);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Step 2: Get partial metadata for agent 0 buffers except the last one

    nixl_reg_dlist_t valid_descs(DRAM_SEG);
    for (size_t i = 0; i < src.buffers.size() - 1; i++) {
        valid_descs.addDesc(src.buffers[i].getBlobDesc());
    }
    nixl_reg_dlist_t invalid_descs(DRAM_SEG);
    invalid_descs.addDesc(src.buffers.back().getBlobDesc());

    status = src.agent->getLocalPartialMD(valid_descs, md, nullptr);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->loadRemoteMD(md, remote_name);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    status = dst.agent->checkRemoteMD(src.name, valid_descs.trim());
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->checkRemoteMD(src.name, invalid_descs.trim());
    ASSERT_EQ(status, NIXL_ERR_NOT_FOUND);

    status = dst.agent->invalidateRemoteMD(src.name);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Step 3: Get and load again but with extra params

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(src.backend_handle);
    extra_params.includeConnInfo = true;

    status = src.agent->getLocalPartialMD(valid_descs, md, &extra_params);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->loadRemoteMD(md, remote_name);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(remote_name, src.name);

    status = dst.agent->checkRemoteMD(src.name, valid_descs.trim());
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->checkRemoteMD(src.name, invalid_descs.trim());
    ASSERT_EQ(status, NIXL_ERR_NOT_FOUND);
}

TEST_F(MetadataExchangeTestFixture, GetLocalPartialWithErrors)
{
    auto &src = agents[0];
    auto &dst = agents[1];

    src.createAgentBackend();

    size_t count = (std::rand() % 10) + 1;
    size_t size = (std::rand() % 1024) + 1;
    src.initAndRegisterBuffers(count, size);

    std::string remote_name;
    nixl_status_t status;
    nixl_blob_t md;

    // Case 1: Use unregistered descriptors
    MemBuffer unregistered_buffer(1024, DRAM_SEG);
    nixl_reg_dlist_t unregistered_descs(DRAM_SEG);
    unregistered_descs.addDesc(unregistered_buffer.getBlobDesc());

    status = src.agent->getLocalPartialMD(unregistered_descs, md, nullptr);
    ASSERT_NE(status, NIXL_SUCCESS);

    // Case 2: Attempt to load connection info on agent without backend

    status = src.agent->getLocalPartialMD({DRAM_SEG}, md, nullptr);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Agent 1 has no backend
    status = dst.agent->loadRemoteMD(md, remote_name);
    ASSERT_NE(status, NIXL_SUCCESS);

    // Case 3: Attempt to load metadata without connection info

    dst.createAgentBackend();

    nixl_reg_dlist_t valid_descs(DRAM_SEG);
    for (const auto& buffer : src.buffers) {
        valid_descs.addDesc(buffer.getBlobDesc());
    }

    status = src.agent->getLocalPartialMD(valid_descs, md, nullptr);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Agent 1 has no backend
    status = dst.agent->loadRemoteMD(md, remote_name);
    ASSERT_NE(status, NIXL_SUCCESS);
}

TEST_F(MetadataExchangeTestFixture, SocketSendLocalAndInvalidateLocal)
{
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].createAgentBackend();

    size_t count = (std::rand() % 10) + 2;
    size_t size = (std::rand() % 1024) + 1;
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].initAndRegisterBuffers(count, size);

    auto &src = agents[0];
    auto &dst = agents[1];

    auto sleep_time = std::chrono::milliseconds(500);
    nixl_status_t status;
    nixl_blob_t md;

    nixl_opt_args_t send_args;
    send_args.ipAddr = dst.ip;
    send_args.port = dst.port;

    status = src.agent->sendLocalMD(&send_args);
    ASSERT_EQ(status, NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = src.agent->invalidateLocalMD(&send_args);
    ASSERT_EQ(status, NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    ASSERT_EQ(status, NIXL_ERR_NOT_FOUND);
}

TEST_F(MetadataExchangeTestFixture, SocketFetchRemoteAndInvalidateLocal)
{
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].createAgentBackend();

    size_t count = (std::rand() % 10) + 2;
    size_t size = (std::rand() % 1024) + 1;
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].initAndRegisterBuffers(count, size);

    auto &src = agents[0];
    auto &dst = agents[1];

    auto sleep_time = std::chrono::milliseconds(500);
    nixl_status_t status;
    nixl_blob_t md;

    nixl_opt_args_t fetch_args;
    fetch_args.ipAddr = src.ip;
    fetch_args.port = src.port;

    status = dst.agent->fetchRemoteMD(src.name, &fetch_args);
    ASSERT_EQ(status, NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    ASSERT_EQ(status, NIXL_SUCCESS);

    // TODO: support invalidate local after remote fetch? (currently raises exception)

    // nixl_opt_args_t send_args;
    // send_args.ipAddr = dst.ip;
    // send_args.port = dst.port;

    // status = src.agent->invalidateLocalMD(&send_args);
    // ASSERT_EQ(status, NIXL_SUCCESS);

    // std::this_thread::sleep_for(sleep_time);

    // status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    // ASSERT_EQ(status, NIXL_ERR_NOT_FOUND);
}

TEST_F(MetadataExchangeTestFixture, SocketSendPartialLocal)
{
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].createAgentBackend();

    size_t count = (std::rand() % 10) + 2;
    size_t size = (std::rand() % 1024) + 1;
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].initAndRegisterBuffers(count, size);

    auto &src = agents[0];
    auto &dst = agents[1];

    auto sleep_time = std::chrono::milliseconds(500);
    nixl_status_t status;
    nixl_blob_t md;

    nixl_opt_args_t send_args;
    send_args.ipAddr = dst.ip;
    send_args.port = dst.port;

    // Step 1: Get and load connection info

    status = src.agent->sendLocalPartialMD({DRAM_SEG}, &send_args);
    ASSERT_EQ(status, NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Step 2: Get partial metadata for agent 0 buffers except the last one

    nixl_reg_dlist_t valid_descs(DRAM_SEG);
    for (size_t i = 0; i < src.buffers.size() - 1; i++) {
        valid_descs.addDesc(src.buffers[i].getBlobDesc());
    }
    nixl_reg_dlist_t invalid_descs(DRAM_SEG);
    invalid_descs.addDesc(src.buffers.back().getBlobDesc());

    status = src.agent->sendLocalPartialMD(valid_descs, &send_args);
    ASSERT_EQ(status, NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    status = dst.agent->checkRemoteMD(src.name, valid_descs.trim());
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->checkRemoteMD(src.name, invalid_descs.trim());
    ASSERT_EQ(status, NIXL_ERR_NOT_FOUND);

    status = src.agent->invalidateLocalMD(&send_args);
    ASSERT_EQ(status, NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    // Step 3: Get and load again but with additional extra params

    send_args.backends.push_back(src.backend_handle);
    send_args.includeConnInfo = true;

    status = src.agent->sendLocalPartialMD(valid_descs, &send_args);
    ASSERT_EQ(status, NIXL_SUCCESS);

    std::this_thread::sleep_for(sleep_time);

    status = dst.agent->checkRemoteMD(src.name, valid_descs.trim());
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = dst.agent->checkRemoteMD(src.name, invalid_descs.trim());
    ASSERT_EQ(status, NIXL_ERR_NOT_FOUND);
}

TEST_F(MetadataExchangeTestFixture, SocketSendLocalPartialWithErrors)
{
    auto &src = agents[0];
    auto &dst = agents[1];

    src.createAgentBackend();

    size_t count = (std::rand() % 10) + 1;
    size_t size = (std::rand() % 1024) + 1;
    src.initAndRegisterBuffers(count, size);

    auto sleep_time = std::chrono::milliseconds(500);
    nixl_status_t status;
    nixl_blob_t md;

    nixl_opt_args_t send_args;
    send_args.ipAddr = dst.ip;
    send_args.port = dst.port;

    // Case 1: Use unregistered descriptors
    MemBuffer unregistered_buffer(1024, DRAM_SEG);
    nixl_reg_dlist_t unregistered_descs(DRAM_SEG);
    unregistered_descs.addDesc(unregistered_buffer.getBlobDesc());

    status = src.agent->sendLocalPartialMD(unregistered_descs, &send_args);
    ASSERT_NE(status, NIXL_SUCCESS);

    // Case 2: Attempt to load connection info on agent without backend
    // TODO: This currently raises exception in commWorker thread

    // status = src.agent->sendLocalPartialMD({DRAM_SEG}, &send_args);
    // ASSERT_EQ(status, NIXL_SUCCESS);

    // std::this_thread::sleep_for(sleep_time);

    // // Agent 1 has no backend
    // status = dst.agent->checkRemoteMD(src.name, {DRAM_SEG});
    // ASSERT_NE(status, NIXL_SUCCESS);
}

} // namespace metadata_exchange
} // namespace gtest
