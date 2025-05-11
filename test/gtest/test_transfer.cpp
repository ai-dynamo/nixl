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

#include "common.h"
#include "gtest/gtest.h"

#include "nixl.h"
#include "nixl_types.h"

#include <gtest/gtest.h>
#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <memory>
#include <string>
#include <vector>

namespace gtest {

class MemBuffer : std::shared_ptr<void> {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type = DRAM_SEG) :
        std::shared_ptr<void>(allocate(size, mem_type),
                              [&mem_type](void *ptr) {
                                  release(ptr, mem_type);
                              }),
        size(size)
    {
    }

    operator uintptr_t() const
    {
        return reinterpret_cast<uintptr_t>(get());
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
};

class TestTransfer : public testing::TestWithParam<std::string> {
protected:
    static nixlAgentConfig getConfig()
    {
        return nixlAgentConfig(false, false, 0,
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 0,
                               100000);
    }

    void SetUp() override
    {
        // Create two agents
        for (size_t i = 0; i < 2; i++) {
            auto &agent = agents.emplace_back(getAgentName(i), getConfig());
            nixlBackendH *backend_handle = nullptr;
            nixl_status_t status = agent.createBackend(getBackendName(), {},
                                                       backend_handle);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(backend_handle, nullptr);
        }
    }

    void TearDown() override
    {
        agents.clear();
    }

    std::string getBackendName() const
    {
        return GetParam();
    }

    template<typename Desc>
    nixlDescList<Desc>
    makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type)
    {
        nixlDescList<Desc> desc_list(mem_type);
        for (const auto &buffer : buffers) {
            desc_list.addDesc(Desc(buffer, buffer.getSize(), DEV_ID));
        }
        return desc_list;
    }

    void registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers,
                     nixl_mem_t mem_type)
    {
        auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
        agent.registerMem(reg_list);
    }

    void exchangeMD()
    {
        // Connect the existing agents and exchange metadata
        for (size_t i = 0; i < agents.size(); i++) {
            nixl_blob_t md;
            nixl_status_t status = agents[i].getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                std::string remote_agent_name;
                status = agents[j].loadRemoteMD(md, remote_agent_name);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_EQ(remote_agent_name, getAgentName(i));
            }
        }
    }

    void invalidateMD()
    {
        // Disconnect the agents and invalidate remote metadata
        for (size_t i = 0; i < agents.size(); i++) {
            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                nixl_status_t status = agents[j].invalidateRemoteMD(
                        getAgentName(i));
                ASSERT_EQ(status, NIXL_SUCCESS);
            }
        }
    }

    void waitForXfer(nixlAgent &from, const std::string &from_name,
                     nixlAgent &to, nixlXferReqH *xfer_req)
    {
        nixl_notifs_t notif_map;
        bool xfer_done;
        do {
            // progress on "from" agent while waiting for notification
            nixl_status_t status = from.getXferStatus(xfer_req);
            EXPECT_TRUE((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG));
            xfer_done = (status == NIXL_SUCCESS);

            // Get notifications and progress all agents to avoid deadlocks
            status = to.getNotifs(notif_map);
            ASSERT_EQ(status, NIXL_SUCCESS);
        } while (notif_map.empty() || !xfer_done);

        // Expect the notification from the right agent
        auto &notif_list = notif_map[from_name];
        EXPECT_EQ(notif_list.size(), 1u);
        EXPECT_EQ(notif_list.front(), NOTIF_MSG);
    }

    void doTransfer(nixlAgent &from, const std::string &from_name,
                    nixlAgent &to, const std::string &to_name, size_t size,
                    size_t count, size_t repeat, nixl_mem_t src_mem_type,
                    nixl_mem_t dst_mem_type)
    {
        std::vector<MemBuffer> src_buffers, dst_buffers;
        for (size_t i = 0; i < count; i++) {
            src_buffers.emplace_back(size, src_mem_type);
            dst_buffers.emplace_back(size, dst_mem_type);
        }

        registerMem(from, src_buffers, src_mem_type);
        registerMem(to, dst_buffers, dst_mem_type);
        exchangeMD();

        nixl_opt_args_t extra_params;
        extra_params.hasNotif = true;
        extra_params.notifMsg = NOTIF_MSG;

        nixlXferReqH *xfer_req = nullptr;
        nixl_status_t status   = from.createXferReq(
                NIXL_WRITE,
                makeDescList<nixlBasicDesc>(src_buffers, src_mem_type),
                makeDescList<nixlBasicDesc>(dst_buffers, dst_mem_type), to_name,
                xfer_req, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(xfer_req, nullptr);

        auto start_time = absl::Now();
        for (size_t i = 0; i < repeat; i++) {
            status = from.postXferReq(xfer_req);
            ASSERT_GE(status, NIXL_SUCCESS);

            waitForXfer(from, from_name, to, xfer_req);

            status = from.getXferStatus(xfer_req);
            EXPECT_EQ(status, NIXL_SUCCESS);
        }
        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);

        auto total_size = size * count * repeat;
        auto bandwidth  = total_size / total_time / (1024 * 1024 * 1024);
        Logger() << size << "x" << count << "x" << repeat << "=" << total_size
                 << " bytes in " << total_time << " seconds "
                 << "(" << bandwidth << " GB/s)";

        status = from.releaseXferReq(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        invalidateMD();
    }

    nixlAgent &getAgent(size_t idx)
    {
        return agents[idx];
    }

    std::string getAgentName(size_t idx)
    {
        return absl::StrFormat("agent_%d", idx);
    }

private:
    static constexpr uint64_t DEV_ID = 0;
    static const std::string NOTIF_MSG;

    std::vector<nixlAgent> agents;
};

const std::string TestTransfer::NOTIF_MSG = "notification";

TEST_P(TestTransfer, RandomSizes)
{
    // Tuple fields are: size, count, repeat
    constexpr std::array<std::tuple<size_t, size_t, size_t>, 3> test_cases = {
        {{4096, 8, 3},
         {32768, 64, 3},
         {1000000, 100, 3}}
    };

    for (const auto &[size, count, repeat] : test_cases) {
        doTransfer(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
                   size, count, repeat, DRAM_SEG, DRAM_SEG);
    }
}

INSTANTIATE_TEST_SUITE_P(ucx, TestTransfer, testing::Values("UCX"));

} // namespace gtest
