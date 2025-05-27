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
#include "mem_buffer.h"
#include "gtest/gtest.h"

#include "nixl.h"
#include "nixl_types.h"

#include <gtest/gtest.h>
#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <random>
#include <set>
#include <thread>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gtest {

class TestTransfer : public testing::TestWithParam<std::string> {
protected:
    TestTransfer() : rd(), gen(rd()), distrib() {}

    static nixlAgentConfig
    getConfig(int listen_port) {
        return nixlAgentConfig(true,
                               listen_port > 0,
                               listen_port,
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
                               0,
                               100000);
    }

    static int getPort(int i)
    {
        return 9000 + i;
    }

    nixl_b_params_t getBackendParams()
    {
        nixl_b_params_t params;

        if (getBackendName() == "UCX" || getBackendName() == "UCX_MO") {
            params["num_workers"] = "2";
        }

        return params;
    }

    void SetUp() override
    {
#ifdef HAVE_CUDA
        m_cuda_device = (cudaSetDevice(0) == cudaSuccess);
#endif

        // Create two agents
        for (size_t i = 0; i < 2; i++) {
            agents.emplace_back(std::make_unique<nixlAgent>(getAgentName(i),
                                                            getConfig(getPort(i))));
            nixlBackendH *backend_handle = nullptr;
            nixl_status_t status = agents.back()->createBackend(getBackendName(), getBackendParams(),
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

    static nixl_opt_args_t extra_params_ip(int remote)
    {
        nixl_opt_args_t extra_params;

        extra_params.ipAddr = "127.0.0.1";
        extra_params.port   = getPort(remote);
        return extra_params;
    }

    nixl_status_t fetchRemoteMD(int local = 0, int remote = 1)
    {
        auto extra_params = extra_params_ip(remote);

        return agents[local]->fetchRemoteMD(getAgentName(remote),
                                            &extra_params);
    }

    nixl_status_t checkRemoteMD(int local = 0, int remote = 1)
    {
        nixl_xfer_dlist_t descs(DRAM_SEG);
        return agents[local]->checkRemoteMD(getAgentName(remote), descs);
    }

    template<typename Desc, nixl_mem_t MemType, typename Iter>
    nixlDescList<Desc>
    makeDescList(Iter begin, Iter end) {
        nixlDescList<Desc> desc_list(MemType);
        for (; begin != end; ++begin) {
            desc_list.addDesc(Desc(begin->data(), begin->size(), DEV_ID));
        }
        return desc_list;
    }

    template<nixl_mem_t MemType>
    void registerMem(nixlAgent &agent, const MemBufferVec<MemType> &buffers)
    {
        auto reg_list = makeDescList<nixlBlobDesc, MemType>(buffers.begin(), buffers.end());
        agent.registerMem(reg_list);
    }

    void exchangeMDIP()
    {
        for (size_t i = 0; i < agents.size(); i++) {
            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j) {
                    continue;
                }

                auto status = fetchRemoteMD(i, j);
                ASSERT_EQ(NIXL_SUCCESS, status);
                do {
                    status = checkRemoteMD(i, j);
                } while (status != NIXL_SUCCESS);
            }
        }
    }

    void exchangeMD()
    {
        // Connect the existing agents and exchange metadata
        for (size_t i = 0; i < agents.size(); i++) {
            nixl_blob_t md;
            nixl_status_t status = agents[i]->getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                std::string remote_agent_name;
                status = agents[j]->loadRemoteMD(md, remote_agent_name);
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
                nixl_status_t status = agents[j]->invalidateRemoteMD(
                        getAgentName(i));
                ASSERT_EQ(status, NIXL_SUCCESS);
            }
        }
    }

    void
    waitForXfer(size_t from_idx, nixlXferReqH *xfer_req) {
        auto &from = getAgent(from_idx);
        bool xfer_done;
        do {
            // progress on "from" agent while waiting for completion
            nixl_status_t status = from.getXferStatus(xfer_req);
            EXPECT_TRUE((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG));
            xfer_done = (status == NIXL_SUCCESS);
        } while (!xfer_done);
    }

    template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType, nixl_xfer_op_t Mode>
    std::pair<MemBufferVec2D<LocalMemType>, MemBufferVec2D<RemoteMemType>>
    initThreadBuffers(nixlAgent &from,
                      nixlAgent &to,
                      size_t num_threads,
                      size_t count,
                      size_t size) {
        MemBufferVec2D<LocalMemType> thread_local_buffers(num_threads);
        MemBufferVec2D<RemoteMemType> thread_remote_buffers(num_threads);
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            initBuffersWithRandomData<LocalMemType, RemoteMemType, Mode>(
                    thread_local_buffers[thread_id],
                    thread_remote_buffers[thread_id],
                    count,
                    size,
                    gen,
                    distrib);
            registerMem(from, thread_local_buffers[thread_id]);
            registerMem(to, thread_remote_buffers[thread_id]);
        }
        return {std::move(thread_local_buffers), std::move(thread_remote_buffers)};
    }

    template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType, nixl_xfer_op_t Mode>
    void
    validateThreadBuffers(const MemBufferVec2D<LocalMemType> &thread_local_buffers,
                          const MemBufferVec2D<RemoteMemType> &thread_remote_buffers,
                          size_t num_threads,
                          size_t count) {
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t i = 0; i < count; i++) {
                EXPECT_EQ(thread_local_buffers[thread_id][i], thread_remote_buffers[thread_id][i])
                        << "Transfer validation failed for thread " << thread_id << " buffer " << i;
            }
        }
    }

    std::vector<std::vector<std::string>>
    initThreadNotifications(size_t num_threads, size_t count, size_t batch_size) {
        std::vector<std::vector<std::string>> thread_notifs(num_threads);
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
                size_t batch_idx = batch_start / batch_size;
                std::string notif =
                        absl::StrFormat("notification_thread_%zu_batch_%zu", thread_id, batch_idx);
                thread_notifs[thread_id].push_back(notif);
            }
        }
        return thread_notifs;
    }

    void
    waitForNotifications(size_t to_idx,
                         size_t from_idx,
                         size_t expected_count,
                         nixl_notifs_t &notif_map) {
        auto &to = getAgent(to_idx);
        // Loop to attempt to get all notifications from agent from_name with exponential retry
        // backoff
        for (size_t attempt = 0; attempt < 10; ++attempt) {
            nixl_status_t status = to.getNotifs(notif_map);
            ASSERT_EQ(status, NIXL_SUCCESS);
            auto notifs_from = notif_map.find(getAgentName(from_idx));
            if (notifs_from != notif_map.end() && notifs_from->second.size() == expected_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1 << attempt));
        }
    }

    void
    validateNotifications(const nixl_notifs_t &notif_map,
                          const std::string &from_name,
                          const std::vector<std::vector<std::string>> &thread_notifs) {
        std::set<std::string> expected_msgs;
        for (const auto &thread_notif_list : thread_notifs) {
            expected_msgs.insert(thread_notif_list.begin(), thread_notif_list.end());
        }

        EXPECT_EQ(notif_map.count(from_name), 1)
                << "Expected notifications from " << from_name << " to be in notifications map";
        auto &notif_list = notif_map.at(from_name);
        EXPECT_EQ(notif_list.size(), expected_msgs.size())
                << "Expected " << expected_msgs.size() << " notifications, got "
                << notif_list.size();

        std::set<std::string> remaining_msgs = expected_msgs;
        for (const auto &msg : notif_list) {
            EXPECT_TRUE(remaining_msgs.erase(msg) > 0)
                    << "Unexpected or duplicate notification: " << msg;
        }
        EXPECT_TRUE(remaining_msgs.empty())
                << "Missing " << remaining_msgs.size() << " notifications";
    }

    template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType, nixl_xfer_op_t Mode>
    void
    zeroThreadBuffers(MemBufferVec2D<LocalMemType> &thread_local_buffers,
                      MemBufferVec2D<RemoteMemType> &thread_remote_buffers,
                      size_t num_threads,
                      size_t count) {
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            zeroBuffers<LocalMemType, RemoteMemType, Mode>(
                    thread_local_buffers[thread_id], thread_remote_buffers[thread_id], count);
        }
    }

    template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType, nixl_xfer_op_t Mode>
    void
    doTransfer(size_t from_idx,
               size_t to_idx,
               size_t size,
               size_t count,
               size_t batch_size,
               std::function<void()> setup_md,
               const std::vector<std::string> &expected_notifs,
               const MemBufferVec<LocalMemType> &local_buffers,
               const MemBufferVec<RemoteMemType> &remote_buffers) {
        auto &from = getAgent(from_idx);
        auto start_time = absl::Now();
        size_t total_transferred = 0;
        size_t notif_idx = 0;

        for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, count);

            nixl_opt_args_t extra_params;
            extra_params.hasNotif = true;
            extra_params.notifMsg = expected_notifs[notif_idx++];

            nixlXferReqH *xfer_req = nullptr;
            nixl_status_t status = from.createXferReq(
                    Mode,
                    makeDescList<nixlBasicDesc, LocalMemType>(local_buffers.begin() + batch_start,
                                                              local_buffers.begin() + batch_end),
                    makeDescList<nixlBasicDesc, RemoteMemType>(remote_buffers.begin() + batch_start,
                                                               remote_buffers.begin() + batch_end),
                    getAgentName(to_idx),
                    xfer_req,
                    &extra_params);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(xfer_req, nullptr);

            status = from.postXferReq(xfer_req);
            ASSERT_GE(status, NIXL_SUCCESS);

            waitForXfer(from_idx, xfer_req);

            status = from.getXferStatus(xfer_req);
            EXPECT_EQ(status, NIXL_SUCCESS);

            status = from.releaseXferReq(xfer_req);
            EXPECT_EQ(status, NIXL_SUCCESS);

            total_transferred += (batch_end - batch_start) * size;
        }

        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
        auto bandwidth = total_transferred / total_time / (1024 * 1024 * 1024);
        Logger() << (Mode == NIXL_WRITE ? "Write" : "Read") << " transfer: " << size << "x" << count
                 << "=" << total_transferred << " bytes in " << total_time << " seconds "
                 << "(" << bandwidth << " GB/s)";
    }

    template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType, nixl_xfer_op_t Mode>
    void
    doTransfers(size_t from_idx,
                size_t to_idx,
                size_t size,
                size_t count,
                size_t batch_size,
                size_t repeat,
                size_t num_threads,
                std::function<void()> setup_md) {
        auto &from = getAgent(from_idx);
        auto &to = getAgent(to_idx);
        auto [thread_local_buffers, thread_remote_buffers] =
                initThreadBuffers<LocalMemType, RemoteMemType, Mode>(
                        from, to, num_threads, count, size);
        setup_md();

        auto base_thread_notifs = initThreadNotifications(num_threads, count, batch_size);

        for (size_t repeat_idx = 0; repeat_idx < repeat; ++repeat_idx) {
            zeroThreadBuffers<LocalMemType, RemoteMemType, Mode>(
                    thread_local_buffers, thread_remote_buffers, num_threads, count);

            std::vector<std::thread> threads;
            for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
                threads.emplace_back([&, thread_id]() {
                    doTransfer<LocalMemType, RemoteMemType, Mode>(from_idx,
                                                                  to_idx,
                                                                  size,
                                                                  count,
                                                                  batch_size,
                                                                  setup_md,
                                                                  base_thread_notifs[thread_id],
                                                                  thread_local_buffers[thread_id],
                                                                  thread_remote_buffers[thread_id]);
                });
            }

            for (auto &thread : threads) {
                thread.join();
            }

            validateThreadBuffers<LocalMemType, RemoteMemType, Mode>(
                    thread_local_buffers, thread_remote_buffers, num_threads, count);

            nixl_notifs_t notif_map;
            nixl_status_t status = to.getNotifs(notif_map);
            ASSERT_EQ(status, NIXL_SUCCESS);

            validateNotifications(notif_map, getAgentName(from_idx), base_thread_notifs);
        }

        invalidateMD();
    }

    void
    doNotificationTest(size_t from_idx,
                       size_t to_idx,
                       size_t num_threads,
                       size_t notifications_per_thread,
                       size_t repeat) {
        auto &from = getAgent(from_idx);
        auto thread_notifs = initThreadNotifications(num_threads, notifications_per_thread, 1);

        exchangeMD();

        for (size_t repeat_idx = 0; repeat_idx < repeat; ++repeat_idx) {

            std::vector<std::thread> threads;
            for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
                threads.emplace_back([&, thread_id]() {
                    for (const auto &notif : thread_notifs[thread_id]) {
                        nixl_status_t status = from.genNotif(getAgentName(to_idx), notif);
                        ASSERT_EQ(status, NIXL_SUCCESS);
                    }
                });
            }

            for (auto &thread : threads) {
                thread.join();
            }

            nixl_notifs_t notif_map;
            waitForNotifications(
                    to_idx, from_idx, num_threads * notifications_per_thread, notif_map);
            validateNotifications(notif_map, getAgentName(from_idx), thread_notifs);
        }

        invalidateMD();
    }

    nixlAgent &getAgent(size_t idx)
    {
        return *agents[idx];
    }

    std::string getAgentName(size_t idx)
    {
        return absl::StrFormat("agent_%d", idx);
    }

    bool m_cuda_device = false;

private:
    static constexpr uint64_t DEV_ID = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
    std::random_device rd;
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint64_t> distrib;
};

TEST_P(TestTransfer, RandomSizes) {
    // Tuple fields are: size, count, batch_size, repeat, num_threads
    constexpr std::array<std::tuple<size_t, size_t, size_t, size_t, size_t>, 4> test_cases = {
            {{40, 250, 1, 1, 4}, {4096, 32, 2, 2, 4}, {32768, 16, 2, 2, 4}, {1000000, 8, 1, 2, 4}}};

    for (const auto &[size, count, batch_size, repeat, num_threads] : test_cases) {
        doTransfers<DRAM_SEG, DRAM_SEG, NIXL_WRITE>(
                0, 1, size, count, batch_size, repeat, num_threads, [this]() { exchangeMD(); });
        doTransfers<DRAM_SEG, DRAM_SEG, NIXL_READ>(
                0, 1, size, count, batch_size, repeat, num_threads, [this]() { exchangeMD(); });
    }
}

TEST_P(TestTransfer, remoteMDFromSocket) {
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 4;

    if (m_cuda_device) {
        doTransfers<VRAM_SEG, VRAM_SEG, NIXL_WRITE>(
                0, 1, size, count, 1, 1, 1, [this]() { exchangeMDIP(); });
    } else {
        doTransfers<DRAM_SEG, DRAM_SEG, NIXL_WRITE>(
                0, 1, size, count, 1, 1, 1, [this]() { exchangeMDIP(); });
    }
}

TEST_P(TestTransfer, NotificationOnly) {
    constexpr size_t num_threads = 4;
    constexpr size_t notifications_per_thread = 10000;
    constexpr size_t repeat = 3;

    doNotificationTest(0, 1, num_threads, notifications_per_thread, repeat);
}

INSTANTIATE_TEST_SUITE_P(ucx, TestTransfer, testing::Values("UCX"));
INSTANTIATE_TEST_SUITE_P(ucx_mo, TestTransfer, testing::Values("UCX_MO"));

} // namespace gtest
