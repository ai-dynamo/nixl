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

#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <thread>
#include <mutex>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

constexpr auto min_chrono_time = std::chrono::steady_clock::time_point::min();

namespace gtest {

class MemBuffer : std::shared_ptr<void> {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type = DRAM_SEG) :
        std::shared_ptr<void>(allocate(size, mem_type),
                              [mem_type](void *ptr) {
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
#ifdef HAVE_CUDA
        case VRAM_SEG:
            void *ptr;
            return cudaSuccess == cudaMalloc(&ptr, size)? ptr : nullptr;
#endif
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
#ifdef HAVE_CUDA
        case VRAM_SEG:
            cudaFree(ptr);
            break;
#endif
        default:
            return; // TODO
        }
    }

    const size_t size;
};

class TestTransfer :
    // Tuple fields are: backend_name, enable_progress_thread, num_workers, num_threads
    public testing::TestWithParam<std::tuple<std::string, bool, size_t, size_t>> {
protected:
    nixlAgentConfig
    getConfig(int listen_port, bool capture_telemetry) {
        return nixlAgentConfig(isProgressThreadEnabled(),
                               listen_port > 0,
                               listen_port,
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
                               1,
                               0,
                               100000,
                               capture_telemetry);
    }

    uint16_t
    getPort(int i) const {
        return ports.at(i);
    }

    nixl_b_params_t getBackendParams()
    {
        nixl_b_params_t params;

        if (getBackendName() == "UCX") {
            params["num_workers"] = std::to_string(getNumWorkers());
            params["num_threads"] = std::to_string(getNumThreads());
            params["split_batch_size"] = "32";
        }

        return params;
    }

    void
    addAgent(unsigned int agent_num, bool capture_telemetry = false) {
        ports.push_back(PortAllocator::next_tcp_port());
        agents.emplace_back(std::make_unique<nixlAgent>(
            getAgentName(agent_num), getConfig(getPort(agent_num), capture_telemetry)));
        nixlBackendH *backend_handle = nullptr;
        nixl_status_t status =
            agents.back()->createBackend(getBackendName(), getBackendParams(), backend_handle);
        ASSERT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(backend_handle, nullptr);
        backend_handles.push_back(backend_handle);
    }

    void SetUp() override
    {
#ifdef HAVE_CUDA
        m_cuda_device = (cudaSetDevice(0) == cudaSuccess);
#endif

        // Disabling Telemetry until the corresponding test
        env.addVar("NIXL_TELEMETRY_ENABLE", "n");

        // Create two agents
        for (size_t i = 0; i < 2; i++) {
            addAgent(i);
        }
    }

    void TearDown() override
    {
        agents.clear();
    }

    std::string getBackendName() const
    {
        return std::get<0>(GetParam());
    }

    bool
    isProgressThreadEnabled() const {
        return std::get<1>(GetParam());
    }

    size_t
    getNumWorkers() const {
        return std::get<2>(GetParam());
    }

    size_t
    getNumThreads() const {
        return std::get<3>(GetParam());
    }

    nixl_opt_args_t
    extra_params_ip(int remote) {
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

    template<typename Desc>
    nixlDescList<Desc>
    makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) const {
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

    static bool wait_until_true(std::function<bool()> func, int retries = 500) {
        bool result;

        while (!(result = func()) && retries-- > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        return result;
    }

    void
    exchangeMDIP(size_t start, size_t end) {
        // Exchange metadata for the agents in the specified range using their IP
        for (size_t i = start; i <= end; i++) {
            for (size_t j = start; j <= end; j++) {
                if (i == j) {
                    continue;
                }

                auto status = fetchRemoteMD(i, j);
                ASSERT_EQ(NIXL_SUCCESS, status);
                ASSERT_TRUE(wait_until_true(
                    [&]() { return checkRemoteMD(i, j) == NIXL_SUCCESS; }));
            }
        }
    }

    void
    exchangeMD(size_t start, size_t end) {
        // Exchange metadata for the agents in the specified range
        for (size_t i = start; i <= end; i++) {
            nixl_blob_t md;
            nixl_status_t status = agents[i]->getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (size_t j = start; j <= end; j++) {
                if (i == j)
                    continue;
                std::string remote_agent_name;
                status = agents[j]->loadRemoteMD(md, remote_agent_name);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_EQ(remote_agent_name, getAgentName(i));
            }
        }
    }

    void
    invalidateMD(size_t start, size_t end) {
        // Invalidate each other's metadata for the agents in the specified range
        for (size_t i = start; i <= end; i++) {
            for (size_t j = start; j < end; j++) {
                if (i == j)
                    continue;
                nixl_status_t status = agents[j]->invalidateRemoteMD(
                        getAgentName(i));
                ASSERT_EQ(status, NIXL_SUCCESS);
            }
        }
    }

    void createRegisteredMem(nixlAgent& agent,
                             size_t size, size_t count,
                             nixl_mem_t mem_type,
                             std::vector<MemBuffer>& out)
    {
        while (count-- != 0) {
            out.emplace_back(size, mem_type);
        }

        registerMem(agent, out, mem_type);
    }

    void
    deregisterMem(nixlAgent &agent,
                  const std::vector<MemBuffer> &buffers,
                  nixl_mem_t mem_type) const {
        const auto desc_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
        agent.deregisterMem(desc_list);
    }

    void
    verifyNotifs(nixlAgent &agent,
                 const std::string &from_name,
                 size_t expected_count,
                 nixl_notifs_t notif_map = {}) {
        for (int i = 0; i < retry_count; i++) {
            nixl_status_t status = agent.getNotifs(notif_map);
            ASSERT_EQ(status, NIXL_SUCCESS);

            if (notif_map[from_name].size() >= expected_count) {
                break;
            }
            std::this_thread::sleep_for(retry_timeout);
        }

        auto& notif_list = notif_map[from_name];
        EXPECT_EQ(notif_list.size(), expected_count);

        for (const auto& notif : notif_list) {
            EXPECT_EQ(notif, NOTIF_MSG);
        }
    }

    void
    doNotificationTest(nixlAgent &from,
                       const std::string &from_name,
                       nixlAgent &to,
                       const std::string &to_name,
                       size_t repeat,
                       size_t num_threads) {
        const size_t total_notifs = repeat * num_threads;

        exchangeMD(0, 1);

        std::vector<std::thread> threads;
        nixl_notifs_t notif_map;
        for (size_t thread = 0; thread < num_threads; ++thread) {
            threads.emplace_back([&]() {
                for (size_t i = 0; i < repeat; ++i) {
                    nixl_status_t status = from.genNotif(to_name, NOTIF_MSG);
                    ASSERT_EQ(status, NIXL_SUCCESS);

                    if (!isProgressThreadEnabled()) {
                        ASSERT_EQ(NIXL_SUCCESS, from.getNotifs(notif_map));
                        ASSERT_EQ(NIXL_SUCCESS, to.getNotifs(notif_map));
                    }
                }
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }

        verifyNotifs(to, from_name, total_notifs, std::move(notif_map));
        invalidateMD(0, 1);
    }

    void
    doTransfer(nixlAgent &from,
               const std::string &from_name,
               nixlAgent &to,
               const std::string &to_name,
               size_t size,
               size_t count,
               size_t repeat,
               size_t num_threads,
               nixl_mem_t src_mem_type,
               std::vector<MemBuffer> src_buffers,
               nixl_mem_t dst_mem_type,
               std::vector<MemBuffer> dst_buffers,
               nixl_status_t expected_telem_status = NIXL_ERR_NO_TELEMETRY) {
        std::mutex logger_mutex;
        std::vector<std::thread> threads;
        nixl_notifs_t notif_map;
        for (size_t thread = 0; thread < num_threads; ++thread) {
            threads.emplace_back([&, thread]() {
                nixl_opt_args_t extra_params;
                extra_params.hasNotif = true;
                extra_params.notifMsg = NOTIF_MSG;

                nixlXferReqH *xfer_req = nullptr;
                nixl_status_t status = from.createXferReq(
                        NIXL_WRITE,
                        makeDescList<nixlBasicDesc>(src_buffers, src_mem_type),
                        makeDescList<nixlBasicDesc>(dst_buffers, dst_mem_type), to_name,
                        xfer_req, &extra_params);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_NE(xfer_req, nullptr);

                auto start_time = absl::Now();

                for (size_t i = 0; i < repeat; i++) {
                    status = from.postXferReq(xfer_req);
                    ASSERT_TRUE((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG));

                    for (int i = 0; i < retry_count; i++) {
                        status = from.getXferStatus(xfer_req);
                        EXPECT_TRUE((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG));
                        if (status == NIXL_SUCCESS) {
                            break;
                        }
                        if (!isProgressThreadEnabled()) {
                            ASSERT_EQ(NIXL_SUCCESS, to.getNotifs(notif_map));
                        }
                        std::this_thread::sleep_for(retry_timeout);
                    }
                    EXPECT_TRUE(status == NIXL_SUCCESS);
                }

                auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
                auto total_size = size * count * repeat;
                auto bandwidth  = total_size / total_time / (1024 * 1024 * 1024);
                {
                    const std::lock_guard<std::mutex> lock(logger_mutex);
                    Logger() << "Thread " << thread << ": " << size << "x" << count << "x" << repeat
                             << "=" << total_size << " bytes in " << total_time << " seconds "
                             << "(" << bandwidth << " GB/s)";
                }

                nixl_xfer_telem_t telemetry;
                status = from.getXferTelemetry(xfer_req, telemetry);
                EXPECT_EQ(status, expected_telem_status);
                if (expected_telem_status == NIXL_SUCCESS) {
                    EXPECT_TRUE(telemetry.startTime > min_chrono_time);
                    EXPECT_TRUE(telemetry.postDuration > chrono_period_us_t(0));
                    EXPECT_TRUE(telemetry.xferDuration > chrono_period_us_t(0));
                    EXPECT_TRUE(telemetry.xferDuration >= telemetry.postDuration);
                }

                status = from.releaseXferReq(xfer_req);
                EXPECT_EQ(status, NIXL_SUCCESS);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        verifyNotifs(to, from_name, repeat * num_threads, std::move(notif_map));
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
    gtest::ScopedEnv env;
    std::vector<nixlBackendH *> backend_handles;

private:
    static constexpr uint64_t DEV_ID = 0;
    static const std::string NOTIF_MSG;
    // TODO: with error handling enabled by default we get poor performance with UCX1.18.
    // Before we upgrade to UCX1.19, we need to temporarily increase the retry count,
    // in order to pass threadpool tests.
    // TODO: revert this to 1000 once we upgrade to UCX1.19.
    static constexpr int retry_count{10000};
    static constexpr std::chrono::milliseconds retry_timeout{1};

    std::vector<std::unique_ptr<nixlAgent>> agents;
    std::vector<uint16_t> ports;
};

const std::string TestTransfer::NOTIF_MSG = "notification";

TEST_P(TestTransfer, RandomSizes)
{
    // Tuple fields are: size, count, repeat, num_threads
    constexpr std::array<std::tuple<size_t, size_t, size_t, size_t>, 4> test_cases = {
        {{4096, 8, 3, 1},
         {32768, 64, 3, 2},
         {1000000, 100, 3, 4},
         {40, 1000, 1, 4}}
    };
    constexpr nixl_mem_t mem_type = DRAM_SEG;

    for (const auto &[size, count, repeat, num_threads] : test_cases) {
        std::vector<MemBuffer> src_buffers, dst_buffers;

        createRegisteredMem(getAgent(0), size, count, mem_type, src_buffers);
        createRegisteredMem(getAgent(1), size, count, mem_type, dst_buffers);

        exchangeMD(0, 1);
        doTransfer(getAgent(0),
                   getAgentName(0),
                   getAgent(1),
                   getAgentName(1),
                   size,
                   count,
                   repeat,
                   num_threads,
                   mem_type,
                   src_buffers,
                   mem_type,
                   dst_buffers);
        invalidateMD(0, 1);
        deregisterMem(getAgent(0), src_buffers, mem_type);
        deregisterMem(getAgent(1), dst_buffers, mem_type);
    }
}

TEST_P(TestTransfer, remoteMDFromSocket)
{
    std::vector<MemBuffer> src_buffers, dst_buffers;
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 4;
    nixl_mem_t mem_type = m_cuda_device? VRAM_SEG : DRAM_SEG;

    createRegisteredMem(getAgent(0), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(1), size, count, mem_type, dst_buffers);

    exchangeMDIP(0, 1);
    doTransfer(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
               size, count, 1, 1,
               mem_type, src_buffers,
               mem_type, dst_buffers);

    invalidateMD(0, 1);
    deregisterMem(getAgent(0), src_buffers, mem_type);
    deregisterMem(getAgent(1), dst_buffers, mem_type);
}

TEST_P(TestTransfer, NotificationOnly) {
    constexpr size_t repeat = 100;
    constexpr size_t num_threads = 4;
    doNotificationTest(
            getAgent(0), getAgentName(0), getAgent(1), getAgentName(1), repeat, num_threads);
}

TEST_P(TestTransfer, SelfNotification) {
    constexpr size_t repeat = 100;
    constexpr size_t num_threads = 4;
    doNotificationTest(
            getAgent(0), getAgentName(0), getAgent(0), getAgentName(0), repeat, num_threads);
}

TEST_P(TestTransfer, ListenerCommSize) {
    std::vector<MemBuffer> buffers;
    createRegisteredMem(getAgent(1), 64, 10000, DRAM_SEG, buffers);
    auto status = fetchRemoteMD(0, 1);
    ASSERT_EQ(NIXL_SUCCESS, status);
    ASSERT_TRUE(
        wait_until_true([&]() { return checkRemoteMD(0, 1) == NIXL_SUCCESS; }));
    deregisterMem(getAgent(1), buffers, DRAM_SEG);
}

TEST_P(TestTransfer, GetXferTelemetryFile) {
    env.addVar("NIXL_TELEMETRY_ENABLE", "y");
    env.addVar("NIXL_TELEMETRY_DIR", "/tmp/");

    // Create fresh agents that read the current env var and add them to the fixture
    addAgent(2);
    addAgent(3);

    constexpr size_t size = 1024;
    constexpr size_t count = 1;
    std::vector<MemBuffer> src_buffers, dst_buffers;
    createRegisteredMem(getAgent(2), size, count, DRAM_SEG, src_buffers);
    createRegisteredMem(getAgent(3), size, count, DRAM_SEG, dst_buffers);

    exchangeMD(2, 3);
    doTransfer(getAgent(2),
               getAgentName(2),
               getAgent(3),
               getAgentName(3),
               size,
               count,
               1,
               1,
               DRAM_SEG,
               src_buffers,
               DRAM_SEG,
               dst_buffers,
               NIXL_SUCCESS);

    invalidateMD(2, 3);
    deregisterMem(getAgent(2), src_buffers, DRAM_SEG);
    deregisterMem(getAgent(3), dst_buffers, DRAM_SEG);
}

TEST_P(TestTransfer, GetXferTelemetryAPI) {
    // Enable telemetry without file output
    env.addVar("NIXL_TELEMETRY_ENABLE", "y");

    // Create fresh agents that read the current env var and add them to the fixture
    addAgent(2);
    addAgent(3);

    constexpr size_t size = 1024;
    constexpr size_t count = 1;
    std::vector<MemBuffer> src_buffers, dst_buffers;
    createRegisteredMem(getAgent(2), size, count, DRAM_SEG, src_buffers);
    createRegisteredMem(getAgent(3), size, count, DRAM_SEG, dst_buffers);

    exchangeMD(2, 3);
    doTransfer(getAgent(2),
               getAgentName(2),
               getAgent(3),
               getAgentName(3),
               size,
               count,
               1,
               1,
               DRAM_SEG,
               src_buffers,
               DRAM_SEG,
               dst_buffers,
               NIXL_SUCCESS);

    invalidateMD(2, 3);
    deregisterMem(getAgent(2), src_buffers, DRAM_SEG);
    deregisterMem(getAgent(3), dst_buffers, DRAM_SEG);
}

TEST_P(TestTransfer, GetXferTelemetryAPICfg) {
    // Disable telemetry from env var but through config, expecting a warning
    env.addVar("NIXL_TELEMETRY_ENABLE", "n");

    // Create fresh agents that read the current env var and add them to the fixture
    // with capture_telemetry set
    addAgent(2, true);
    addAgent(3, true);

    constexpr size_t size = 1024;
    constexpr size_t count = 1;
    std::vector<MemBuffer> src_buffers, dst_buffers;
    createRegisteredMem(getAgent(2), size, count, DRAM_SEG, src_buffers);
    createRegisteredMem(getAgent(3), size, count, DRAM_SEG, dst_buffers);

    exchangeMD(2, 3);
    doTransfer(getAgent(2),
               getAgentName(2),
               getAgent(3),
               getAgentName(3),
               size,
               count,
               1,
               1,
               DRAM_SEG,
               src_buffers,
               DRAM_SEG,
               dst_buffers,
               NIXL_SUCCESS);

    invalidateMD(2, 3);
    deregisterMem(getAgent(2), src_buffers, DRAM_SEG);
    deregisterMem(getAgent(3), dst_buffers, DRAM_SEG);
}


TEST_P(TestTransfer, GetXferTelemetryDisabled) {
    env.addVar("NIXL_TELEMETRY_ENABLE", "n");

    // Create fresh agents that read the current env var and add them to the fixture
    addAgent(2);
    addAgent(3);

    constexpr size_t size = 512;
    constexpr size_t count = 1;
    std::vector<MemBuffer> src_buffers, dst_buffers;
    createRegisteredMem(getAgent(2), size, count, DRAM_SEG, src_buffers);
    createRegisteredMem(getAgent(3), size, count, DRAM_SEG, dst_buffers);

    exchangeMD(2, 3);
    doTransfer(getAgent(2),
               getAgentName(2),
               getAgent(3),
               getAgentName(3),
               size,
               count,
               1,
               1,
               DRAM_SEG,
               src_buffers,
               DRAM_SEG,
               dst_buffers,
               NIXL_ERR_NO_TELEMETRY);

    invalidateMD(2, 3);
    deregisterMem(getAgent(2), src_buffers, DRAM_SEG);
    deregisterMem(getAgent(3), dst_buffers, DRAM_SEG);
}

TEST_P(TestTransfer, PrepGpuSignal) {
#ifndef HAVE_UCX_GPU_DEVICE_API
    GTEST_SKIP() << "UCX GPU device API not available, skipping test";
#else
    size_t gpu_signal_size = 0;
    nixl_opt_args_t extra_params = {.backends = {backend_handles[0]}};
    nixl_status_t size_status = getAgent(0).getGpuSignalSize(gpu_signal_size, &extra_params);
    ASSERT_EQ(size_status, NIXL_SUCCESS) << "getGpuSignalSize failed";
    ASSERT_GT(gpu_signal_size, 0) << "GPU signal size is 0";

    // Allocate a buffer on the GPU with the size of the signal
    std::vector<MemBuffer> signal_buffer;
    createRegisteredMem(getAgent(0), gpu_signal_size, 1, VRAM_SEG, signal_buffer);

    auto signal_desc_list = makeDescList<nixlBlobDesc>(signal_buffer, VRAM_SEG);

    nixl_status_t status = getAgent(0).prepGpuSignal(signal_desc_list, &extra_params);

    EXPECT_EQ(status, NIXL_SUCCESS)
        << "prepGpuSignal returned unexpected status: " << nixlEnumStrings::statusStr(status);

    deregisterMem(getAgent(0), signal_buffer, VRAM_SEG);
#endif
}

INSTANTIATE_TEST_SUITE_P(ucx, TestTransfer, testing::Values(std::make_tuple("UCX", true, 2, 0)));
INSTANTIATE_TEST_SUITE_P(ucx_no_pt,
                         TestTransfer,
                         testing::Values(std::make_tuple("UCX", false, 2, 0)));
INSTANTIATE_TEST_SUITE_P(ucx_threadpool,
                         TestTransfer,
                         testing::Values(std::make_tuple("UCX", true, 6, 4)));
INSTANTIATE_TEST_SUITE_P(ucx_threadpool_no_pt,
                         TestTransfer,
                         testing::Values(std::make_tuple("UCX", false, 6, 4)));
} // namespace gtest
