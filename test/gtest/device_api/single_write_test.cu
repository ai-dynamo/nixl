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

#include "utils.cuh"

namespace gtest::nixl::gpu::single_write {

template<nixl_gpu_level_t level>
__global__ void
TestSingleWriteKernel(nixlGpuXferReqH req_hdnl,
                      unsigned index,
                      const void *src_addr,
                      uint64_t remote_addr,
                      size_t size,
                      size_t num_iters,
                      bool is_no_delay,
                      unsigned long long *start_time_ptr,
                      unsigned long long *end_time_ptr) {
    __shared__ nixlGpuXferStatusH xfer_status[MAX_THREADS];
    nixlGpuXferStatusH *xfer_status_ptr = &xfer_status[GetReqIdx<level>()];
    nixl_status_t status;

    assert(GetReqIdx<level>() < MAX_THREADS);

    if (threadIdx.x == 0) {
        unsigned long long start_time = GetTimeNs();
        *start_time_ptr = start_time;
    }

    __syncthreads();

    for (size_t i = 0; i < num_iters; ++i) {
        status = nixlGpuPostSingleWriteXferReq<level>(
            req_hdnl, index, src_addr, remote_addr, size, is_no_delay, xfer_status_ptr);
        if (status != NIXL_SUCCESS) {
            printf("Thread %d: nixlGpuPostSingleWriteXferReq failed iteration %lu: status=%d (0x%x)\n",
                   threadIdx.x,
                   (unsigned long)i,
                   status,
                   static_cast<unsigned int>(status));
            return;
        }

        status = nixlGpuGetXferStatus<level>(*xfer_status_ptr);
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            printf("Thread %d: Failed to progress single write transfer iteration %zu: status=%d\n",
                   threadIdx.x,
                   i,
                   status);
            return;
        }

        while (status == NIXL_IN_PROG) {
            status = nixlGpuGetXferStatus<level>(*xfer_status_ptr);
            if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                printf("Thread %d: Failed to progress single write transfer iteration %zu: status=%d\n",
                       threadIdx.x,
                       i,
                       status);
                return;
            }
        }

        if (status != NIXL_SUCCESS) {
            printf("Thread %d: Transfer completion failed iteration %zu: status=%d\n",
                   threadIdx.x,
                   i,
                   status);
            return;
        }
    }

    if (threadIdx.x == 0) {
        unsigned long long end_time = GetTimeNs();
        *end_time_ptr = end_time;
    }
}

template<nixl_gpu_level_t level>
nixl_status_t
LaunchSingleWriteTest(unsigned num_threads,
                      nixlGpuXferReqH req_hdnl,
                      unsigned index,
                      const void *src_addr,
                      uint64_t remote_addr,
                      size_t size,
                      size_t num_iters,
                      bool is_no_delay,
                      unsigned long long *start_time_ptr,
                      unsigned long long *end_time_ptr) {
    nixl_status_t ret = NIXL_SUCCESS;
    cudaError_t err;

    TestSingleWriteKernel<level><<<1, num_threads>>>(req_hdnl,
                                                     index,
                                                     src_addr,
                                                     remote_addr,
                                                     size,
                                                     num_iters,
                                                     is_no_delay,
                                                     start_time_ptr,
                                                     end_time_ptr);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    return ret;
}

class SingleWriteTest : public DeviceApiTestBase {
protected:
    std::string getBackendName() const { return "UCX"; }

    static nixlAgentConfig
    getConfig() {
        return nixlAgentConfig(true,
                               false,
                               0,
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
                               0,
                               100000);
    }

    nixl_b_params_t
    getBackendParams() {
        nixl_b_params_t params;

        if (getBackendName() == "UCX") {
            params["num_workers"] = "2";
        }

        return params;
    }

    void
    SetUp() override {
        if (cudaSetDevice(0) != cudaSuccess) {
            FAIL() << "Failed to set CUDA device 0";
        }

        for (size_t i = 0; i < 2; i++) {
            agents.emplace_back(std::make_unique<nixlAgent>(getAgentName(i), getConfig()));
            nixlBackendH *backend_handle = nullptr;
            nixl_status_t status =
                agents.back()->createBackend(getBackendName(), getBackendParams(), backend_handle);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(backend_handle, nullptr);
            backend_handles.push_back(backend_handle);
        }
    }

    void
    TearDown() override {
        agents.clear();
        backend_handles.clear();
    }

    template<typename Desc>
    nixlDescList<Desc>
    makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
        nixlDescList<Desc> desc_list(mem_type);
        for (const auto &buffer : buffers) {
            desc_list.addDesc(Desc(buffer, buffer.getSize(), uint64_t(DEV_ID)));
        }
        return desc_list;
    }

    void
    registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
        auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
        agent.registerMem(reg_list);
    }

    void
    completeWireup(size_t from_agent, size_t to_agent) {
        nixl_notifs_t notifs;
        nixl_status_t status = getAgent(from_agent).genNotif(getAgentName(to_agent), NOTIF_MSG);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to complete wireup";

        do {
            nixl_status_t ret = getAgent(to_agent).getNotifs(notifs);
            ASSERT_EQ(ret, NIXL_SUCCESS) << "Failed to get notifications during wireup";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (notifs.size() == 0);
    }

    void
    exchangeMD(size_t from_agent, size_t to_agent) {
        for (size_t i = 0; i < agents.size(); i++) {
            nixl_blob_t md;
            nixl_status_t status = agents[i]->getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j) continue;
                std::string remote_agent_name;
                status = agents[j]->loadRemoteMD(md, remote_agent_name);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_EQ(remote_agent_name, getAgentName(i));
            }
        }

        completeWireup(from_agent, to_agent);
    }

    void
    invalidateMD() {
        for (size_t i = 0; i < agents.size(); i++) {
            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j) continue;
                nixl_status_t status = agents[j]->invalidateRemoteMD(getAgentName(i));
                ASSERT_EQ(status, NIXL_SUCCESS);
            }
        }
    }

    void
    createRegisteredMem(nixlAgent &agent,
                        size_t size,
                        size_t count,
                        nixl_mem_t mem_type,
                        std::vector<MemBuffer> &out) {
        while (count-- != 0) {
            out.emplace_back(size, mem_type);
        }

        registerMem(agent, out, mem_type);
    }

    nixlAgent &
    getAgent(size_t idx) {
        return *agents[idx];
    }

    std::string
    getAgentName(size_t idx) {
        return absl::StrFormat("agent_%d", idx);
    }

    nixl_status_t
    dispatchLaunchSingleWriteTest(nixl_gpu_level_t level,
                                  unsigned num_threads,
                                  nixlGpuXferReqH req_hdnl,
                                  unsigned index,
                                  const void *src_addr,
                                  uint64_t remote_addr,
                                  size_t size,
                                  size_t num_iters,
                                  bool is_no_delay,
                                  unsigned long long *start_time_ptr,
                                  unsigned long long *end_time_ptr) {
        switch (level) {
        case nixl_gpu_level_t::BLOCK:
            return LaunchSingleWriteTest<nixl_gpu_level_t::BLOCK>(num_threads,
                                                                  req_hdnl,
                                                                  index,
                                                                  src_addr,
                                                                  remote_addr,
                                                                  size,
                                                                  num_iters,
                                                                  is_no_delay,
                                                                  start_time_ptr,
                                                                  end_time_ptr);
        case nixl_gpu_level_t::WARP:
            return LaunchSingleWriteTest<nixl_gpu_level_t::WARP>(num_threads,
                                                                 req_hdnl,
                                                                 index,
                                                                 src_addr,
                                                                 remote_addr,
                                                                 size,
                                                                 num_iters,
                                                                 is_no_delay,
                                                                 start_time_ptr,
                                                                 end_time_ptr);
        case nixl_gpu_level_t::THREAD:
            return LaunchSingleWriteTest<nixl_gpu_level_t::THREAD>(
                num_threads,
                req_hdnl,
                index,
                src_addr,
                remote_addr,
                size,
                num_iters,
                is_no_delay,
                start_time_ptr,
                end_time_ptr);
        default:
            ADD_FAILURE() << "Unknown level: " << static_cast<int>(level);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

protected:
    static constexpr size_t SENDER_AGENT = 0;
    static constexpr size_t RECEIVER_AGENT = 1;

private:
    static constexpr uint64_t DEV_ID = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
    std::vector<nixlBackendH *> backend_handles;

    void
    initTiming(unsigned long long **start_time_ptr, unsigned long long **end_time_ptr) {
        cudaMalloc(start_time_ptr, sizeof(unsigned long long));
        cudaMalloc(end_time_ptr, sizeof(unsigned long long));
        cudaMemset(*start_time_ptr, 0, sizeof(unsigned long long));
        cudaMemset(*end_time_ptr, 0, sizeof(unsigned long long));
    }

    void
    getTiming(unsigned long long *start_time_ptr,
              unsigned long long *end_time_ptr,
              unsigned long long &start_time_cpu,
              unsigned long long &end_time_cpu) {
        cudaMemcpy(
            &start_time_cpu, start_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&end_time_cpu, end_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }

    void
    logResults(size_t size,
               size_t count,
               size_t num_iters,
               unsigned long long start_time_cpu,
               unsigned long long end_time_cpu) {
        auto total_time = NS_TO_SEC(end_time_cpu - start_time_cpu);
        double total_size = size * count * num_iters;
        auto bandwidth = total_size / total_time / (1024 * 1024);
        Logger() << "SingleWrite Results: " << size << "x" << count << "x" << num_iters << "="
                 << total_size << " bytes in " << total_time << " seconds " << "(" << bandwidth
                 << " MB/s)";
    }

public:
    void
    initTimingPublic(unsigned long long **start_time_ptr, unsigned long long **end_time_ptr) {
        initTiming(start_time_ptr, end_time_ptr);
    }

    void
    getTimingPublic(unsigned long long *start_time_ptr,
                    unsigned long long *end_time_ptr,
                    unsigned long long &start_time_cpu,
                    unsigned long long &end_time_cpu) {
        getTiming(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
    }

    void
    logResultsPublic(size_t size,
                     size_t count,
                     size_t num_iters,
                     unsigned long long start_time_cpu,
                     unsigned long long end_time_cpu) {
        logResults(size, count, num_iters, start_time_cpu, end_time_cpu);
    }
};

TEST_P(SingleWriteTest, BasicSingleWriteTest) {
    std::vector<MemBuffer> src_buffers, dst_buffers;
    constexpr size_t size = 4 * 1024;
    constexpr size_t count = 1;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;
    const size_t num_iters = 10000;
    constexpr unsigned index = 0;
    const bool is_no_delay = false; // TODO: Change to true when UCX supports it

    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);

    uint32_t *src_data = static_cast<uint32_t *>(static_cast<void *>(src_buffers[0]));
    uint32_t pattern = 0xDEADBEEF;

    cudaMemset(src_data, 0, size);
    cudaMemcpy(src_data, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    nixl_opt_args_t extra_params = {};
    extra_params.hasNotif = true;
    extra_params.notifMsg = NOTIF_MSG;

    nixlXferReqH *xfer_req = nullptr;
    nixl_status_t status = getAgent(SENDER_AGENT)
                               .createXferReq(NIXL_WRITE,
                                              makeDescList<nixlBasicDesc>(src_buffers, mem_type),
                                              makeDescList<nixlBasicDesc>(dst_buffers, mem_type),
                                              getAgentName(RECEIVER_AGENT),
                                              xfer_req,
                                              &extra_params);

    ASSERT_EQ(status, NIXL_SUCCESS)
        << "Failed to create xfer request " << nixlEnumStrings::statusStr(status);
    EXPECT_NE(xfer_req, nullptr);

    nixlGpuXferReqH gpu_req_hndl;
    status = getAgent(SENDER_AGENT).createGpuXferReq(*xfer_req, gpu_req_hndl);
    ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create GPU xfer request";

    ASSERT_NE(gpu_req_hndl, nullptr) << "GPU request handle is null after createGpuXferReq";

    uint64_t remote_addr = static_cast<uintptr_t>(dst_buffers[0]);
    const void *src_addr = static_cast<const void *>(src_buffers[0]);

    unsigned long long *start_time_ptr = nullptr;
    unsigned long long *end_time_ptr = nullptr;
    nixl_status_t *result_status = nullptr;

    initTimingPublic(&start_time_ptr, &end_time_ptr);
    cudaMalloc(&result_status, sizeof(nixl_status_t));
    cudaMemset(result_status, 0, sizeof(nixl_status_t));

    status = dispatchLaunchSingleWriteTest(GetParam(),
                                           num_threads,
                                           gpu_req_hndl,
                                           index,
                                           src_addr,
                                           remote_addr,
                                           size,
                                           num_iters,
                                           is_no_delay,
                                           start_time_ptr,
                                           end_time_ptr);

    ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel launch failed with status: " << status;

    nixl_status_t gpu_result;
    cudaMemcpy(&gpu_result, result_status, sizeof(nixl_status_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(gpu_result, NIXL_SUCCESS) << "GPU kernel reported error: " << gpu_result;

    unsigned long long start_time_cpu = 0;
    unsigned long long end_time_cpu = 0;
    getTimingPublic(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
    logResultsPublic(size, count, num_iters, start_time_cpu, end_time_cpu);

    uint32_t dst_data;
    cudaMemcpy(&dst_data,
               static_cast<uint32_t *>(static_cast<void *>(dst_buffers[0])),
               sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    EXPECT_EQ(dst_data, pattern) << "Data transfer verification failed. Expected: 0x" << std::hex
                                 << pattern << ", Got: 0x" << dst_data;

    cudaFree(start_time_ptr);
    cudaFree(end_time_ptr);
    cudaFree(result_status);

    getAgent(SENDER_AGENT).releaseGpuXferReq(gpu_req_hndl);

    status = getAgent(SENDER_AGENT).releaseXferReq(xfer_req);
    EXPECT_EQ(status, NIXL_SUCCESS);

    invalidateMD();
}

TEST_P(SingleWriteTest, VariableSizeTest) {
    std::vector<size_t> test_sizes = {64, 256, 1024, 4096, 16384};

    for (size_t test_size : test_sizes) {
        std::vector<MemBuffer> src_buffers, dst_buffers;
        constexpr size_t count = 1;
        nixl_mem_t mem_type = VRAM_SEG;
        size_t num_threads = 32;
        const size_t num_iters = 50000;
        constexpr unsigned index = 0;
        const bool is_no_delay = false; // TODO: Change to true when UCX supports it

        createRegisteredMem(getAgent(SENDER_AGENT), test_size, count, mem_type, src_buffers);
        createRegisteredMem(getAgent(RECEIVER_AGENT), test_size, count, mem_type, dst_buffers);

        std::vector<uint8_t> pattern(test_size);
        for (size_t i = 0; i < test_size; ++i) {
            pattern[i] = static_cast<uint8_t>(i % 256);
        }

        cudaMemcpy(
            static_cast<void *>(src_buffers[0]), pattern.data(), test_size, cudaMemcpyHostToDevice);

        exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

        nixl_opt_args_t extra_params = {};
        extra_params.hasNotif = true;
        extra_params.notifMsg = NOTIF_MSG;

        nixlXferReqH *xfer_req = nullptr;
        nixl_status_t status =
            getAgent(SENDER_AGENT)
                .createXferReq(NIXL_WRITE,
                               makeDescList<nixlBasicDesc>(src_buffers, mem_type),
                               makeDescList<nixlBasicDesc>(dst_buffers, mem_type),
                               getAgentName(RECEIVER_AGENT),
                               xfer_req,
                               &extra_params);

        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create xfer request for size " << test_size;

        nixlGpuXferReqH gpu_req_hndl;
        status = getAgent(SENDER_AGENT).createGpuXferReq(*xfer_req, gpu_req_hndl);
        ASSERT_EQ(status, NIXL_SUCCESS)
            << "Failed to create GPU xfer request for size " << test_size;

        ASSERT_NE(gpu_req_hndl, nullptr) << "GPU request handle is null after createGpuXferReq";

        unsigned long long *start_time_ptr = nullptr;
        unsigned long long *end_time_ptr = nullptr;
        nixl_status_t *result_status = nullptr;

        initTimingPublic(&start_time_ptr, &end_time_ptr);
        cudaMalloc(&result_status, sizeof(nixl_status_t));
        cudaMemset(result_status, 0, sizeof(nixl_status_t));

        uint64_t remote_addr = static_cast<uintptr_t>(dst_buffers[0]);
        const void *src_addr = static_cast<const void *>(src_buffers[0]);

        status = dispatchLaunchSingleWriteTest(GetParam(),
                                               num_threads,
                                               gpu_req_hndl,
                                               index,
                                               src_addr,
                                               remote_addr,
                                               test_size,
                                               num_iters,
                                               is_no_delay,
                                               start_time_ptr,
                                               end_time_ptr);

        ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel launch failed for size " << test_size;

        nixl_status_t gpu_result;
        cudaMemcpy(&gpu_result, result_status, sizeof(nixl_status_t), cudaMemcpyDeviceToHost);
        ASSERT_EQ(gpu_result, NIXL_SUCCESS) << "GPU kernel failed for size " << test_size;

        std::vector<uint8_t> received_data(test_size);
        cudaMemcpy(received_data.data(),
                   static_cast<void *>(dst_buffers[0]),
                   test_size,
                   cudaMemcpyDeviceToHost);

        EXPECT_EQ(received_data, pattern) << "Data verification failed for size " << test_size;

        cudaFree(start_time_ptr);
        cudaFree(end_time_ptr);
        cudaFree(result_status);

        getAgent(SENDER_AGENT).releaseGpuXferReq(gpu_req_hndl);
        getAgent(SENDER_AGENT).releaseXferReq(xfer_req);
        invalidateMD();
    }
}

} // namespace gtest::nixl::gpu::single_write

using gtest::nixl::gpu::single_write::SingleWriteTest;

INSTANTIATE_TEST_SUITE_P(
    ucxDeviceApi,
    SingleWriteTest,
    testing::ValuesIn(gtest::gpu::_test_levels),
    [](const testing::TestParamInfo<nixl_gpu_level_t> &info) {
        return std::string("UCX_") + gtest::gpu::GetGpuXferLevelStr(info.param);
    });
