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

#include "utils.cuh"
#include "common.h"

#include <memory>
#include <gtest/gtest.h>

namespace gtest::nixl::gpu::single_write {
struct putParams {
    nixlMemViewElem src;
    nixlMemViewElem dst;
    size_t size;
    unsigned channelId{0};
    uint64_t flags{0};
};

template<nixl_gpu_level_t level>
__global__ void
putKernel(putParams put_params,
          size_t num_iters,
          unsigned long long *start_time,
          unsigned long long *end_time) {
    __shared__ nixlGpuXferStatusH xfer_statuses[MAX_THREADS];
    nixlGpuXferStatusH xfer_status = xfer_statuses[GetReqIdx<level>()];

    assert(GetReqIdx<level>() < MAX_THREADS);

    if (start_time && (threadIdx.x == 0)) {
        *start_time = GetTimeNs();
    }

    __syncthreads();

    for (size_t i = 0; i < num_iters; ++i) {
        auto status = nixlPut<level>(put_params.src,
                                     put_params.dst,
                                     put_params.size,
                                     put_params.channelId,
                                     put_params.flags,
                                     &xfer_status);
        if (status != NIXL_IN_PROG) {
            printf("Thread %d: nixlPut failed iteration %zu: status=%d (0x%x)\n",
                   threadIdx.x,
                   i,
                   status,
                   static_cast<unsigned int>(status));
            return;
        }

        do {
            status = nixlGpuGetXferStatus<level>(xfer_status);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            printf("Thread %d: Transfer completion failed iteration %zu: status=%d\n",
                   threadIdx.x,
                   i,
                   status);
            return;
        }
    }

    if (end_time && (threadIdx.x == 0)) {
        *end_time = GetTimeNs();
    }
}

__global__ void
getPtrKernel(nixlMemViewH mvh, size_t index, void **ptr) {
    *ptr = nixlGetPtr(mvh, index);
}

template<typename T> class gpuVar {
public:
    gpuVar() : ptr_{allocate(), &deallocate} {
        cudaMemset(ptr_.get(), 0, sizeof(T));
    }

    T *
    get() const {
        return ptr_.get();
    }

    T
    operator*() const {
        T value;
        cudaMemcpy(&value, ptr_.get(), sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

private:
    static T *
    allocate() {
        T *ptr;
        cudaMalloc(&ptr, sizeof(T));
        return ptr;
    }

    static void
    deallocate(T *ptr) {
        cudaFree(ptr);
    }

    std::unique_ptr<T, void (*)(T *)> ptr_;
};

struct gpuTimer {
    gpuVar<unsigned long long> start_;
    gpuVar<unsigned long long> end_;
};

template<nixl_gpu_level_t level>
nixl_status_t
launchPutKernel(const putParams &put_params,
                size_t num_iters,
                gpuTimer *gpu_timer,
                unsigned num_threads = 32) {
    unsigned long long *start_time{nullptr};
    unsigned long long *end_time{nullptr};
    if (gpu_timer) {
        start_time = gpu_timer->start_.get();
        end_time = gpu_timer->end_.get();
    }
    putKernel<level><<<1, num_threads>>>(put_params, num_iters, start_time, end_time);

    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
        return NIXL_ERR_BACKEND;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
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
            params["num_workers"] = std::to_string(numWorkers);
        }

        return params;
    }

    void
    SetUp() override {
        if (!hasCudaGpu()) {
            GTEST_SKIP() << "No CUDA-capable GPU is available, skipping test.";
        }
        if (cudaSetDevice(0) != cudaSuccess) {
            FAIL() << "Failed to set CUDA device 0";
        }

        lig_ = std::make_unique<LogIgnoreGuard>(
            "IB device\\(s\\) were detected, but accelerated IB support was not found");

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

    template<typename Desc>
    nixlDescList<Desc>
    makeDescList(const std::vector<MemBuffer> &buffers,
                 nixl_mem_t mem_type,
                 const std::string &agent_name) {
        nixlDescList<Desc> desc_list(mem_type);
        for (const auto &buffer : buffers) {
            desc_list.addDesc(Desc(buffer, buffer.getSize(), uint64_t(DEV_ID), agent_name));
        }
        return desc_list;
    }

    void
    registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
        auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
        agent.registerMem(reg_list);
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
    dispatchLaunchPutKernel(nixl_gpu_level_t level,
                            const putParams &put_params,
                            size_t num_iters,
                            gpuTimer *gpu_timer = nullptr) {
        switch (level) {
        case nixl_gpu_level_t::BLOCK:
            return launchPutKernel<nixl_gpu_level_t::BLOCK>(put_params, num_iters, gpu_timer);
        case nixl_gpu_level_t::WARP:
            return launchPutKernel<nixl_gpu_level_t::WARP>(put_params, num_iters, gpu_timer);
        case nixl_gpu_level_t::THREAD:
            return launchPutKernel<nixl_gpu_level_t::THREAD>(put_params, num_iters, gpu_timer);
        default:
            ADD_FAILURE() << "Unknown level: " << static_cast<int>(level);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

protected:
    static constexpr size_t SENDER_AGENT = 0;
    static constexpr size_t RECEIVER_AGENT = 1;
    static constexpr size_t numWorkers = 32;

private:
    static constexpr uint64_t DEV_ID = 0;

    std::unique_ptr<LogIgnoreGuard> lig_;
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

TEST_P(SingleWriteTest, SingleWorkerPut) {
    std::vector<MemBuffer> src_buffers, dst_buffers;
    constexpr size_t size = 4 * 1024;
    constexpr size_t count = 1;
    constexpr nixl_mem_t mem_type = VRAM_SEG;
    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);

    auto src_data = static_cast<uint32_t *>(static_cast<void *>(src_buffers[0]));
    cudaMemset(src_data, 0, size);

    constexpr uint32_t pattern = 0xDEADBEEF;
    cudaMemcpy(src_data, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    nixlMemViewH src_mvh;
    auto status = getAgent(SENDER_AGENT)
                      .prepMemView(makeDescList<nixlBasicDesc>(src_buffers, mem_type), src_mvh);
    ASSERT_EQ(status, NIXL_SUCCESS);

    nixlMemViewH dst_mvh;
    status = getAgent(SENDER_AGENT)
                 .prepMemView(makeDescList<nixlRemoteDesc>(
                                  dst_buffers, mem_type, getAgentName(RECEIVER_AGENT)),
                              dst_mvh);
    ASSERT_EQ(status, NIXL_SUCCESS);

    putParams put_params{{src_mvh, 0, 0}, {dst_mvh, 0, 0}, size};
    constexpr size_t num_iters = 1000;
    gpuTimer gpu_timer;
    status = dispatchLaunchPutKernel(GetParam(), put_params, num_iters, &gpu_timer);
    ASSERT_EQ(status, NIXL_SUCCESS);

    logResultsPublic(size, count, num_iters, *gpu_timer.start_, *gpu_timer.end_);

    uint32_t dst_data;
    cudaMemcpy(&dst_data,
               static_cast<uint32_t *>(static_cast<void *>(dst_buffers[0])),
               sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    EXPECT_EQ(dst_data, pattern) << "Data transfer verification failed. Expected: 0x" << std::hex
                                 << pattern << ", Got: 0x" << dst_data;

    getAgent(SENDER_AGENT).releaseMemView(dst_mvh);
    getAgent(SENDER_AGENT).releaseMemView(src_mvh);
    invalidateMD();
}

TEST_P(SingleWriteTest, MultipleWorkersPut) {
    constexpr size_t size = 4 * 1024;
    constexpr nixl_mem_t mem_type = VRAM_SEG;

    std::vector<std::vector<MemBuffer>> src_buffers(numWorkers);
    std::vector<std::vector<MemBuffer>> dst_buffers(numWorkers);
    std::vector<std::vector<uint32_t>> patterns(numWorkers);

    for (size_t worker_id = 0; worker_id < numWorkers; worker_id++) {
        createRegisteredMem(getAgent(SENDER_AGENT), size, 1, mem_type, src_buffers[worker_id]);
        createRegisteredMem(getAgent(RECEIVER_AGENT), size, 1, mem_type, dst_buffers[worker_id]);

        constexpr size_t num_elements = size / sizeof(uint32_t);
        patterns[worker_id].resize(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            patterns[worker_id][i] = 0xDEAD0000 | worker_id;
        }

        cudaMemcpy(static_cast<void *>(src_buffers[worker_id][0]),
                   patterns[worker_id].data(),
                   size,
                   cudaMemcpyHostToDevice);
    }

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    std::vector<nixlMemViewH> src_mvhs(numWorkers);
    std::vector<nixlMemViewH> dst_mvhs(numWorkers);
    nixl_opt_args_t extra_params;

    for (size_t worker_id = 0; worker_id < numWorkers; worker_id++) {
        extra_params.customParam = "worker_id=" + std::to_string(worker_id);

        auto status =
            getAgent(SENDER_AGENT)
                .prepMemView(makeDescList<nixlBasicDesc>(src_buffers[worker_id], mem_type),
                             src_mvhs[worker_id],
                             &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        status =
            getAgent(SENDER_AGENT)
                .prepMemView(makeDescList<nixlRemoteDesc>(
                                 dst_buffers[worker_id], mem_type, getAgentName(RECEIVER_AGENT)),
                             dst_mvhs[worker_id],
                             &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

    for (size_t worker_id = 0; worker_id < numWorkers; worker_id++) {
        putParams put_params{{src_mvhs[worker_id], 0, 0}, {dst_mvhs[worker_id], 0, 0}, size};
        constexpr size_t num_iters = 1;
        const auto status = dispatchLaunchPutKernel(GetParam(), put_params, num_iters);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel launch failed for worker " << worker_id;
    }

    for (size_t worker_id = 0; worker_id < numWorkers; worker_id++) {
        std::vector<uint32_t> received(size / sizeof(uint32_t));
        cudaMemcpy(received.data(),
                   static_cast<void *>(dst_buffers[worker_id][0]),
                   size,
                   cudaMemcpyDeviceToHost);

        EXPECT_EQ(received, patterns[worker_id])
            << "Worker " << worker_id << " full buffer verification failed";
    }

    Logger() << "MultipleWorkers test: " << numWorkers
             << " workers with explicit selection verified";

    for (size_t worker_id = 0; worker_id < numWorkers; worker_id++) {
        getAgent(SENDER_AGENT).releaseMemView(src_mvhs[worker_id]);
        getAgent(SENDER_AGENT).releaseMemView(dst_mvhs[worker_id]);
    }

    invalidateMD();
}

TEST_P(SingleWriteTest, SingleWorkerPutGap) {
    std::vector<MemBuffer> src_buffers, dst_buffers;
    constexpr size_t size = 4 * 1024;
    constexpr size_t count = 1;
    constexpr nixl_mem_t mem_type = VRAM_SEG;
    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);

    auto src_data = static_cast<uint32_t *>(static_cast<void *>(src_buffers[0]));
    cudaMemset(src_data, 0, size);

    constexpr uint32_t pattern = 0xDEADBEEF;
    cudaMemcpy(src_data, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    const auto local_dlist = makeDescList<nixlBasicDesc>(src_buffers, mem_type);
    nixlMemViewH src_mvh;
    auto status = getAgent(SENDER_AGENT).prepMemView(local_dlist, src_mvh);
    ASSERT_EQ(status, NIXL_SUCCESS);

    auto remote_dlist =
        makeDescList<nixlRemoteDesc>(dst_buffers, mem_type, getAgentName(RECEIVER_AGENT));
    remote_dlist.addDesc({{}, nixl_null_agent});
    nixlMemViewH dst_mvh;
    status = getAgent(SENDER_AGENT).prepMemView(remote_dlist, dst_mvh);
    ASSERT_EQ(status, NIXL_SUCCESS);

    putParams put_params{{src_mvh, 0, 0}, {dst_mvh, 0, 0}, size};
    constexpr size_t num_iters = 1000;
    gpuTimer gpu_timer;
    status = dispatchLaunchPutKernel(GetParam(), put_params, num_iters, &gpu_timer);
    ASSERT_EQ(status, NIXL_SUCCESS);

    void *ptr;
    getPtrKernel<<<1, 1>>>(dst_mvh, 0, &ptr);
    ASSERT_NE(ptr, nullptr);

    logResultsPublic(size, count, num_iters, *gpu_timer.start_, *gpu_timer.end_);

    uint32_t dst_data;
    cudaMemcpy(&dst_data,
               static_cast<uint32_t *>(static_cast<void *>(dst_buffers[0])),
               sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    EXPECT_EQ(dst_data, pattern) << "Data transfer verification failed. Expected: 0x" << std::hex
                                 << pattern << ", Got: 0x" << dst_data;

    getAgent(SENDER_AGENT).releaseMemView(dst_mvh);
    getAgent(SENDER_AGENT).releaseMemView(src_mvh);
    invalidateMD();
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
