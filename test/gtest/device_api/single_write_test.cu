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

#include <algorithm>
#include <initializer_list>
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
timedPutKernel(putParams put_params, nixl_status_t *completion_status) {
    constexpr unsigned long long completion_timeout_ns = 30ULL * 1000 * 1000 * 1000;
    nixlGpuXferStatusH xfer_status{};
    nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(put_params.src,
                                                             put_params.dst,
                                                             put_params.size,
                                                             put_params.channelId,
                                                             put_params.flags,
                                                             &xfer_status);
    if (status == NIXL_IN_PROG) {
        const unsigned long long start = GetTimeNs();
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(xfer_status);
        } while (status == NIXL_IN_PROG && GetTimeNs() - start < completion_timeout_ns);
    }
    *completion_status = status;
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

static nixl_status_t
launchTimedProxyPut(const putParams &put_params) {
    gpuVar<nixl_status_t> completion_status;
    timedPutKernel<<<1, 1>>>(put_params, completion_status.get());
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    return *completion_status;
}

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

class SingleWriteIntegrationTest : public ::testing::Test {
protected:
    explicit SingleWriteIntegrationTest(bool use_proxy)
        : use_proxy_(use_proxy),
          agents(use_proxy ? shared_proxy_agents : local_agents),
          backend_handles(use_proxy ? shared_proxy_backend_handles : local_backend_handles) {}

    std::string
    getBackendName() const {
        return "UCX";
    }

    nixlAgentConfig
    getConfig(bool proxy_owner) const {
        nixlAgentConfig cfg;
        cfg.useProgThread = !proxy_owner;
        cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        cfg.pthrDelay = proxy_owner ? 1000 : 100000;
        cfg.enableDeviceProxy = proxy_owner;
        if (proxy_owner) {
            cfg.proxyMaxPeers = kProxyPeerCapacity;
            cfg.proxyChannelCount = kProxyTransferChannelCount;
            cfg.proxyWorkerCount = kProxyCpuWorkerCount;
        }
        return cfg;
    }

    nixl_b_params_t
    getBackendParams() const {
        nixl_b_params_t params;

        if (getBackendName() == "UCX") {
            params["num_workers"] =
                std::to_string(useProxy() ? kProxyUcxWorkerCount : kDirectUcxWorkerCount);
        }

        return params;
    }

    void
    addAgent(bool proxy_owner = false) {
        const size_t agent_index = agents.size();
        agents.emplace_back(
            std::make_unique<nixlAgent>(getAgentName(agent_index), getConfig(proxy_owner)));
        nixlBackendH *backend_handle = nullptr;
        const nixl_status_t status =
            agents.back()->createBackend(getBackendName(), getBackendParams(), backend_handle);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend_handle, nullptr);
        backend_handles.push_back(backend_handle);
    }

    void
    ensureAgentCount(size_t count) {
        while (agents.size() < count) {
            ASSERT_NO_FATAL_FAILURE(addAgent(useProxy() && agents.empty()));
        }
    }

    static void
    resetSharedEnvironment() {
        shared_proxy_agents.clear();
        shared_proxy_backend_handles.clear();
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

        ASSERT_NO_FATAL_FAILURE(ensureAgentCount(2));
    }

    void
    TearDown() override {
        if (useProxy()) {
            if (HasFailure()) {
                resetSharedEnvironment();
            }
        } else {
            agents.clear();
            backend_handles.clear();
        }
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
    deregisterMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
        const auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
        ASSERT_EQ(agent.deregisterMem(reg_list), NIXL_SUCCESS);
    }

    void
    exchangeMD(std::initializer_list<size_t> agent_indices) {
        for (const size_t i : agent_indices) {
            nixl_blob_t md;
            nixl_status_t status = agents[i]->getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (const size_t j : agent_indices) {
                if (i == j) {
                    continue;
                }
                std::string remote_agent_name;
                status = agents[j]->loadRemoteMD(md, remote_agent_name);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_EQ(remote_agent_name, getAgentName(i));
            }
        }
    }

    void
    invalidateMD(std::initializer_list<size_t> agent_indices) {
        for (const size_t i : agent_indices) {
            for (const size_t j : agent_indices) {
                if (i == j) {
                    continue;
                }
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

    bool
    useProxy() const {
        return use_proxy_;
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

    void
    runSingleWorkerPut(const std::vector<nixl_gpu_level_t> &levels) {
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

        exchangeMD({SENDER_AGENT, RECEIVER_AGENT});

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
        constexpr size_t num_iters = 10;
        for (const nixl_gpu_level_t level : levels) {
            SCOPED_TRACE(::testing::Message()
                         << "GPU transfer level: " << gtest::gpu::GetGpuXferLevelStr(level));
            ASSERT_EQ(cudaMemset(static_cast<void *>(dst_buffers[0]), 0, size), cudaSuccess);

            gpuTimer gpu_timer;
            status = dispatchLaunchPutKernel(level, put_params, num_iters, &gpu_timer);
            ASSERT_EQ(status, NIXL_SUCCESS);

            Logger() << "SingleWorkerPut level: " << gtest::gpu::GetGpuXferLevelStr(level);
            logResultsPublic(size, count, num_iters, *gpu_timer.start_, *gpu_timer.end_);

            uint32_t dst_data;
            cudaMemcpy(&dst_data,
                       static_cast<uint32_t *>(static_cast<void *>(dst_buffers[0])),
                       sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            EXPECT_EQ(dst_data, pattern) << "Data transfer verification failed. Expected: 0x"
                                         << std::hex << pattern << ", Got: 0x" << dst_data;
        }

        getAgent(SENDER_AGENT).releaseMemView(dst_mvh);
        getAgent(SENDER_AGENT).releaseMemView(src_mvh);
        invalidateMD({SENDER_AGENT, RECEIVER_AGENT});
        if (useProxy()) {
            deregisterMem(getAgent(SENDER_AGENT), src_buffers, mem_type);
            deregisterMem(getAgent(RECEIVER_AGENT), dst_buffers, mem_type);
        }
    }

    void
    runMultipleWorkersPut();

    void
    runSingleWorkerPutWithGetPtr(nixl_gpu_level_t level, size_t num_iters);

protected:
    static constexpr size_t SENDER_AGENT = 0;
    static constexpr size_t RECEIVER_AGENT = 1;
    static constexpr size_t SECOND_RECEIVER_AGENT = 2;
    static constexpr size_t kProxyTransferChannelCount = 4;
    static constexpr uint32_t kProxyCpuWorkerCount = kProxyTransferChannelCount;
    static constexpr uint32_t kProxyPeerCapacity = 2;
    // The UCX proxy adapter isolates one UCX worker per (channel, peer) pair -
    // see nixlUcxProxyBackendAdapter::init(), which rejects any other count.
    static constexpr size_t kProxyUcxWorkerCount =
        kProxyTransferChannelCount * static_cast<size_t>(kProxyPeerCapacity);
    static constexpr size_t kDirectTransferChannelCount = 32;
    // Each logical transfer channel maps directly to one UCX worker.
    static constexpr size_t kDirectUcxWorkerCount = kDirectTransferChannelCount;

    size_t
    transferChannelCount() const {
        return useProxy() ? kProxyTransferChannelCount : kDirectTransferChannelCount;
    }

private:
    static constexpr uint64_t DEV_ID = 0;

    std::unique_ptr<LogIgnoreGuard> lig_;
    bool use_proxy_;
    std::vector<std::unique_ptr<nixlAgent>> local_agents;
    std::vector<nixlBackendH *> local_backend_handles;
    inline static std::vector<std::unique_ptr<nixlAgent>> shared_proxy_agents;
    inline static std::vector<nixlBackendH *> shared_proxy_backend_handles;
    std::vector<std::unique_ptr<nixlAgent>> &agents;
    std::vector<nixlBackendH *> &backend_handles;

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

class ProxySingleWriteTest : public SingleWriteIntegrationTest {
public:
    ProxySingleWriteTest() : SingleWriteIntegrationTest(true) {}

    static void
    TearDownTestSuite() {
        resetSharedEnvironment();
    }
};

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
class UcxSingleWriteIntegrationTest : public SingleWriteIntegrationTest {
public:
    UcxSingleWriteIntegrationTest() : SingleWriteIntegrationTest(false) {}
};

class UcxSingleWriteTest : public UcxSingleWriteIntegrationTest,
                           public ::testing::WithParamInterface<nixl_gpu_level_t> {};
#endif

TEST_F(ProxySingleWriteTest, SingleWorkerPut) {
    runSingleWorkerPut(gtest::gpu::_test_levels);
}

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
TEST_P(UcxSingleWriteTest, SingleWorkerPut) {
    runSingleWorkerPut({GetParam()});
}
#endif

void
SingleWriteIntegrationTest::runMultipleWorkersPut() {
    if (useProxy()) {
        GTEST_LOG_(INFO) << "Exercising " << kProxyTransferChannelCount
                         << " logical proxy channels with " << kProxyCpuWorkerCount
                         << " CPU proxy workers";
    }
    constexpr size_t size = 4 * 1024;
    constexpr nixl_mem_t mem_type = VRAM_SEG;
    const size_t channel_count = transferChannelCount();

    std::vector<std::vector<MemBuffer>> src_buffers(channel_count);
    std::vector<std::vector<MemBuffer>> dst_buffers(channel_count);
    std::vector<std::vector<uint32_t>> patterns(channel_count);

    for (size_t channel_id = 0; channel_id < channel_count; channel_id++) {
        createRegisteredMem(getAgent(SENDER_AGENT), size, 1, mem_type, src_buffers[channel_id]);
        createRegisteredMem(getAgent(RECEIVER_AGENT), size, 1, mem_type, dst_buffers[channel_id]);

        constexpr size_t num_elements = size / sizeof(uint32_t);
        patterns[channel_id].resize(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            patterns[channel_id][i] = 0xDEAD0000 | channel_id;
        }

        cudaMemcpy(static_cast<void *>(src_buffers[channel_id][0]),
                   patterns[channel_id].data(),
                   size,
                   cudaMemcpyHostToDevice);
    }

    exchangeMD({SENDER_AGENT, RECEIVER_AGENT});

    std::vector<nixlMemViewH> src_mvhs(channel_count);
    std::vector<nixlMemViewH> dst_mvhs(channel_count);
    nixl_opt_args_t extra_params;

    for (size_t channel_id = 0; channel_id < channel_count; channel_id++) {
        extra_params.customParam = "worker_id=" + std::to_string(channel_id);

        auto status =
            getAgent(SENDER_AGENT)
                .prepMemView(makeDescList<nixlBasicDesc>(src_buffers[channel_id], mem_type),
                             src_mvhs[channel_id],
                             &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        status =
            getAgent(SENDER_AGENT)
                .prepMemView(makeDescList<nixlRemoteDesc>(
                                 dst_buffers[channel_id], mem_type, getAgentName(RECEIVER_AGENT)),
                             dst_mvhs[channel_id],
                             &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

    for (size_t channel_id = 0; channel_id < channel_count; channel_id++) {
        putParams put_params{{src_mvhs[channel_id], 0, 0}, {dst_mvhs[channel_id], 0, 0}, size};
        if (useProxy()) {
            put_params.channelId = static_cast<unsigned>(channel_id);
            ASSERT_EQ(launchTimedProxyPut(put_params), NIXL_SUCCESS)
                << "Transfer failed or timed out for logical channel " << channel_id;
        } else {
            constexpr size_t num_iters = 1;
            const auto status =
                launchPutKernel<nixl_gpu_level_t::THREAD>(put_params, num_iters, nullptr, 1);
            ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel launch failed for worker " << channel_id;
        }
    }

    for (size_t channel_id = 0; channel_id < channel_count; channel_id++) {
        std::vector<uint32_t> received(size / sizeof(uint32_t));
        cudaMemcpy(received.data(),
                   static_cast<void *>(dst_buffers[channel_id][0]),
                   size,
                   cudaMemcpyDeviceToHost);

        EXPECT_EQ(received, patterns[channel_id])
            << "Channel " << channel_id << " full buffer verification failed";
    }

    Logger() << "MultipleWorkersPut: " << channel_count
             << " explicitly selected transfer channels verified";

    for (size_t channel_id = 0; channel_id < channel_count; channel_id++) {
        getAgent(SENDER_AGENT).releaseMemView(src_mvhs[channel_id]);
        getAgent(SENDER_AGENT).releaseMemView(dst_mvhs[channel_id]);
    }

    invalidateMD({SENDER_AGENT, RECEIVER_AGENT});
    if (useProxy()) {
        for (size_t channel_id = 0; channel_id < channel_count; channel_id++) {
            deregisterMem(getAgent(SENDER_AGENT), src_buffers[channel_id], mem_type);
            deregisterMem(getAgent(RECEIVER_AGENT), dst_buffers[channel_id], mem_type);
        }
    }
}

TEST_F(ProxySingleWriteTest, MultipleWorkersPut) {
    runMultipleWorkersPut();
}

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
TEST_F(UcxSingleWriteIntegrationTest, MultipleWorkersPut) {
    runMultipleWorkersPut();
}
#endif

TEST_F(ProxySingleWriteTest, MultiplePeersMultipleChannelsPut) {
    ASSERT_NO_FATAL_FAILURE(ensureAgentCount(SECOND_RECEIVER_AGENT + 1));

    constexpr size_t size = 4 * 1024;
    constexpr size_t peer_count = 2;
    constexpr size_t channel_count = kProxyTransferChannelCount;
    constexpr size_t num_elements = size / sizeof(uint32_t);
    constexpr nixl_mem_t mem_type = VRAM_SEG;

    using ChannelBuffers = std::vector<std::vector<MemBuffer>>;
    std::vector<ChannelBuffers> src_buffers(peer_count);
    std::vector<ChannelBuffers> dst_buffers(peer_count);
    std::vector<std::vector<std::vector<uint32_t>>> patterns(peer_count);
    for (size_t peer_slot = 0; peer_slot < peer_count; ++peer_slot) {
        src_buffers[peer_slot].resize(channel_count);
        dst_buffers[peer_slot].resize(channel_count);
        patterns[peer_slot].resize(channel_count);

        const size_t receiver_index = RECEIVER_AGENT + peer_slot;
        for (size_t channel_id = 0; channel_id < channel_count; ++channel_id) {
            createRegisteredMem(
                getAgent(SENDER_AGENT), size, 1, mem_type, src_buffers[peer_slot][channel_id]);
            createRegisteredMem(
                getAgent(receiver_index), size, 1, mem_type, dst_buffers[peer_slot][channel_id]);

            const uint32_t pattern = 0xCA000000u | (static_cast<uint32_t>(peer_slot) << 8) |
                static_cast<uint32_t>(channel_id);
            patterns[peer_slot][channel_id].assign(num_elements, pattern);
            ASSERT_EQ(cudaMemcpy(static_cast<void *>(src_buffers[peer_slot][channel_id][0]),
                                 patterns[peer_slot][channel_id].data(),
                                 size,
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
    }

    exchangeMD({SENDER_AGENT, RECEIVER_AGENT, SECOND_RECEIVER_AGENT});

    std::vector<std::vector<nixlMemViewH>> src_mvhs(peer_count);
    for (auto &peer_mvhs : src_mvhs) {
        peer_mvhs.resize(channel_count);
    }
    std::vector<nixlMemViewH> dst_mvhs(channel_count);
    nixl_opt_args_t extra_params;

    for (size_t channel_id = 0; channel_id < channel_count; ++channel_id) {
        extra_params.customParam = "worker_id=" + std::to_string(channel_id);

        nixlDescList<nixlRemoteDesc> remote_dlist(mem_type);
        for (size_t peer_slot = 0; peer_slot < peer_count; ++peer_slot) {
            const auto &buffer = dst_buffers[peer_slot][channel_id][0];
            remote_dlist.addDesc(nixlRemoteDesc(
                buffer, buffer.getSize(), uint64_t{0}, getAgentName(RECEIVER_AGENT + peer_slot)));

            ASSERT_EQ(getAgent(SENDER_AGENT)
                          .prepMemView(makeDescList<nixlBasicDesc>(
                                           src_buffers[peer_slot][channel_id], mem_type),
                                       src_mvhs[peer_slot][channel_id],
                                       &extra_params),
                      NIXL_SUCCESS);
        }

        ASSERT_EQ(
            getAgent(SENDER_AGENT).prepMemView(remote_dlist, dst_mvhs[channel_id], &extra_params),
            NIXL_SUCCESS);
    }

    for (size_t peer_slot = 0; peer_slot < peer_count; ++peer_slot) {
        for (size_t channel_id = 0; channel_id < channel_count; ++channel_id) {
            putParams put_params{{src_mvhs[peer_slot][channel_id], 0, 0},
                                 {dst_mvhs[channel_id], peer_slot, 0},
                                 size,
                                 static_cast<unsigned>(channel_id)};
            ASSERT_EQ(launchTimedProxyPut(put_params), NIXL_SUCCESS)
                << "Transfer failed for peer slot " << peer_slot << ", logical channel "
                << channel_id;
        }
    }

    for (size_t peer_slot = 0; peer_slot < peer_count; ++peer_slot) {
        for (size_t channel_id = 0; channel_id < channel_count; ++channel_id) {
            std::vector<uint32_t> received(num_elements);
            ASSERT_EQ(cudaMemcpy(received.data(),
                                 static_cast<void *>(dst_buffers[peer_slot][channel_id][0]),
                                 size,
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(received, patterns[peer_slot][channel_id])
                << "Data isolation failed for peer slot " << peer_slot << ", logical channel "
                << channel_id;
            getAgent(SENDER_AGENT).releaseMemView(src_mvhs[peer_slot][channel_id]);
        }
    }
    for (const nixlMemViewH dst_mvh : dst_mvhs) {
        getAgent(SENDER_AGENT).releaseMemView(dst_mvh);
    }

    invalidateMD({SENDER_AGENT, RECEIVER_AGENT, SECOND_RECEIVER_AGENT});
    for (size_t peer_slot = 0; peer_slot < peer_count; ++peer_slot) {
        const size_t receiver_index = RECEIVER_AGENT + peer_slot;
        for (size_t channel_id = 0; channel_id < channel_count; ++channel_id) {
            deregisterMem(getAgent(SENDER_AGENT), src_buffers[peer_slot][channel_id], mem_type);
            deregisterMem(getAgent(receiver_index), dst_buffers[peer_slot][channel_id], mem_type);
        }
    }
}

void
SingleWriteIntegrationTest::runSingleWorkerPutWithGetPtr(nixl_gpu_level_t level, size_t num_iters) {
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

    exchangeMD({SENDER_AGENT, RECEIVER_AGENT});

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
    gpuTimer gpu_timer;
    status = dispatchLaunchPutKernel(level, put_params, num_iters, &gpu_timer);
    ASSERT_EQ(status, NIXL_SUCCESS);

    gpuVar<void *> ptr;
    getPtrKernel<<<1, 1>>>(dst_mvh, 0, ptr.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_NE(*ptr, nullptr);

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
    invalidateMD({SENDER_AGENT, RECEIVER_AGENT});
    if (useProxy()) {
        deregisterMem(getAgent(SENDER_AGENT), src_buffers, mem_type);
        deregisterMem(getAgent(RECEIVER_AGENT), dst_buffers, mem_type);
    }
}

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
TEST_P(UcxSingleWriteTest, SingleWorkerPutGap) {
    runSingleWorkerPutWithGetPtr(GetParam(), 1000);
}
#endif

TEST_F(ProxySingleWriteTest, SingleWorkerPutAndGetPtr) {
    runSingleWorkerPutWithGetPtr(nixl_gpu_level_t::THREAD, 1);
}
} // namespace gtest::nixl::gpu::single_write

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
using gtest::nixl::gpu::single_write::UcxSingleWriteTest;
INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         UcxSingleWriteTest,
                         testing::ValuesIn(gtest::gpu::_test_levels),
                         [](const testing::TestParamInfo<nixl_gpu_level_t> &info) {
                             return std::string("UCX_") +
                                 gtest::gpu::GetGpuXferLevelStr(info.param);
                         });
#endif
