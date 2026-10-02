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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "device/device_memview.h"
#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_runtime.h"
#include "device/proxy/proxy_transport.h"
#include "common.h"

/** Completes every operation at submission. */
class ImmediateTransport : public nixl::proxyTransport {
public:
    nixl_status_t
    init(const nixl::proxyConfig &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const nixl::proxyBackendSubmission &, nixl::proxyBackendRequest &request) override {
        request = {};
        return NIXL_SUCCESS;
    }

    nixl_status_t
    checkCompletion(uint32_t, uint32_t, const nixl::proxyBackendRequest &) override {
        return NIXL_SUCCESS;
    }

    void
    progress(uint32_t, uint32_t) noexcept override {}

    nixl_status_t
    quiesce(uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    shutdown() override {
        return NIXL_SUCCESS;
    }
};

static nixl::proxyConfig
makeProxyConfig(uint32_t max_peers, uint32_t channel_count, uint32_t thread_count) {
    nixl::proxyConfig config;
    config.enabled = true;
    config.max_peers = max_peers;
    config.channel_count = channel_count;
    config.thread_count = thread_count;
    return config;
}

/** Holds every request until the test completes it. */
class ControllableBackend : public ImmediateTransport {
public:
    struct Entry {
        nixl::proxyBackendSubmission submission;
        nixl_status_t status = NIXL_IN_PROG;
    };

    nixl_status_t
    submit(const nixl::proxyBackendSubmission &submission,
           nixl::proxyBackendRequest &request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.push_back({submission});
        request = {entries_.size()};
        return NIXL_IN_PROG;
    }

    nixl_status_t
    checkCompletion(uint32_t, uint32_t, const nixl::proxyBackendRequest &request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.at(request.token - 1).status;
    }

    std::vector<Entry>
    entries() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_;
    }

    void
    complete(size_t index, nixl_status_t status = NIXL_SUCCESS) {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.at(index).status = status;
    }

private:
    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

class DummyBackendMD : public nixlBackendMD {
public:
    DummyBackendMD() : nixlBackendMD(false) {}
};

struct DummyProxyMemViews {
    nixlMemViewH src = nullptr;
    nixlMemViewH dst = nullptr;

    explicit DummyProxyMemViews(nixl::proxyRuntime &runtime, uint32_t peer_count = 1);

    ~DummyProxyMemViews() {
        nixlDeviceMemViewFree(src);
        nixlDeviceMemViewFree(dst);
    }

    DummyProxyMemViews(const DummyProxyMemViews &) = delete;
    DummyProxyMemViews &
    operator=(const DummyProxyMemViews &) = delete;
};

DummyProxyMemViews::DummyProxyMemViews(nixl::proxyRuntime &runtime, uint32_t peer_count) {
    static DummyBackendMD local_md;
    static DummyBackendMD remote_md;

    nixlMemViewH src_raw = nullptr, dst_raw = nullptr;

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
    EXPECT_EQ(runtime.prepMemView(local_dlist, &src_raw), NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
    for (uint32_t peer = 0; peer < peer_count; ++peer) {
        nixlRemoteMetaDesc remote_desc("peer" + std::to_string(peer));
        remote_desc.addr = 0x2000 + peer * 0x100;
        remote_desc.len = 64;
        remote_desc.devId = 0;
        remote_desc.metadataP = &remote_md;
        remote_dlist.addDesc(remote_desc);
    }
    EXPECT_EQ(runtime.prepMemView(remote_dlist, &dst_raw), NIXL_SUCCESS);

    EXPECT_EQ(nixlDeviceMemViewAllocate(nixl_device_exec_mode_t::PROXY, src_raw, src),
              NIXL_SUCCESS);
    EXPECT_EQ(nixlDeviceMemViewAllocate(nixl_device_exec_mode_t::PROXY, dst_raw, dst),
              NIXL_SUCCESS);
}

struct DeviceResult {
    nixl_status_t submit;
    nixlGpuXferStatusH transfer;
    nixl_status_t poll;
};

__global__ void
submitKernel(nixlMemViewH src,
             nixlMemViewH dst,
             DeviceResult *result,
             bool atomic = false,
             uint32_t peer = 0,
             uint32_t channel = 0) {
    nixlMemViewElem source{src, 0, 0}, destination{dst, peer, 0};
    result->submit = atomic ? nixlAtomicAdd(42, destination, channel, 0, &result->transfer) :
                              nixlPut(source, destination, 0, channel, 0, &result->transfer);
}

__global__ void
pollKernel(DeviceResult *result, bool wait = false) {
    do {
        result->poll = nixlGpuGetXferStatus(result->transfer);
    } while (wait && result->poll == NIXL_IN_PROG);
}

__global__ void
putLoopKernel(nixlMemViewH src, nixlMemViewH dst, uint32_t count, DeviceResult *results) {
    nixlMemViewElem source{src, 0, 0}, destination{dst, 0, 0};
    for (uint32_t i = 0; i < count; ++i) {
        auto &result = results[i];
        result.submit = nixlPut(source, destination, 0, 0, 0, &result.transfer);
        if (result.submit != NIXL_IN_PROG) {
            return;
        }
        do {
            result.poll = nixlGpuGetXferStatus(result.transfer);
        } while (result.poll == NIXL_IN_PROG);
        if (result.poll != NIXL_SUCCESS) {
            return;
        }
    }
}

__global__ void
putBurstKernel(nixlMemViewH src, nixlMemViewH dst, uint32_t count, nixl_status_t *statuses) {
    nixlMemViewElem source{src, 0, 0}, destination{dst, 0, 0};
    for (uint32_t i = 0; i < count; ++i) {
        statuses[i] = nixlPut(source, destination, 0, 0);
    }
}

class ProxyDeviceApiTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        if (!gtest::hasCudaGpu()) {
            GTEST_SKIP() << "No CUDA-capable GPU, skipping proxy device API test.";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    template<typename T>
    T
    deviceGet(T *ptr) {
        T value{};
        EXPECT_EQ(cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        return value;
    }

    template<typename T>
    T *
    deviceAlloc(size_t count = 1) {
        allocations_.emplace_back();
        auto *ops = nixl::getDeviceOps();
        EXPECT_NE(ops, nullptr);
        if (ops == nullptr) {
            return nullptr;
        }
        EXPECT_EQ(ops->allocDeviceMem(sizeof(T) * count, allocations_.back()), NIXL_SUCCESS);
        EXPECT_EQ(cudaMemset(allocations_.back().get(), 0, sizeof(T) * count), cudaSuccess);
        return static_cast<T *>(allocations_.back().get());
    }

    nixl_status_t
    poll(DeviceResult *result) {
        pollKernel<<<1, 1>>>(result);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return deviceGet(result).poll;
    }

    template<typename Predicate>
    bool
    waitForCondition(Predicate predicate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return predicate();
    }

private:
    std::vector<nixl::deviceMem> allocations_;
};

__global__ void
proxyTokenLayoutKernel(nixlProxyDeviceMemView *view, uint64_t *out) {
    out[0] = nixl::gpu::impl::proxy::proxyHostViewFromHandle(view);
    out[1] = reinterpret_cast<uintptr_t>(nixl::gpu::impl::proxy::getPtr(view, 0));
    out[2] = nixl::gpu::impl::proxy::proxyContextFromMemView(view) == nullptr;
}

TEST_F(ProxyDeviceApiTest, TokenLayoutAndGetPtr) {
    auto *allocator = nixl::getDeviceOps();
    ASSERT_NE(allocator, nullptr);
    nixl::deviceMem context_mem, view_mem, output;
    ASSERT_EQ(allocator->allocDeviceMem(sizeof(nixlProxyDeviceContextData), context_mem),
              NIXL_SUCCESS);
    ASSERT_EQ(allocator->allocDeviceMem(nixlProxyDeviceMemViewBytes(1), view_mem), NIXL_SUCCESS);
    ASSERT_EQ(allocator->allocDeviceMem(3 * sizeof(uint64_t), output), NIXL_SUCCESS);
    nixlProxyDeviceContextData context;
    nixlProxyDeviceMemView view{
        0xfedcba9876543210ULL, static_cast<nixlProxyDeviceContextData *>(context_mem.get()), 1};
    void *direct = reinterpret_cast<void *>(uintptr_t{0x12340000});
    auto *device_view = static_cast<nixlProxyDeviceMemView *>(view_mem.get());
    ASSERT_EQ(allocator->copy(
                  device_view, &view, sizeof(view), nixl::deviceOps::copyDirection::HostToDevice),
              NIXL_SUCCESS);
    ASSERT_EQ(allocator->copy(nixlProxyDeviceMemViewDirectPtrs(device_view),
                              &direct,
                              sizeof(direct),
                              nixl::deviceOps::copyDirection::HostToDevice),
              NIXL_SUCCESS);
    ASSERT_EQ(allocator->copy(context_mem.get(),
                              &context,
                              sizeof(context),
                              nixl::deviceOps::copyDirection::HostToDevice),
              NIXL_SUCCESS);
    proxyTokenLayoutKernel<<<1, 1>>>(device_view, static_cast<uint64_t *>(output.get()));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint64_t result[3]{};
    ASSERT_EQ(
        allocator->copy(
            result, output.get(), sizeof(result), nixl::deviceOps::copyDirection::DeviceToHost),
        NIXL_SUCCESS);
    EXPECT_EQ(result[0], view.host_view);
    EXPECT_EQ(result[1], reinterpret_cast<uintptr_t>(direct));
    EXPECT_EQ(result[2], 0u);
}

TEST_F(ProxyDeviceApiTest, ImmediatePutAndAtomicCompletionRoundTrip) {
    std::unique_ptr<nixl::proxyRuntime> runtime;
    ASSERT_EQ(nixl::proxyRuntime::create(
                  std::make_unique<ImmediateTransport>(), makeProxyConfig(1, 1, 1), runtime),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime->startWorkers(), NIXL_SUCCESS);
    const DummyProxyMemViews views(*runtime);
    auto *result = deviceAlloc<DeviceResult>();
    for (bool atomic : {false, true}) {
        SCOPED_TRACE(atomic ? "atomic" : "put");
        submitKernel<<<1, 1>>>(views.src, views.dst, result, atomic);
        pollKernel<<<1, 1>>>(result, true);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(deviceGet(result).submit, NIXL_IN_PROG);
        EXPECT_EQ(deviceGet(result).poll, NIXL_SUCCESS);
    }
    ASSERT_EQ(runtime->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, PutPutAtomicAddCompletionFrontier) {
    auto transport = std::make_unique<ControllableBackend>();
    ControllableBackend &backend = *transport;
    std::unique_ptr<nixl::proxyRuntime> runtime;
    ASSERT_EQ(nixl::proxyRuntime::create(std::move(transport), makeProxyConfig(1, 1, 1), runtime),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime->startWorkers(), NIXL_SUCCESS);
    const DummyProxyMemViews views(*runtime);
    const auto ring = deviceGet(runtime->deviceChannelViews()[0].work_ring);
    auto *results = deviceAlloc<DeviceResult>(3);
    for (int i = 0; i < 3; ++i) {
        submitKernel<<<1, 1>>>(views.src, views.dst, results + i, i == 2);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(waitForCondition([&] { return backend.entries().size() == 3; }));
    const auto entries = backend.entries();
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(deviceGet(results + i).submit, NIXL_IN_PROG);
        EXPECT_EQ(entries[i].submission.opcode,
                  i == 2 ? nixl_proxy_opcode_t::ATOMIC_ADD : nixl_proxy_opcode_t::PUT);
        EXPECT_EQ(poll(results + i), NIXL_IN_PROG);
    }

    pollKernel<<<1, 1>>>(results, true);
    EXPECT_EQ(cudaStreamQuery(nullptr), cudaErrorNotReady);
    backend.complete(0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(results).poll, NIXL_SUCCESS);
    for (int i = 1; i < 3; ++i) {
        EXPECT_EQ(poll(results + i), NIXL_IN_PROG);
        backend.complete(i);
        ASSERT_TRUE(waitForCondition([&] { return poll(results + i) == NIXL_SUCCESS; }));
    }
    ASSERT_TRUE(waitForCondition([&] { return deviceGet(ring.consumer_idx) == 3; }));
    ASSERT_EQ(runtime->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, EarlierCompletionStaysSuccessfulAfterLaterError) {
    auto transport = std::make_unique<ControllableBackend>();
    ControllableBackend &backend = *transport;
    std::unique_ptr<nixl::proxyRuntime> runtime;
    ASSERT_EQ(nixl::proxyRuntime::create(std::move(transport), makeProxyConfig(1, 1, 1), runtime),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime->startWorkers(), NIXL_SUCCESS);
    const DummyProxyMemViews views(*runtime);
    auto *results = deviceAlloc<DeviceResult>(2);
    for (int i = 0; i < 2; ++i) {
        submitKernel<<<1, 1>>>(views.src, views.dst, results + i);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(waitForCondition([&] { return backend.entries().size() == 2; }));
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(deviceGet(results + i).submit, NIXL_IN_PROG);
        backend.complete(i, i == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
        ASSERT_TRUE(waitForCondition(
            [&] { return poll(results + i) == (i == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND); }));
    }
    EXPECT_EQ(poll(results), NIXL_SUCCESS);
    ASSERT_EQ(runtime->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, SubmitFailurePropagatesErrorStatus) {
    const gtest::LogIgnoreGuard lig("proxyChannel::submitRecord: backend submit failed");
    std::atomic<unsigned> submits{0}, checks{0};

    struct FailingTransport : ImmediateTransport {
        std::atomic<unsigned> &submits;
        std::atomic<unsigned> &checks;

        FailingTransport(std::atomic<unsigned> &submit_calls, std::atomic<unsigned> &check_calls)
            : submits(submit_calls),
              checks(check_calls) {}

        nixl_status_t
        submit(const nixl::proxyBackendSubmission &, nixl::proxyBackendRequest &) override {
            ++submits;
            return NIXL_ERR_BACKEND;
        }

        nixl_status_t
        checkCompletion(uint32_t, uint32_t, const nixl::proxyBackendRequest &) override {
            ++checks;
            return NIXL_SUCCESS;
        }
    };

    std::unique_ptr<nixl::proxyRuntime> runtime;
    ASSERT_EQ(nixl::proxyRuntime::create(std::make_unique<FailingTransport>(submits, checks),
                                         makeProxyConfig(1, 1, 1),
                                         runtime),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime->startWorkers(), NIXL_SUCCESS);
    const DummyProxyMemViews views(*runtime);
    auto *result = deviceAlloc<DeviceResult>();
    submitKernel<<<1, 1>>>(views.src, views.dst, result);
    pollKernel<<<1, 1>>>(result, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(result).submit, NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(result).poll, NIXL_ERR_BACKEND);
    EXPECT_EQ(submits.load(), 1u);
    EXPECT_EQ(checks.load(), 0u);
    ASSERT_EQ(runtime->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, RingSlotsAreReusedAfterWraparound) {
    std::unique_ptr<nixl::proxyRuntime> runtime;
    ASSERT_EQ(nixl::proxyRuntime::create(
                  std::make_unique<ImmediateTransport>(), makeProxyConfig(1, 1, 1), runtime),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime->startWorkers(), NIXL_SUCCESS);
    const DummyProxyMemViews views(*runtime);
    constexpr uint32_t count = nixl::kDefaultProxyRingDepth + 3;
    auto *results = deviceAlloc<DeviceResult>(count);
    putLoopKernel<<<1, 1>>>(views.src, views.dst, count, results);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<DeviceResult> host(count);
    ASSERT_EQ(
        cudaMemcpy(host.data(), results, sizeof(DeviceResult) * count, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(host[i].submit, NIXL_IN_PROG) << i;
        EXPECT_EQ(host[i].poll, NIXL_SUCCESS) << i;
    }
    const auto ring = deviceGet(runtime->deviceChannelViews()[0].work_ring);
    EXPECT_EQ(deviceGet(ring.producer_idx), count);
    ASSERT_TRUE(waitForCondition([&] { return deviceGet(ring.consumer_idx) == count; }));
    ASSERT_EQ(runtime->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, FullRingResumesWhenWorkersStart) {
    std::unique_ptr<nixl::proxyRuntime> runtime;
    ASSERT_EQ(nixl::proxyRuntime::create(
                  std::make_unique<ImmediateTransport>(), makeProxyConfig(1, 1, 1), runtime),
              NIXL_SUCCESS);
    const DummyProxyMemViews views(*runtime);
    constexpr uint32_t count = nixl::kDefaultProxyRingDepth + 1;
    auto *statuses = deviceAlloc<nixl_status_t>(count);
    putBurstKernel<<<1, 1>>>(views.src, views.dst, count, statuses);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(cudaStreamQuery(nullptr), cudaErrorNotReady);
    ASSERT_EQ(runtime->startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<nixl_status_t> host(count);
    ASSERT_EQ(
        cudaMemcpy(host.data(), statuses, sizeof(nixl_status_t) * count, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (const auto status : host) {
        EXPECT_EQ(status, NIXL_IN_PROG);
    }
    ASSERT_EQ(runtime->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, PeerAndChannelRoutingKeepsCompletionsIndependent) {
    auto transport = std::make_unique<ControllableBackend>();
    ControllableBackend &backend = *transport;
    std::unique_ptr<nixl::proxyRuntime> runtime;
    ASSERT_EQ(nixl::proxyRuntime::create(std::move(transport), makeProxyConfig(2, 2, 2), runtime),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime->startWorkers(), NIXL_SUCCESS);
    const DummyProxyMemViews views(*runtime, 2);
    auto *results = deviceAlloc<DeviceResult>(2);
    submitKernel<<<1, 1>>>(views.src, views.dst, results, false, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(results).submit, NIXL_ERR_INVALID_PARAM);
    submitKernel<<<1, 1>>>(views.src, views.dst, results, false, 0, 0);
    submitKernel<<<1, 1>>>(views.src, views.dst, results + 1, false, 1, 5);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(waitForCondition([&] { return backend.entries().size() == 2; }));
    const auto entries = backend.entries();
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto &submission = entries[i].submission;
        EXPECT_EQ(submission.channel_id, submission.peer_index);
        EXPECT_EQ(deviceGet(results + i).submit, NIXL_IN_PROG);
    }
    // Backend arrival order may differ across workers.
    for (uint32_t channel : {1u, 0u}) {
        EXPECT_EQ(poll(results), NIXL_IN_PROG);
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].submission.channel_id == channel) {
                backend.complete(i);
            }
        }
        ASSERT_TRUE(waitForCondition([&] { return poll(results + channel) == NIXL_SUCCESS; }));
    }
    EXPECT_EQ(poll(results + 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime->shutdown(), NIXL_SUCCESS);
}
