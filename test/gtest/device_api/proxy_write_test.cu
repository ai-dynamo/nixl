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

// Verifies that the proxy backend compiles, links, and that GPU kernels can
// reach the ProxyDeviceContext published by ProxyRuntime::startWorkers().
//
// nixl_device.cuh resolves to proxy/nixl_device.cuh via the proxy include
// path supplied by the build system - no backend macro is needed.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <nixl_device.cuh>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "device_proxy/proxy_runtime.h"
#include "device_proxy/backend_adapter.h"
#include "common.h"

// ---------------------------------------------------------------------------
// Minimal stub backend — satisfies the pure-virtual interface without doing
// any real I/O.  Sufficient for testing the runtime lifecycle and the GPU
// device-context path.
// ---------------------------------------------------------------------------
class StubProxyBackendAdapter : public nixlDeviceProxyBackendAdapter {
public:
    nixl_status_t
    init(uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const nixlBackendProxySubmission &, uint64_t &token) override {
        token = 0;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    checkCompletion(uint64_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    progress() override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    shutdown() override {
        return NIXL_SUCCESS;
    }
};

// ---------------------------------------------------------------------------
// Controllable stub — lets the test thread decide when each submission
// completes.  submit() assigns unique monotonic tokens; checkCompletion()
// returns NIXL_IN_PROG until markComplete() is called for a token.
// ---------------------------------------------------------------------------
class ControllableStubAdapter : public nixlDeviceProxyBackendAdapter {
public:
    nixl_status_t
    init(uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const nixlBackendProxySubmission &submission, uint64_t &token) override {
        std::lock_guard<std::mutex> lk(mu_);
        token = next_token_++;
        pending_.insert(token);
        token_channel_[token] = submission.channel_id;
        submitted_opcodes_.push_back(submission.opcode);
        return NIXL_SUCCESS;
    }

    nixl_status_t
    checkCompletion(uint64_t token) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = completed_.find(token);
        if (it != completed_.end()) {
            nixl_status_t status = it->second;
            completed_.erase(it);
            return status;
        }
        return NIXL_IN_PROG;
    }

    nixl_status_t
    progress() override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    shutdown() override {
        return NIXL_SUCCESS;
    }

    void
    markComplete(uint64_t token) {
        markCompleteWithStatus(token, NIXL_SUCCESS);
    }

    void
    markCompleteWithStatus(uint64_t token, nixl_status_t status) {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.erase(token);
        completed_[token] = status;
    }

    bool
    hasPending() const {
        std::lock_guard<std::mutex> lk(mu_);
        return !pending_.empty();
    }

    size_t
    pendingCount() const {
        std::lock_guard<std::mutex> lk(mu_);
        return pending_.size();
    }

    bool
    hasPendingForChannel(uint32_t channel_id) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (uint64_t token : pending_) {
            auto it = token_channel_.find(token);
            if (it != token_channel_.end() && it->second == channel_id) {
                return true;
            }
        }
        return false;
    }

    bool
    markFirstPendingForChannel(uint32_t channel_id, uint64_t *token = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        for (uint64_t pending_token : pending_) {
            auto it = token_channel_.find(pending_token);
            if (it != token_channel_.end() && it->second == channel_id) {
                pending_.erase(pending_token);
                completed_[pending_token] = NIXL_SUCCESS;
                if (token != nullptr) {
                    *token = pending_token;
                }
                return true;
            }
        }
        return false;
    }

    std::vector<nixl_proxy_opcode_t>
    submittedOpcodes() const {
        std::lock_guard<std::mutex> lk(mu_);
        return submitted_opcodes_;
    }

private:
    mutable std::mutex mu_;
    uint64_t next_token_ = 1;
    std::set<uint64_t> pending_;
    std::map<uint64_t, nixl_status_t> completed_;
    std::map<uint64_t, uint32_t> token_channel_;
    std::vector<nixl_proxy_opcode_t> submitted_opcodes_;
};

// ---------------------------------------------------------------------------
// Error-returning stub — submit succeeds but checkCompletion always returns
// NIXL_ERR_BACKEND, simulating a backend failure.
// ---------------------------------------------------------------------------
class ErrorStubAdapter : public nixlDeviceProxyBackendAdapter {
public:
    nixl_status_t
    init(uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const nixlBackendProxySubmission &, uint64_t &token) override {
        token = 0;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    checkCompletion(uint64_t) override {
        return NIXL_ERR_BACKEND;
    }

    nixl_status_t
    progress() override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    shutdown() override {
        return NIXL_SUCCESS;
    }
};

// ---------------------------------------------------------------------------
// Submit-error stub - submit() fails immediately and should be published back
// to the GPU without going through checkCompletion().
// ---------------------------------------------------------------------------
class SubmitErrorStubAdapter : public nixlDeviceProxyBackendAdapter {
public:
    nixl_status_t
    init(uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const nixlBackendProxySubmission &, uint64_t &) override {
        ++submit_calls_;
        return NIXL_ERR_BACKEND;
    }

    nixl_status_t
    checkCompletion(uint64_t) override {
        ++check_completion_calls_;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    progress() override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    shutdown() override {
        return NIXL_SUCCESS;
    }

    std::atomic<uint64_t> submit_calls_{0};
    std::atomic<uint64_t> check_completion_calls_{0};
};

class DummyBackendMD : public nixlBackendMD {
public:
    DummyBackendMD() : nixlBackendMD(false) {}
};

struct DummyProxyMemViews {
    nixlMemViewH src = nullptr;
    nixlMemViewH dst = nullptr;
};

static DummyProxyMemViews
registerDummyMemViews(nixlProxyRuntime &runtime);

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Writes true if load_proxy_context() returns a non-null pointer.
__global__ void
proxyContextKernel(bool *out_has_ctx) {
    *out_has_ctx = (load_proxy_context() != nullptr);
}

// Calls nixlPut with zero-initialised operands and records the status.
__global__ void
proxyPutKernel(nixlMemViewH src_mvh, nixlMemViewH dst_mvh, nixl_status_t *out_status) {
    nixlMemViewElem src{src_mvh, 0, 0}, dst{dst_mvh, 0, 0};
    *out_status = nixlPut(src, dst, /*size=*/0);
}

__global__ void
proxyAtomicAddKernel(nixlMemViewH counter_mvh, uint64_t value, nixl_status_t *out_status) {
    nixlMemViewElem counter{counter_mvh, 0, 0};
    *out_status = nixlAtomicAdd(value, counter);
}

static void
publishProxyContext(nixlProxyRuntime &runtime) {
    bool *d_warmup = nullptr;
    ASSERT_EQ(cudaMalloc(&d_warmup, sizeof(bool)), cudaSuccess);
    proxyContextKernel<<<1, 1>>>(d_warmup);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaFree(d_warmup), cudaSuccess);

    ASSERT_NE(runtime.deviceContext(), nullptr);
    ASSERT_EQ(nixlProxyPublishContext(runtime.deviceContext()), cudaSuccess);
}

static void
clearProxyContext() {
    ASSERT_EQ(nixlProxyClearContext(), cudaSuccess);
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

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
    deviceGet(T *d_ptr) {
        T val{};
        cudaMemcpy(&val, d_ptr, sizeof(T), cudaMemcpyDeviceToHost);
        return val;
    }

    template<typename T>
    T *
    deviceAlloc() {
        T *ptr = nullptr;
        EXPECT_EQ(cudaMalloc(&ptr, sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMemset(ptr, 0, sizeof(T)), cudaSuccess);
        return ptr;
    }

    template<typename Predicate>
    bool
    waitForCondition(Predicate predicate,
                     std::chrono::milliseconds timeout = std::chrono::milliseconds(500)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return predicate();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// After startWorkers() the GPU should see a non-null proxy context.
TEST_F(ProxyDeviceApiTest, ContextPublishedAfterStartWorkers) {
    auto adapter = std::make_unique<StubProxyBackendAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    bool *d_has_ctx = deviceAlloc<bool>();
    proxyContextKernel<<<1, 1>>>(d_has_ctx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_TRUE(deviceGet(d_has_ctx));
    cudaFree(d_has_ctx);

    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// After shutdown() the GPU should no longer see a proxy context.
TEST_F(ProxyDeviceApiTest, ContextClearedAfterShutdown) {
    auto adapter = std::make_unique<StubProxyBackendAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
    clearProxyContext();

    bool *d_has_ctx = deviceAlloc<bool>();
    // Initialise to true so a no-op kernel would give a false pass.
    bool init_val = true;
    cudaMemcpy(d_has_ctx, &init_val, sizeof(bool), cudaMemcpyHostToDevice);

    proxyContextKernel<<<1, 1>>>(d_has_ctx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_FALSE(deviceGet(d_has_ctx));
    cudaFree(d_has_ctx);
}

// nixlPut() via the proxy backend should report NIXL_IN_PROG once the
// submission is accepted into the proxy ring.
TEST_F(ProxyDeviceApiTest, PutReturnsInProgWhenEnqueued) {
    auto adapter = std::make_unique<StubProxyBackendAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);
    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_status = deviceAlloc<nixl_status_t>();
    proxyPutKernel<<<1, 1>>>(mvhs.src, mvhs.dst, d_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_status), NIXL_IN_PROG);
    cudaFree(d_status);

    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, AtomicAddReturnsInProgWhenEnqueued) {
    auto adapter = std::make_unique<StubProxyBackendAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);
    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_status = deviceAlloc<nixl_status_t>();
    proxyAtomicAddKernel<<<1, 1>>>(mvhs.dst, 42, d_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_status), NIXL_IN_PROG);
    cudaFree(d_status);

    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// ---------------------------------------------------------------------------
// Completion round-trip kernels
//
// Kernels that spin on pollXferStatus require valid proxy memview handles so
// the worker's dispatch succeeds and a completion is actually published.
// The test registers a dummy memview and passes the proxy handle here.
// ---------------------------------------------------------------------------

// Enqueues a put and spins until pollXferStatus returns a final status.
__global__ void
proxyPutAndPollKernel(nixlMemViewH src_mvh,
                      nixlMemViewH dst_mvh,
                      uint32_t channel_id,
                      nixl_status_t *out_put_status,
                      nixl_status_t *out_poll_status) {
    nixlMemViewElem src{src_mvh, 0, 0}, dst{dst_mvh, 0, 0};
    nixlGpuXferStatusH xfer_status{};
    *out_put_status = nixlPut(src,
                              dst,
                              /*size=*/0,
                              channel_id,
                              /*flags=*/0,
                              &xfer_status);

    nixl_status_t poll;
    do {
        poll = nixlGpuGetXferStatus(xfer_status);
    } while (poll == NIXL_IN_PROG);
    *out_poll_status = poll;
}

__global__ void
proxyAtomicAddAndPollKernel(nixlMemViewH counter_mvh,
                            uint64_t value,
                            uint32_t channel_id,
                            nixl_status_t *out_atomic_status,
                            nixl_status_t *out_poll_status) {
    nixlMemViewElem counter{counter_mvh, 0, 0};
    nixlGpuXferStatusH xfer_status{};
    *out_atomic_status = nixlAtomicAdd(value,
                                       counter,
                                       channel_id,
                                       /*flags=*/0,
                                       &xfer_status);

    nixl_status_t poll;
    do {
        poll = nixlGpuGetXferStatus(xfer_status);
    } while (poll == NIXL_IN_PROG);
    *out_poll_status = poll;
}

// Enqueues a put and immediately returns; saves xfer_status to device memory
// so the test thread can later launch a poll kernel.
__global__ void
proxyPutAsyncKernel(nixlMemViewH src_mvh,
                    nixlMemViewH dst_mvh,
                    uint32_t channel_id,
                    nixl_status_t *out_put_status,
                    nixlGpuXferStatusH *out_xfer_status) {
    nixlMemViewElem src{src_mvh, 0, 0}, dst{dst_mvh, 0, 0};
    *out_put_status = nixlPut(src,
                              dst,
                              /*size=*/0,
                              channel_id,
                              /*flags=*/0,
                              out_xfer_status);
}

__global__ void
proxyAtomicAddAsyncKernel(nixlMemViewH counter_mvh,
                          uint64_t value,
                          uint32_t channel_id,
                          nixl_status_t *out_atomic_status,
                          nixlGpuXferStatusH *out_xfer_status) {
    nixlMemViewElem counter{counter_mvh, 0, 0};
    *out_atomic_status = nixlAtomicAdd(value,
                                       counter,
                                       channel_id,
                                       /*flags=*/0,
                                       out_xfer_status);
}

// Enqueues op_count puts on one channel and records each immediate enqueue
// status. The final submission may block if the ring is full.
__global__ void
proxyPutBurstKernel(uint32_t op_count, uint32_t channel_id, nixl_status_t *out_put_statuses) {
    nixlMemViewElem src{}, dst{};
    for (uint32_t i = 0; i < op_count; ++i) {
        out_put_statuses[i] = nixlPut(src, dst, /*size=*/0, channel_id);
    }
}

// Non-blocking single poll: returns current status without spinning.
__global__ void
proxyPollOnceKernel(nixlGpuXferStatusH *xfer_status, nixl_status_t *out_poll_status) {
    *out_poll_status = nixlGpuGetXferStatus(*xfer_status);
}

// ---------------------------------------------------------------------------
// Completion round-trip helpers
// ---------------------------------------------------------------------------

// Register one local and one remote proxy memview so dispatch can prepare
// transport-ready descriptors before submit().
static DummyProxyMemViews
registerDummyMemViews(nixlProxyRuntime &runtime) {
    static DummyBackendMD local_md;
    static DummyBackendMD remote_md;

    DummyProxyMemViews handles;
    nixlMemViewH dummy_local_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0xBEEF});
    nixlMemViewH dummy_remote_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0xFEED});

    EXPECT_EQ(runtime.registerProxyMemView(dummy_local_backend, &handles.src), NIXL_SUCCESS);
    EXPECT_EQ(runtime.registerProxyMemView(dummy_remote_backend, &handles.dst), NIXL_SUCCESS);

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
    EXPECT_EQ(runtime.storeMetadata(handles.src, local_dlist), NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    EXPECT_EQ(runtime.storeMetadata(handles.dst, remote_dlist), NIXL_SUCCESS);

    return handles;
}

static uint32_t *
shutdownWordHostFromRuntime(nixlProxyRuntime &runtime) {
    nixlProxyDeviceContextData device_ctx{};
    if (runtime.deviceContext() == nullptr) {
        return nullptr;
    }
    if (cudaMemcpy(
            &device_ctx, runtime.deviceContext(), sizeof(device_ctx), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        return nullptr;
    }
    if (device_ctx.shutdown_word == nullptr) {
        return nullptr;
    }

    cudaPointerAttributes attrs{};
    if (cudaPointerGetAttributes(&attrs, device_ctx.shutdown_word) != cudaSuccess) {
        return nullptr;
    }
    return static_cast<uint32_t *>(attrs.hostPointer);
}

static void
signalProxyShutdown(uint32_t *shutdown_host) {
    __atomic_store_n(shutdown_host,
                     static_cast<uint32_t>(nixl_proxy_control_state_t::SHUTDOWN),
                     __ATOMIC_RELEASE);
}

// ---------------------------------------------------------------------------
// Completion round-trip tests
// ---------------------------------------------------------------------------

// Full round-trip: GPU enqueues -> worker dequeues -> backend completes
// (immediately via StubProxyBackendAdapter) -> worker publishes -> GPU polls
// NIXL_SUCCESS.
TEST_F(ProxyDeviceApiTest, PutCompletionRoundTrip) {
    auto adapter = std::make_unique<StubProxyBackendAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_put_status = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    proxyPutAndPollKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status, d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_SUCCESS);

    cudaFree(d_put_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, AtomicAddCompletionRoundTrip) {
    auto adapter = std::make_unique<StubProxyBackendAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_atomic_status = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    proxyAtomicAddAndPollKernel<<<1, 1>>>(mvhs.dst, 42, 0, d_atomic_status, d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_atomic_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_SUCCESS);

    cudaFree(d_atomic_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Verifies that the GPU kernel stays spinning until the test thread
// explicitly marks the backend token complete.
TEST_F(ProxyDeviceApiTest, CompletionNotVisibleUntilPublished) {
    auto adapter_owner = std::make_unique<ControllableStubAdapter>();
    auto *adapter = adapter_owner.get();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter_owner), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_put_status = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    // Launch async — kernel will spin on pollXferStatus.
    proxyPutAndPollKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status, d_poll_status);

    // Give the worker time to pick up and submit the request.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Kernel should still be running (spinning on completion).
    ASSERT_EQ(cudaStreamQuery(nullptr), cudaErrorNotReady);

    // Release the completion from the test thread.
    adapter->markComplete(1);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_SUCCESS);

    cudaFree(d_put_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Enqueue 3 operations, complete them in order, and verify the collapsed-CQ
// frontier semantics: each pollXferStatus returns NIXL_SUCCESS only after its
// op_idx has been reached.
TEST_F(ProxyDeviceApiTest, MultipleSubmissionsCompletionFrontier) {
    auto adapter_owner = std::make_unique<ControllableStubAdapter>();
    auto *adapter = adapter_owner.get();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter_owner), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    constexpr int kOps = 3;
    nixl_status_t *d_put_status[kOps];
    nixlGpuXferStatusH *d_xfer_status[kOps];

    for (int i = 0; i < kOps; i++) {
        d_put_status[i] = deviceAlloc<nixl_status_t>();
        ASSERT_EQ(cudaMalloc(&d_xfer_status[i], sizeof(nixlGpuXferStatusH)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_xfer_status[i], 0, sizeof(nixlGpuXferStatusH)), cudaSuccess);
    }

    // Enqueue 3 operations sequentially (each kernel returns after enqueue).
    for (int i = 0; i < kOps; i++) {
        proxyPutAsyncKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status[i], d_xfer_status[i]);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_put_status[i]), NIXL_IN_PROG);
    }

    // All three should still be in-progress.
    nixl_status_t *d_poll = deviceAlloc<nixl_status_t>();
    for (int i = 0; i < kOps; i++) {
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[i], d_poll);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_poll), NIXL_IN_PROG)
            << "op " << i << " should be in-progress before any markComplete";
    }

    // Complete them one at a time and verify frontier advances.
    for (int i = 0; i < kOps; i++) {
        adapter->markComplete(static_cast<uint64_t>(i + 1));

        // Give worker time to publish.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Poll this op — should now be complete.
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[i], d_poll);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_poll), NIXL_SUCCESS)
            << "op " << i << " should be complete after markComplete";
    }

    cudaFree(d_poll);
    for (int i = 0; i < kOps; i++) {
        cudaFree(d_put_status[i]);
        cudaFree(d_xfer_status[i]);
    }
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyDeviceApiTest, PutPutAtomicAddCompletionFrontier) {
    auto adapter_owner = std::make_unique<ControllableStubAdapter>();
    auto *adapter = adapter_owner.get();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter_owner), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    constexpr int kOps = 3;
    nixl_status_t *d_submit_status[kOps];
    nixlGpuXferStatusH *d_xfer_status[kOps];

    for (int i = 0; i < kOps; i++) {
        d_submit_status[i] = deviceAlloc<nixl_status_t>();
        ASSERT_EQ(cudaMalloc(&d_xfer_status[i], sizeof(nixlGpuXferStatusH)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_xfer_status[i], 0, sizeof(nixlGpuXferStatusH)), cudaSuccess);
    }

    proxyPutAsyncKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_submit_status[0], d_xfer_status[0]);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_submit_status[0]), NIXL_IN_PROG);

    proxyPutAsyncKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_submit_status[1], d_xfer_status[1]);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_submit_status[1]), NIXL_IN_PROG);

    proxyAtomicAddAsyncKernel<<<1, 1>>>(mvhs.dst, 42, 0, d_submit_status[2], d_xfer_status[2]);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_submit_status[2]), NIXL_IN_PROG);

    ASSERT_TRUE(waitForCondition([adapter]() { return adapter->pendingCount() == kOps; }));
    EXPECT_EQ(
        adapter->submittedOpcodes(),
        std::vector<nixl_proxy_opcode_t>(
            {nixl_proxy_opcode_t::PUT, nixl_proxy_opcode_t::PUT, nixl_proxy_opcode_t::ATOMIC_ADD}));

    nixl_status_t *d_poll = deviceAlloc<nixl_status_t>();
    for (int i = 0; i < kOps; i++) {
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[i], d_poll);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_poll), NIXL_IN_PROG)
            << "op " << i << " should be in-progress before any markComplete";
    }

    for (int i = 0; i < kOps; i++) {
        adapter->markComplete(static_cast<uint64_t>(i + 1));
        ASSERT_TRUE(waitForCondition([&]() {
            proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[i], d_poll);
            return cudaDeviceSynchronize() == cudaSuccess && deviceGet(d_poll) == NIXL_SUCCESS;
        })) << "op "
            << i << " should complete after markComplete";
    }

    cudaFree(d_poll);
    for (int i = 0; i < kOps; i++) {
        cudaFree(d_submit_status[i]);
        cudaFree(d_xfer_status[i]);
    }
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Once a later op publishes an error, an earlier op whose op_idx is already
// behind the completion frontier must still observe NIXL_SUCCESS.
TEST_F(ProxyDeviceApiTest, EarlierCompletionStaysSuccessfulAfterLaterError) {
    auto adapter_owner = std::make_unique<ControllableStubAdapter>();
    auto *adapter = adapter_owner.get();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter_owner), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_put_status[2];
    nixlGpuXferStatusH *d_xfer_status[2];
    for (int i = 0; i < 2; ++i) {
        d_put_status[i] = deviceAlloc<nixl_status_t>();
        ASSERT_EQ(cudaMalloc(&d_xfer_status[i], sizeof(nixlGpuXferStatusH)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_xfer_status[i], 0, sizeof(nixlGpuXferStatusH)), cudaSuccess);
        proxyPutAsyncKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status[i], d_xfer_status[i]);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_put_status[i]), NIXL_IN_PROG);
    }

    ASSERT_TRUE(waitForCondition([&]() { return adapter->pendingCount() == 2; }));

    nixl_status_t *d_poll = deviceAlloc<nixl_status_t>();
    adapter->markCompleteWithStatus(1, NIXL_SUCCESS);
    ASSERT_TRUE(waitForCondition([&]() {
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[0], d_poll);
        return cudaDeviceSynchronize() == cudaSuccess && deviceGet(d_poll) == NIXL_SUCCESS;
    }));

    adapter->markCompleteWithStatus(2, NIXL_ERR_BACKEND);
    ASSERT_TRUE(waitForCondition([&]() {
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[1], d_poll);
        return cudaDeviceSynchronize() == cudaSuccess && deviceGet(d_poll) == NIXL_ERR_BACKEND;
    }));

    proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[0], d_poll);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_poll), NIXL_SUCCESS);

    cudaFree(d_poll);
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_put_status[i]);
        cudaFree(d_xfer_status[i]);
    }
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Once an earlier op publishes a terminal error, later queued ops must also
// observe that error instead of spinning forever.
TEST_F(ProxyDeviceApiTest, EarlierErrorPropagatesToLaterQueuedOp) {
    auto adapter_owner = std::make_unique<ControllableStubAdapter>();
    auto *adapter = adapter_owner.get();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter_owner), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_put_status[2];
    nixlGpuXferStatusH *d_xfer_status[2];
    for (int i = 0; i < 2; ++i) {
        d_put_status[i] = deviceAlloc<nixl_status_t>();
        ASSERT_EQ(cudaMalloc(&d_xfer_status[i], sizeof(nixlGpuXferStatusH)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_xfer_status[i], 0, sizeof(nixlGpuXferStatusH)), cudaSuccess);
        proxyPutAsyncKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status[i], d_xfer_status[i]);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_put_status[i]), NIXL_IN_PROG);
    }

    ASSERT_TRUE(waitForCondition([&]() { return adapter->pendingCount() == 2; }));

    nixl_status_t *d_poll = deviceAlloc<nixl_status_t>();
    adapter->markCompleteWithStatus(1, NIXL_ERR_BACKEND);
    ASSERT_TRUE(waitForCondition([&]() {
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[1], d_poll);
        return cudaDeviceSynchronize() == cudaSuccess && deviceGet(d_poll) == NIXL_ERR_BACKEND;
    }));

    proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[0], d_poll);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_poll), NIXL_ERR_BACKEND);

    cudaFree(d_poll);
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_put_status[i]);
        cudaFree(d_xfer_status[i]);
    }
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Backend returns NIXL_ERR_BACKEND on checkCompletion; verify the GPU kernel
// receives the error status through the completion slot.
TEST_F(ProxyDeviceApiTest, CompletionPropagatesErrorStatus) {
    auto adapter = std::make_unique<ErrorStubAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_put_status = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    proxyPutAndPollKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status, d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_ERR_BACKEND);

    cudaFree(d_put_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Backend submit() failure should be published to the GPU as the terminal
// transfer status, without going through checkCompletion().
TEST_F(ProxyDeviceApiTest, SubmitFailurePropagatesErrorStatus) {
    const gtest::LogIgnoreGuard lig("ProxyWorker::submitToBackend: backend submit failed");
    auto adapter_owner = std::make_unique<SubmitErrorStubAdapter>();
    auto *adapter = adapter_owner.get();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter_owner), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_put_status = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    proxyPutAndPollKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status, d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_ERR_BACKEND);
    EXPECT_EQ(adapter->submit_calls_.load(), 1u);
    EXPECT_EQ(adapter->check_completion_calls_.load(), 0u);

    cudaFree(d_put_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// When the ring is full and no worker can drain it, the next enqueue should
// spin until shutdown is signalled and then return NIXL_ERR_BACKEND.
TEST_F(ProxyDeviceApiTest, RingOverflowReturnsBackendErrorOnShutdown) {
    auto adapter = std::make_unique<StubProxyBackendAdapter>();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter), /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    publishProxyContext(runtime);
    uint32_t *shutdown_host = shutdownWordHostFromRuntime(runtime);
    ASSERT_NE(shutdown_host, nullptr);

    constexpr uint32_t kBurstOps = kDefaultProxyRingDepth + 1;
    nixl_status_t *d_statuses = nullptr;
    ASSERT_EQ(cudaMalloc(&d_statuses, sizeof(nixl_status_t) * kBurstOps), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_statuses, 0, sizeof(nixl_status_t) * kBurstOps), cudaSuccess);

    proxyPutBurstKernel<<<1, 1>>>(kBurstOps, 0, d_statuses);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(cudaStreamQuery(nullptr), cudaErrorNotReady);

    signalProxyShutdown(shutdown_host);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<nixl_status_t> statuses(kBurstOps);
    ASSERT_EQ(
        cudaMemcpy(
            statuses.data(), d_statuses, sizeof(nixl_status_t) * kBurstOps, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (uint32_t i = 0; i < kDefaultProxyRingDepth; ++i) {
        EXPECT_EQ(statuses[i], NIXL_IN_PROG) << "unexpected status at op " << i;
    }
    EXPECT_EQ(statuses.back(), NIXL_ERR_BACKEND);

    cudaFree(d_statuses);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Completions are tracked per-channel, so publishing one channel should not
// advance an unrelated channel's xfer status.
TEST_F(ProxyDeviceApiTest, ChannelCompletionsAdvanceIndependently) {
    auto adapter_owner = std::make_unique<ControllableStubAdapter>();
    auto *adapter = adapter_owner.get();
    nixlProxyRuntime runtime;

    ASSERT_EQ(runtime.init(std::move(adapter_owner), /*channel_count=*/2, /*worker_count=*/2),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    const auto mvhs = registerDummyMemViews(runtime);

    nixl_status_t *d_put_status[2];
    nixlGpuXferStatusH *d_xfer_status[2];
    for (int i = 0; i < 2; ++i) {
        d_put_status[i] = deviceAlloc<nixl_status_t>();
        ASSERT_EQ(cudaMalloc(&d_xfer_status[i], sizeof(nixlGpuXferStatusH)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_xfer_status[i], 0, sizeof(nixlGpuXferStatusH)), cudaSuccess);
    }

    proxyPutAsyncKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 0, d_put_status[0], d_xfer_status[0]);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    proxyPutAsyncKernel<<<1, 1>>>(mvhs.src, mvhs.dst, 1, d_put_status[1], d_xfer_status[1]);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status[0]), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_put_status[1]), NIXL_IN_PROG);
    ASSERT_TRUE(waitForCondition([adapter]() {
        return adapter->pendingCount() == 2 && adapter->hasPendingForChannel(0) &&
            adapter->hasPendingForChannel(1);
    }));

    uint64_t completed_token = 0;
    ASSERT_TRUE(adapter->markFirstPendingForChannel(1, &completed_token));
    EXPECT_GT(completed_token, 0u);
    ASSERT_TRUE(waitForCondition([adapter]() {
        return adapter->hasPendingForChannel(0) && !adapter->hasPendingForChannel(1);
    }));

    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();
    proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[0], d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_IN_PROG);

    proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[1], d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_SUCCESS);

    ASSERT_TRUE(adapter->markFirstPendingForChannel(0));
    ASSERT_TRUE(waitForCondition([adapter]() { return !adapter->hasPending(); }));

    proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[0], d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_SUCCESS);

    cudaFree(d_poll_status);
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_put_status[i]);
        cudaFree(d_xfer_status[i]);
    }
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}
