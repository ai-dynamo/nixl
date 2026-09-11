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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "device/device_allocator.h"
#include "device/proxy/proxy_backend_ops.h"
#include "device/proxy/proxy_registry.h"
#include "device/proxy/proxy_runtime.h"
#include "device/proxy/proxy_worker.h"

namespace gtest {
namespace proxy_runtime {

class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
};

struct StubBackendState {
    mutable std::mutex released_mutex;
    std::vector<nixlBackendProxyRequest> released_requests;
};

class StubBackend {
    public:
        /** The mock the runtime actually sees. */
        nixlProxyBackendOps
        ops() {
            nixlProxyBackendOps ops;
            ops.init = [this](const nixlProxyConfig &config) { return init(config); };
            ops.submit = [this](const nixlBackendProxySubmission &submission,
                                nixlBackendProxyRequest &request) {
                return submit(submission, request);
            };
            ops.check_completion = [this](const nixlBackendProxyRequest &request) {
                return checkCompletion(request);
            };
            ops.release_request = [this](const nixlBackendProxyRequest &request) {
                releaseRequest(request);
            };
            ops.progress = [this](uint32_t, uint32_t) { return progress(); };
            ops.shutdown = []() { return NIXL_SUCCESS; };
            ops.on_remote_loaded = [](const std::string &, const nixl_blob_t &) {
                return NIXL_SUCCESS;
            };
            ops.resolve_direct_ptrs = [this](const nixl_remote_meta_dlist_t &dlist,
                                             std::vector<void *> &direct_ptrs) {
                return resolveDirectPointers(dlist, direct_ptrs);
            };
            return ops;
        }

        nixl_status_t
        init(const nixlProxyConfig &config) {
            init_called_ = true;
            init_worker_count_ = config.effectiveThreadCount();
            init_channel_count_ = config.channel_count;
            init_max_peers_ = config.max_peers;
            return init_rc_;
        }

        nixl_status_t
        resolveDirectPointers(const nixl_remote_meta_dlist_t &dlist,
                              std::vector<void *> &direct_ptrs) {
            ++resolve_direct_pointer_calls_;
            last_resolved_desc_count_ = dlist.descCount();
            if (resolve_direct_pointer_rc_ == NIXL_SUCCESS) {
                direct_ptrs = direct_ptrs_to_return_;
            }
            return resolve_direct_pointer_rc_;
        }

        nixl_status_t
        submit(const nixlBackendProxySubmission &submission, nixlBackendProxyRequest &request) {
            nixl_status_t status = submit_rc_;
            {
                std::lock_guard<std::mutex> lock(submit_mutex_);
                submissions_.push_back(submission);
                if (!submit_rcs_.empty()) {
                    status = submit_rcs_.front();
                    submit_rcs_.erase(submit_rcs_.begin());
                }
            }
            request = request_to_return_;
            if (status == NIXL_IN_PROG && !request) {
                request = nixlBackendProxyRequest{++next_request_token_, 0};
            }
            return status;
        }

        nixl_status_t
        checkCompletion(const nixlBackendProxyRequest &request) {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            last_checked_request_ = request;
            ++check_completion_calls_;
            const auto status = completion_status_by_token_.find(request.token);
            if (status != completion_status_by_token_.end()) {
                return status->second;
            }
            return completion_rc_;
        }

        nixl_status_t
        progress() {
            ++progress_calls_;
            return NIXL_SUCCESS;
        }

        void
        releaseRequest(const nixlBackendProxyRequest &request) {
            std::lock_guard<std::mutex> lock(state_->released_mutex);
            state_->released_requests.push_back(request);
        }

        void
        setCompletionStatus(uint64_t token, nixl_status_t status) {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            completion_status_by_token_[token] = status;
        }

        bool init_called_ = false;
        uint32_t init_worker_count_ = 0;
        uint32_t init_channel_count_ = 0;
        uint32_t init_max_peers_ = 0;
        nixl_status_t init_rc_ = NIXL_SUCCESS;
        std::atomic<uint64_t> progress_calls_{0};
        mutable std::mutex submit_mutex_;
        std::vector<nixlBackendProxySubmission> submissions_;
        std::vector<nixl_status_t> submit_rcs_;
        uint64_t next_request_token_ = 0;
        nixl_status_t submit_rc_ = NIXL_SUCCESS;
        nixl_status_t completion_rc_ = NIXL_SUCCESS;
        nixlBackendProxyRequest request_to_return_{};
        mutable std::mutex completion_mutex_;
        nixlBackendProxyRequest last_checked_request_{};
        uint64_t check_completion_calls_ = 0;
        std::unordered_map<uint64_t, nixl_status_t> completion_status_by_token_;
        std::shared_ptr<StubBackendState> state_ = std::make_shared<StubBackendState>();
        uint64_t resolve_direct_pointer_calls_ = 0;
        size_t last_resolved_desc_count_ = 0;
        nixl_status_t resolve_direct_pointer_rc_ = NIXL_SUCCESS;
        std::vector<void *> direct_ptrs_to_return_;
};

static nixlProxyConfig
makeConfig(uint32_t channel_count, uint32_t thread_count, uint32_t max_peers = 4) {
    nixlProxyConfig config;
    config.enabled = true;
    config.channel_count = channel_count;
    config.thread_count = thread_count;
    config.max_peers = max_peers;
    return config;
}

class ProxyRuntimeTest : public testing::Test {
    protected:
        nixl_status_t
        initRuntime(uint32_t channel_count,
                    uint32_t worker_count,
                    nixl_status_t init_rc = NIXL_SUCCESS,
                    uint32_t max_peers = 4,
                    bool with_direct_ptr_resolver = true) {
            // Tear the previous runtime down before its backend goes away.
            runtime_.reset();
            backend_storage_ = std::make_unique<StubBackend>();
            backend_ = backend_storage_.get();
            backend_->init_rc_ = init_rc;
            nixlProxyBackendOps ops = backend_->ops();
            if (!with_direct_ptr_resolver) {
                ops.resolve_direct_ptrs = nullptr;
            }
            return nixlProxyRuntime::create(
                std::move(ops), makeConfig(channel_count, worker_count, max_peers), runtime_);
        }

        void
        TearDown() override {
            runtime_.reset();
        }

        // Declared before runtime_: the runtime's callbacks reference the stub,
        // so the stub has to outlive it.
        std::unique_ptr<StubBackend> backend_storage_;
        StubBackend *backend_ = nullptr;
        std::unique_ptr<nixlProxyRuntime> runtime_;
};

static nixlProxyWorkRing
copyDeviceWorkRing(const nixlProxyChannelView &view) {
    nixlProxyWorkRing ring{};
    EXPECT_EQ(cudaMemcpy(&ring, view.work_ring, sizeof(ring), cudaMemcpyDeviceToHost), cudaSuccess);
    return ring;
}

// Resolve the pinned-host alias of a device-mapped submission or completion buffer.
// GDR-backed control words do not have CUDA host aliases and must be read as device memory.
template<class T>
static T *
hostAliasOf(T *device_alias) {
    cudaPointerAttributes attrs{};
    EXPECT_EQ(cudaPointerGetAttributes(&attrs, device_alias), cudaSuccess);
    EXPECT_NE(attrs.hostPointer, nullptr);
    return static_cast<T *>(attrs.hostPointer);
}

static size_t
channelViewIndex(uint32_t peer, uint32_t channel, uint32_t max_peers = 4) {
    return static_cast<size_t>(channel) * max_peers + peer;
}

static uint32_t
proxyMemViewId(nixlMemViewH proxy_memview) {
    if (proxy_memview == nullptr) {
        return 0;
    }
    nixlProxyDeviceMemView device_memview{};
    EXPECT_EQ(
        cudaMemcpy(&device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return device_memview.proxy_memview_id;
}

static nixlProxyDeviceMemView
copyDeviceMemView(nixlMemViewH proxy_memview) {
    nixlProxyDeviceMemView device_memview{};
    EXPECT_EQ(
        cudaMemcpy(&device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return device_memview;
}

static std::vector<void *>
copyDirectPointers(nixlMemViewH proxy_memview, size_t count) {
    std::vector<void *> direct_ptrs(count, nullptr);
    if (count != 0) {
        auto *direct_ptrs_dev =
            nixlProxyDeviceMemViewDirectPtrs(static_cast<nixlProxyDeviceMemView *>(proxy_memview));
        EXPECT_EQ(cudaMemcpy(direct_ptrs.data(),
                             direct_ptrs_dev,
                             sizeof(void *) * count,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    return direct_ptrs;
}

static std::vector<nixlBackendProxySubmission>
waitForSubmissions(StubBackend *backend, size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(backend->submit_mutex_);
            if (backend->submissions_.size() >= count) {
                return backend->submissions_;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::lock_guard<std::mutex> lock(backend->submit_mutex_);
    return backend->submissions_;
}

static bool
waitForCompletedIdx(const nixlProxyChannelView &view, uint64_t completed_idx) {
    auto *completion_slot = hostAliasOf(view.completion_slot);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        if (__atomic_load_n(&completion_slot->completed_idx, __ATOMIC_ACQUIRE) >= completed_idx) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return __atomic_load_n(&completion_slot->completed_idx, __ATOMIC_ACQUIRE) >= completed_idx;
}

static nixl_status_t
allocateDirectChannel(nixlProxyChannelState &channel,
                      nixlProxyControlBuffer &control_slots,
                      uint32_t depth) {
    nixlDeviceAllocator &allocator = nixlGetDeviceAllocator();
    nixl_status_t status = control_slots.allocate(allocator, kProxyCiSlotBase + 1);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    return channel.allocate(allocator, depth, &control_slots, kProxyCiSlotBase);
}

static uint64_t
deviceConsumerIdx(const nixlProxyChannelState &channel) {
    uint64_t consumer_idx = 0;
    EXPECT_EQ(
        cudaMemcpy(
            &consumer_idx, channel.consumer_idx_dev_, sizeof(consumer_idx), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return consumer_idx;
}

static nixlProxySubmission
makeAtomicAddSubmission(nixlMemViewH dst_proxy, uint64_t value = 42) {
    nixlProxySubmission submission{};
    submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
    submission.channel_id = 0;
    submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
    submission.dst_offset = 0;
    submission.size = sizeof(uint64_t);
    submission.value = value;
    return submission;
}

static nixlProxySubmission
makeInvalidAtomicAddSubmission() {
    return makeAtomicAddSubmission(nullptr);
}

static void
publishRecord(nixlProxySubmission *records,
              uint32_t slot,
              const nixlProxySubmission &submission,
              uint64_t op_idx) {
    nixlProxySubmission record = submission;
    record.op_idx = 0;
    records[slot] = record;
    __atomic_store_n(&records[slot].op_idx, op_idx, __ATOMIC_RELEASE);
}

static std::unique_ptr<ProxyWorker>
makeDirectWorker(const nixlProxyBackendOps *ops,
                 const nixlProxyMemViewRegistry *registry,
                 std::atomic<uint64_t> *shutdown_state,
                 nixlProxyChannelState *channel) {
    return std::make_unique<ProxyWorker>(
        ops, registry, shutdown_state, channel, 1, 1, 0, 1, 0);
}

static nixl_remote_meta_dlist_t
makeRemotePeerDlist(const std::vector<std::string> &agents, nixlBackendMD *md) {
    nixl_remote_meta_dlist_t dlist(VRAM_SEG);
    for (const auto &agent : agents) {
        if (agent.empty()) {
            dlist.addDesc(nixlRemoteMetaDesc(nixl_null_agent));
        } else {
            nixlRemoteMetaDesc desc(agent);
            desc.addr = 0x4000;
            desc.len = 64;
            desc.devId = 0;
            desc.metadataP = md;
            dlist.addDesc(desc);
        }
    }
    return dlist;
}

TEST_F(ProxyRuntimeTest, CreateCallsBackendInit) {
    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_TRUE(backend_->init_called_);
    EXPECT_EQ(backend_->init_worker_count_, 2u);
    EXPECT_EQ(backend_->init_channel_count_, 4u);
}

TEST_F(ProxyRuntimeTest, CreateRejectsIncompleteOps) {
    StubBackend backend;
    nixlProxyBackendOps ops = backend.ops();
    ops.submit = nullptr;
    EXPECT_EQ(nixlProxyRuntime::create(std::move(ops), makeConfig(4, 2), runtime_),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(runtime_, nullptr);
}

TEST_F(ProxyRuntimeTest, CreateRejectsZeroPeerCapacity) {
    EXPECT_EQ(initRuntime(2, 1, NIXL_SUCCESS, 0), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, CreateRejectsZeroChannels) {
    EXPECT_EQ(initRuntime(0, 2), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, CreateRejectsZeroWorkers) {
    EXPECT_EQ(initRuntime(4, 0), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, CreatePropagatesBackendFailure) {
    EXPECT_EQ(initRuntime(4, 2, NIXL_ERR_BACKEND), NIXL_ERR_BACKEND);
    // A failed create() leaves no half-built runtime behind.
    EXPECT_EQ(runtime_, nullptr);
}

TEST_F(ProxyRuntimeTest, DeviceChannelViewMatrixStartsAllocated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_->deviceChannelViews();
    ASSERT_NE(views, nullptr);
    for (uint32_t peer = 0; peer < 4; ++peer) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            const auto &view = views[channelViewIndex(peer, channel)];
            EXPECT_NE(view.work_ring, nullptr);
            EXPECT_NE(view.completion_slot, nullptr);
        }
    }
}

TEST_F(ProxyRuntimeTest, WorkRingIndicesStartAtZero) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    nixlMemViewH remote_mvh = nullptr;
    ASSERT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
              NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_->deviceChannelViews();
    for (uint32_t channel = 0; channel < 2; ++channel) {
        const nixlProxyWorkRing ring = copyDeviceWorkRing(views[channelViewIndex(0, channel)]);
        uint64_t producer = 0;
        uint64_t consumer = 0;
        ASSERT_EQ(
            cudaMemcpy(&producer, ring.producer_idx, sizeof(producer), cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(&consumer, ring.consumer_idx, sizeof(consumer), cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_EQ(producer, 0u);
        EXPECT_EQ(consumer, 0u);
    }
}

TEST_F(ProxyRuntimeTest, CompletionSlotsInitialized) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    nixlMemViewH remote_mvh = nullptr;
    ASSERT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
              NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_->deviceChannelViews();
    for (uint32_t channel = 0; channel < 2; ++channel) {
        nixlProxyCompletionSlot slot{};
        ASSERT_EQ(cudaMemcpy(&slot,
                             views[channelViewIndex(0, channel)].completion_slot,
                             sizeof(nixlProxyCompletionSlot),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(slot.completed_idx, 0u);
        EXPECT_EQ(slot.next_status, NIXL_IN_PROG);
    }
}

TEST_F(ProxyRuntimeTest, WorkerCountIsNotClampedToPeerCapacity) {
    ASSERT_EQ(initRuntime(8, 8, NIXL_SUCCESS, 2), NIXL_SUCCESS);
    EXPECT_EQ(backend_->init_worker_count_, 8u);
    EXPECT_EQ(backend_->init_channel_count_, 8u);
}

TEST_F(ProxyRuntimeTest, WorkerCountClampedToChannelCount) {
    ASSERT_EQ(initRuntime(2, 8, NIXL_SUCCESS, 4), NIXL_SUCCESS);
    EXPECT_EQ(backend_->init_worker_count_, 2u);
    EXPECT_EQ(backend_->init_channel_count_, 2u);
}

TEST_F(ProxyRuntimeTest, DeviceContextPopulated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    auto *device_ctx = runtime_->deviceContext();
    ASSERT_NE(device_ctx, nullptr);
    nixlProxyDeviceContextData ctx{};
    ASSERT_EQ(cudaMemcpy(&ctx, device_ctx, sizeof(ctx), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(ctx.max_peers, 4u);
    EXPECT_EQ(ctx.num_channels, 3u);
    EXPECT_NE(ctx.channels, nullptr);
    EXPECT_NE(ctx.shutdown_word, nullptr);
}

TEST_F(ProxyRuntimeTest, DeviceContextCarriedByMemView) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    nixlMemViewH remote_mvh = nullptr;
    ASSERT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
              NIXL_SUCCESS);
    EXPECT_EQ(copyDeviceMemView(remote_mvh).context, runtime_->deviceContext());
}

TEST_F(ProxyRuntimeTest, DeviceContextNullAfterShutdown) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_NE(runtime_->deviceContext(), nullptr);
    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_->deviceContext(), nullptr);
}

TEST_F(ProxyRuntimeTest, StartWorkersAndShutdown) {
    ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, RepeatedStartWorkersIsRejected) {
    ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(runtime_->startWorkers(), NIXL_ERR_INVALID_PARAM);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ShutdownWithoutStartIsHarmless) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
}

// create() replaced the two-phase ctor+init(): there is no observable
// pre-init state left to shut down, only a runtime that was never created.
TEST_F(ProxyRuntimeTest, FailedCreateLeavesNoRuntime) {
    EXPECT_EQ(initRuntime(0, 1), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(runtime_, nullptr);
}

TEST_F(ProxyRuntimeTest, DoubleShutdownIsHarmless) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, CreateAfterShutdownWorks) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, SingleChannelSingleWorker) {
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ManyChannelsManyWorkers) {
    ASSERT_EQ(initRuntime(16, 4), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, PrepMemViewProducesReadyEntries) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_->prepMemView(local_dlist, &src_proxy), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->prepMemView(remote_dlist, &dst_proxy), NIXL_SUCCESS);

    nixlMemViewH resolved = nullptr;
    EXPECT_TRUE(runtime_->resolveProxyMemView(src_proxy, resolved));
    EXPECT_TRUE(runtime_->resolveProxyMemView(dst_proxy, resolved));

    nixlProxySubmission submission{};
    submission.opcode = nixl_proxy_opcode_t::PUT;
    submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
    submission.src_offset = 4;
    submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
    submission.dst_offset = 8;
    submission.size = 32;

    nixlBackendProxySubmission prepared_submission;
    ASSERT_EQ(runtime_->memviewRegistry().prepareSubmission(submission, prepared_submission),
              NIXL_SUCCESS);
    EXPECT_EQ(prepared_submission.local.desc.addr, 0x1004u);
    EXPECT_EQ(prepared_submission.local.desc.len, 32u);
    EXPECT_EQ(prepared_submission.local.desc.metadataP, &local_md);
    EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared_submission.remote.desc.len, 32u);
    EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md);
    EXPECT_EQ(prepared_submission.remote_agent, "peer");
}

TEST_F(ProxyRuntimeTest, PrepMemViewRejectsNullOutput) {
    DummyBackendMD local_md;
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    EXPECT_EQ(runtime_->prepMemView(local_dlist, nullptr), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, PrepRemoteMemViewRejectsNonVramMetadata) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);

    nixlMemViewH dst_proxy = nullptr;
    EXPECT_EQ(runtime_->prepMemView(remote_dlist, &dst_proxy), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, PrepRemoteMemViewStoresResolvedDirectPointers) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    backend_->direct_ptrs_to_return_ = {reinterpret_cast<void *>(uintptr_t{0xabc00000}), nullptr};

    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer0", "peer1"}, &remote_md), &dst_proxy),
              NIXL_SUCCESS);

    EXPECT_EQ(backend_->resolve_direct_pointer_calls_, 1u);
    EXPECT_EQ(backend_->last_resolved_desc_count_, 2u);
    const nixlProxyDeviceMemView device_memview = copyDeviceMemView(dst_proxy);
    EXPECT_EQ(device_memview.proxy_memview_id, proxyMemViewId(dst_proxy));
    EXPECT_EQ(device_memview.direct_ptr_count, backend_->direct_ptrs_to_return_.size());
    EXPECT_EQ(copyDirectPointers(dst_proxy, backend_->direct_ptrs_to_return_.size()),
              backend_->direct_ptrs_to_return_);
}

// An unset resolve_direct_ptrs callback means the backend does not resolve
// direct pointers - the prep still succeeds, with none stored.
TEST_F(ProxyRuntimeTest, PrepRemoteMemViewSkipsDirectPointersWhenResolverUnset) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(1, 1, NIXL_SUCCESS, 4, /*with_direct_ptr_resolver=*/false),
              NIXL_SUCCESS);

    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
              NIXL_SUCCESS);

    EXPECT_EQ(backend_->resolve_direct_pointer_calls_, 0u);
    EXPECT_EQ(copyDeviceMemView(dst_proxy).direct_ptr_count, 0u);
}

TEST_F(ProxyRuntimeTest, PrepRemoteMemViewPropagatesDirectPointerResolverErrors) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    backend_->resolve_direct_pointer_rc_ = NIXL_ERR_INVALID_PARAM;

    nixlMemViewH dst_proxy = nullptr;
    EXPECT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(dst_proxy, nullptr);
    EXPECT_EQ(backend_->resolve_direct_pointer_calls_, 1u);
}

TEST_F(ProxyRuntimeTest, WorkerSubmitsPreparedTransportDescriptors) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    backend_->submit_rc_ = NIXL_IN_PROG;
    backend_->completion_rc_ = NIXL_SUCCESS;
    backend_->request_to_return_ = nixlBackendProxyRequest{101, 7};

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
    ASSERT_EQ(runtime_->prepMemView(local_dlist, &src_proxy), NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->prepMemView(remote_dlist, &dst_proxy), NIXL_SUCCESS);

    nixlProxySubmission submission{};
    submission.op_idx = 11;
    submission.opcode = nixl_proxy_opcode_t::PUT;
    submission.channel_id = 0;
    submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
    submission.src_offset = 4;
    submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
    submission.dst_offset = 8;
    submission.size = 32;

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_->deviceChannelViews()[0]);
    auto *records = hostAliasOf(ring.records);
    ASSERT_NE(records, nullptr);
    submission.op_idx = 0;
    records[0] = submission;
    __atomic_store_n(&records[0].op_idx, uint64_t{11}, __ATOMIC_RELEASE);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
            if (!backend_->submissions_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::vector<nixlBackendProxySubmission> submissions;
    {
        std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
        submissions = backend_->submissions_;
    }
    ASSERT_TRUE(waitForCompletedIdx(runtime_->deviceChannelViews()[0], 11));

    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 1u);
    const auto &prepared = submissions.front();
    EXPECT_EQ(prepared.op_idx, 11u);
    EXPECT_EQ(prepared.channel_id, 0u);
    EXPECT_EQ(prepared.peer_index, 0u);
    EXPECT_EQ(prepared.local.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.local.desc.addr, 0x1004u);
    EXPECT_EQ(prepared.local.desc.len, 32u);
    EXPECT_EQ(prepared.local.desc.metadataP, &local_md);
    EXPECT_EQ(prepared.remote.mem_type, VRAM_SEG);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.remote.desc.len, 32u);
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
    EXPECT_EQ(prepared.remote_agent, "peer");
    EXPECT_EQ(backend_->last_checked_request_.token, 101u);
    EXPECT_EQ(backend_->last_checked_request_.context, 7u);
    EXPECT_GT(backend_->check_completion_calls_, 0u);
}

TEST_F(ProxyRuntimeTest, WorkerSubmitsPreparedAtomicAddDescriptor) {
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    backend_->submit_rc_ = NIXL_IN_PROG;
    backend_->completion_rc_ = NIXL_SUCCESS;
    backend_->request_to_return_ = nixlBackendProxyRequest{202, 8};

    nixlMemViewH dst_proxy = nullptr;
    nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->prepMemView(remote_dlist, &dst_proxy), NIXL_SUCCESS);

    nixlProxySubmission submission{};
    submission.op_idx = 11;
    submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
    submission.channel_id = 0;
    submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
    submission.dst_offset = 8;
    submission.size = sizeof(uint64_t);
    submission.value = 42;

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_->deviceChannelViews()[0]);
    auto *records = hostAliasOf(ring.records);
    ASSERT_NE(records, nullptr);
    submission.op_idx = 0;
    records[0] = submission;
    __atomic_store_n(&records[0].op_idx, uint64_t{11}, __ATOMIC_RELEASE);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
            if (!backend_->submissions_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::vector<nixlBackendProxySubmission> submissions;
    {
        std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
        submissions = backend_->submissions_;
    }
    ASSERT_TRUE(waitForCompletedIdx(runtime_->deviceChannelViews()[0], 11));

    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 1u);
    const auto &prepared = submissions.front();
    EXPECT_EQ(prepared.op_idx, 11u);
    EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
    EXPECT_EQ(prepared.channel_id, 0u);
    EXPECT_EQ(prepared.peer_index, 0u);
    EXPECT_EQ(prepared.remote.mem_type, VRAM_SEG);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.remote.desc.len, sizeof(uint64_t));
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
    EXPECT_EQ(prepared.remote_agent, "peer");
    EXPECT_EQ(prepared.value, 42u);
    EXPECT_EQ(backend_->last_checked_request_.token, 202u);
    EXPECT_EQ(backend_->last_checked_request_.context, 8u);
    EXPECT_GT(backend_->check_completion_calls_, 0u);
}

TEST_F(ProxyRuntimeTest, ShutdownReleasesPendingBackendRequests) {
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    backend_->submit_rc_ = NIXL_IN_PROG;
    backend_->completion_rc_ = NIXL_IN_PROG;
    backend_->request_to_return_ = nixlBackendProxyRequest{303, 9};
    auto backend_state = backend_->state_;

    nixlMemViewH dst_proxy = nullptr;
    nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->prepMemView(remote_dlist, &dst_proxy), NIXL_SUCCESS);

    nixlProxySubmission submission{};
    submission.op_idx = 31;
    submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
    submission.channel_id = 0;
    submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
    submission.dst_offset = 8;
    submission.size = sizeof(uint64_t);
    submission.value = 42;

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_->deviceChannelViews()[0]);
    auto *records = hostAliasOf(ring.records);
    ASSERT_NE(records, nullptr);
    submission.op_idx = 0;
    records[0] = submission;
    __atomic_store_n(&records[0].op_idx, uint64_t{31}, __ATOMIC_RELEASE);

    const auto submissions = waitForSubmissions(backend_, 1);
    ASSERT_EQ(submissions.size(), 1u);
    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);

    std::lock_guard<std::mutex> lock(backend_state->released_mutex);
    ASSERT_EQ(backend_state->released_requests.size(), 1u);
    EXPECT_EQ(backend_state->released_requests.front().token, 303u);
    EXPECT_EQ(backend_state->released_requests.front().context, 9u);
}

TEST_F(ProxyRuntimeTest, WorkerSubmitsReadyPeersForOwnedChannel) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1, NIXL_SUCCESS, 2), NIXL_SUCCESS);

    nixlMemViewH src_proxy = nullptr;
    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
    ASSERT_EQ(runtime_->prepMemView(local_dlist, &src_proxy), NIXL_SUCCESS);

    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer0", "peer1"}, &remote_md), &dst_proxy),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

    nixlProxySubmission peer0{};
    peer0.opcode = nixl_proxy_opcode_t::PUT;
    peer0.channel_id = 0;
    peer0.src_proxy_memview_id = proxyMemViewId(src_proxy);
    peer0.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
    peer0.dst_index = 0;
    peer0.size = 32;

    nixlProxySubmission peer1 = peer0;
    peer1.dst_index = 1;

    const nixlProxyWorkRing ring0 =
        copyDeviceWorkRing(runtime_->deviceChannelViews()[channelViewIndex(0, 0, 2)]);
    const nixlProxyWorkRing ring1 =
        copyDeviceWorkRing(runtime_->deviceChannelViews()[channelViewIndex(1, 0, 2)]);
    auto *records0 = hostAliasOf(ring0.records);
    auto *records1 = hostAliasOf(ring1.records);
    ASSERT_NE(records0, nullptr);
    ASSERT_NE(records1, nullptr);

    records0[0] = peer0;
    records1[0] = peer1;
    __atomic_store_n(&records0[0].op_idx, uint64_t{21}, __ATOMIC_RELEASE);
    __atomic_store_n(&records1[0].op_idx, uint64_t{22}, __ATOMIC_RELEASE);

    const auto submissions = waitForSubmissions(backend_, 2);
    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 2u);
    std::vector<bool> seen(2, false);
    for (const auto &submission : submissions) {
        ASSERT_LT(submission.peer_index, 2u);
        EXPECT_EQ(submission.channel_id, 0u);
        seen[submission.peer_index] = true;
    }
    EXPECT_TRUE(seen[0]);
    EXPECT_TRUE(seen[1]);
}

TEST_F(ProxyRuntimeTest, ConsumerIndexAdvancesOnlyAfterBackendCompletion) {
    DummyBackendMD remote_md;
    StubBackend backend;
    backend.submit_rc_ = NIXL_IN_PROG;
    backend.completion_rc_ = NIXL_IN_PROG;

    nixlProxyMemViewRegistry registry(nixlGetDeviceAllocator(), nullptr);
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(registry.prepRemote(makeRemotePeerDlist({"peer"}, &remote_md), {}, dst_proxy),
              NIXL_SUCCESS);

    nixlProxyChannelState channel;
    nixlProxyControlBuffer control_slots;
    ASSERT_EQ(allocateDirectChannel(channel, control_slots, 2), NIXL_SUCCESS);
    std::atomic<uint64_t> shutdown_state{
        static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
    const nixlProxyBackendOps ops = backend.ops();
    auto worker = makeDirectWorker(&ops, &registry, &shutdown_state, &channel);

    publishRecord(channel.recordsHost(), 0, makeAtomicAddSubmission(dst_proxy), 1);

    worker->runOnce();
    ASSERT_EQ(backend.submissions_.size(), 1u);
    EXPECT_EQ(deviceConsumerIdx(channel), 0u);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 0u);

    backend.setCompletionStatus(1, NIXL_SUCCESS);
    worker->runOnce();

    EXPECT_EQ(deviceConsumerIdx(channel), 1u);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 1u);
    EXPECT_EQ(channel.completionSlotHost()->next_status, NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, InFlightRequestsAreBoundedByRingDepth) {
    DummyBackendMD remote_md;
    StubBackend backend;
    backend.submit_rc_ = NIXL_IN_PROG;
    backend.completion_rc_ = NIXL_IN_PROG;

    nixlProxyMemViewRegistry registry(nixlGetDeviceAllocator(), nullptr);
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(registry.prepRemote(makeRemotePeerDlist({"peer"}, &remote_md), {}, dst_proxy),
              NIXL_SUCCESS);

    nixlProxyChannelState channel;
    nixlProxyControlBuffer control_slots;
    ASSERT_EQ(allocateDirectChannel(channel, control_slots, 2), NIXL_SUCCESS);
    std::atomic<uint64_t> shutdown_state{
        static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
    const nixlProxyBackendOps ops = backend.ops();
    auto worker = makeDirectWorker(&ops, &registry, &shutdown_state, &channel);

    const auto submission = makeAtomicAddSubmission(dst_proxy);
    publishRecord(channel.recordsHost(), 0, submission, 1);
    publishRecord(channel.recordsHost(), 1, submission, 2);

    worker->runOnce();
    worker->runOnce();
    ASSERT_EQ(backend.submissions_.size(), 2u);
    EXPECT_EQ(deviceConsumerIdx(channel), 0u);

    publishRecord(channel.recordsHost(), 0, submission, 3);
    worker->runOnce();
    EXPECT_EQ(backend.submissions_.size(), 2u);

    backend.setCompletionStatus(1, NIXL_SUCCESS);
    worker->runOnce();
    EXPECT_EQ(backend.submissions_.size(), 2u);
    EXPECT_EQ(deviceConsumerIdx(channel), 1u);

    worker->runOnce();
    EXPECT_EQ(backend.submissions_.size(), 3u);
    EXPECT_EQ(backend.submissions_.back().op_idx, 3u);
}

TEST_F(ProxyRuntimeTest, CompletionsPublishInSubmissionOrder) {
    DummyBackendMD remote_md;
    StubBackend backend;
    backend.submit_rc_ = NIXL_IN_PROG;
    backend.completion_rc_ = NIXL_IN_PROG;

    nixlProxyMemViewRegistry registry(nixlGetDeviceAllocator(), nullptr);
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(registry.prepRemote(makeRemotePeerDlist({"peer"}, &remote_md), {}, dst_proxy),
              NIXL_SUCCESS);

    nixlProxyChannelState channel;
    nixlProxyControlBuffer control_slots;
    ASSERT_EQ(allocateDirectChannel(channel, control_slots, 3), NIXL_SUCCESS);
    std::atomic<uint64_t> shutdown_state{
        static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
    const nixlProxyBackendOps ops = backend.ops();
    auto worker = makeDirectWorker(&ops, &registry, &shutdown_state, &channel);

    const auto submission = makeAtomicAddSubmission(dst_proxy);
    publishRecord(channel.recordsHost(), 0, submission, 1);
    publishRecord(channel.recordsHost(), 1, submission, 2);

    worker->runOnce();
    worker->runOnce();
    ASSERT_EQ(backend.submissions_.size(), 2u);

    backend.setCompletionStatus(2, NIXL_SUCCESS);
    worker->runOnce();
    EXPECT_EQ(deviceConsumerIdx(channel), 0u);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 0u);

    backend.setCompletionStatus(1, NIXL_SUCCESS);
    worker->runOnce();
    EXPECT_EQ(deviceConsumerIdx(channel), 2u);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 2u);
}

TEST_F(ProxyRuntimeTest, PreparationErrorLatchesStatusButLaterWorkIsReclaimed) {
    DummyBackendMD remote_md;
    StubBackend backend;
    backend.submit_rc_ = NIXL_IN_PROG;
    backend.completion_rc_ = NIXL_SUCCESS;

    nixlProxyMemViewRegistry registry(nixlGetDeviceAllocator(), nullptr);
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(registry.prepRemote(makeRemotePeerDlist({"peer"}, &remote_md), {}, dst_proxy),
              NIXL_SUCCESS);

    nixlProxyChannelState channel;
    nixlProxyControlBuffer control_slots;
    ASSERT_EQ(allocateDirectChannel(channel, control_slots, 3), NIXL_SUCCESS);
    std::atomic<uint64_t> shutdown_state{
        static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
    const nixlProxyBackendOps ops = backend.ops();
    auto worker = makeDirectWorker(&ops, &registry, &shutdown_state, &channel);

    publishRecord(channel.recordsHost(), 0, makeInvalidAtomicAddSubmission(), 1);
    worker->runOnce();
    EXPECT_EQ(deviceConsumerIdx(channel), 1u);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 1u);
    EXPECT_LT(channel.completionSlotHost()->next_status, 0);

    publishRecord(channel.recordsHost(), 1, makeAtomicAddSubmission(dst_proxy), 2);
    worker->runOnce();
    EXPECT_EQ(deviceConsumerIdx(channel), 2u);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 1u);
    ASSERT_EQ(backend.submissions_.size(), 1u);
    EXPECT_EQ(backend.submissions_.front().op_idx, 2u);
}

TEST_F(ProxyRuntimeTest, SubmitAndCompletionErrorsLatchFirstStatusAndRetireWork) {
    DummyBackendMD remote_md;
    StubBackend backend;
    backend.submit_rcs_ = {NIXL_ERR_BACKEND, NIXL_IN_PROG, NIXL_IN_PROG};
    backend.completion_rc_ = NIXL_IN_PROG;

    nixlProxyMemViewRegistry registry(nixlGetDeviceAllocator(), nullptr);
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(registry.prepRemote(makeRemotePeerDlist({"peer"}, &remote_md), {}, dst_proxy),
              NIXL_SUCCESS);

    nixlProxyChannelState channel;
    nixlProxyControlBuffer control_slots;
    ASSERT_EQ(allocateDirectChannel(channel, control_slots, 4), NIXL_SUCCESS);
    std::atomic<uint64_t> shutdown_state{
        static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
    const nixlProxyBackendOps ops = backend.ops();
    auto worker = makeDirectWorker(&ops, &registry, &shutdown_state, &channel);

    const auto submission = makeAtomicAddSubmission(dst_proxy);
    publishRecord(channel.recordsHost(), 0, submission, 1);
    publishRecord(channel.recordsHost(), 1, submission, 2);
    publishRecord(channel.recordsHost(), 2, submission, 3);

    worker->runOnce();
    EXPECT_EQ(deviceConsumerIdx(channel), 1u);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 1u);
    const nixl_status_t first_error = channel.completionSlotHost()->next_status;
    EXPECT_LT(first_error, 0);

    worker->runOnce();
    ASSERT_EQ(backend.submissions_.size(), 2u);
    backend.setCompletionStatus(1, NIXL_ERR_BACKEND);
    worker->runOnce();
    EXPECT_EQ(deviceConsumerIdx(channel), 2u);
    EXPECT_EQ(channel.completionSlotHost()->next_status, first_error);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 1u);

    backend.setCompletionStatus(2, NIXL_SUCCESS);
    worker->runOnce();
    EXPECT_EQ(deviceConsumerIdx(channel), 3u);
    EXPECT_EQ(channel.completionSlotHost()->next_status, first_error);
    EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost()->completed_idx, __ATOMIC_ACQUIRE), 1u);
}

TEST_F(ProxyRuntimeTest, ShutdownReleasesAllPendingBackendRequests) {
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    backend_->submit_rc_ = NIXL_IN_PROG;
    backend_->completion_rc_ = NIXL_IN_PROG;
    auto backend_state = backend_->state_;

    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_->prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
              NIXL_SUCCESS);

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_->deviceChannelViews()[0]);
    auto *records = hostAliasOf(ring.records);
    ASSERT_NE(records, nullptr);

    const auto submission = makeAtomicAddSubmission(dst_proxy);
    publishRecord(records, 0, submission, 1);
    publishRecord(records, 1, submission, 2);

    const auto submissions = waitForSubmissions(backend_, 2);
    ASSERT_EQ(submissions.size(), 2u);
    ASSERT_EQ(runtime_->shutdown(), NIXL_SUCCESS);

    std::lock_guard<std::mutex> lock(backend_state->released_mutex);
    ASSERT_EQ(backend_state->released_requests.size(), 2u);
    EXPECT_EQ(backend_state->released_requests[0].token, 1u);
    EXPECT_EQ(backend_state->released_requests[1].token, 2u);
}

} // namespace proxy_runtime
} // namespace gtest
