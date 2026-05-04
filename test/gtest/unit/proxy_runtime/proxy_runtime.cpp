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
#include <vector>

#include "device_proxy/backend_adapter.h"
#include "device_proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_runtime {

    class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
    };

    class StubBackend : public nixlDeviceProxyBackendAdapter {
    public:
        nixl_status_t
        init(uint32_t worker_count, uint32_t channel_count) override {
            init_called_ = true;
            init_worker_count_ = worker_count;
            init_channel_count_ = channel_count;
            return init_rc_;
        }

        nixl_status_t
        loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        submit(const nixlBackendProxySubmission &submission, uint64_t &request_token) override {
            {
                std::lock_guard<std::mutex> lock(submit_mutex_);
                submissions_.push_back(submission);
            }
            request_token = ++next_request_token_;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        checkCompletion(uint64_t) override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        progress() override {
            ++progress_calls_;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        shutdown() override {
            return NIXL_SUCCESS;
        }

        bool init_called_ = false;
        uint32_t init_worker_count_ = 0;
        uint32_t init_channel_count_ = 0;
        nixl_status_t init_rc_ = NIXL_SUCCESS;
        std::atomic<uint64_t> progress_calls_{0};
        mutable std::mutex submit_mutex_;
        std::vector<nixlBackendProxySubmission> submissions_;
        uint64_t next_request_token_ = 0;
    };

    class ProxyRuntimeTest : public testing::Test {
    protected:
        nixl_status_t
        initRuntime(uint32_t channel_count,
                    uint32_t worker_count,
                    nixl_status_t init_rc = NIXL_SUCCESS) {
            auto backend = std::make_unique<StubBackend>();
            backend_ = backend.get();
            backend_->init_rc_ = init_rc;
            return runtime_.init(std::move(backend), channel_count, worker_count);
        }

        void TearDown() override {
            runtime_.shutdown();
        }

        StubBackend *backend_ = nullptr;
        nixlProxyRuntime runtime_;
};

static nixlProxyWorkRing
copyDeviceWorkRing(const nixlProxyChannelView &view) {
    nixlProxyWorkRing ring{};
    EXPECT_EQ(cudaMemcpy(&ring, view.work_ring, sizeof(ring), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return ring;
}

static nixlProxySubmission *
hostRecordsFromDeviceAlias(nixlProxySubmission *records_host_dev) {
    cudaPointerAttributes attrs{};
    EXPECT_EQ(cudaPointerGetAttributes(&attrs, records_host_dev), cudaSuccess);
    EXPECT_NE(attrs.hostPointer, nullptr);
    return static_cast<nixlProxySubmission *>(attrs.hostPointer);
}

TEST_F(ProxyRuntimeTest, InitCallsBackendInit) {
    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_TRUE(backend_->init_called_);
    EXPECT_EQ(backend_->init_worker_count_, 2u);
    EXPECT_EQ(backend_->init_channel_count_, 4u);
}

TEST_F(ProxyRuntimeTest, InitRejectsNullBackend) {
    EXPECT_EQ(runtime_.init(nullptr, 4, 2), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitRejectsZeroChannels) {
    EXPECT_EQ(initRuntime(0, 2), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitRejectsZeroWorkers) {
    EXPECT_EQ(initRuntime(4, 0), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitPropagatesBackendFailure) {
    EXPECT_EQ(initRuntime(4, 2, NIXL_ERR_BACKEND), NIXL_ERR_BACKEND);
}

TEST_F(ProxyRuntimeTest, InitSetsChannelCount) {
    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 4u);
}

TEST_F(ProxyRuntimeTest, DeviceChannelViewsPopulated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    ASSERT_NE(views, nullptr);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(views[i].channel_id, i);
        EXPECT_NE(views[i].work_ring, nullptr);
        const nixlProxyWorkRing ring = copyDeviceWorkRing(views[i]);
        EXPECT_NE(ring.records, nullptr);
        EXPECT_NE(ring.producer_idx, nullptr);
        EXPECT_NE(ring.consumer_idx, nullptr);
        EXPECT_NE(views[i].completion_slot, nullptr);
        EXPECT_EQ(ring.depth, kDefaultProxyRingDepth);
    }
}

TEST_F(ProxyRuntimeTest, WorkRingIndicesStartAtZero) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t i = 0; i < 2; ++i) {
        const nixlProxyWorkRing ring = copyDeviceWorkRing(views[i]);
        uint64_t producer = 0;
        uint64_t consumer = 0;
        ASSERT_EQ(cudaMemcpy(&producer,
                             ring.producer_idx,
                             sizeof(producer),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&consumer,
                             ring.consumer_idx,
                             sizeof(consumer),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(producer, 0u);
        EXPECT_EQ(consumer, 0u);
    }
}

TEST_F(ProxyRuntimeTest, CompletionSlotsInitialized) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t i = 0; i < 2; ++i) {
        nixlProxyCompletionSlot slot{};
        ASSERT_EQ(cudaMemcpy(&slot,
                             views[i].completion_slot,
                             sizeof(nixlProxyCompletionSlot),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(slot.completed_idx, 0u);
        EXPECT_EQ(slot.next_status, NIXL_IN_PROG);
    }
}

TEST_F(ProxyRuntimeTest, WorkerCountClampedToChannels) {
    ASSERT_EQ(initRuntime(2, 8), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 2u);
    EXPECT_EQ(backend_->init_worker_count_, 2u);
    EXPECT_EQ(backend_->init_channel_count_, 2u);
}

TEST_F(ProxyRuntimeTest, DeviceContextPopulated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    auto *device_ctx = runtime_.deviceContext();
    ASSERT_NE(device_ctx, nullptr);
    nixlProxyDeviceContextData ctx{};
    ASSERT_EQ(cudaMemcpy(&ctx, device_ctx, sizeof(ctx), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(ctx.num_channels, 3u);
    EXPECT_NE(ctx.channels, nullptr);
    EXPECT_NE(ctx.shutdown_word, nullptr);
}

TEST_F(ProxyRuntimeTest, DeviceContextNullAfterShutdown) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_NE(runtime_.deviceContext(), nullptr);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.deviceContext(), nullptr);
}

TEST_F(ProxyRuntimeTest, StartWorkersAndShutdown) {
    ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, RepeatedStartWorkersIsRejected) {
    ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(runtime_.startWorkers(), NIXL_ERR_INVALID_PARAM);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ShutdownWithoutStartIsHarmless) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ShutdownBeforeInitIsHarmless) {
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, DoubleShutdownIsHarmless) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, InitAfterShutdownWorks) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 4u);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, SingleChannelSingleWorker) {
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 0u);
}

TEST_F(ProxyRuntimeTest, ManyChannelsManyWorkers) {
    ASSERT_EQ(initRuntime(16, 4), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, PrepMemViewProducesReadyEntries) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;
    const auto local_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x10});
    const auto remote_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x20});

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.prepMemView(local_backend, local_dlist, &src_proxy),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime_.prepMemView(remote_backend, remote_dlist, &dst_proxy),
              NIXL_SUCCESS);

    nixlMemViewH resolved = nullptr;
    EXPECT_TRUE(runtime_.resolveProxyMemView(src_proxy, resolved));
    EXPECT_EQ(resolved, local_backend);
    EXPECT_TRUE(runtime_.resolveProxyMemView(dst_proxy, resolved));
    EXPECT_EQ(resolved, remote_backend);

    nixlProxySubmission submission{};
    submission.opcode = nixl_proxy_opcode_t::PUT;
    submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
    submission.src_offset = 4;
    submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
    submission.dst_offset = 8;
    submission.size = 32;

    nixlBackendProxySubmission prepared_submission;
    ASSERT_EQ(runtime_.memviewRegistry().prepareSubmission(submission, prepared_submission),
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
    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    EXPECT_EQ(runtime_.prepMemView(local_dlist, nullptr),
              NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, WorkerSubmitsPreparedTransportDescriptors) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x10}),
                                           &src_proxy),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x20}),
                                           &dst_proxy),
              NIXL_SUCCESS);

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
    ASSERT_EQ(runtime_.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_.storeMetadata(dst_proxy, remote_dlist), NIXL_SUCCESS);

    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    nixlProxySubmission submission{};
    submission.op_idx = 11;
    submission.opcode = nixl_proxy_opcode_t::PUT;
    submission.channel_id = 0;
    submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
    submission.src_offset = 4;
    submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
    submission.dst_offset = 8;
    submission.size = 32;

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_.deviceChannelViews()[0]);
    auto *records = hostRecordsFromDeviceAlias(ring.records);
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

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 1u);
    const auto &prepared = submissions.front();
    EXPECT_EQ(prepared.op_idx, 11u);
    EXPECT_EQ(prepared.channel_id, 0u);
    EXPECT_EQ(prepared.local.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.local.desc.addr, 0x1004u);
    EXPECT_EQ(prepared.local.desc.len, 32u);
    EXPECT_EQ(prepared.local.desc.metadataP, &local_md);
    EXPECT_EQ(prepared.remote.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.remote.desc.len, 32u);
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
    EXPECT_EQ(prepared.remote_agent, "peer");
}

TEST_F(ProxyRuntimeTest, WorkerSubmitsPreparedAtomicAddDescriptor) {
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x20}),
                                           &dst_proxy),
              NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_.storeMetadata(dst_proxy, remote_dlist), NIXL_SUCCESS);

    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    nixlProxySubmission submission{};
    submission.op_idx = 11;
    submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
    submission.channel_id = 0;
    submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
    submission.dst_offset = 8;
    submission.size = sizeof(uint64_t);
    submission.value = 42;

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_.deviceChannelViews()[0]);
    auto *records = hostRecordsFromDeviceAlias(ring.records);
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

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 1u);
    const auto &prepared = submissions.front();
    EXPECT_EQ(prepared.op_idx, 11u);
    EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
    EXPECT_EQ(prepared.channel_id, 0u);
    EXPECT_EQ(prepared.remote.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.remote.desc.len, sizeof(uint64_t));
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
    EXPECT_EQ(prepared.remote_agent, "peer");
    EXPECT_EQ(prepared.value, 42u);
}

} // namespace proxy_runtime
} // namespace gtest
