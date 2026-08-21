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

#include "device_proxy/backend_adapter.h"
#include "device_proxy/proxy_runtime.h"
#include "device_proxy/proxy_worker.h"

namespace gtest {
namespace proxy_runtime {
    namespace {

        class dummyBackendMd : public nixlBackendMD {
        public:
            dummyBackendMd() : nixlBackendMD(false) {}
        };

        struct stubBackendState {
            mutable std::mutex releasedMutex;
            std::vector<nixlBackendProxyRequest> releasedRequests;
        };

        class stubBackend : public nixlDeviceProxyBackendAdapter {
        public:
            nixl_status_t
            init(uint32_t worker_count, uint32_t channel_count, uint32_t max_peers) override {
                initCalled = true;
                initWorkerCount = worker_count;
                initChannelCount = channel_count;
                initMaxPeers = max_peers;
                return initRc;
            }

            nixl_status_t
            loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
                return NIXL_SUCCESS;
            }

            nixl_status_t
            resolveDirectPointers(const nixl_remote_meta_dlist_t &dlist,
                                  std::vector<void *> &direct_ptrs) override {
                ++resolveDirectPointerCalls;
                lastResolvedDescCount = dlist.descCount();
                if (resolveDirectPointerRc == NIXL_SUCCESS) {
                    direct_ptrs = directPtrsToReturn;
                }
                return resolveDirectPointerRc;
            }

            nixl_status_t
            submit(const nixlBackendProxySubmission &submission,
                   nixlBackendProxyRequest &request) override {
                nixl_status_t status = submitRc;
                {
                    std::lock_guard<std::mutex> lock(submitMutex);
                    submissions.push_back(submission);
                    if (!submitRcs.empty()) {
                        status = submitRcs.front();
                        submitRcs.erase(submitRcs.begin());
                    }
                }
                request = requestToReturn;
                if (status == NIXL_IN_PROG && !request) {
                    request = nixlBackendProxyRequest{++nextRequestToken, 0};
                }
                return status;
            }

            nixl_status_t
            checkCompletion(const nixlBackendProxyRequest &request) override {
                std::lock_guard<std::mutex> lock(completionMutex);
                lastCheckedRequest = request;
                ++checkCompletionCalls;
                const auto status = completionStatusByToken.find(request.token);
                if (status != completionStatusByToken.end()) {
                    return status->second;
                }
                return completionRc;
            }

            nixl_status_t
            progress() override {
                ++progressCalls;
                return NIXL_SUCCESS;
            }

            nixl_status_t
            progress(uint32_t, uint32_t) override {
                return progress();
            }

            nixl_status_t
            shutdown() override {
                return NIXL_SUCCESS;
            }

            void
            releaseRequest(const nixlBackendProxyRequest &request) override {
                std::lock_guard<std::mutex> lock(state->releasedMutex);
                state->releasedRequests.push_back(request);
            }

            void
            setCompletionStatus(uint64_t token, nixl_status_t status) {
                std::lock_guard<std::mutex> lock(completionMutex);
                completionStatusByToken[token] = status;
            }

            bool initCalled = false;
            uint32_t initWorkerCount = 0;
            uint32_t initChannelCount = 0;
            uint32_t initMaxPeers = 0;
            nixl_status_t initRc = NIXL_SUCCESS;
            std::atomic<uint64_t> progressCalls{0};
            mutable std::mutex submitMutex;
            std::vector<nixlBackendProxySubmission> submissions;
            std::vector<nixl_status_t> submitRcs;
            uint64_t nextRequestToken = 0;
            nixl_status_t submitRc = NIXL_SUCCESS;
            nixl_status_t completionRc = NIXL_SUCCESS;
            nixlBackendProxyRequest requestToReturn{};
            mutable std::mutex completionMutex;
            nixlBackendProxyRequest lastCheckedRequest{};
            uint64_t checkCompletionCalls = 0;
            std::unordered_map<uint64_t, nixl_status_t> completionStatusByToken;
            std::shared_ptr<stubBackendState> state = std::make_shared<stubBackendState>();
            uint64_t resolveDirectPointerCalls = 0;
            size_t lastResolvedDescCount = 0;
            nixl_status_t resolveDirectPointerRc = NIXL_ERR_NOT_SUPPORTED;
            std::vector<void *> directPtrsToReturn;
        };

        class proxyRuntimeTest : public testing::Test {
        protected:
            nixl_status_t
            initRuntime(uint32_t channel_count,
                        uint32_t worker_count,
                        nixl_status_t init_rc = NIXL_SUCCESS,
                        uint32_t max_peers = 4) {
                auto backend_owner = std::make_unique<stubBackend>();
                backend = backend_owner.get();
                backend->initRc = init_rc;
                return runtime.init(
                    std::move(backend_owner), max_peers, channel_count, worker_count);
            }

            void
            TearDown() override {
                runtime.shutdown();
            }

            stubBackend *backend = nullptr;
            nixlProxyRuntime runtime;
        };

        nixlProxyWorkRing
        copyDeviceWorkRing(const nixlProxyChannelView &view) {
            nixlProxyWorkRing ring{};
            EXPECT_EQ(cudaMemcpy(&ring, view.workRing, sizeof(ring), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            return ring;
        }

        // Resolve the pinned-host alias of a device-mapped submission or completion buffer.
        // GDR-backed control words do not have CUDA host aliases and must be read as device memory.
        template<class T>
        T *
        hostAliasOf(T *device_alias) {
            cudaPointerAttributes attrs{};
            EXPECT_EQ(cudaPointerGetAttributes(&attrs, device_alias), cudaSuccess);
            EXPECT_NE(attrs.hostPointer, nullptr);
            return static_cast<T *>(attrs.hostPointer);
        }

        size_t
        channelViewIndex(uint32_t peer, uint32_t channel, uint32_t max_peers = 4) {
            return static_cast<size_t>(channel) * max_peers + peer;
        }

        uint32_t
        proxyMemViewId(nixlMemViewH proxy_memview) {
            if (proxy_memview == nullptr) {
                return 0;
            }
            nixlProxyDeviceMemView device_memview{};
            EXPECT_EQ(
                cudaMemcpy(
                    &device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
                cudaSuccess);
            return device_memview.proxyMemViewId;
        }

        nixlProxyDeviceMemView
        copyDeviceMemView(nixlMemViewH proxy_memview) {
            nixlProxyDeviceMemView device_memview{};
            EXPECT_EQ(
                cudaMemcpy(
                    &device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
                cudaSuccess);
            return device_memview;
        }

        std::vector<void *>
        copyDirectPointers(nixlMemViewH proxy_memview, size_t count) {
            std::vector<void *> direct_ptrs(count, nullptr);
            if (count != 0) {
                auto *direct_ptrs_dev =
                    static_cast<nixlProxyDeviceMemView *>(proxy_memview)->directPtrs;
                EXPECT_EQ(cudaMemcpy(direct_ptrs.data(),
                                     direct_ptrs_dev,
                                     sizeof(void *) * count,
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
            }
            return direct_ptrs;
        }

        std::vector<nixlBackendProxySubmission>
        waitForSubmissions(stubBackend *backend, size_t count) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            while (std::chrono::steady_clock::now() < deadline) {
                {
                    std::lock_guard<std::mutex> lock(backend->submitMutex);
                    if (backend->submissions.size() >= count) {
                        return backend->submissions;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            std::lock_guard<std::mutex> lock(backend->submitMutex);
            return backend->submissions;
        }

        bool
        waitForCompletedIdx(const nixlProxyChannelView &view, uint64_t completed_idx) {
            auto *completion_slot = hostAliasOf(view.completionSlot);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            while (std::chrono::steady_clock::now() < deadline) {
                if (__atomic_load_n(&completion_slot->completedIdx, __ATOMIC_ACQUIRE) >=
                    completed_idx) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            return __atomic_load_n(&completion_slot->completedIdx, __ATOMIC_ACQUIRE) >=
                completed_idx;
        }

        nixl_status_t
        allocateDirectChannel(nixlProxyChannelState &channel,
                              nixlProxyControlBuffer &control_slots,
                              uint32_t depth) {
            nixl_status_t status = control_slots.allocate(proxy_ci_slot_base + 1);
            if (status != NIXL_SUCCESS) {
                return status;
            }
            return channel.allocate(depth, &control_slots, proxy_ci_slot_base);
        }

        uint64_t
        deviceConsumerIdx(const nixlProxyChannelState &channel) {
            uint64_t consumer_idx = 0;
            EXPECT_EQ(cudaMemcpy(&consumer_idx,
                                 channel.consumerIdxDev,
                                 sizeof(consumer_idx),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            return consumer_idx;
        }

        nixlProxySubmission
        makeAtomicAddSubmission(nixlMemViewH dst_proxy, uint64_t value = 42) {
            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.channelId = 0;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 0;
            submission.size = sizeof(uint64_t);
            submission.value = value;
            return submission;
        }

        nixlProxySubmission
        makeInvalidAtomicAddSubmission() {
            return makeAtomicAddSubmission(nullptr);
        }

        void
        publishRecord(nixlProxySubmission *records,
                      uint32_t slot,
                      const nixlProxySubmission &submission,
                      uint64_t op_idx) {
            nixlProxySubmission record = submission;
            record.opIdx = 0;
            records[slot] = record;
            __atomic_store_n(&records[slot].opIdx, op_idx, __ATOMIC_RELEASE);
        }

        std::unique_ptr<nixlProxyWorker>
        makeDirectWorker(stubBackend *backend,
                         const nixlProxyMemViewRegistry *registry,
                         std::atomic<uint64_t> *shutdown_state,
                         nixlProxyChannelState *channel) {
            return std::make_unique<nixlProxyWorker>(
                backend, registry, shutdown_state, channel, 1, 1, 0, 1, 0);
        }

        nixl_remote_meta_dlist_t
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

        TEST_F(proxyRuntimeTest, InitCallsBackendInit) {
            ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
            EXPECT_TRUE(backend->initCalled);
            EXPECT_EQ(backend->initWorkerCount, 2u);
            EXPECT_EQ(backend->initChannelCount, 4u);
        }

        TEST_F(proxyRuntimeTest, InitRejectsNullBackend) {
            EXPECT_EQ(runtime.init(nullptr, 4, 4, 2), NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyRuntimeTest, InitRejectsZeroPeerCapacity) {
            EXPECT_EQ(initRuntime(2, 1, NIXL_SUCCESS, 0), NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyRuntimeTest, InitRejectsZeroChannels) {
            EXPECT_EQ(initRuntime(0, 2), NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyRuntimeTest, InitRejectsZeroWorkers) {
            EXPECT_EQ(initRuntime(4, 0), NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyRuntimeTest, InitPropagatesBackendFailure) {
            EXPECT_EQ(initRuntime(4, 2, NIXL_ERR_BACKEND), NIXL_ERR_BACKEND);
        }

        TEST_F(proxyRuntimeTest, DeviceChannelViewMatrixStartsAllocated) {
            ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
            const nixlProxyChannelView *views = runtime.deviceChannelViews();
            ASSERT_NE(views, nullptr);
            for (uint32_t peer = 0; peer < 4; ++peer) {
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const auto &view = views[channelViewIndex(peer, channel)];
                    EXPECT_NE(view.workRing, nullptr);
                    EXPECT_NE(view.completionSlot, nullptr);
                }
            }
        }

        TEST_F(proxyRuntimeTest, WorkRingIndicesStartAtZero) {
            dummyBackendMd remote_md;
            ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
            nixlMemViewH remote_mvh = nullptr;
            ASSERT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
                      NIXL_SUCCESS);
            const nixlProxyChannelView *views = runtime.deviceChannelViews();
            for (uint32_t channel = 0; channel < 2; ++channel) {
                const nixlProxyWorkRing ring =
                    copyDeviceWorkRing(views[channelViewIndex(0, channel)]);
                uint64_t producer = 0;
                uint64_t consumer = 0;
                ASSERT_EQ(
                    cudaMemcpy(
                        &producer, ring.producerIdx, sizeof(producer), cudaMemcpyDeviceToHost),
                    cudaSuccess);
                ASSERT_EQ(
                    cudaMemcpy(
                        &consumer, ring.consumerIdx, sizeof(consumer), cudaMemcpyDeviceToHost),
                    cudaSuccess);
                EXPECT_EQ(producer, 0u);
                EXPECT_EQ(consumer, 0u);
            }
        }

        TEST_F(proxyRuntimeTest, CompletionSlotsInitialized) {
            dummyBackendMd remote_md;
            ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
            nixlMemViewH remote_mvh = nullptr;
            ASSERT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
                      NIXL_SUCCESS);
            const nixlProxyChannelView *views = runtime.deviceChannelViews();
            for (uint32_t channel = 0; channel < 2; ++channel) {
                nixlProxyCompletionSlot slot{};
                ASSERT_EQ(cudaMemcpy(&slot,
                                     views[channelViewIndex(0, channel)].completionSlot,
                                     sizeof(nixlProxyCompletionSlot),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                EXPECT_EQ(slot.completedIdx, 0u);
                EXPECT_EQ(slot.nextStatus, NIXL_IN_PROG);
            }
        }

        TEST_F(proxyRuntimeTest, WorkerCountIsNotClampedToPeerCapacity) {
            ASSERT_EQ(initRuntime(8, 8, NIXL_SUCCESS, 2), NIXL_SUCCESS);
            EXPECT_EQ(backend->initWorkerCount, 8u);
            EXPECT_EQ(backend->initChannelCount, 8u);
        }

        TEST_F(proxyRuntimeTest, WorkerCountClampedToChannelCount) {
            ASSERT_EQ(initRuntime(2, 8, NIXL_SUCCESS, 4), NIXL_SUCCESS);
            EXPECT_EQ(backend->initWorkerCount, 2u);
            EXPECT_EQ(backend->initChannelCount, 2u);
        }

        TEST_F(proxyRuntimeTest, DeviceContextPopulated) {
            ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
            auto *device_ctx = runtime.deviceContext();
            ASSERT_NE(device_ctx, nullptr);
            nixlProxyDeviceContextData ctx{};
            ASSERT_EQ(cudaMemcpy(&ctx, device_ctx, sizeof(ctx), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(ctx.maxPeers, 4u);
            EXPECT_EQ(ctx.numChannels, 3u);
            EXPECT_NE(ctx.channels, nullptr);
            EXPECT_NE(ctx.shutdownWord, nullptr);
        }

        TEST_F(proxyRuntimeTest, DeviceContextCarriedByMemView) {
            dummyBackendMd remote_md;
            ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
            nixlMemViewH remote_mvh = nullptr;
            ASSERT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
                      NIXL_SUCCESS);
            EXPECT_EQ(copyDeviceMemView(remote_mvh).context, runtime.deviceContext());
        }

        TEST_F(proxyRuntimeTest, DeviceContextNullAfterShutdown) {
            ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
            ASSERT_NE(runtime.deviceContext(), nullptr);
            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
            EXPECT_EQ(runtime.deviceContext(), nullptr);
        }

        TEST_F(proxyRuntimeTest, StartWorkersAndShutdown) {
            ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);

            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, RepeatedStartWorkersIsRejected) {
            ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            EXPECT_EQ(runtime.startWorkers(), NIXL_ERR_INVALID_PARAM);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, ShutdownWithoutStartIsHarmless) {
            ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
            EXPECT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, ShutdownBeforeInitIsHarmless) {
            EXPECT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, DoubleShutdownIsHarmless) {
            ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
            EXPECT_EQ(runtime.shutdown(), NIXL_SUCCESS);
            EXPECT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, InitAfterShutdownWorks) {
            ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);

            ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
            EXPECT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, SingleChannelSingleWorker) {
            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, ManyChannelsManyWorkers) {
            ASSERT_EQ(initRuntime(16, 4), NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);

            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, PrepMemViewProducesReadyEntries) {
            dummyBackendMd local_md;
            dummyBackendMd remote_md;
            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
            const auto local_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x10});
            const auto remote_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x20});

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
            ASSERT_EQ(runtime.prepMemView(local_backend, local_dlist, &src_proxy), NIXL_SUCCESS);
            ASSERT_EQ(runtime.prepMemView(remote_backend, remote_dlist, &dst_proxy), NIXL_SUCCESS);

            nixlMemViewH resolved = nullptr;
            EXPECT_TRUE(runtime.resolveProxyMemView(src_proxy, resolved));
            EXPECT_EQ(resolved, local_backend);
            EXPECT_TRUE(runtime.resolveProxyMemView(dst_proxy, resolved));
            EXPECT_EQ(resolved, remote_backend);

            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.srcOffset = 4;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 8;
            submission.size = 32;

            nixlBackendProxySubmission prepared_submission;
            ASSERT_EQ(runtime.memviewRegistry().prepareSubmission(submission, prepared_submission),
                      NIXL_SUCCESS);
            EXPECT_EQ(prepared_submission.local.desc.addr, 0x1004u);
            EXPECT_EQ(prepared_submission.local.desc.len, 32u);
            EXPECT_EQ(prepared_submission.local.desc.metadataP, &local_md);
            EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2008u);
            EXPECT_EQ(prepared_submission.remote.desc.len, 32u);
            EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md);
            EXPECT_EQ(prepared_submission.remoteAgent, "peer");
        }

        TEST_F(proxyRuntimeTest, PrepMemViewRejectsNullOutput) {
            dummyBackendMd local_md;
            nixl_meta_dlist_t local_dlist(DRAM_SEG);
            local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

            EXPECT_EQ(runtime.prepMemView(local_dlist, nullptr), NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyRuntimeTest, PrepRemoteMemViewRejectsNonVramMetadata) {
            dummyBackendMd remote_md;

            nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
            nixlRemoteMetaDesc remote_desc("peer");
            remote_desc.addr = 0x2000;
            remote_desc.len = 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remote_md;
            remote_dlist.addDesc(remote_desc);

            nixlMemViewH dst_proxy = nullptr;
            EXPECT_EQ(runtime.prepMemView(remote_dlist, &dst_proxy), NIXL_ERR_INVALID_PARAM);
        }

        TEST_F(proxyRuntimeTest, PrepRemoteMemViewStoresResolvedDirectPointers) {
            dummyBackendMd remote_md;
            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

            backend->resolveDirectPointerRc = NIXL_SUCCESS;
            backend->directPtrsToReturn = {reinterpret_cast<void *>(uintptr_t{0xabc00000}),
                                           nullptr};

            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer0", "peer1"}, &remote_md),
                                          &dst_proxy),
                      NIXL_SUCCESS);

            EXPECT_EQ(backend->resolveDirectPointerCalls, 1u);
            EXPECT_EQ(backend->lastResolvedDescCount, 2u);
            const nixlProxyDeviceMemView device_memview = copyDeviceMemView(dst_proxy);
            EXPECT_EQ(device_memview.proxyMemViewId, proxyMemViewId(dst_proxy));
            EXPECT_EQ(device_memview.directPtrCount, backend->directPtrsToReturn.size());
            EXPECT_EQ(copyDirectPointers(dst_proxy, backend->directPtrsToReturn.size()),
                      backend->directPtrsToReturn);
        }

        TEST_F(proxyRuntimeTest, PrepRemoteMemViewFallsBackWhenDirectPointersUnsupported) {
            dummyBackendMd remote_md;
            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_SUCCESS);

            EXPECT_EQ(backend->resolveDirectPointerCalls, 1u);
            EXPECT_EQ(copyDeviceMemView(dst_proxy).directPtrCount, 0u);
        }

        TEST_F(proxyRuntimeTest, PrepRemoteMemViewPropagatesDirectPointerResolverErrors) {
            dummyBackendMd remote_md;
            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
            backend->resolveDirectPointerRc = NIXL_ERR_INVALID_PARAM;

            nixlMemViewH dst_proxy = nullptr;
            EXPECT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_ERR_INVALID_PARAM);
            EXPECT_EQ(dst_proxy, nullptr);
            EXPECT_EQ(backend->resolveDirectPointerCalls, 1u);
        }

        TEST_F(proxyRuntimeTest, WorkerSubmitsPreparedTransportDescriptors) {
            dummyBackendMd local_md;
            dummyBackendMd remote_md;

            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
            backend->submitRc = NIXL_IN_PROG;
            backend->completionRc = NIXL_SUCCESS;
            backend->requestToReturn = nixlBackendProxyRequest{101, 7};

            nixlMemViewH src_proxy = nullptr;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(runtime.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x10}),
                                                   &src_proxy),
                      NIXL_SUCCESS);

            nixl_meta_dlist_t local_dlist(DRAM_SEG);
            local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
            ASSERT_EQ(runtime.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);

            nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
            nixlRemoteMetaDesc remote_desc("peer");
            remote_desc.addr = 0x2000;
            remote_desc.len = 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remote_md;
            remote_dlist.addDesc(remote_desc);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
            ASSERT_EQ(runtime.prepMemView(remote_dlist, &dst_proxy), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opIdx = 11;
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.channelId = 0;
            submission.srcProxyMemViewId = proxyMemViewId(src_proxy);
            submission.srcOffset = 4;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 8;
            submission.size = 32;

            const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime.deviceChannelViews()[0]);
            auto *records = hostAliasOf(ring.records);
            ASSERT_NE(records, nullptr);
            submission.opIdx = 0;
            records[0] = submission;
            __atomic_store_n(&records[0].opIdx, uint64_t{11}, __ATOMIC_RELEASE);

            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            while (std::chrono::steady_clock::now() < deadline) {
                {
                    std::lock_guard<std::mutex> lock(backend->submitMutex);
                    if (!backend->submissions.empty()) {
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            std::vector<nixlBackendProxySubmission> submissions;
            {
                std::lock_guard<std::mutex> lock(backend->submitMutex);
                submissions = backend->submissions;
            }
            ASSERT_TRUE(waitForCompletedIdx(runtime.deviceChannelViews()[0], 11));

            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);

            ASSERT_EQ(submissions.size(), 1u);
            const auto &prepared = submissions.front();
            EXPECT_EQ(prepared.opIdx, 11u);
            EXPECT_EQ(prepared.channelId, 0u);
            EXPECT_EQ(prepared.peerIndex, 0u);
            EXPECT_EQ(prepared.local.memType, DRAM_SEG);
            EXPECT_EQ(prepared.local.desc.addr, 0x1004u);
            EXPECT_EQ(prepared.local.desc.len, 32u);
            EXPECT_EQ(prepared.local.desc.metadataP, &local_md);
            EXPECT_EQ(prepared.remote.memType, VRAM_SEG);
            EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
            EXPECT_EQ(prepared.remote.desc.len, 32u);
            EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
            EXPECT_EQ(prepared.remoteAgent, "peer");
            EXPECT_EQ(backend->lastCheckedRequest.token, 101u);
            EXPECT_EQ(backend->lastCheckedRequest.context, 7u);
            EXPECT_GT(backend->checkCompletionCalls, 0u);
        }

        TEST_F(proxyRuntimeTest, WorkerSubmitsPreparedAtomicAddDescriptor) {
            dummyBackendMd remote_md;

            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
            backend->submitRc = NIXL_IN_PROG;
            backend->completionRc = NIXL_SUCCESS;
            backend->requestToReturn = nixlBackendProxyRequest{202, 8};

            nixlMemViewH dst_proxy = nullptr;
            nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
            nixlRemoteMetaDesc remote_desc("peer");
            remote_desc.addr = 0x2000;
            remote_desc.len = 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remote_md;
            remote_dlist.addDesc(remote_desc);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
            ASSERT_EQ(runtime.prepMemView(remote_dlist, &dst_proxy), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opIdx = 11;
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.channelId = 0;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 8;
            submission.size = sizeof(uint64_t);
            submission.value = 42;

            const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime.deviceChannelViews()[0]);
            auto *records = hostAliasOf(ring.records);
            ASSERT_NE(records, nullptr);
            submission.opIdx = 0;
            records[0] = submission;
            __atomic_store_n(&records[0].opIdx, uint64_t{11}, __ATOMIC_RELEASE);

            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            while (std::chrono::steady_clock::now() < deadline) {
                {
                    std::lock_guard<std::mutex> lock(backend->submitMutex);
                    if (!backend->submissions.empty()) {
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            std::vector<nixlBackendProxySubmission> submissions;
            {
                std::lock_guard<std::mutex> lock(backend->submitMutex);
                submissions = backend->submissions;
            }
            ASSERT_TRUE(waitForCompletedIdx(runtime.deviceChannelViews()[0], 11));

            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);

            ASSERT_EQ(submissions.size(), 1u);
            const auto &prepared = submissions.front();
            EXPECT_EQ(prepared.opIdx, 11u);
            EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
            EXPECT_EQ(prepared.channelId, 0u);
            EXPECT_EQ(prepared.peerIndex, 0u);
            EXPECT_EQ(prepared.remote.memType, VRAM_SEG);
            EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
            EXPECT_EQ(prepared.remote.desc.len, sizeof(uint64_t));
            EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
            EXPECT_EQ(prepared.remoteAgent, "peer");
            EXPECT_EQ(prepared.value, 42u);
            EXPECT_EQ(backend->lastCheckedRequest.token, 202u);
            EXPECT_EQ(backend->lastCheckedRequest.context, 8u);
            EXPECT_GT(backend->checkCompletionCalls, 0u);
        }

        TEST_F(proxyRuntimeTest, ShutdownReleasesPendingBackendRequests) {
            dummyBackendMd remote_md;

            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
            backend->submitRc = NIXL_IN_PROG;
            backend->completionRc = NIXL_IN_PROG;
            backend->requestToReturn = nixlBackendProxyRequest{303, 9};
            auto backend_state = backend->state;

            nixlMemViewH dst_proxy = nullptr;
            nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
            nixlRemoteMetaDesc remote_desc("peer");
            remote_desc.addr = 0x2000;
            remote_desc.len = 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remote_md;
            remote_dlist.addDesc(remote_desc);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
            ASSERT_EQ(runtime.prepMemView(remote_dlist, &dst_proxy), NIXL_SUCCESS);

            nixlProxySubmission submission{};
            submission.opIdx = 31;
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.channelId = 0;
            submission.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            submission.dstOffset = 8;
            submission.size = sizeof(uint64_t);
            submission.value = 42;

            const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime.deviceChannelViews()[0]);
            auto *records = hostAliasOf(ring.records);
            ASSERT_NE(records, nullptr);
            submission.opIdx = 0;
            records[0] = submission;
            __atomic_store_n(&records[0].opIdx, uint64_t{31}, __ATOMIC_RELEASE);

            const auto submissions = waitForSubmissions(backend, 1);
            ASSERT_EQ(submissions.size(), 1u);
            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);

            std::lock_guard<std::mutex> lock(backend_state->releasedMutex);
            ASSERT_EQ(backend_state->releasedRequests.size(), 1u);
            EXPECT_EQ(backend_state->releasedRequests.front().token, 303u);
            EXPECT_EQ(backend_state->releasedRequests.front().context, 9u);
        }

        TEST_F(proxyRuntimeTest, WorkerSubmitsReadyPeersForOwnedChannel) {
            dummyBackendMd local_md;
            dummyBackendMd remote_md;

            ASSERT_EQ(initRuntime(1, 1, NIXL_SUCCESS, 2), NIXL_SUCCESS);

            nixlMemViewH src_proxy = nullptr;
            ASSERT_EQ(runtime.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x10}),
                                                   &src_proxy),
                      NIXL_SUCCESS);

            nixl_meta_dlist_t local_dlist(DRAM_SEG);
            local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
            ASSERT_EQ(runtime.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);

            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer0", "peer1"}, &remote_md),
                                          &dst_proxy),
                      NIXL_SUCCESS);
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);

            nixlProxySubmission peer0{};
            peer0.opcode = nixl_proxy_opcode_t::PUT;
            peer0.channelId = 0;
            peer0.srcProxyMemViewId = proxyMemViewId(src_proxy);
            peer0.dstProxyMemViewId = proxyMemViewId(dst_proxy);
            peer0.dstIndex = 0;
            peer0.size = 32;

            nixlProxySubmission peer1 = peer0;
            peer1.dstIndex = 1;

            const nixlProxyWorkRing ring0 =
                copyDeviceWorkRing(runtime.deviceChannelViews()[channelViewIndex(0, 0, 2)]);
            const nixlProxyWorkRing ring1 =
                copyDeviceWorkRing(runtime.deviceChannelViews()[channelViewIndex(1, 0, 2)]);
            auto *records0 = hostAliasOf(ring0.records);
            auto *records1 = hostAliasOf(ring1.records);
            ASSERT_NE(records0, nullptr);
            ASSERT_NE(records1, nullptr);

            records0[0] = peer0;
            records1[0] = peer1;
            __atomic_store_n(&records0[0].opIdx, uint64_t{21}, __ATOMIC_RELEASE);
            __atomic_store_n(&records1[0].opIdx, uint64_t{22}, __ATOMIC_RELEASE);

            const auto submissions = waitForSubmissions(backend, 2);
            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);

            ASSERT_EQ(submissions.size(), 2u);
            std::vector<bool> seen(2, false);
            for (const auto &submission : submissions) {
                ASSERT_LT(submission.peerIndex, 2u);
                EXPECT_EQ(submission.channelId, 0u);
                seen[submission.peerIndex] = true;
            }
            EXPECT_TRUE(seen[0]);
            EXPECT_TRUE(seen[1]);
        }

        TEST_F(proxyRuntimeTest, ConsumerIndexAdvancesOnlyAfterBackendCompletion) {
            dummyBackendMd remote_md;
            stubBackend backend;
            backend.submitRc = NIXL_IN_PROG;
            backend.completionRc = NIXL_IN_PROG;

            nixlProxyMemViewRegistry registry;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_SUCCESS);

            nixlProxyChannelState channel;
            nixlProxyControlBuffer control_slots;
            ASSERT_EQ(allocateDirectChannel(channel, control_slots, 2), NIXL_SUCCESS);
            std::atomic<uint64_t> shutdown_state{
                static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
            auto worker = makeDirectWorker(&backend, &registry, &shutdown_state, &channel);

            publishRecord(channel.recordsHost, 0, makeAtomicAddSubmission(dst_proxy), 1);

            worker->runOnce();
            ASSERT_EQ(backend.submissions.size(), 1u);
            EXPECT_EQ(deviceConsumerIdx(channel), 0u);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      0u);

            backend.setCompletionStatus(1, NIXL_SUCCESS);
            worker->runOnce();

            EXPECT_EQ(deviceConsumerIdx(channel), 1u);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      1u);
            EXPECT_EQ(channel.completionSlotHost->nextStatus, NIXL_SUCCESS);
        }

        TEST_F(proxyRuntimeTest, InFlightRequestsAreBoundedByRingDepth) {
            dummyBackendMd remote_md;
            stubBackend backend;
            backend.submitRc = NIXL_IN_PROG;
            backend.completionRc = NIXL_IN_PROG;

            nixlProxyMemViewRegistry registry;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_SUCCESS);

            nixlProxyChannelState channel;
            nixlProxyControlBuffer control_slots;
            ASSERT_EQ(allocateDirectChannel(channel, control_slots, 2), NIXL_SUCCESS);
            std::atomic<uint64_t> shutdown_state{
                static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
            auto worker = makeDirectWorker(&backend, &registry, &shutdown_state, &channel);

            const auto submission = makeAtomicAddSubmission(dst_proxy);
            publishRecord(channel.recordsHost, 0, submission, 1);
            publishRecord(channel.recordsHost, 1, submission, 2);

            worker->runOnce();
            worker->runOnce();
            ASSERT_EQ(backend.submissions.size(), 2u);
            EXPECT_EQ(deviceConsumerIdx(channel), 0u);

            publishRecord(channel.recordsHost, 0, submission, 3);
            worker->runOnce();
            EXPECT_EQ(backend.submissions.size(), 2u);

            backend.setCompletionStatus(1, NIXL_SUCCESS);
            worker->runOnce();
            EXPECT_EQ(backend.submissions.size(), 2u);
            EXPECT_EQ(deviceConsumerIdx(channel), 1u);

            worker->runOnce();
            EXPECT_EQ(backend.submissions.size(), 3u);
            EXPECT_EQ(backend.submissions.back().opIdx, 3u);
        }

        TEST_F(proxyRuntimeTest, CompletionsPublishInSubmissionOrder) {
            dummyBackendMd remote_md;
            stubBackend backend;
            backend.submitRc = NIXL_IN_PROG;
            backend.completionRc = NIXL_IN_PROG;

            nixlProxyMemViewRegistry registry;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_SUCCESS);

            nixlProxyChannelState channel;
            nixlProxyControlBuffer control_slots;
            ASSERT_EQ(allocateDirectChannel(channel, control_slots, 3), NIXL_SUCCESS);
            std::atomic<uint64_t> shutdown_state{
                static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
            auto worker = makeDirectWorker(&backend, &registry, &shutdown_state, &channel);

            const auto submission = makeAtomicAddSubmission(dst_proxy);
            publishRecord(channel.recordsHost, 0, submission, 1);
            publishRecord(channel.recordsHost, 1, submission, 2);

            worker->runOnce();
            worker->runOnce();
            ASSERT_EQ(backend.submissions.size(), 2u);

            backend.setCompletionStatus(2, NIXL_SUCCESS);
            worker->runOnce();
            EXPECT_EQ(deviceConsumerIdx(channel), 0u);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      0u);

            backend.setCompletionStatus(1, NIXL_SUCCESS);
            worker->runOnce();
            EXPECT_EQ(deviceConsumerIdx(channel), 2u);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      2u);
        }

        TEST_F(proxyRuntimeTest, PreparationErrorLatchesStatusButLaterWorkIsReclaimed) {
            dummyBackendMd remote_md;
            stubBackend backend;
            backend.submitRc = NIXL_IN_PROG;
            backend.completionRc = NIXL_SUCCESS;

            nixlProxyMemViewRegistry registry;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_SUCCESS);

            nixlProxyChannelState channel;
            nixlProxyControlBuffer control_slots;
            ASSERT_EQ(allocateDirectChannel(channel, control_slots, 3), NIXL_SUCCESS);
            std::atomic<uint64_t> shutdown_state{
                static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
            auto worker = makeDirectWorker(&backend, &registry, &shutdown_state, &channel);

            publishRecord(channel.recordsHost, 0, makeInvalidAtomicAddSubmission(), 1);
            worker->runOnce();
            EXPECT_EQ(deviceConsumerIdx(channel), 1u);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      1u);
            EXPECT_LT(channel.completionSlotHost->nextStatus, 0);

            publishRecord(channel.recordsHost, 1, makeAtomicAddSubmission(dst_proxy), 2);
            worker->runOnce();
            EXPECT_EQ(deviceConsumerIdx(channel), 2u);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      1u);
            ASSERT_EQ(backend.submissions.size(), 1u);
            EXPECT_EQ(backend.submissions.front().opIdx, 2u);
        }

        TEST_F(proxyRuntimeTest, SubmitAndCompletionErrorsLatchFirstStatusAndRetireWork) {
            dummyBackendMd remote_md;
            stubBackend backend;
            backend.submitRcs = {NIXL_ERR_BACKEND, NIXL_IN_PROG, NIXL_IN_PROG};
            backend.completionRc = NIXL_IN_PROG;

            nixlProxyMemViewRegistry registry;
            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(registry.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_SUCCESS);

            nixlProxyChannelState channel;
            nixlProxyControlBuffer control_slots;
            ASSERT_EQ(allocateDirectChannel(channel, control_slots, 4), NIXL_SUCCESS);
            std::atomic<uint64_t> shutdown_state{
                static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
            auto worker = makeDirectWorker(&backend, &registry, &shutdown_state, &channel);

            const auto submission = makeAtomicAddSubmission(dst_proxy);
            publishRecord(channel.recordsHost, 0, submission, 1);
            publishRecord(channel.recordsHost, 1, submission, 2);
            publishRecord(channel.recordsHost, 2, submission, 3);

            worker->runOnce();
            EXPECT_EQ(deviceConsumerIdx(channel), 1u);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      1u);
            const nixl_status_t first_error = channel.completionSlotHost->nextStatus;
            EXPECT_LT(first_error, 0);

            worker->runOnce();
            ASSERT_EQ(backend.submissions.size(), 2u);
            backend.setCompletionStatus(1, NIXL_ERR_BACKEND);
            worker->runOnce();
            EXPECT_EQ(deviceConsumerIdx(channel), 2u);
            EXPECT_EQ(channel.completionSlotHost->nextStatus, first_error);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      1u);

            backend.setCompletionStatus(2, NIXL_SUCCESS);
            worker->runOnce();
            EXPECT_EQ(deviceConsumerIdx(channel), 3u);
            EXPECT_EQ(channel.completionSlotHost->nextStatus, first_error);
            EXPECT_EQ(__atomic_load_n(&channel.completionSlotHost->completedIdx, __ATOMIC_ACQUIRE),
                      1u);
        }

        TEST_F(proxyRuntimeTest, ShutdownReleasesAllPendingBackendRequests) {
            dummyBackendMd remote_md;

            ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
            backend->submitRc = NIXL_IN_PROG;
            backend->completionRc = NIXL_IN_PROG;
            auto backend_state = backend->state;

            nixlMemViewH dst_proxy = nullptr;
            ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
            ASSERT_EQ(runtime.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy),
                      NIXL_SUCCESS);

            const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime.deviceChannelViews()[0]);
            auto *records = hostAliasOf(ring.records);
            ASSERT_NE(records, nullptr);

            const auto submission = makeAtomicAddSubmission(dst_proxy);
            publishRecord(records, 0, submission, 1);
            publishRecord(records, 1, submission, 2);

            const auto submissions = waitForSubmissions(backend, 2);
            ASSERT_EQ(submissions.size(), 2u);
            ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);

            std::lock_guard<std::mutex> lock(backend_state->releasedMutex);
            ASSERT_EQ(backend_state->releasedRequests.size(), 2u);
            EXPECT_EQ(backend_state->releasedRequests[0].token, 1u);
            EXPECT_EQ(backend_state->releasedRequests[1].token, 2u);
        }

    } // namespace
} // namespace proxy_runtime
} // namespace gtest
