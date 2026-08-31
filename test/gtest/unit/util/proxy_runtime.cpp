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

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_protocol.h"
#include "device/proxy/proxy_runtime.h"
#include "mocks/proxy_mocks.h"

namespace gtest {
namespace proxy_runtime {

    using proxy_mocks::DummyBackendMD;
    using proxy_mocks::makeLocalDlist;
    using proxy_mocks::makeRemoteDesc;
    using proxy_mocks::MockBackend;
    using proxy_mocks::MockDeviceOps;

    constexpr uint32_t kRingDepth = 4;

    class ProxyRuntimeTest : public testing::Test {
    protected:
        void
        TearDown() override {
            backend_.completeEverything();
            runtime_.reset();
            EXPECT_EQ(allocator_.liveAllocations(), 0u);
        }

        nixl::proxyConfig
        makeConfig(uint32_t channel_count, uint32_t max_peers, uint32_t thread_count) const {
            nixl::proxyConfig config;
            config.enabled = true;
            config.channel_count = channel_count;
            config.thread_count = thread_count == 0 ? channel_count : thread_count;
            config.max_peers = max_peers;
            config.ring_depth = kRingDepth;
            return config;
        }

        nixl_status_t
        createRuntime(uint32_t channel_count = 1,
                      uint32_t max_peers = 1,
                      uint32_t thread_count = 0) {
            max_peers_ = max_peers;
            return nixl::proxyRuntime::create(backend_.ops(),
                                              makeConfig(channel_count, max_peers, thread_count),
                                              runtime_,
                                              allocator_);
        }

        /** Translate mapped device aliases so incorrect host-pointer publication fails. */
        struct ChannelAccess {
            const nixlProxyWorkRing *ring = nullptr;
            nixlProxySubmission *records = nullptr;
            const nixlProxyCompletionSlot *completion = nullptr;
            const uint64_t *consumer_idx = nullptr;
        };

        ChannelAccess
        channel(uint32_t channel_id = 0, uint32_t peer = 0) const {
            const size_t slot = static_cast<size_t>(channel_id) * max_peers_ + peer;
            const nixlProxyChannelView &view = runtime_->deviceChannelViews()[slot];
            ChannelAccess access;
            access.ring = view.work_ring;
            access.records = allocator_.hostAlias(access.ring->records);
            access.completion = allocator_.hostAlias(view.completion_slot);
            access.consumer_idx = allocator_.hostAlias(access.ring->consumer_idx);
            EXPECT_NE(access.records, nullptr);
            EXPECT_NE(access.completion, nullptr);
            EXPECT_NE(access.consumer_idx, nullptr);
            return access;
        }

        /** Publish the record body before its readiness word, as the GPU does. */
        static void
        publish(const ChannelAccess &access,
                uint64_t producer_idx,
                const nixlProxySubmission &record,
                uint64_t op_idx) {
            *access.ring->producer_idx = producer_idx + 1;
            const uint32_t slot = static_cast<uint32_t>(producer_idx % kRingDepth);
            std::memcpy(reinterpret_cast<char *>(&access.records[slot]) + sizeof(uint64_t),
                        reinterpret_cast<const char *>(&record) + sizeof(uint64_t),
                        sizeof(record) - sizeof(uint64_t));
            __atomic_store_n(&access.records[slot].op_idx, op_idx, __ATOMIC_RELEASE);
        }

        static uint64_t
        consumerIdx(const ChannelAccess &access) {
            return __atomic_load_n(access.consumer_idx, __ATOMIC_ACQUIRE);
        }

        static uint64_t
        completedIdx(const ChannelAccess &access) {
            return __atomic_load_n(&access.completion->completed_idx, __ATOMIC_ACQUIRE);
        }

        template<typename Predicate>
        static bool
        waitFor(Predicate predicate) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                if (predicate()) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return predicate();
        }

        /** A local source view and a remote view with one 64-byte descriptor per agent. */
        void
        prepMemViews(nixlMemViewH &src,
                     nixlMemViewH &dst,
                     const std::vector<std::string> &agents = {"peer"}) {
            ASSERT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), &src),
                      NIXL_SUCCESS);
            nixl_remote_meta_dlist_t remote(VRAM_SEG);
            for (const auto &agent : agents) {
                remote.addDesc(makeRemoteDesc(agent, 0x2000, 64, 0, &remote_md_));
            }
            ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        }

        static uint64_t
        hostViewToken(nixlMemViewH view) {
            return static_cast<const nixlProxyDeviceMemView *>(view)->host_view;
        }

        static nixlProxySubmission
        makePut(nixlMemViewH src,
                nixlMemViewH dst,
                uint32_t channel_id = 0,
                uint32_t dst_index = 0) {
            nixlProxySubmission record{};
            record.opcode = nixl_proxy_opcode_t::PUT;
            record.channel_id = static_cast<uint16_t>(channel_id);
            record.src_view = hostViewToken(src);
            record.dst_view = hostViewToken(dst);
            record.dst_index = dst_index;
            record.operand = 4;
            record.dst_offset = 8;
            record.size = 32;
            return record;
        }

        static nixlProxySubmission
        makeAtomicAdd(nixlMemViewH dst, uint64_t value = 42) {
            nixlProxySubmission record{};
            record.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            record.dst_view = hostViewToken(dst);
            record.dst_offset = 8;
            record.size = sizeof(uint64_t);
            record.operand = value;
            return record;
        }

        MockDeviceOps allocator_;
        MockBackend backend_;
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;
        uint32_t max_peers_ = 1;
        std::unique_ptr<nixl::proxyRuntime> runtime_;
    };

    TEST_F(ProxyRuntimeTest, CreateRejectsBadInputsWithoutLeaking) {
        struct Row {
            const char *name;
            std::function<void(nixl::proxyConfig &, nixl::proxyBackendOps &)> mutate;
            nixl_status_t expected;
        };

        const std::vector<Row> rows = {
            {"incomplete callbacks",
             [](nixl::proxyConfig &, nixl::proxyBackendOps &ops) { ops.submit = nullptr; },
             NIXL_ERR_INVALID_PARAM},
            {"missing quiesce",
             [](nixl::proxyConfig &, nixl::proxyBackendOps &ops) { ops.quiesce = nullptr; },
             NIXL_ERR_INVALID_PARAM},
            {"zero peers",
             [](nixl::proxyConfig &config, nixl::proxyBackendOps &) { config.max_peers = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"zero channels",
             [](nixl::proxyConfig &config, nixl::proxyBackendOps &) { config.channel_count = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"zero threads",
             [](nixl::proxyConfig &config, nixl::proxyBackendOps &) { config.thread_count = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"zero ring depth",
             [](nixl::proxyConfig &config, nixl::proxyBackendOps &) { config.ring_depth = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"backend init failure",
             [](nixl::proxyConfig &, nixl::proxyBackendOps &ops) {
                 ops.init = [](const nixl::proxyConfig &) { return NIXL_ERR_BACKEND; };
             },
             NIXL_ERR_BACKEND},
        };
        for (const auto &row : rows) {
            nixl::proxyConfig config = makeConfig(1, 1, 1);
            nixl::proxyBackendOps ops = backend_.ops();
            row.mutate(config, ops);
            EXPECT_EQ(nixl::proxyRuntime::create(std::move(ops), config, runtime_, allocator_),
                      row.expected)
                << row.name;
            EXPECT_EQ(runtime_, nullptr) << row.name;
            EXPECT_EQ(allocator_.liveAllocations(), 0u) << row.name;
        }
        auto ops = backend_.ops();
        ops.init = [](const nixl::proxyConfig &config) {
            EXPECT_EQ(config.effectiveThreadCount(), 3u);
            return NIXL_SUCCESS;
        };
        ASSERT_EQ(nixl::proxyRuntime::create(ops, makeConfig(3, 2, 8), runtime_, allocator_),
                  NIXL_SUCCESS);
        EXPECT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), nullptr),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyRuntimeTest, PartialAllocationRollsBack) {
        for (int allocation = 0; allocation < 20; ++allocation) {
            allocator_.fail_after = allocation;
            const auto status = createRuntime(2, 2);
            if (status == NIXL_SUCCESS) {
                runtime_.reset();
            } else {
                EXPECT_EQ(runtime_, nullptr);
            }
            EXPECT_EQ(allocator_.liveAllocations(), 0u) << allocation;
        }
        allocator_.fail_after = -1;
    }

    TEST_F(ProxyRuntimeTest, QuiescenceFailureNeverFreesViews) {
        EXPECT_DEATH(
            {
                auto ops = backend_.ops();
                ops.quiesce = [](uint32_t, uint32_t) { return NIXL_ERR_BACKEND; };
                ASSERT_EQ(
                    nixl::proxyRuntime::create(ops, makeConfig(1, 1, 1), runtime_, allocator_),
                    NIXL_SUCCESS);
                ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
                static_cast<void>(runtime_->shutdown());
            },
            "Failed to quiesce proxy backend");
    }

    TEST_F(ProxyRuntimeTest, InitializedRingsFollowStartAndShutdownLifecycle) {
        ASSERT_EQ(createRuntime(/*channel_count=*/3, /*max_peers=*/2), NIXL_SUCCESS);

        const nixlProxyDeviceContextData *context = runtime_->deviceContext();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->max_peers, 2u);
        EXPECT_EQ(context->num_channels, 3u);
        ASSERT_NE(context->channels, nullptr);
        ASSERT_NE(context->shutdown_word, nullptr);

        std::set<const void *> record_arrays;
        for (uint32_t channel_id = 0; channel_id < 3; ++channel_id) {
            for (uint32_t peer = 0; peer < 2; ++peer) {
                const size_t slot = channel_id * 2 + peer;
                const ChannelAccess access = channel(channel_id, peer);
                EXPECT_EQ(context->channels[slot].work_ring, access.ring);
                EXPECT_EQ(context->channels[slot].completion_slot,
                          runtime_->deviceChannelViews()[slot].completion_slot);
                EXPECT_EQ(access.ring->depth, kRingDepth);
                EXPECT_EQ(*access.ring->producer_idx, 0u);
                EXPECT_EQ(*access.ring->consumer_idx_cache, 0u);
                EXPECT_EQ(consumerIdx(access), 0u);
                EXPECT_EQ(completedIdx(access), 0u);
                EXPECT_EQ(access.completion->completion_status, NIXL_IN_PROG);
                record_arrays.insert(access.records);
            }
        }
        EXPECT_EQ(record_arrays.size(), 6u);

        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->startWorkers(), NIXL_ERR_INVALID_PARAM);
        const uint64_t *shutdown_word = allocator_.hostAlias(context->shutdown_word);
        ASSERT_NE(shutdown_word, nullptr);
        EXPECT_EQ(__atomic_load_n(shutdown_word, __ATOMIC_ACQUIRE),
                  static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING));

        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        EXPECT_EQ(static_cast<const nixlProxyDeviceMemView *>(src)->context, context);
        EXPECT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->unregisterProxyMemView(src), NIXL_SUCCESS);
        EXPECT_EQ(backend_.submissionCount(), 0u);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->deviceContext(), nullptr);
        EXPECT_EQ(backend_.shutdownCalls(), 1u);
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(backend_.shutdownCalls(), 1u);

        runtime_.reset();
        EXPECT_EQ(allocator_.liveAllocations(), 0u);
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(backend_.shutdownCalls(), 2u);
    }

    TEST_F(ProxyRuntimeTest, RecordPathSubmitsAndCompletesInOrder) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst, {"remote-agent"});
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        publish(access, 0, makePut(src, dst), 7);
        publish(access, 1, makeAtomicAdd(dst, 42), 8);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));

        const auto submissions = backend_.submissions();
        const nixl::proxyBackendSubmission &put = submissions[0];
        EXPECT_EQ(put.op_idx, 7u);
        EXPECT_EQ(put.opcode, nixl_proxy_opcode_t::PUT);
        EXPECT_EQ(put.channel_id, 0u);
        EXPECT_EQ(put.peer_index, 0u);
        EXPECT_EQ(put.size, 32u);
        EXPECT_EQ(put.local.desc.addr, 0x1004u);
        EXPECT_EQ(put.remote.desc.addr, 0x2008u);
        const nixl::proxyBackendSubmission &atomic = submissions[1];
        EXPECT_EQ(atomic.op_idx, 8u);
        EXPECT_EQ(atomic.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
        EXPECT_EQ(atomic.size, sizeof(uint64_t));
        EXPECT_EQ(atomic.value, 42u);
        EXPECT_EQ(atomic.remote.desc.addr, 0x2008u);

        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(consumerIdx(access), 0u);
        backend_.complete(backend_.token(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(consumerIdx(access), 0u);

        backend_.complete(backend_.token(0));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 2u; }));
        EXPECT_EQ(completedIdx(access), 8u);
        EXPECT_EQ(access.completion->completion_status, NIXL_SUCCESS);
    }

    TEST_F(ProxyRuntimeTest, FullRingBackpressuresUntilTheConsumerIndexAdvances) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        for (uint64_t i = 0; i < kRingDepth; ++i) {
            publish(access, i, record, i + 1);
        }
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == kRingDepth; }));
        EXPECT_EQ(consumerIdx(access), 0u);

        publish(access, kRingDepth, record, kRingDepth + 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_EQ(backend_.submissionCount(), size_t{kRingDepth});

        for (uint32_t i = 0; i < kRingDepth; ++i) {
            backend_.complete(backend_.token(i));
        }
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == kRingDepth; }));
        EXPECT_EQ(completedIdx(access), uint64_t{kRingDepth});

        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == kRingDepth + 1; }));
        EXPECT_EQ(backend_.submissions().back().op_idx, uint64_t{kRingDepth} + 1);
        backend_.complete(backend_.token(kRingDepth));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == uint64_t{kRingDepth} + 1; }));
    }

    TEST_F(ProxyRuntimeTest, FirstErrorLatchesWhileLaterWorkIsRetired) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        const ChannelAccess access = channel();
        const nixlProxySubmission good = makePut(src, dst);

        nixlProxySubmission unknown = good;
        unknown.dst_view = 0;
        publish(access, 0, unknown, 1);
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 1u; }));
        EXPECT_EQ(completedIdx(access), 1u);
        const nixl_status_t first_error = access.completion->completion_status;
        EXPECT_LT(first_error, 0);
        EXPECT_EQ(backend_.submissionCount(), 0u);

        publish(access, 1, good, 2);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));
        backend_.complete(backend_.token(0), NIXL_ERR_BACKEND);
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 2u; }));

        backend_.failNextSubmit(NIXL_ERR_BACKEND);
        publish(access, 2, good, 3);
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 3u; }));

        publish(access, 3, good, 4);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 3; }));
        backend_.complete(backend_.token(2));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 4u; }));

        EXPECT_EQ(completedIdx(access), 1u);
        EXPECT_EQ(access.completion->completion_status, first_error);
    }

    TEST_F(ProxyRuntimeTest, RemoteDirectPointersFollowTheResolver) {
        nixl_remote_meta_dlist_t remote(VRAM_SEG);
        remote.addDesc(makeRemoteDesc("peer0", 0x2000, 64, 0, &remote_md_));
        remote.addDesc(makeRemoteDesc("peer1", 0x3000, 64, 1, &remote_md_));
        const std::vector<void *> direct_ptrs{reinterpret_cast<void *>(uintptr_t{0xabc00000}),
                                              nullptr};
        nixl_status_t status = NIXL_SUCCESS;
        unsigned calls = 0;
        auto ops = backend_.ops();
        ops.resolve_direct_ptrs = [&](const auto &descs, auto &pointers) {
            ++calls;
            EXPECT_EQ(descs.descCount(), 2u);
            pointers = direct_ptrs;
            return status;
        };
        ASSERT_EQ(nixl::proxyRuntime::create(ops, makeConfig(1, 2, 1), runtime_, allocator_),
                  NIXL_SUCCESS);
        nixlMemViewH dst = nullptr;
        ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        const auto *view = static_cast<const nixlProxyDeviceMemView *>(dst);
        ASSERT_EQ(view->direct_ptr_count, 2u);
        void *const *stored = nixlProxyDeviceMemViewDirectPtrs(view);
        EXPECT_EQ(std::vector<void *>(stored, stored + 2), direct_ptrs);

        status = NIXL_ERR_INVALID_PARAM;
        const size_t live = allocator_.liveAllocations();
        nixlMemViewH failed = nullptr;
        EXPECT_EQ(runtime_->prepMemView(remote, &failed), status);
        EXPECT_EQ(failed, nullptr);
        EXPECT_EQ(allocator_.liveAllocations(), live);
        EXPECT_EQ(calls, 2u);

        runtime_.reset();
        ASSERT_EQ(createRuntime(1, 2), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        EXPECT_EQ(static_cast<const nixlProxyDeviceMemView *>(dst)->direct_ptr_count, 0u);
    }

    TEST_F(ProxyRuntimeTest, ChannelsAndPeersKeepToTheirOwnRings) {
        ASSERT_EQ(createRuntime(/*channel_count=*/3, /*max_peers=*/2, /*thread_count=*/2),
                  NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        ASSERT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), &src),
                  NIXL_SUCCESS);
        nixl_remote_meta_dlist_t remote(VRAM_SEG);
        remote.addDesc(makeRemoteDesc("peer0", 0x2000, 64, 0, &remote_md_));
        remote.addDesc(makeRemoteDesc("peer1", 0x3000, 64, 0, &remote_md_));
        ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess striped = channel(2, 1);
        const ChannelAccess middle = channel(1, 0);
        const ChannelAccess idle = channel(0, 0);
        publish(striped, 0, makePut(src, dst, /*channel_id=*/2, /*dst_index=*/1), 5);
        publish(middle, 0, makePut(src, dst, /*channel_id=*/1, /*dst_index=*/0), 6);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));

        for (const auto &submission : backend_.submissions()) {
            if (submission.op_idx == 5) {
                EXPECT_EQ(submission.channel_id, 2u);
                EXPECT_EQ(submission.peer_index, 1u);
                EXPECT_EQ(submission.remote.desc.addr, 0x3008u);
            } else {
                EXPECT_EQ(submission.op_idx, 6u);
                EXPECT_EQ(submission.channel_id, 1u);
                EXPECT_EQ(submission.peer_index, 0u);
                EXPECT_EQ(submission.remote.desc.addr, 0x2008u);
            }
        }

        backend_.complete(backend_.token(0));
        backend_.complete(backend_.token(1));
        ASSERT_TRUE(
            waitFor([&]() { return consumerIdx(striped) == 1u && consumerIdx(middle) == 1u; }));
        EXPECT_EQ(completedIdx(striped), 5u);
        EXPECT_EQ(completedIdx(middle), 6u);
        EXPECT_EQ(consumerIdx(striped), 1u);
        EXPECT_EQ(consumerIdx(middle), 1u);
        EXPECT_EQ(consumerIdx(idle), 0u);
        EXPECT_EQ(completedIdx(idle), 0u);
    }

    TEST_F(ProxyRuntimeTest, DrainSubmitsQueuedRecordsBeforeRetire) {
        nixlMemViewH src = nullptr, dst = nullptr;
        const auto caller = std::this_thread::get_id();
        unsigned quiesced = 0;
        auto ops = backend_.ops();
        ops.quiesce = [&](uint32_t, uint32_t) {
            EXPECT_NE(std::this_thread::get_id(), caller);
            if (quiesced++ == 0) {
                EXPECT_EQ(consumerIdx(channel()), kRingDepth);
                EXPECT_EQ(completedIdx(channel()), kRingDepth);
                EXPECT_FALSE(allocator_.wasFreed(dst));
            }
            return NIXL_SUCCESS;
        };
        ASSERT_EQ(nixl::proxyRuntime::create(ops, makeConfig(1, 1, 1), runtime_, allocator_),
                  NIXL_SUCCESS);
        prepMemViews(src, dst);
        backend_.completeEverything();
        const ChannelAccess access = channel();
        for (uint64_t i = 0; i < kRingDepth; ++i) {
            publish(access, i, makePut(src, dst), i + 1);
        }
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        EXPECT_EQ(backend_.submissionCount(), size_t{kRingDepth});
        for (const auto &submission : backend_.submissions()) {
            EXPECT_EQ(submission.remote.desc.addr, 0x2008u);
        }
        EXPECT_TRUE(allocator_.wasFreed(dst));
        EXPECT_EQ(consumerIdx(access), 0u);
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(access.completion->completion_status, NIXL_IN_PROG);
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(quiesced, 2u);
    }

    TEST_F(ProxyRuntimeTest, RingsAreUsableAfterDrain) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        publish(access, 0, makePut(src, dst), 1);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));

        backend_.complete(backend_.token(0), NIXL_ERR_BACKEND);
        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        EXPECT_TRUE(allocator_.wasFreed(dst));
        EXPECT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(consumerIdx(access), 0u);
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(access.completion->completion_status, NIXL_IN_PROG);
        ASSERT_EQ(runtime_->unregisterProxyMemView(src), NIXL_SUCCESS);

        nixlMemViewH new_src = nullptr, new_dst = nullptr;
        ASSERT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), &new_src),
                  NIXL_SUCCESS);
        nixl_remote_meta_dlist_t replacement(VRAM_SEG);
        replacement.addDesc(makeRemoteDesc("peer", 0x9000, 64, 0, &remote_md_));
        ASSERT_EQ(runtime_->prepMemView(replacement, &new_dst), NIXL_SUCCESS);
        backend_.completeEverything();
        publish(access, 0, makePut(new_src, new_dst), 7);

        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 1u; }));
        EXPECT_EQ(completedIdx(access), 7u);
        EXPECT_EQ(access.completion->completion_status, NIXL_SUCCESS);
        EXPECT_EQ(backend_.submissionCount(), 2u);
        EXPECT_EQ(backend_.submissions().back().remote.desc.addr, 0x9008u);
    }

    TEST_F(ProxyRuntimeTest, ReleaseWaitsForTerminalErrorsBeyondOldDeadline) {
        ASSERT_EQ(createRuntime(1, 2), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst, {"slow", "healthy"});
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        publish(channel(0, 0), 0, makePut(src, dst, 0, 0), 1);
        publish(channel(0, 1), 0, makePut(src, dst, 0, 1), 1);
        ASSERT_TRUE(waitFor([&] { return backend_.submissionCount() == 2; }));
        const auto submissions = backend_.submissions();
        for (size_t i = 0; i < submissions.size(); ++i) {
            if (submissions[i].peer_index == 1) {
                backend_.complete(backend_.token(i));
            }
        }
        std::atomic<bool> done{false};
        std::thread release([&] {
            EXPECT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
            done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        EXPECT_FALSE(done.load(std::memory_order_acquire));
        EXPECT_FALSE(allocator_.wasFreed(dst));
        EXPECT_EQ(backend_.quiesceCalls(), 0u);
        EXPECT_TRUE(waitFor([&] { return completedIdx(channel(0, 1)) == 1; }));
        for (size_t i = 0; i < submissions.size(); ++i) {
            if (submissions[i].peer_index == 0) {
                backend_.complete(backend_.token(i), NIXL_ERR_REMOTE_DISCONNECT);
            }
        }
        release.join();
        EXPECT_TRUE(done.load());
        EXPECT_TRUE(allocator_.wasFreed(dst));
        EXPECT_EQ(backend_.quiesceCalls(), 2u);
    }

    TEST_F(ProxyRuntimeTest, ShutdownRetiresCompletedAndPendingWork) {
        for (bool pending : {false, true}) {
            SCOPED_TRACE(pending);
            ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
            nixlMemViewH src = nullptr, dst = nullptr;
            prepMemViews(src, dst);
            ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
            const auto *shutdown = allocator_.hostAlias(runtime_->deviceContext()->shutdown_word);
            const size_t previous = backend_.submissionCount();
            publish(channel(), 0, makePut(src, dst), 1);
            ASSERT_TRUE(waitFor([&] { return backend_.submissionCount() == previous + 1; }));
            const auto token = backend_.token(previous);
            if (!pending) {
                backend_.complete(token);
                ASSERT_TRUE(waitFor([&] { return consumerIdx(channel()) == 1; }));
            }
            std::atomic<bool> done{false};
            std::thread stop([&] {
                EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
                done.store(true, std::memory_order_release);
            });
            if (pending) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                EXPECT_FALSE(done.load(std::memory_order_acquire));
                EXPECT_EQ(__atomic_load_n(shutdown, __ATOMIC_ACQUIRE),
                          static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN));
                EXPECT_FALSE(allocator_.wasFreed(dst));
                backend_.complete(token, NIXL_ERR_REMOTE_DISCONNECT);
            }
            stop.join();
            EXPECT_TRUE(done.load(std::memory_order_acquire));
            EXPECT_TRUE(allocator_.wasFreed(src));
            EXPECT_TRUE(allocator_.wasFreed(dst));
            EXPECT_EQ(backend_.shutdownCalls(), pending ? 2u : 1u);
            runtime_.reset();
        }
    }

    TEST_F(ProxyRuntimeTest, UnpublishedTicketsNeverRearmSilently) {
        EXPECT_DEATH(
            {
                ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
                nixlMemViewH src = nullptr;
                nixlMemViewH dst = nullptr;
                prepMemViews(src, dst);
                backend_.completeEverything();
                publish(channel(), 0, makePut(src, dst), 1);
                publish(channel(), 2, makePut(src, dst), 3);
                ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
                static_cast<void>(runtime_->unregisterProxyMemView(dst));
            },
            "unpublished producer tickets");
    }

    TEST_F(ProxyRuntimeTest, UnstartedRuntimeCannotDiscardQueuedWork) {
        EXPECT_DEATH(
            {
                ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
                nixlMemViewH src = nullptr;
                nixlMemViewH dst = nullptr;
                prepMemViews(src, dst);
                publish(channel(), 0, makePut(src, dst), 1);
                static_cast<void>(runtime_->shutdown());
            },
            "unfinished or unpublished");
    }

} // namespace proxy_runtime
} // namespace gtest
