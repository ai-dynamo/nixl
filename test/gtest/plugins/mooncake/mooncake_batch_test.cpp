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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "mooncake_backend.h"

namespace {
// Parameters: operation, notification, split across submit threads.
class MooncakeBatchTest : public testing::TestWithParam<std::tuple<nixl_xfer_op_t, bool, bool>> {
protected:
    static constexpr size_t stride = 32;
    static constexpr size_t length = 16;
    static constexpr size_t max_count = 2048;
    std::vector<uint8_t> local_buffer = std::vector<uint8_t>(max_count * stride);
    std::vector<uint8_t> remote_buffer = std::vector<uint8_t>(max_count * stride);
    std::unique_ptr<nixlMooncakeEngine> initiator;
    std::unique_ptr<nixlMooncakeEngine> target;
    nixlBackendMD *local_md = nullptr;
    nixlBackendMD *target_md = nullptr;
    nixlBackendMD *remote_md = nullptr;
    nixlBackendReqH *handle = nullptr;

    std::unique_ptr<nixlMooncakeEngine>
    makeEngine(const std::string &name, bool split = false) {
        nixl_b_params_t params;
        if (split) {
            // Requests 1023 to 2048 below become four batches each.
            params["num_threads"] = "4";
            params["split_batch_size"] = "256";
        }
        nixlBackendInitParams init{};
        init.localAgent = name;
        init.type = "MOONCAKE";
        init.customParams = &params;
        return std::make_unique<nixlMooncakeEngine>(&init);
    }

    void
    SetUp() override {
        initiator = makeEngine("batch-initiator", std::get<2>(GetParam()));
        target = makeEngine("batch-target");
        ASSERT_FALSE(initiator->getInitErr());
        ASSERT_FALSE(target->getInitErr());
        nixlBlobDesc local_desc(
            reinterpret_cast<uintptr_t>(local_buffer.data()), local_buffer.size(), 0, "");
        nixlBlobDesc remote_desc(
            reinterpret_cast<uintptr_t>(remote_buffer.data()), remote_buffer.size(), 0, "");
        ASSERT_EQ(initiator->registerMem(local_desc, DRAM_SEG, local_md), NIXL_SUCCESS);
        ASSERT_EQ(target->registerMem(remote_desc, DRAM_SEG, target_md), NIXL_SUCCESS);
        std::string conn_info;
        ASSERT_EQ(target->getConnInfo(conn_info), NIXL_SUCCESS);
        ASSERT_EQ(initiator->loadRemoteConnInfo("batch-target", conn_info), NIXL_SUCCESS);
        ASSERT_EQ(target->getPublicData(target_md, remote_desc.metaInfo), NIXL_SUCCESS);
        ASSERT_EQ(initiator->loadRemoteMD(remote_desc, DRAM_SEG, "batch-target", remote_md),
                  NIXL_SUCCESS);
    }

    void
    TearDown() override {
        if (handle) {
            EXPECT_EQ(initiator->releaseReqH(handle), NIXL_SUCCESS);
        }
        if (remote_md) {
            EXPECT_EQ(initiator->unloadMD(remote_md), NIXL_SUCCESS);
        }
        if (local_md) {
            EXPECT_EQ(initiator->deregisterMem(local_md), NIXL_SUCCESS);
        }
        if (target_md) {
            EXPECT_EQ(target->deregisterMem(target_md), NIXL_SUCCESS);
        }
    }
};

TEST_P(MooncakeBatchTest, DescriptorBoundariesAndCompletedHandleReuse) {
    const auto [operation, notify, split] = GetParam();
    auto &source = operation == NIXL_READ ? remote_buffer : local_buffer;
    auto &destination = operation == NIXL_READ ? local_buffer : remote_buffer;
    size_t round = 0;
    // Warm the peer, then cross the old capacity limit. The same backend handle
    // is reused after completion, including when the next request grows.
    for (const size_t count : {1, 1023, 1024, 1025, 1, 2048, 1025}) {
        SCOPED_TRACE(testing::Message() << "count=" << count << " round=" << round);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<uint8_t>((i + 37 * round) % 251);
        }
        std::fill(destination.begin(), destination.end(), 0xff);
        nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
        for (size_t i = 0; i < count; ++i) {
            // Gaps ensure the test exercises this many independent descriptors.
            nixlMetaDesc desc;
            desc.addr = reinterpret_cast<uintptr_t>(local_buffer.data()) + i * stride;
            desc.len = length;
            desc.devId = 0;
            desc.metadataP = local_md;
            local.addDesc(desc);
            desc.addr = reinterpret_cast<uintptr_t>(remote_buffer.data()) + i * stride;
            desc.metadataP = remote_md;
            remote.addDesc(desc);
        }
        nixl_opt_b_args_t opts{};
        opts.hasNotif = notify;
        opts.notifMsg = "batch-round-" + std::to_string(round++);
        const auto *opt_args = notify ? &opts : nullptr;
        if (!handle) {
            ASSERT_EQ(
                initiator->prepXfer(operation, local, remote, "batch-target", handle, opt_args),
                NIXL_SUCCESS);
        }
        auto status =
            initiator->postXfer(operation, local, remote, "batch-target", handle, opt_args);
        ASSERT_TRUE(status == NIXL_IN_PROG || status == NIXL_SUCCESS) << status;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
            status = initiator->checkXfer(handle);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_EQ(status, NIXL_SUCCESS);
        EXPECT_EQ(initiator->checkXfer(handle), NIXL_SUCCESS);
        for (size_t i = 0; i < destination.size(); ++i) {
            const auto expected = i / stride < count && i % stride < length ? source[i] : 0xff;
            ASSERT_EQ(destination[i], expected) << "byte=" << i;
        }
        if (notify) {
            notif_list_t notifications;
            while (notifications.empty() && std::chrono::steady_clock::now() < deadline) {
                ASSERT_EQ(target->getNotifs(notifications), NIXL_SUCCESS);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ASSERT_EQ(notifications.size(), 1U);
            EXPECT_EQ(notifications.front().first, "batch-initiator");
            EXPECT_EQ(notifications.front().second, opts.notifMsg);
        }
    }
    // Every transfer notified exactly once, a split one after all of its
    // batches completed. Repeated checks must not notify again.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    notif_list_t late;
    ASSERT_EQ(target->getNotifs(late), NIXL_SUCCESS);
    EXPECT_TRUE(late.empty()) << "split=" << split;
}

INSTANTIATE_TEST_SUITE_P(ReadWrite,
                         MooncakeBatchTest,
                         testing::Combine(testing::Values(NIXL_READ, NIXL_WRITE),
                                          testing::Bool(),
                                          testing::Values(false)));

// Splitting does not depend on the operation. Over TCP, Transfer Engines older
// than v0.3.12 (kvcache-ai/Mooncake#2850) report a WRITE complete before the
// target stored it, which concurrent batches expose.
INSTANTIATE_TEST_SUITE_P(Split,
                         MooncakeBatchTest,
                         testing::Combine(testing::Values(NIXL_READ),
                                          testing::Bool(),
                                          testing::Values(true)));
} // namespace
