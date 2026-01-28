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

#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <chrono>

#include "plugins_common.h"
#include "transfer_handler.h"
#include "obj/obj_backend.h"

namespace gtest::plugins::obj {

/**
 * @note To run OBJ plugin tests, the following environment variables must be set:
 *       - AWS_ACCESS_KEY_ID
 *       - AWS_SECRET_ACCESS_KEY
 *       - AWS_DEFAULT_REGION
 *       - AWS_DEFAULT_BUCKET
 *
 * These variables are required for authenticating and interacting with the S3 bucket
 * used during the tests.
 */

nixl_b_params_t obj_params = {{"crtMinLimit", "0"}};
const std::string local_agent_name = "Agent1";
const nixlBackendInitParams obj_test_params = {.localAgent = local_agent_name,
                                               .type = "OBJ",
                                               .customParams = &obj_params,
                                               .enableProgTh = false,
                                               .pthrDelay = 0,
                                               .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

class ObjCheckXferListTest : public setupBackendTestFixture {
protected:
    ObjCheckXferListTest() {
        localBackendEngine_ = std::make_shared<nixlObjEngine>(&GetParam());
    }
};

// Test batch transfer with all entries succeeding
TEST_P(ObjCheckXferListTest, BatchTransferAllSuccess) {
    const int num_buffers = 3;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    // Perform write transfer
    nixlBackendReqH *handle = nullptr;
    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle);

    ASSERT_EQ(status, NIXL_IN_PROG);
    ASSERT_NE(handle, nullptr);

    // Poll until complete
    std::vector<nixl_status_t> entry_status;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferList(handle, entry_status);
        if (overall_status == NIXL_IN_PROG) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (overall_status == NIXL_IN_PROG && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);
    ASSERT_EQ(entry_status.size(), num_buffers);

    // Verify all entries succeeded
    for (int i = 0; i < num_buffers; ++i) {
        EXPECT_EQ(entry_status[i], NIXL_SUCCESS) << "Entry " << i << " failed";
    }

    localBackendEngine_->releaseReqH(handle);
}

// Test that shared_future pattern allows multiple calls
TEST_P(ObjCheckXferListTest, SharedFutureMultipleCalls) {
    const int num_buffers = 2;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    // Perform write transfer
    nixlBackendReqH *handle = nullptr;
    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle);

    ASSERT_EQ(status, NIXL_IN_PROG);
    ASSERT_NE(handle, nullptr);

    // Poll until complete
    std::vector<nixl_status_t> entry_status1, entry_status2;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferList(handle, entry_status1);
        if (overall_status == NIXL_IN_PROG) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (overall_status == NIXL_IN_PROG && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);

    // Call checkXferList again - should work with shared_future
    auto overall_status2 = localBackendEngine_->checkXferList(handle, entry_status2);
    ASSERT_EQ(overall_status2, NIXL_SUCCESS);

    // Verify both calls return the same results
    ASSERT_EQ(entry_status1.size(), entry_status2.size());
    for (size_t i = 0; i < entry_status1.size(); ++i) {
        EXPECT_EQ(entry_status1[i], entry_status2[i]);
    }

    // Also verify checkXfer still works
    auto check_status = localBackendEngine_->checkXfer(handle);
    EXPECT_EQ(check_status, NIXL_SUCCESS);

    localBackendEngine_->releaseReqH(handle);
}

// Test result caching works correctly
TEST_P(ObjCheckXferListTest, ResultCaching) {
    const int num_buffers = 3;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    // Perform write transfer
    nixlBackendReqH *handle = nullptr;
    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle);

    ASSERT_EQ(status, NIXL_IN_PROG);
    ASSERT_NE(handle, nullptr);

    // Poll while in progress
    std::vector<nixl_status_t> entry_status_progress;
    auto status_progress = localBackendEngine_->checkXferList(handle, entry_status_progress);

    // Should be in progress or success
    EXPECT_TRUE(status_progress == NIXL_IN_PROG || status_progress == NIXL_SUCCESS);

    // Wait for completion
    std::vector<nixl_status_t> entry_status_final;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferList(handle, entry_status_final);
        if (overall_status == NIXL_IN_PROG) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (overall_status == NIXL_IN_PROG && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);
    ASSERT_EQ(entry_status_final.size(), num_buffers);

    // Verify cached results are consistent on subsequent calls
    for (int call = 0; call < 5; ++call) {
        std::vector<nixl_status_t> cached_status;
        auto cached_overall = localBackendEngine_->checkXferList(handle, cached_status);

        EXPECT_EQ(cached_overall, NIXL_SUCCESS);
        EXPECT_EQ(cached_status, entry_status_final);
    }

    localBackendEngine_->releaseReqH(handle);
}

// Test per-entry status collection
TEST_P(ObjCheckXferListTest, PerEntryStatus) {
    const int num_buffers = 4;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    // Perform write transfer
    nixlBackendReqH *handle = nullptr;
    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle);

    ASSERT_EQ(status, NIXL_IN_PROG);
    ASSERT_NE(handle, nullptr);

    // Track status changes for each entry
    std::vector<std::vector<nixl_status_t>> status_history;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        std::vector<nixl_status_t> entry_status;
        overall_status = localBackendEngine_->checkXferList(handle, entry_status);
        status_history.push_back(entry_status);

        if (overall_status == NIXL_IN_PROG) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (overall_status == NIXL_IN_PROG && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);

    // Verify final status
    const auto &final_status = status_history.back();
    ASSERT_EQ(final_status.size(), num_buffers);

    for (int i = 0; i < num_buffers; ++i) {
        EXPECT_EQ(final_status[i], NIXL_SUCCESS) << "Entry " << i << " failed";
    }

    // Verify status progresses from NIXL_IN_PROG to NIXL_SUCCESS (never backwards)
    for (size_t i = 0; i < status_history.size(); ++i) {
        for (int j = 0; j < num_buffers; ++j) {
            if (status_history[i][j] == NIXL_SUCCESS) {
                // Once success, should remain success
                for (size_t k = i + 1; k < status_history.size(); ++k) {
                    EXPECT_EQ(status_history[k][j], NIXL_SUCCESS)
                        << "Entry " << j << " regressed at poll " << k;
                }
            }
        }
    }

    localBackendEngine_->releaseReqH(handle);
}

// Test compatibility with existing checkXfer API
TEST_P(ObjCheckXferListTest, CheckXferCompatibility) {
    const int num_buffers = 2;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    // Perform write transfer
    nixlBackendReqH *handle = nullptr;
    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle);

    ASSERT_EQ(status, NIXL_IN_PROG);
    ASSERT_NE(handle, nullptr);

    // Use both APIs interleaved
    std::vector<nixl_status_t> entry_status;
    nixl_status_t overall_status_list, overall_status_single;
    int max_polls = 100;
    int poll_count = 0;

    do {
        // Call checkXferList
        overall_status_list = localBackendEngine_->checkXferList(handle, entry_status);

        // Call checkXfer
        overall_status_single = localBackendEngine_->checkXfer(handle);

        // Both should return the same overall status
        EXPECT_EQ(overall_status_list, overall_status_single);

        if (overall_status_list == NIXL_IN_PROG) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (overall_status_list == NIXL_IN_PROG && poll_count < max_polls);

    ASSERT_EQ(overall_status_list, NIXL_SUCCESS);
    ASSERT_EQ(overall_status_single, NIXL_SUCCESS);

    localBackendEngine_->releaseReqH(handle);
}

INSTANTIATE_TEST_SUITE_P(ObjCheckXferListTests,
                         ObjCheckXferListTest,
                         testing::Values(obj_test_params));

} // namespace gtest::plugins::obj
