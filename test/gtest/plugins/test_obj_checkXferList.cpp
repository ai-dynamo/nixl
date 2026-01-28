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
#include <cstdlib>
#include <vector>
#include <thread>
#include <chrono>

#include "plugins_common.h"
#include "transfer_handler.h"
#include "obj/obj_backend.h"
#include "backend/backend_aux.h"

namespace gtest::plugins::obj::checkxferlist {

/**
 * @note To run OBJ plugin tests, the following environment variables must be set:
 *       - AWS_ACCESS_KEY_ID
 *       - AWS_SECRET_ACCESS_KEY
 *       - AWS_DEFAULT_REGION
 *       - AWS_DEFAULT_BUCKET
 *       - AWS_ENDPOINT_OVERRIDE (optional, for LocalStack testing)
 *
 * These variables are required for authenticating and interacting with the S3 bucket
 * used during the tests.
 */

// Get endpoint override from environment variable for LocalStack testing
static std::string getEndpointOverride() {
    const char* endpoint = std::getenv("AWS_ENDPOINT_OVERRIDE");
    return endpoint ? endpoint : "";
}

static nixl_b_params_t obj_params = {
    {"crtMinLimit", "0"},
    {"use_virtual_addressing", "false"},
    {"scheme", "http"},
    {"endpoint_override", getEndpointOverride()}
};
static const std::string local_agent_name = "Agent1";
static const nixlBackendInitParams obj_test_params = {.localAgent = local_agent_name,
                                               .type = "OBJ",
                                               .customParams = &obj_params,
                                               .enableProgTh = false,
                                               .pthrDelay = 0,
                                               .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

// Helper to check if status indicates transfer is still in progress
static bool isInProgress(nixl_status_t status) {
    return status == NIXL_IN_PROG || status == NIXL_ERR_IN_PROG;
}

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

    nixl_opt_b_args_t opt_args;
    opt_args.trackFlags = NIXL_XFER_TRACK_ERRORS | NIXL_XFER_TRACK_SUCCESSES;

    // Prepare and perform write transfer
    nixlBackendReqH *handle = nullptr;
    auto prep_status = localBackendEngine_->prepXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);
    ASSERT_EQ(prep_status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);

    ASSERT_EQ(status, NIXL_IN_PROG);

    // Poll until complete using append-only events
    nixl_xfer_entry_events_t events;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferEvents(handle, events);
        if (isInProgress(overall_status)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (isInProgress(overall_status) && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);
    ASSERT_EQ(events.size(), static_cast<size_t>(num_buffers));

    // Verify all entries succeeded
    for (size_t i = 0; i < events.size(); ++i) {
        EXPECT_EQ(events[i].status, NIXL_SUCCESS) << "Entry " << events[i].index << " failed";
    }

    localBackendEngine_->releaseReqH(handle);
}

// Test that shared_future pattern allows multiple calls
TEST_P(ObjCheckXferListTest, SharedFutureMultipleCalls) {
    const int num_buffers = 2;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    nixl_opt_b_args_t opt_args;
    opt_args.trackFlags = NIXL_XFER_TRACK_ERRORS | NIXL_XFER_TRACK_SUCCESSES;

    nixlBackendReqH *handle = nullptr;
    auto prep_status = localBackendEngine_->prepXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);
    ASSERT_EQ(prep_status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);

    ASSERT_EQ(status, NIXL_IN_PROG);
    ASSERT_NE(handle, nullptr);

    nixl_xfer_entry_events_t events1;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferEvents(handle, events1);
        if (isInProgress(overall_status)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (isInProgress(overall_status) && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);

    // Call checkXferEvents again - should work with shared_future (no new events)
    nixl_xfer_entry_events_t events2;
    auto overall_status2 = localBackendEngine_->checkXferEvents(handle, events2);
    ASSERT_EQ(overall_status2, NIXL_SUCCESS);
    ASSERT_EQ(events1.size(), events2.size());

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

    nixl_opt_b_args_t opt_args;
    opt_args.trackFlags = NIXL_XFER_TRACK_ERRORS | NIXL_XFER_TRACK_SUCCESSES;

    nixlBackendReqH *handle = nullptr;
    auto prep_status = localBackendEngine_->prepXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);
    ASSERT_EQ(prep_status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);

    ASSERT_EQ(status, NIXL_IN_PROG);

    nixl_xfer_entry_events_t events;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferEvents(handle, events);
        if (isInProgress(overall_status)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (isInProgress(overall_status) && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);
    ASSERT_EQ(events.size(), static_cast<size_t>(num_buffers));

    // Verify results are consistent on subsequent calls (no new events)
    for (int call = 0; call < 5; ++call) {
        nixl_xfer_entry_events_t cached_events;
        auto cached_overall = localBackendEngine_->checkXferEvents(handle, cached_events);

        EXPECT_EQ(cached_overall, NIXL_SUCCESS);
        EXPECT_EQ(cached_events.size(), events.size());
    }

    localBackendEngine_->releaseReqH(handle);
}

// Test per-entry status collection via append-only events
TEST_P(ObjCheckXferListTest, PerEntryStatus) {
    const int num_buffers = 4;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    nixl_opt_b_args_t opt_args;
    opt_args.trackFlags = NIXL_XFER_TRACK_ERRORS | NIXL_XFER_TRACK_SUCCESSES;

    nixlBackendReqH *handle = nullptr;
    auto prep_status = localBackendEngine_->prepXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);
    ASSERT_EQ(prep_status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);

    ASSERT_EQ(status, NIXL_IN_PROG);

    nixl_xfer_entry_events_t events;
    nixl_status_t overall_status;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferEvents(handle, events);
        if (isInProgress(overall_status)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (isInProgress(overall_status) && poll_count < max_polls);

    ASSERT_EQ(overall_status, NIXL_SUCCESS);
    ASSERT_EQ(events.size(), static_cast<size_t>(num_buffers));

    for (size_t i = 0; i < events.size(); ++i) {
        EXPECT_EQ(events[i].status, NIXL_SUCCESS) << "Entry " << events[i].index << " failed";
    }

    localBackendEngine_->releaseReqH(handle);
}

// Test compatibility with existing checkXfer API
TEST_P(ObjCheckXferListTest, CheckXferCompatibility) {
    const int num_buffers = 2;
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, num_buffers);

    transfer.setLocalMem();

    nixl_opt_b_args_t opt_args;
    opt_args.trackFlags = NIXL_XFER_TRACK_ERRORS | NIXL_XFER_TRACK_SUCCESSES;

    nixlBackendReqH *handle = nullptr;
    auto prep_status = localBackendEngine_->prepXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);
    ASSERT_EQ(prep_status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto status = localBackendEngine_->postXfer(
        NIXL_WRITE,
        transfer.getLocalMeta(),
        transfer.getRemoteMeta(),
        local_agent_name,
        handle,
        &opt_args);

    ASSERT_EQ(status, NIXL_IN_PROG);

    nixl_xfer_entry_events_t events;
    nixl_status_t overall_status_events, overall_status_single;
    int max_polls = 100;
    int poll_count = 0;

    do {
        overall_status_events = localBackendEngine_->checkXferEvents(handle, events);
        overall_status_single = localBackendEngine_->checkXfer(handle);

        EXPECT_EQ(overall_status_events, overall_status_single);

        if (isInProgress(overall_status_events)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (isInProgress(overall_status_events) && poll_count < max_polls);

    ASSERT_EQ(overall_status_events, NIXL_SUCCESS);
    ASSERT_EQ(overall_status_single, NIXL_SUCCESS);

    localBackendEngine_->releaseReqH(handle);
}

// Test partial failures - some entries succeed, some fail
// This test writes to some S3 objects, then attempts to read from both
// existing and non-existing objects to verify partial failure handling
TEST_P(ObjCheckXferListTest, PartialFailureOnRead) {
    const int num_buffers = 4;
    const size_t buffer_size = 64;

    // Create local memory buffers
    std::vector<std::vector<char>> local_buffers(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
        local_buffers[i].resize(buffer_size);
        std::fill(local_buffers[i].begin(), local_buffers[i].end(), static_cast<char>(0x10 + i));
    }

    // Register object keys - use unique keys for this test
    std::vector<nixlBackendMD*> obj_metadata(num_buffers);
    std::vector<std::string> obj_keys(num_buffers);

    for (int i = 0; i < num_buffers; ++i) {
        obj_keys[i] = "partial_failure_test_obj_" + std::to_string(i) + "_" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        nixlBlobDesc desc;
        desc.addr = 0;  // offset
        desc.len = buffer_size;
        desc.devId = i;

        auto status = localBackendEngine_->registerMem(desc, OBJ_SEG, obj_metadata[i]);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to register object key " << i;
    }

    // Create source (DRAM) metadata for local buffers
    nixl_meta_dlist_t src_descs(DRAM_SEG);
    for (int i = 0; i < num_buffers; ++i) {
        nixlMetaDesc desc;
        desc.addr = reinterpret_cast<uintptr_t>(local_buffers[i].data());
        desc.len = buffer_size;
        desc.devId = i;
        desc.metadataP = nullptr;
        src_descs.addDesc(desc);
    }

    // Create destination (OBJ) metadata
    nixl_meta_dlist_t dst_descs(OBJ_SEG);
    for (int i = 0; i < num_buffers; ++i) {
        nixlMetaDesc desc;
        desc.addr = 0;  // offset in object
        desc.len = buffer_size;
        desc.devId = i;
        desc.metadataP = obj_metadata[i];
        dst_descs.addDesc(desc);
    }

    // Write to ONLY first 2 objects (indices 0 and 1)
    // Objects 2 and 3 will NOT be written to
    nixl_meta_dlist_t partial_src_descs(DRAM_SEG);
    nixl_meta_dlist_t partial_dst_descs(OBJ_SEG);

    for (int i = 0; i < 2; ++i) {  // Only first 2
        nixlMetaDesc src_desc;
        src_desc.addr = reinterpret_cast<uintptr_t>(local_buffers[i].data());
        src_desc.len = buffer_size;
        src_desc.devId = i;
        src_desc.metadataP = nullptr;
        partial_src_descs.addDesc(src_desc);

        nixlMetaDesc dst_desc;
        dst_desc.addr = 0;
        dst_desc.len = buffer_size;
        dst_desc.devId = i;
        dst_desc.metadataP = obj_metadata[i];
        partial_dst_descs.addDesc(dst_desc);
    }

    // Prepare and perform write to first 2 objects
    nixlBackendReqH *write_handle = nullptr;
    auto write_prep_status = localBackendEngine_->prepXfer(
        NIXL_WRITE, partial_src_descs, partial_dst_descs, local_agent_name, write_handle);
    ASSERT_EQ(write_prep_status, NIXL_SUCCESS);
    ASSERT_NE(write_handle, nullptr);

    auto write_status = localBackendEngine_->postXfer(
        NIXL_WRITE, partial_src_descs, partial_dst_descs, local_agent_name, write_handle);
    ASSERT_EQ(write_status, NIXL_IN_PROG);

    // Wait for write to complete
    int max_polls = 100;
    int poll_count = 0;
    nixl_status_t status;
    do {
        status = localBackendEngine_->checkXfer(write_handle);
        if (isInProgress(status)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (isInProgress(status) && poll_count < max_polls);
    ASSERT_EQ(status, NIXL_SUCCESS) << "Write to first 2 objects failed";
    localBackendEngine_->releaseReqH(write_handle);

    // Clear local buffers for read verification
    for (auto &buf : local_buffers) {
        std::fill(buf.begin(), buf.end(), 0);
    }

    // Now try to READ from ALL 4 objects
    // Objects 0 and 1 exist (should succeed)
    // Objects 2 and 3 do NOT exist (should fail)
    nixl_opt_b_args_t read_opt_args;
    read_opt_args.trackFlags = NIXL_XFER_TRACK_ERRORS | NIXL_XFER_TRACK_SUCCESSES;

    nixlBackendReqH *read_handle = nullptr;
    auto read_prep_status = localBackendEngine_->prepXfer(
        NIXL_READ, src_descs, dst_descs, local_agent_name, read_handle, &read_opt_args);
    ASSERT_EQ(read_prep_status, NIXL_SUCCESS);
    ASSERT_NE(read_handle, nullptr);

    auto read_status = localBackendEngine_->postXfer(
        NIXL_READ, src_descs, dst_descs, local_agent_name, read_handle, &read_opt_args);
    ASSERT_EQ(read_status, NIXL_IN_PROG);

    nixl_xfer_entry_events_t events;
    nixl_status_t overall_status;
    poll_count = 0;

    do {
        overall_status = localBackendEngine_->checkXferEvents(read_handle, events);
        if (isInProgress(overall_status)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }
    } while (isInProgress(overall_status) && poll_count < max_polls);

    EXPECT_LT(overall_status, 0) << "Expected overall failure status";

    ASSERT_EQ(events.size(), static_cast<size_t>(num_buffers));

    // Build index->status map from events (order may vary)
    std::vector<nixl_status_t> entry_status(num_buffers, NIXL_IN_PROG);
    for (const auto &e : events) {
        if (e.index < num_buffers)
            entry_status[e.index] = e.status;
    }

    EXPECT_EQ(entry_status[0], NIXL_SUCCESS) << "Entry 0 should have succeeded";
    EXPECT_EQ(entry_status[1], NIXL_SUCCESS) << "Entry 1 should have succeeded";
    EXPECT_LT(entry_status[2], 0) << "Entry 2 should have failed (object doesn't exist)";
    EXPECT_LT(entry_status[3], 0) << "Entry 3 should have failed (object doesn't exist)";

    for (int i = 0; i < num_buffers; ++i) {
        NIXL_INFO << "Entry " << i << " status: " << entry_status[i];
    }

    localBackendEngine_->releaseReqH(read_handle);

    // Cleanup - deregister all object keys
    for (int i = 0; i < num_buffers; ++i) {
        localBackendEngine_->deregisterMem(obj_metadata[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(ObjCheckXferListTests,
                         ObjCheckXferListTest,
                         testing::Values(obj_test_params));

} // namespace gtest::plugins::obj::checkxferlist
