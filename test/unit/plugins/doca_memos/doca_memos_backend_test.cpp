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
#include <memory>
#include <vector>
#include <cstring>

// Include mocks before the backend
#include "doca_memos_mocks.h"

// Now we can include the backend header (which will use our mocked DOCA types)
#include "doca_memos_backend.h"
#include "nixl_types.h"

class DocMemosBackendTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        // Reset mock state before each test
        DocaMockControl::instance().reset();

        // Setup default init params
        setupInitParams();
    }

    void
    TearDown() override {
        // Clean up engine if created
        if (engine_) {
            engine_.reset();
        }
    }

    void
    setupInitParams(bool enable_progress_thread = false,
                    nixl_thread_sync_t sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE,
                    const std::string &device_name = "/dev/nvme0n1",
                    const std::string &num_tasks = "8192") {
        // Setup backend parameters
        backend_params_ = std::make_unique<nixl_b_params_t>();
        (*backend_params_)["device_name"] = device_name;
        (*backend_params_)["num_tasks"] = num_tasks;

        // Setup init params
        init_params_ = std::make_unique<nixlBackendInitParams>();
        init_params_->customParams = backend_params_.get();
        init_params_->enableProgTh = enable_progress_thread;
        init_params_->syncMode = sync_mode;
        init_params_->pthrDelay = 1000; // 1ms
    }

    void
    createEngine() {
        engine_ = std::make_unique<nixlDocaMemosEngine>(init_params_.get());
    }

    // Helper method to register memory and return metadata
    // Returns the metadata object that should be attached to the descriptor
    nixlBackendMD *
    registerMemory(uint64_t dev_id, const std::string &key) {
        if (!engine_) {
            return nullptr;
        }

        nixlBlobDesc blob;
        blob.devId = dev_id;
        blob.metaInfo = key;
        blob.addr = 0; // Not used for KV
        blob.len = 0; // Not used for KV

        nixlBackendMD *out = nullptr;
        engine_->registerMem(blob, OBJ_SEG, out);
        return out;
    }

    std::unique_ptr<nixlDocaMemosEngine> engine_;
    std::unique_ptr<nixlBackendInitParams> init_params_;
    std::unique_ptr<nixl_b_params_t> backend_params_;
};

// ============================================================================
// Constructor/Destructor Tests
// ============================================================================

TEST_F(DocMemosBackendTest, ConstructorSuccess) {
    createEngine();
    EXPECT_TRUE(engine_ != nullptr);
    EXPECT_FALSE(engine_->supportsRemote());
    EXPECT_TRUE(engine_->supportsLocal());
}

TEST_F(DocMemosBackendTest, ConstructorMissingDeviceName) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE, ""); // Empty device name
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, ConstructorDocaKvdevCreateFailure) {
    DocaMockControl::instance().nvme_kvdev_create_result = DOCA_ERROR_INVALID_VALUE;
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, ConstructorDocaPECreateFailure) {
    DocaMockControl::instance().pe_create_result = DOCA_ERROR_NO_MEMORY;
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, ConstructorWithProgressThread) {
    setupInitParams(true); // Enable progress thread
    createEngine();
    EXPECT_TRUE(engine_ != nullptr);
}

// ============================================================================
// Memory Registration Tests
// ============================================================================

TEST_F(DocMemosBackendTest, RegisterMemObjSeg) {
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 100;
    desc.metaInfo = "test_key_123";

    nixl_mem_t nixl_mem = OBJ_SEG;
    nixlBackendMD *out = nullptr;

    nixl_status_t status = engine_->registerMem(desc, nixl_mem, out);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(out, nullptr);

    // Cleanup
    engine_->deregisterMem(out);
}

TEST_F(DocMemosBackendTest, RegisterMemObjSegWithoutMetaInfo) {
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 42;
    desc.metaInfo = ""; // Empty, should use devId

    nixl_mem_t nixl_mem = OBJ_SEG;
    nixlBackendMD *out = nullptr;

    nixl_status_t status = engine_->registerMem(desc, nixl_mem, out);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(out, nullptr);

    // Cleanup
    engine_->deregisterMem(out);
}

TEST_F(DocMemosBackendTest, RegisterMemUnsupportedType) {
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 1;

    nixl_mem_t unsupported_mem = VRAM_SEG; // Not supported by DOCA KV
    nixlBackendMD *out = nullptr;

    nixl_status_t status = engine_->registerMem(desc, unsupported_mem, out);
    EXPECT_NE(status, NIXL_SUCCESS);
}

// ============================================================================
// Transfer Preparation Tests
// ============================================================================

// A handle can be re-posted after its prior submission completes; the second
// post must succeed and produce a fresh round of submissions.
TEST_F(DocMemosBackendTest, PostXferHandleReuse) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 9101;
    remote_desc.addr = 0;
    remote_desc.len = 1024;
    remote_desc.metadataP = registerMemory(remote_desc.devId, "reuse_key");
    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);

    for (int round = 0; round < 2; round++) {
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle),
                  NIXL_IN_PROG)
            << "round " << round;
        nixl_status_t status = NIXL_IN_PROG;
        for (int i = 0; i < 100 && status == NIXL_IN_PROG; i++) {
            status = engine_->checkXfer(handle);
        }
        EXPECT_EQ(status, NIXL_SUCCESS) << "round " << round;
    }

    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// ============================================================================
// Query Memory Tests
// ============================================================================

TEST_F(DocMemosBackendTest, QueryMemAssumeSuccessDefault) {
    // Default mode: queryMem should return success for all descriptors
    // without actually querying the device
    createEngine();

    // Register some OBJ_SEG memory
    nixlBlobDesc desc1;
    desc1.devId = 500;
    desc1.metaInfo = "key_1";
    nixlBackendMD *md1 = nullptr;
    ASSERT_EQ(engine_->registerMem(desc1, OBJ_SEG, md1), NIXL_SUCCESS);

    nixlBlobDesc desc2;
    desc2.devId = 501;
    desc2.metaInfo = "key_2";
    nixlBackendMD *md2 = nullptr;
    ASSERT_EQ(engine_->registerMem(desc2, OBJ_SEG, md2), NIXL_SUCCESS);

    // Build a registered descriptor list
    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc reg_desc1;
    reg_desc1.devId = 500;
    reg_desc1.metaInfo = "key_1";
    reg_dlist.addDesc(reg_desc1);

    nixlBlobDesc reg_desc2;
    reg_desc2.devId = 501;
    reg_desc2.metaInfo = "key_2";
    reg_dlist.addDesc(reg_desc2);

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = engine_->queryMem(reg_dlist, resp);

    EXPECT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 2);
    EXPECT_TRUE(resp[0].has_value())
        << "All descriptors should report success in assume_success mode";
    EXPECT_TRUE(resp[1].has_value())
        << "All descriptors should report success in assume_success mode";

    // No DOCA tasks should have been submitted
    EXPECT_EQ(DocaMockControl::instance().submitted_tasks.size(), 0)
        << "No DOCA tasks should be submitted in assume_success mode";

    engine_->deregisterMem(md1);
    engine_->deregisterMem(md2);
}

TEST_F(DocMemosBackendTest, QueryMemAssumeSuccessExplicit) {
    // Explicitly set assume_success mode
    (*backend_params_)["query_mem_mode"] = "assume_success";
    createEngine();

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc reg_desc;
    reg_desc.devId = 600;
    reg_desc.metaInfo = "some_key";
    reg_dlist.addDesc(reg_desc);

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = engine_->queryMem(reg_dlist, resp);

    EXPECT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1);
    EXPECT_TRUE(resp[0].has_value());
}

TEST_F(DocMemosBackendTest, QueryMemInvalidMode) {
    (*backend_params_)["query_mem_mode"] = "bogus_value";
    EXPECT_THROW(createEngine(), std::runtime_error);
}

// queryMem in "actual" mode submits an EXIST task per descriptor and reports
// presence based on the DOCA error callback.
TEST_F(DocMemosBackendTest, QueryMemActualKeyExists) {
    setupInitParams(false);
    (*backend_params_)["query_mem_mode"] = "actual";
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // The mock keys EXIST results off this map; the key string here must
    // match what resolveMemosKey produces. metaInfo "exists" is not valid
    // hex (odd length) and <= 16 bytes, so the plugin uses it raw.
    mock.kv_store["exists"] = std::vector<uint8_t>(8, 0x42);

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc desc;
    desc.devId = 700;
    desc.metaInfo = "exists";
    reg_dlist.addDesc(desc);

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_TRUE(resp[0].has_value());
}

TEST_F(DocMemosBackendTest, QueryMemActualKeyDoesNotExist) {
    setupInitParams(false);
    (*backend_params_)["query_mem_mode"] = "actual";
    createEngine();
    DocaMockControl::instance().auto_complete_tasks = true;

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    nixlBlobDesc desc;
    desc.devId = 701;
    desc.metaInfo = "missing";
    reg_dlist.addDesc(desc);

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_FALSE(resp[0].has_value());
}

// Mix of present and absent keys in a single batch.
TEST_F(DocMemosBackendTest, QueryMemActualMixed) {
    setupInitParams(false);
    (*backend_params_)["query_mem_mode"] = "actual";
    createEngine();
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.kv_store["alpha"] = std::vector<uint8_t>(4, 0x11);
    mock.kv_store["gamma"] = std::vector<uint8_t>(4, 0x33);

    nixl_reg_dlist_t reg_dlist(OBJ_SEG);
    for (const char *name : {"alpha", "beta", "gamma", "delta"}) {
        nixlBlobDesc desc;
        desc.devId = 800;
        desc.metaInfo = name;
        reg_dlist.addDesc(desc);
    }

    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine_->queryMem(reg_dlist, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 4u);
    EXPECT_TRUE(resp[0].has_value()); // alpha
    EXPECT_FALSE(resp[1].has_value()); // beta
    EXPECT_TRUE(resp[2].has_value()); // gamma
    EXPECT_FALSE(resp[3].has_value()); // delta
}

// ============================================================================
// Threading Mode Tests
// ============================================================================

TEST_F(DocMemosBackendTest, SingleThreadedMode) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Verify no progress thread is created and no locks are required
    // This is implicit in the implementation, but we can verify behavior
}

TEST_F(DocMemosBackendTest, MultiThreadedModeWithProgressThread) {
    setupInitParams(true, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Progress thread should be created
}

TEST_F(DocMemosBackendTest, MultiThreadedModeWithSyncMode) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    createEngine();

    // Locks should be required even without progress thread
}

TEST_F(DocMemosBackendTest, ThreadedEngineQueuedRequestsComplete) {
    setupInitParams(true);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 2;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 4; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 20000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "thr_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // Submission runs on the background progress thread, so wait until it has
    // actually hit the exhaustion window (2 successful allocs followed by at
    // least one failed retry) before lifting the fault. Clearing it eagerly
    // lets the thread drain the whole batch without ever queuing, leaving the
    // resume path untested.
    bool exhaustion_observed = false;
    for (int i = 0; i < 1000; i++) {
        {
            auto lk = DocaMockControl::lock();
            if (mock.task_alloc_call_count > 2) {
                exhaustion_observed = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        auto lk = DocaMockControl::lock();
        ASSERT_TRUE(exhaustion_observed)
            << "Progress thread never reached the task-pool exhaustion window";
        mock.task_alloc_fail_after_n = -1;
        mock.task_alloc_call_count = 0;
    }

    nixl_status_t status = NIXL_IN_PROG;
    for (int i = 0; i < 200; i++) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(status, NIXL_SUCCESS)
        << "Threaded engine should drain pending queue and complete all tasks";

    for (int i = 0; i < 4; i++) {
        free(buffers[i]);
        engine_->deregisterMem(mds[i]);
    }
    engine_->releaseReqH(handle);
}

// ============================================================================
// Edge Cases
// ============================================================================

// prepXfer must reject empty descriptor lists rather than allocating a
// zero-task request handle.
TEST_F(DocMemosBackendTest, EmptyDescriptorList) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);
}

// =============================================================================
// Failure Mode Tests
// =============================================================================

// Test 1: Task Pool Exhaustion - With queueing, requests succeed after resources free up
TEST_F(DocMemosBackendTest, TaskPoolExhaustion) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Configure mock to fail after 2 successful task allocations initially
    auto &mock = DocaMockControl::instance();
    mock.task_alloc_fail_after_n = 2;

    // Create 3 descriptors, but pool can only handle 2 initially
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
        local_desc.len = 1024;
        remote_desc.devId = 1000 + i; // Unique devId for each descriptor
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "key_" + std::to_string(i);
        // Register memory and get metadata
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    // Prepare transfer - should succeed
    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // Post transfer - first 2 tasks submit, 3rd gets queued due to resource exhaustion
    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Verify partial submission: 2 tasks should be submitted, 1 queued
    EXPECT_EQ(mock.submitted_tasks.size(), 2)
        << "Should have submitted 2 tasks (partial submission)";

    // Now allow more allocations and let progress drain the queue
    mock.task_alloc_fail_after_n = -1;
    mock.task_alloc_call_count = 0;
    mock.auto_complete_tasks = true;

    // Check status - should eventually succeed after queued task is retried
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after resources become available";

    // Cleanup
    for (int i = 0; i < 3; i++) {
        free(reinterpret_cast<void *>(local_dlist[i].addr));
        engine_->deregisterMem(remote_dlist[i].metadataP);
    }
    engine_->releaseReqH(handle);
}

// Test 2: Key Not Found - RETRIEVE operation on non-existent key
TEST_F(DocMemosBackendTest, KeyNotFoundOnRetrieve) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Don't store any key in the mock KV store
    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.kv_store.clear(); // Ensure store is empty

    // Try to retrieve a non-existent key
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 2000;
    remote_desc.addr = 0;
    remote_desc.len = 1024;

    std::string key = "nonexistent_key";
    remote_desc.metadataP = registerMemory(remote_desc.devId, key);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = engine_->postXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_IN_PROG);

    // Wait for completion - should fail with NOT_FOUND
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(status, NIXL_ERR_BACKEND); // Key not found

    // Cleanup
    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// Test 2b: Key Not Found with ignore_read_not_found - should succeed
TEST_F(DocMemosBackendTest, KeyNotFoundOnRetrieveIgnored) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    (*backend_params_)["ignore_read_not_found"] = "true";
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;
    mock.kv_store.clear();

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 2001;
    remote_desc.addr = 0;
    remote_desc.len = 1024;

    std::string key = "nonexistent_key";
    remote_desc.metadataP = registerMemory(remote_desc.devId, key);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = engine_->postXfer(NIXL_READ, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_IN_PROG);

    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(status, NIXL_SUCCESS);

    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// Test 3: Device I/O Error - Simulated during STORE operation
TEST_F(DocMemosBackendTest, DeviceIOErrorOnStore) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Force task error to simulate I/O failure
    auto &mock = DocaMockControl::instance();
    mock.force_task_error = true;
    mock.forced_task_error_code = DOCA_ERROR_IO_FAILED;
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 3000;
    remote_desc.addr = 0;
    remote_desc.len = 1024;

    std::string key = "test_key";
    remote_desc.metadataP = registerMemory(remote_desc.devId, key);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_IN_PROG);

    // Wait for completion - should fail with I/O error
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    // Cleanup
    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// Test 4: Task Queue Full - With queueing, requests succeed when queue has space
TEST_F(DocMemosBackendTest, TaskQueueFull) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    // Configure mock to fail after 1 successful submission initially
    auto &mock = DocaMockControl::instance();
    mock.task_submit_fail_after_n = 1;
    mock.auto_complete_tasks = true;

    // Create 2 descriptors
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    for (int i = 0; i < 2; i++) {
        nixlMetaDesc local_desc, remote_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
        local_desc.len = 1024;
        remote_desc.devId = 4000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Post transfer - first task submits, second gets queued due to queue full
    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Now allow more submissions (simulating queue having space)
    mock.task_submit_fail_after_n = -1;
    mock.task_submit_call_count = 0;

    // Check status - should eventually succeed after queued task is retried
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after queue has space";

    // Cleanup
    for (int i = 0; i < 2; i++) {
        free(reinterpret_cast<void *>(local_dlist[i].addr));
        engine_->deregisterMem(remote_dlist[i].metadataP);
    }
    engine_->releaseReqH(handle);
}

TEST_F(DocMemosBackendTest, DeviceNotFound) {
    DocaMockControl::instance().nvme_kvdev_create_result = DOCA_ERROR_NOT_FOUND;
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    EXPECT_THROW(createEngine(), std::runtime_error);
}

TEST_F(DocMemosBackendTest, GetMaxValueLenFailure) {
    DocaMockControl::instance().get_max_value_len_result = DOCA_ERROR_INVALID_VALUE;
    EXPECT_THROW(createEngine(), std::runtime_error);
}

// Value larger than the device's max value length is rejected host-side before
// any DOCA submission, not with an opaque DOCA_ERROR_DRIVER from the device.
TEST_F(DocMemosBackendTest, ValueExceedsDeviceMax) {
    DocaMockControl::instance().max_value_len = 4096; // 4 KiB device limit
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    // Allocate a buffer larger than the device's 4 KiB max.
    const size_t oversized = 8192;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(oversized));
    ASSERT_NE(local_desc.addr, static_cast<uintptr_t>(0));
    local_desc.len = oversized;
    local_desc.devId = 0;

    remote_desc.addr = 0;
    remote_desc.len = oversized;
    remote_desc.devId = 1;
    remote_desc.metadataP = registerMemory(remote_desc.devId, "key0001");
    ASSERT_NE(remote_desc.metadataP, nullptr);

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    engine_->releaseReqH(handle);
    engine_->deregisterMem(remote_desc.metadataP);
    free(reinterpret_cast<void *>(local_desc.addr));
}

// Plugin caps non-hex metaInfo at DOCA_MEMOS_MAX_OBJECT_KEY_LEN (16) bytes.
TEST_F(DocMemosBackendTest, KeyTooLarge) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    nixlBlobDesc desc;
    desc.devId = 6000;
    desc.metaInfo = "this_key_is_way_too_long"; // 24 bytes, not valid hex
    nixlBackendMD *md = nullptr;

    nixl_status_t status = engine_->registerMem(desc, OBJ_SEG, md);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md, nullptr);
}

// Verify the plugin propagates DOCA_ERROR_TOO_BIG from a STORE task as a
// backend error, mirroring how a real device would reject an oversized value.
TEST_F(DocMemosBackendTest, StoreReturnsTooBig) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.force_task_error = true;
    mock.forced_task_error_code = DOCA_ERROR_TOO_BIG;
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
    local_desc.len = 1024;
    remote_desc.devId = 6100;
    remote_desc.addr = 0;
    remote_desc.len = 1024;
    remote_desc.metadataP = registerMemory(remote_desc.devId, "too_big_key");

    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    nixl_status_t status = NIXL_IN_PROG;
    for (int i = 0; i < 100 && status == NIXL_IN_PROG; i++) {
        status = engine_->checkXfer(handle);
    }
    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    free(reinterpret_cast<void *>(local_desc.addr));
    engine_->deregisterMem(remote_desc.metadataP);
    engine_->releaseReqH(handle);
}

// ============================================================================
// Request Queueing Tests - Comprehensive coverage for resource exhaustion
// ============================================================================

// Test that a request completes successfully after temporary resource exhaustion
TEST_F(DocMemosBackendTest, QueuedRequestSucceedsAfterResourcesFreed) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Configure mock to fail allocation after 1 task (simulating resource exhaustion)
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 1; // Allow 1 allocation, then fail

    // Create request with 3 tasks - first will succeed, rest will be queued
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(malloc(1024));
        local_desc.len = 1024;
        remote_desc.devId = 7000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // Request should be partially submitted (1 task) and queued (2 remaining)

    // Now allow unlimited allocations
    mock.task_alloc_fail_after_n = -1; // Stop failing (negative = disabled)
    mock.task_alloc_call_count = 0; // Reset counter

    // Call checkXfer to progress and complete the queued request
    nixl_status_t status = NIXL_IN_PROG;
    for (int i = 0; i < 50; i++) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Request should complete successfully
    EXPECT_EQ(status, NIXL_SUCCESS);

    // Cleanup
    for (int i = 0; i < 3; i++) {
        free(reinterpret_cast<void *>(local_dlist[i].addr));
        engine_->deregisterMem(remote_dlist[i].metadataP);
    }
    engine_->releaseReqH(handle);
}

// Test FIFO ordering with multiple queued requests
TEST_F(DocMemosBackendTest, MultipleQueuedRequestsFIFOOrdering) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Allow only 2 task allocations initially
    mock.task_alloc_fail_after_n = 2;

    // Create 3 requests, each with 2 tasks (total 6 tasks, but only 2 can submit)
    std::vector<nixlBackendReqH *> handles;
    std::vector<nixl_meta_dlist_t> local_dlists;
    std::vector<nixl_meta_dlist_t> remote_dlists;
    std::vector<std::vector<void *>> buffers;
    std::vector<std::vector<nixlBackendMD *>> metadatas;

    for (int req = 0; req < 3; req++) {
        local_dlists.emplace_back(DRAM_SEG);
        remote_dlists.emplace_back(OBJ_SEG);
        buffers.emplace_back();
        metadatas.emplace_back();

        for (int i = 0; i < 2; i++) {
            nixlMetaDesc local_desc, remote_desc;
            void *buf = malloc(1024);
            // Write a unique marker to verify ordering
            memset(buf, 'A' + req, 1024);
            buffers[req].push_back(buf);

            local_desc.addr = reinterpret_cast<uintptr_t>(buf);
            local_desc.len = 1024;
            remote_desc.devId = 8000 + req * 10 + i;
            remote_desc.addr = 0;
            remote_desc.len = 1024;

            std::string key = "fifo_key_" + std::to_string(req) + "_" + std::to_string(i);
            auto *kv_md = registerMemory(remote_desc.devId, key);
            metadatas[req].push_back(kv_md);
            remote_desc.metadataP = kv_md;

            local_dlists[req].addDesc(local_desc);
            remote_dlists[req].addDesc(remote_desc);
        }

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_IN_PROG);
        handles.push_back(handle);
    }

    // At this point:
    // - Request 0: fully submitted (2 tasks)
    // - Request 1: queued (0 tasks submitted due to allocation limit)
    // - Request 2: queued (0 tasks submitted due to allocation limit)

    // Now allow unlimited allocations
    mock.task_alloc_fail_after_n = -1; // Negative = disabled
    mock.task_alloc_call_count = 0; // Reset counter

    // Check status to progress and complete everything
    for (int i = 0; i < 100; i++) { // More iterations for multiple requests
        // Check if all completed
        bool all_done = true;
        for (auto *h : handles) {
            if (engine_->checkXfer(h) == NIXL_IN_PROG) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // All requests should complete successfully in FIFO order
    for (auto *h : handles) {
        EXPECT_EQ(engine_->checkXfer(h), NIXL_SUCCESS);
    }

    // Verify the KV store has all keys
    for (int req = 0; req < 3; req++) {
        for (int i = 0; i < 2; i++) {
            std::string key = "fifo_key_" + std::to_string(req) + "_" + std::to_string(i);
            EXPECT_TRUE(mock.kv_store.find(key) != mock.kv_store.end())
                << "Key " << key << " not found in KV store";
        }
    }

    // Cleanup
    for (size_t req = 0; req < handles.size(); req++) {
        for (size_t i = 0; i < buffers[req].size(); i++) {
            free(buffers[req][i]);
            engine_->deregisterMem(metadatas[req][i]);
        }
        engine_->releaseReqH(handles[req]);
    }
}

// Test that queue handles submit failures (DOCA_ERROR_FULL) correctly
TEST_F(DocMemosBackendTest, QueuedRequestsWithSubmitFailures) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Allow task allocation but fail submission after 2 tasks
    mock.task_alloc_fail_after_n = -1; // Allocations succeed (negative = disabled)
    mock.task_submit_fail_after_n = 2; // Submissions fail after 2

    // Create 2 requests, each with 3 tasks
    std::vector<nixlBackendReqH *> handles;
    std::vector<nixl_meta_dlist_t> local_dlists;
    std::vector<nixl_meta_dlist_t> remote_dlists;
    std::vector<std::vector<void *>> buffers;
    std::vector<std::vector<nixlBackendMD *>> metadatas;

    for (int req = 0; req < 2; req++) {
        local_dlists.emplace_back(DRAM_SEG);
        remote_dlists.emplace_back(OBJ_SEG);
        buffers.emplace_back();
        metadatas.emplace_back();

        for (int i = 0; i < 3; i++) {
            nixlMetaDesc local_desc, remote_desc;
            void *buf = malloc(1024);
            buffers[req].push_back(buf);

            local_desc.addr = reinterpret_cast<uintptr_t>(buf);
            local_desc.len = 1024;
            remote_desc.devId = 9000 + req * 10 + i;
            remote_desc.addr = 0;
            remote_desc.len = 1024;

            std::string key = "submit_key_" + std::to_string(req) + "_" + std::to_string(i);
            auto *kv_md = registerMemory(remote_desc.devId, key);
            metadatas[req].push_back(kv_md);
            remote_desc.metadataP = kv_md;

            local_dlists[req].addDesc(local_desc);
            remote_dlists[req].addDesc(remote_desc);
        }

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_IN_PROG);
        handles.push_back(handle);
    }

    // At this point:
    // - Request 0: 2 tasks submitted, 1 queued
    // - Request 1: 0 tasks submitted, 3 queued

    // Allow unlimited submissions
    mock.task_submit_fail_after_n = -1; // Negative = disabled
    mock.task_submit_call_count = 0; // Reset counter

    // Check status to progress and complete everything
    for (int i = 0; i < 100; i++) { // More iterations for multiple requests
        bool all_done = true;
        for (auto *h : handles) {
            if (engine_->checkXfer(h) == NIXL_IN_PROG) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // All requests should complete successfully
    for (auto *h : handles) {
        EXPECT_EQ(engine_->checkXfer(h), NIXL_SUCCESS);
    }

    // Cleanup
    for (size_t req = 0; req < handles.size(); req++) {
        for (size_t i = 0; i < buffers[req].size(); i++) {
            free(buffers[req][i]);
            engine_->deregisterMem(metadatas[req][i]);
        }
        engine_->releaseReqH(handles[req]);
    }
}

// Test queue with many concurrent requests
TEST_F(DocMemosBackendTest, ManyQueuedRequestsConcurrent) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Severely limit resources: only 3 tasks can be allocated/submitted initially
    mock.task_alloc_call_count = 0; // Reset counter
    mock.task_alloc_fail_after_n = 3; // Allow 3 allocations, then fail

    const int NUM_REQUESTS = 10;
    const int TASKS_PER_REQUEST = 2;

    std::vector<nixlBackendReqH *> handles;
    std::vector<nixl_meta_dlist_t> local_dlists;
    std::vector<nixl_meta_dlist_t> remote_dlists;
    std::vector<std::vector<void *>> buffers;
    std::vector<std::vector<nixlBackendMD *>> metadatas;

    // Submit many requests quickly
    for (int req = 0; req < NUM_REQUESTS; req++) {
        local_dlists.emplace_back(DRAM_SEG);
        remote_dlists.emplace_back(OBJ_SEG);
        buffers.emplace_back();
        metadatas.emplace_back();

        for (int i = 0; i < TASKS_PER_REQUEST; i++) {
            nixlMetaDesc local_desc, remote_desc;
            void *buf = malloc(1024);
            buffers[req].push_back(buf);

            local_desc.addr = reinterpret_cast<uintptr_t>(buf);
            local_desc.len = 1024;
            remote_desc.devId = 10000 + req * 10 + i;
            remote_desc.addr = 0;
            remote_desc.len = 1024;

            std::string key = "many_key_" + std::to_string(req) + "_" + std::to_string(i);
            auto *kv_md = registerMemory(remote_desc.devId, key);
            metadatas[req].push_back(kv_md);
            remote_desc.metadataP = kv_md;

            local_dlists[req].addDesc(local_desc);
            remote_dlists[req].addDesc(remote_desc);
        }

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlists[req], remote_dlists[req], "", handle),
                  NIXL_IN_PROG);
        handles.push_back(handle);
    }

    // Most requests should be queued

    // Now allow unlimited resources
    mock.task_alloc_fail_after_n = -1; // Negative = disabled
    mock.task_alloc_call_count = 0; // Reset counter

    // Check status to progress until all complete
    int max_iterations = 400; // More iterations for many requests
    for (int i = 0; i < max_iterations; i++) {
        bool all_done = true;
        for (auto *h : handles) {
            if (engine_->checkXfer(h) == NIXL_IN_PROG) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // All requests should complete successfully
    int success_count = 0;
    for (auto *h : handles) {
        if (engine_->checkXfer(h) == NIXL_SUCCESS) {
            success_count++;
        }
    }
    EXPECT_EQ(success_count, NUM_REQUESTS) << "Not all requests completed successfully";

    // Cleanup
    for (size_t req = 0; req < handles.size(); req++) {
        for (size_t i = 0; i < buffers[req].size(); i++) {
            free(buffers[req][i]);
            engine_->deregisterMem(metadatas[req][i]);
        }
        engine_->releaseReqH(handles[req]);
    }
}

// ============================================================================
// High Queue Depth Tests - Partial Submission and Forward Progress
// ============================================================================

// Test 1: New request can't submit any tasks - whole request gets queued
// After resources become available, the request is fully submitted
TEST_F(DocMemosBackendTest, QueuedRequestNoPartialSubmission) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Configure mock to fail ALL task allocations (simulating complete resource exhaustion)
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 0; // Fail immediately (0 = fail first call)

    // Create request with 5 tasks - none should be able to submit
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> metadatas;

    for (int i = 0; i < 5; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);

        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 11000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "no_partial_key_" + std::to_string(i);
        auto *kv_md = registerMemory(remote_desc.devId, key);
        metadatas.push_back(kv_md);
        remote_desc.metadataP = kv_md;

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);

    // Post transfer - should return IN_PROG even though no tasks submitted (request is queued)
    nixl_status_t status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG) << "Request should be queued even if no tasks submitted";

    // Verify no tasks were actually submitted (allocation failed immediately on first attempt)
    // The code stops trying after the first allocation fails with DOCA_ERROR_FULL
    EXPECT_EQ(mock.task_alloc_call_count, 1)
        << "Should have attempted 1 allocation (stops on first failure)";
    EXPECT_EQ(mock.submitted_tasks.size(), 0) << "No tasks should be submitted";

    // Now allow unlimited allocations (simulating resources becoming available)
    mock.task_alloc_fail_after_n = -1; // Disable failure
    mock.task_alloc_call_count = 0; // Reset counter

    // Call checkXfer to trigger retry of queued request
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Request should complete successfully after resources become available
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after resources become available";
    EXPECT_EQ(mock.submitted_tasks.size(), 0) << "All tasks should have completed";

    // Cleanup
    for (size_t i = 0; i < buffers.size(); i++) {
        free(buffers[i]);
        engine_->deregisterMem(metadatas[i]);
    }
    engine_->releaseReqH(handle);
}

// Test 2: Queued request, when retried, can still only be partially submitted
// This tests that we make forward progress even when retrying queued requests
TEST_F(DocMemosBackendTest, QueuedRequestPartialRetryForwardProgress) {
    setupInitParams(false, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    // Configure mock to allow 2 task allocations initially, then fail
    mock.task_alloc_call_count = 0;
    mock.task_alloc_fail_after_n = 2; // Allow 2 allocations, then fail

    // Create request with 5 tasks
    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> metadatas;

    for (int i = 0; i < 5; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);

        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 13000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;

        std::string key = "rp_key_" + std::to_string(i);
        auto *kv_md = registerMemory(remote_desc.devId, key);
        metadatas.push_back(kv_md);
        remote_desc.metadataP = kv_md;

        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);

    // Post transfer - should partially submit (2 tasks) and queue the rest (3 tasks)
    nixl_status_t status = engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Verify initial partial submission: 2 tasks submitted
    EXPECT_EQ(mock.submitted_tasks.size(), 2) << "Initial: 2 tasks should be submitted";

    // Reset counter but keep fail_after_n at 2 to simulate limited resources
    // When we retry, we should be able to submit 2 more tasks (partial progress)
    mock.task_alloc_call_count = 0; // Reset for retry attempt

    // Call checkXfer to trigger retry of queued tasks
    // This should submit 2 more tasks (partial progress), leaving 1 still queued
    status = engine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_IN_PROG) << "Should still be in progress after partial retry";

    // Verify partial retry: should have submitted 2 more tasks (total 4 now)
    // Note: The first 2 tasks may have completed, so we check that we made progress
    // by verifying that more allocations were attempted
    EXPECT_GE(mock.task_alloc_call_count, 2)
        << "Should have attempted at least 2 more allocations on retry";

    // Now allow unlimited allocations for final submission
    mock.task_alloc_fail_after_n = -1; // Disable failure
    mock.task_alloc_call_count = 0; // Reset counter

    // Call checkXfer again to complete remaining tasks
    int retries = 100;
    while (retries-- > 0) {
        status = engine_->checkXfer(handle);
        if (status != NIXL_IN_PROG) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Request should complete successfully
    EXPECT_EQ(status, NIXL_SUCCESS) << "Request should succeed after all tasks are submitted";

    // Cleanup
    for (size_t i = 0; i < buffers.size(); i++) {
        free(buffers[i]);
        engine_->deregisterMem(metadatas[i]);
    }
    engine_->releaseReqH(handle);
}

// ============================================================================
// Release While In-Flight Tests
// ============================================================================

TEST_F(DocMemosBackendTest, ReleaseInFlightRequest) {
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 30000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "cancel_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    EXPECT_EQ(mock.submitted_tasks.size(), 3u);

    // Release while 3 tasks are in flight — must not crash or leak
    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    // Now complete the orphaned tasks via progress — callbacks must safely
    // detect the cancelled request and delete it
    mock.auto_complete_tasks = true;
    for (int i = 0; i < 5; i++) {
        engine_->checkXfer(nullptr);
    }

    for (int i = 0; i < 3; i++) {
        free(buffers[i]);
        engine_->deregisterMem(mds[i]);
    }
}

TEST_F(DocMemosBackendTest, ReleasePendingRequest) {
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_call_count = 0;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 5; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 31000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "pend_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // 2 tasks submitted, 3 tasks queued in pendingRequests_
    EXPECT_EQ(mock.submitted_tasks.size(), 2u);

    // Release while request is pending — removes from queue, cancels in-flight tasks
    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    // Complete orphaned tasks
    mock.auto_complete_tasks = true;
    mock.task_alloc_fail_after_n = -1;
    for (int i = 0; i < 5; i++) {
        engine_->checkXfer(nullptr);
    }

    for (int i = 0; i < 5; i++) {
        free(buffers[i]);
        engine_->deregisterMem(mds[i]);
    }
}

TEST_F(DocMemosBackendTest, ReleaseCompletedRequest) {
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = true;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    void *buf = malloc(1024);
    local_desc.addr = reinterpret_cast<uintptr_t>(buf);
    local_desc.len = 1024;
    remote_desc.devId = 32000;
    remote_desc.addr = 0;
    remote_desc.len = 1024;
    auto *md = registerMemory(remote_desc.devId, "done_key");
    remote_desc.metadataP = md;
    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    nixl_status_t status = NIXL_IN_PROG;
    for (int i = 0; i < 100 && status == NIXL_IN_PROG; i++) {
        status = engine_->checkXfer(handle);
    }
    EXPECT_EQ(status, NIXL_SUCCESS);

    // Release an already-completed request — immediate delete path
    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    free(buf);
    engine_->deregisterMem(md);
}

TEST_F(DocMemosBackendTest, ThreadedCancelBeforeProgressPickup) {
    setupInitParams(true);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;
    // Fail every task allocation so the progress thread can never hand buf to
    // DOCA, regardless of whether it picks the request up before releaseReqH().
    // Without this the test races: a winning progress thread submits the store
    // task and the free(buf) below becomes a use-after-free.
    mock.task_alloc_fail_after_n = 0;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    nixlMetaDesc local_desc, remote_desc;
    void *buf = malloc(1024);
    local_desc.addr = reinterpret_cast<uintptr_t>(buf);
    local_desc.len = 1024;
    remote_desc.devId = 33000;
    remote_desc.addr = 0;
    remote_desc.len = 1024;
    auto *md = registerMemory(remote_desc.devId, "cancel_qd_key_0");
    remote_desc.metadataP = md;
    local_dlist.addDesc(local_desc);
    remote_dlist.addDesc(remote_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    free(buf);
    engine_->deregisterMem(md);
}

// ============================================================================
// Cancel Request Safety Tests — verify use-after-free fix
//
// The bug: processCancellations unconditionally set total_tasks = submitted_tasks.
// When handleSubmissionFailure had already bumped completed_tasks by +1 (for the
// failed task) and set total_tasks = submitted_tasks + 1, the overwrite made
// completed_tasks satisfy the completion threshold one callback early, causing
// premature deletion of req_h while DOCA callbacks were still pending.
// ============================================================================

// No-thread engine: cancel a request where some tasks submitted and a later
// allocation failed fatally (triggering handleSubmissionFailure).  The fixed
// cancelRequest must drain all in-flight callbacks before deleting.
TEST_F(DocMemosBackendTest, CancelAfterFatalSubmissionError_NoThread) {
    setupInitParams(false);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    // First 2 allocs succeed, 3rd fails with a fatal error (not FULL).
    // This triggers handleSubmissionFailure instead of queueing for retry.
    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_fail_error = DOCA_ERROR_BAD_STATE;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 40000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "fatal_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);

    // 2 tasks submitted OK, 3rd alloc fails with BAD_STATE → handleSubmissionFailure:
    //   total_tasks = submitted_tasks + 1 = 3
    //   ++completed_tasks = 1
    // postXfer returns IN_PROG (2 tasks still in-flight) or ERR_BACKEND.
    EXPECT_NE(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    EXPECT_EQ(mock.submitted_tasks.size(), 2u);

    // releaseReqH → cancelRequest marks cancelled, adds to cancelledRequests_.
    // Tasks are NOT drained here (no-thread engine returns immediately).
    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    // Tasks still in flight after cancel.
    EXPECT_EQ(mock.submitted_tasks.size(), 2u);

    // Enable auto-complete.  The engine destructor drain loop will call
    // doca_pe_progress to fire callbacks, then clean up cancelledRequests_.
    // With old code, the callbacks would see wrong total_tasks (from
    // processCancellations overwriting it) and could delete req_h early.
    // With the fix, cancelRequest preserves total_tasks so the callback
    // accounting stays correct.
    mock.auto_complete_tasks = true;

    for (int i = 0; i < 3; i++) {
        free(buffers[i]);
        engine_->deregisterMem(mds[i]);
    }

    // Engine destructor drains DOCA PE and cleans up cancelledRequests_.
    // Must not crash or double-free.
}

// Threaded engine: same handleSubmissionFailure scenario.  The progress
// thread's processCancellations must not overwrite total_tasks when the
// request was NOT in pendingRequests_ (i.e. handleSubmissionFailure already
// adjusted it).
//
// Timeline with old (buggy) code:
//   1. Progress thread submits 2 tasks; 3rd alloc fails fatally.
//      handleSubmissionFailure: total_tasks=3, completed_tasks=1.
//   2. doca_pe_progress completes 1 task → completed_tasks=2.
//   3. processCancellations: total_tasks = submitted_tasks = 2.
//      completed_tasks(2) >= total_tasks(2) → delete req_h.
//   4. doca_pe_progress fires callback for 2nd task → UAF on line 67.
//
// With the fix, processCancellations leaves total_tasks at 3 and adds the
// request to pendingDeletes_.  The 2nd callback safely completes it.
TEST_F(DocMemosBackendTest, CancelAfterFatalSubmissionError_Threaded) {
    setupInitParams(true);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_fail_error = DOCA_ERROR_BAD_STATE;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 41000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "thr_fatal_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // Wait for the progress thread to attempt all 3 allocations (2 succeed + 1 fatal).
    for (int i = 0; i < 200; i++) {
        if (mock.task_alloc_call_count >= 3) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_GE(mock.task_alloc_call_count, 3)
        << "Progress thread should have attempted 3 allocations";
    EXPECT_EQ(mock.submitted_tasks.size(), 2u);

    // Enable auto-complete so callbacks start firing, then immediately cancel.
    // The progress thread will interleave PE progress (callbacks) with
    // processCancellations.  With the old bug, this is the exact race that
    // causes UAF.
    mock.auto_complete_tasks = true;
    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    // Wait for the progress thread to drain everything.
    for (int i = 0; i < 200; i++) {
        if (mock.submitted_tasks.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(mock.submitted_tasks.size(), 0u)
        << "All tasks must be completed after cancel + drain";

    for (int i = 0; i < 3; i++) {
        free(buffers[i]);
        engine_->deregisterMem(mds[i]);
    }
}

// Threaded engine: cancel a request that has both in-flight tasks AND
// remaining descriptors queued in pendingRequests_ (partial submission
// due to DOCA_ERROR_FULL).  processCancellations must correctly adjust
// total_tasks for the pending portion without corrupting the in-flight
// task accounting.
TEST_F(DocMemosBackendTest, CancelWithInflightAndPendingTasks_Threaded) {
    setupInitParams(true);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    // Allow 2 allocations, then return FULL (queuing, not fatal).
    mock.task_alloc_fail_after_n = 2;
    mock.task_alloc_fail_error = DOCA_ERROR_FULL;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 5; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 42000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "pend_infl_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // Wait for partial submission (2 submitted, 3 pending).
    for (int i = 0; i < 200; i++) {
        if (mock.submitted_tasks.size() >= 2) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(mock.submitted_tasks.size(), 2u);

    // Cancel while request has both in-flight and pending tasks.
    // processCancellations will find it in pendingRequests_ and correctly
    // set total_tasks = submitted_tasks = 2, then add to pendingDeletes_
    // (since those 2 callbacks haven't fired yet).
    mock.auto_complete_tasks = true;
    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);

    // Wait for drain.
    for (int i = 0; i < 200; i++) {
        if (mock.submitted_tasks.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(mock.submitted_tasks.size(), 0u);

    for (int i = 0; i < 5; i++) {
        free(buffers[i]);
        engine_->deregisterMem(mds[i]);
    }
}

// Threaded engine destructor: verify that pendingDeletes_ entries are
// cleaned up when the engine is destroyed (progress thread joins, DOCA
// PE is drained, then remaining cancelled requests are deleted).
TEST_F(DocMemosBackendTest, DestructorCleansPendingDeletes) {
    setupInitParams(true);
    createEngine();

    auto &mock = DocaMockControl::instance();
    mock.auto_complete_tasks = false;

    nixl_meta_dlist_t local_dlist(DRAM_SEG), remote_dlist(OBJ_SEG);
    std::vector<void *> buffers;
    std::vector<nixlBackendMD *> mds;
    for (int i = 0; i < 3; i++) {
        nixlMetaDesc local_desc, remote_desc;
        void *buf = malloc(1024);
        buffers.push_back(buf);
        local_desc.addr = reinterpret_cast<uintptr_t>(buf);
        local_desc.len = 1024;
        remote_desc.devId = 43000 + i;
        remote_desc.addr = 0;
        remote_desc.len = 1024;
        std::string key = "dtor_key_" + std::to_string(i);
        remote_desc.metadataP = registerMemory(remote_desc.devId, key);
        mds.push_back(remote_desc.metadataP);
        local_dlist.addDesc(local_desc);
        remote_dlist.addDesc(remote_desc);
    }

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine_->prepXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_SUCCESS);
    ASSERT_EQ(engine_->postXfer(NIXL_WRITE, local_dlist, remote_dlist, "", handle), NIXL_IN_PROG);

    // Wait for submission.
    for (int i = 0; i < 200; i++) {
        if (mock.submitted_tasks.size() >= 3) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Cancel while tasks are in flight — req_h goes into pendingDeletes_.
    ASSERT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Enable auto-complete and destroy the engine.  The destructor must
    // join the thread, drain DOCA PE (firing callbacks), and delete
    // pendingDeletes_ entries.  Must not leak or crash.
    mock.auto_complete_tasks = true;
    for (auto *md : mds) {
        engine_->deregisterMem(md);
    }
    engine_.reset();

    for (int i = 0; i < 3; i++) {
        free(buffers[i]);
    }
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
