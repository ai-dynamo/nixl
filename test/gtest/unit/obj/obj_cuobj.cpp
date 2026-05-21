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

/**
 * Unit tests for cuobjclient-dependent accelerated engine paths.
 *
 * This file is only built when the cuobjclient library is available.
 * It exercises the Dell S3-over-RDMA code path and adds the Accel/Dell
 * configurations to the common parameterized test suite.
 *
 * Note on mock vs real behaviour:
 *   When the Dell engine is constructed with an injected mock iS3Client,
 *   CuObjTokenManager is NOT created (tokenMgr_ == nullptr).  This means
 *   registerMem for DRAM/VRAM takes the early-return path (NIXL_SUCCESS
 *   with out = nullptr).  Tests that require real cuObject behaviour
 *   (e.g. size validation, RDMA descriptor coverage) are placed in the
 *   CuObjTokenManager section and guarded by isConnected().
 */
#include "obj_test_base.h"
#include "s3_accel/dell/cuobj_token_manager.h"
#include <cstdlib>
#include <cstring>

namespace gtest::obj {

// ---------------------------------------------------------------------------
// Accel / Dell parameterized test configurations
// ---------------------------------------------------------------------------

static const ObjTestConfig accelConfig = {"Accel",
                                          {{"accelerated", "true"}},
                                          "test-accel-agent",
                                          false};
static const ObjTestConfig dellConfig = {"Dell",
                                         {{"accelerated", "true"}, {"type", "dell"}},
                                         "test-dell-agent",
                                         true};

// Add Accel/Dell to the common parameterized suite defined in obj.cpp.
INSTANTIATE_TEST_SUITE_P(ObjAccelClientTests,
                         objParamTestFixture,
                         testing::Values(accelConfig, dellConfig),
                         [](const testing::TestParamInfo<ObjTestConfig> &info) {
                             return info.param.name;
                         });

// ---------------------------------------------------------------------------
// Dell-specific test fixture
// ---------------------------------------------------------------------------

/**
 * Dell ObjectScale accelerated engine test fixture.
 *
 * Exercises the Dell S3-over-RDMA code path by configuring the OBJ engine
 * with accelerated=true and type=dell.  Uses the common mockS3Client to
 * simulate S3 operations.  The RDMA-specific behaviour is transparent at
 * the iS3Client level (Pattern B).
 */
class objDellTestFixture : public objTestBase, public testing::Test {
protected:
    void
    SetUp() override {
        setupEngine("test-dell-agent", {{"accelerated", "true"}, {"type", "dell"}});
    }
};

// ---------------------------------------------------------------------------
// Dell engine: basic transfer tests
// ---------------------------------------------------------------------------

/** End-to-end write through the Dell accelerated engine. */
TEST_F(objDellTestFixture, DellRdmaWriteWithDescriptor) {
    testTransferWithSize(NIXL_WRITE, 1024, "-dell-rdma");
}

/** Read at a non-zero object offset; verifies data correctness. */
TEST_F(objDellTestFixture, DellRdmaReadWithOffset) {
    mockS3Client_->setSimulateSuccess(true);

    std::vector<char> test_buffer(256);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "dell-rdma-read-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    const size_t offset = 512;
    const size_t length = 256;
    nixlMetaDesc local_meta_desc(local_desc.addr, length, local_desc.devId);
    nixlMetaDesc remote_meta_desc(offset, length, remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(objEngine_->prepXfer(
                  NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
              NIXL_SUCCESS);

    nixl_status_t status = objEngine_->postXfer(
        NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_IN_PROG);

    mockS3Client_->execAsync();
    status = objEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(test_buffer[0], 'A' + (offset % 26));

    objEngine_->releaseReqH(handle);
    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** Dell engine advertises VRAM_SEG support and can register VRAM memory. */
TEST_F(objDellTestFixture, DellVramMemorySupport) {
    auto supported_mems = objEngine_->getSupportedMems();
    EXPECT_TRUE(std::find(supported_mems.begin(), supported_mems.end(), VRAM_SEG) !=
                supported_mems.end());

    std::vector<char> test_buffer1(512);
    std::fill(test_buffer1.begin(), test_buffer1.end(), 'A');
    nixlBlobDesc vram_desc;

    vram_desc.addr = reinterpret_cast<uintptr_t>(test_buffer1.data());
    vram_desc.len = test_buffer1.size();
    vram_desc.devId = 3;
    nixlBackendMD *vram_metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(vram_desc, VRAM_SEG, vram_metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
    // With mock client, tokenMgr_ is null so metadata will be nullptr.
    // This is expected — real cuObject registration only happens with
    // actual RDMA hardware.

    if (vram_metadata) {
        objEngine_->deregisterMem(vram_metadata);
    }
}

/** Multi-descriptor write with two different buffer sizes. */
TEST_F(objDellTestFixture, DellMultiDescriptorRdmaOperations) {
    testMultiDescriptorWithSizes(NIXL_WRITE, 512, 1024, "-dell-multi");
}

// ---------------------------------------------------------------------------
// Dell engine: prepXfer parameter validation
//
// isValidPrepXferParams checks: operation type, local mem type, remote mem type.
// Descriptor count and devId registration are checked in postXfer.
// ---------------------------------------------------------------------------

/** prepXfer rejects an invalid (out-of-range) transfer operation enum. */
TEST_F(objDellTestFixture, DellEngineInvalidOperationType) {
    std::vector<char> test_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "invalid-op-test";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, 1024, remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;

    nixl_xfer_op_t invalid_op = static_cast<nixl_xfer_op_t>(999);
    nixl_status_t status = objEngine_->prepXfer(
        invalid_op, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** prepXfer rejects OBJ_SEG as the local descriptor segment type. */
TEST_F(objDellTestFixture, DellEngineInvalidLocalMemoryType) {
    nixlBlobDesc local_desc, remote_desc;
    nixlBackendMD *local_metadata = nullptr, *remote_metadata = nullptr;

    local_desc.devId = 1;
    local_desc.metaInfo = "local-obj-key";
    ASSERT_EQ(objEngine_->registerMem(local_desc, OBJ_SEG, local_metadata), NIXL_SUCCESS);

    remote_desc.devId = 2;
    remote_desc.metaInfo = "remote-obj-key";
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(OBJ_SEG); // Invalid - should be DRAM/VRAM
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    char dummy[1024];
    nixlMetaDesc local_meta_desc(reinterpret_cast<uintptr_t>(dummy), 1024, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, 1024, remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** prepXfer rejects a non-OBJ_SEG remote descriptor segment type. */
TEST_F(objDellTestFixture, DellEngineInvalidRemoteMemoryType) {
    std::vector<char> test_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(DRAM_SEG); // Invalid - should be OBJ_SEG

    nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, 1024, 2);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    objEngine_->deregisterMem(local_metadata);
}

// ---------------------------------------------------------------------------
// Dell engine: postXfer validation
//
// These errors are caught at postXfer time, not prepXfer.
// ---------------------------------------------------------------------------

/** postXfer fails when the remote devId has no registered OBJ_SEG mapping. */
TEST_F(objDellTestFixture, DellUnregisteredRemoteDevId) {
    std::vector<char> test_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    // Register an OBJ_SEG with devId=2, but use devId=999 in the descriptor
    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "registered-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
    // Use devId=999 which was never registered
    nixlMetaDesc remote_meta_desc(0, 1024, 999);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    // prepXfer succeeds — it does not validate devId registration
    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // postXfer fails — devId 999 is not in devIdToObjKey_
    status = objEngine_->postXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    objEngine_->releaseReqH(handle);
    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

// ---------------------------------------------------------------------------
// Dell engine: registerMem edge cases
// ---------------------------------------------------------------------------

/** registerMem rejects unsupported segment types (BLK_SEG, FILE_SEG). */
TEST_F(objDellTestFixture, DellRegisterMemUnsupportedType) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 1;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, BLK_SEG, metadata);
    EXPECT_EQ(status, NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(metadata, nullptr);

    // Also test FILE_SEG
    status = objEngine_->registerMem(mem_desc, FILE_SEG, metadata);
    EXPECT_EQ(status, NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(metadata, nullptr);
}

/**
 * With mock client (no tokenMgr_), DRAM/VRAM registerMem returns
 * NIXL_SUCCESS with nullptr metadata regardless of buffer size/address.
 * This verifies the mock path works correctly.
 */
TEST_F(objDellTestFixture, DellRegisterMemDramVramMockPath) {
    // Zero-length buffer — with real tokenMgr_ this would fail,
    // but with mock the early-return path returns NIXL_SUCCESS.
    std::vector<char> dummy(1);
    nixlBlobDesc mem_desc;
    mem_desc.addr = reinterpret_cast<uintptr_t>(dummy.data());
    mem_desc.len = 0;
    mem_desc.devId = 1;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, DRAM_SEG, metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(metadata, nullptr); // No tokenMgr_ → nullptr

    // Large buffer — same mock path
    mem_desc.len = 1024 * 1024;
    metadata = nullptr;
    status = objEngine_->registerMem(mem_desc, VRAM_SEG, metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(metadata, nullptr);
}

// ---------------------------------------------------------------------------
// Dell engine: failure propagation and lifecycle
// ---------------------------------------------------------------------------

/** Simulated write failure propagates error status through checkXfer. */
TEST_F(objDellTestFixture, DellWriteTransferFailure) {
    testTransferFailure(NIXL_WRITE, 1024, "-dell-fail-write");
}

/** Simulated read failure propagates error status through checkXfer. */
TEST_F(objDellTestFixture, DellReadTransferFailure) {
    testTransferFailure(NIXL_READ, 1024, "-dell-fail-read");
}

/** Multi-descriptor read with data-content verification. */
TEST_F(objDellTestFixture, DellMultiDescriptorReadVerifyData) {
    testMultiDescriptorWithSizes(NIXL_READ, 512, 256, "-dell-multi-read");
}

/** Agent name mismatch between prepXfer and engine logs a warning but succeeds. */
TEST_F(objDellTestFixture, DellAgentMismatchWarning) {
    mockS3Client_->setSimulateSuccess(true);

    std::vector<char> test_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "agent-mismatch-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, test_buffer.size(), remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    // Pass a different remote_agent than the engine's localAgent
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, "different-agent", handle, nullptr);

    // Should succeed (mismatch is only a warning, not an error)
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(handle, nullptr);

    objEngine_->releaseReqH(handle);
    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** OBJ_SEG registration with empty metaInfo auto-generates a key from devId. */
TEST_F(objDellTestFixture, DellObjSegAutoKeyGeneration) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 42;
    mem_desc.metaInfo = ""; // Empty key - engine will generate from devId

    nixlBackendMD *metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(mem_desc, OBJ_SEG, metadata), NIXL_SUCCESS);
    ASSERT_NE(metadata, nullptr);

    // Verify the key is used by attempting a transfer
    std::vector<char> test_buffer(256);
    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, test_buffer.size(), mem_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_SUCCESS);

    if (handle) {
        objEngine_->releaseReqH(handle);
    }
    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(metadata);
}

/** Deregistering an OBJ_SEG removes its devId-to-key mapping; subsequent transfers fail. */
TEST_F(objDellTestFixture, DellDeregisterObjSegCleansMapping) {
    nixlBlobDesc remote_desc;
    remote_desc.devId = 10;
    remote_desc.metaInfo = "deregister-cleanup-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    // Deregister the OBJ_SEG
    ASSERT_EQ(objEngine_->deregisterMem(remote_metadata), NIXL_SUCCESS);

    // Now try to use the devId in a transfer - should fail because mapping was cleaned
    std::vector<char> test_buffer(256);
    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, 256, 10); // devId=10, no longer registered
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    // prepXfer succeeds (doesn't check devId registration)
    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // postXfer fails because devId 10 mapping was removed
    status = objEngine_->postXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    objEngine_->releaseReqH(handle);
    objEngine_->deregisterMem(local_metadata);
}

/** checkXfer returns NIXL_IN_PROG before mock callbacks are executed. */
TEST_F(objDellTestFixture, DellCheckXferBeforeExecAsync) {
    mockS3Client_->setSimulateSuccess(true);

    std::vector<char> test_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
    local_desc.len = test_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "check-before-exec-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, test_buffer.size(), remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(objEngine_->prepXfer(
                  NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
              NIXL_SUCCESS);

    nixl_status_t status = objEngine_->postXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Check before executing async callbacks - should still be in progress
    status = objEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Now execute and verify completion
    mockS3Client_->execAsync();
    status = objEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_SUCCESS);

    objEngine_->releaseReqH(handle);
    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** prepXfer handles zero-length descriptors without crashing. */
TEST_F(objDellTestFixture, DellEngineZeroSizePrep) {
    mockS3Client_->setSimulateSuccess(true);

    nixlBlobDesc local_desc;
    std::vector<char> dummy(1);
    local_desc.addr = reinterpret_cast<uintptr_t>(dummy.data());
    local_desc.len = 1;
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "zero-size-test";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    // Create meta descriptions with zero size
    nixlMetaDesc local_meta_desc(local_desc.addr, 0, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, 0, remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    // Should handle zero size gracefully
    EXPECT_EQ(status, NIXL_SUCCESS);

    if (handle) {
        objEngine_->releaseReqH(handle);
    }
    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

} // namespace gtest::obj

// ---------------------------------------------------------------------------
// CuObjTokenManager integration tests (require cuObject library)
//
// These tests exercise the CuObjTokenManager directly against the real
// cuObjClient library.  They validate:
//   - Construction and connection status
//   - Input validation (null ptr, zero size, oversized regions)
//   - Memory registration and deregistration with system memory
//   - Token generation for registered memory (requires RDMA hardware)
//
// Tests that require RDMA hardware (token generation) are skipped
// gracefully if the cuObject client cannot connect.
// ---------------------------------------------------------------------------

class CuObjTokenManagerTest : public testing::Test {
protected:
    // Allocate page-aligned system memory for RDMA registration tests.
    // cuMemObjGetDescriptor requires the buffer to remain valid until
    // cuMemObjPutDescriptor is called.
    static constexpr size_t kPageSize = 4096;
    static constexpr size_t kBufferSize = 64 * 1024; // 64 KiB

    void *buffer_ = nullptr;

    void
    SetUp() override {
        // posix_memalign guarantees page alignment, which cuObject prefers.
        int rc = posix_memalign(&buffer_, kPageSize, kBufferSize);
        ASSERT_EQ(rc, 0);
        ASSERT_NE(buffer_, nullptr);
        std::memset(buffer_, 0xAB, kBufferSize);
    }

    void
    TearDown() override {
        if (buffer_) {
            free(buffer_);
            buffer_ = nullptr;
        }
    }
};

// --- Construction tests ---

TEST_F(CuObjTokenManagerTest, ConstructionDefault) {
    // The token manager should construct without throwing.
    // Connection may fail if no RDMA hardware is available — that's OK,
    // isConnected() will return false.
    CuObjTokenManager mgr;
    // isConnected() returns a valid bool (true or false) — no crash.
    (void)mgr.isConnected();
}

TEST_F(CuObjTokenManagerTest, ConstructionWithProtocol) {
    CuObjTokenManager mgr(CUOBJ_PROTO_RDMA_DC_V1);
    (void)mgr.isConnected();
}

// --- Input validation tests (do not require RDMA hardware) ---

TEST_F(CuObjTokenManagerTest, RegisterNullPtrFails) {
    CuObjTokenManager mgr;
    EXPECT_EQ(mgr.registerMemory(nullptr, kBufferSize), CU_OBJ_FAIL);
}

TEST_F(CuObjTokenManagerTest, RegisterZeroSizeFails) {
    CuObjTokenManager mgr;
    EXPECT_EQ(mgr.registerMemory(buffer_, 0), CU_OBJ_FAIL);
}

TEST_F(CuObjTokenManagerTest, RegisterOversizedFails) {
    CuObjTokenManager mgr;
    // 4 GiB is the cuObject limit (CUOBJ_MAX_MEMORY_REG_SIZE).
    size_t too_large = 4ULL * 1024 * 1024 * 1024;
    EXPECT_EQ(mgr.registerMemory(buffer_, too_large), CU_OBJ_FAIL);
}

TEST_F(CuObjTokenManagerTest, DeregisterNullPtrSucceeds) {
    CuObjTokenManager mgr;
    // Deregistering nullptr is a no-op, should not fail.
    EXPECT_EQ(mgr.deregisterMemory(nullptr), CU_OBJ_SUCCESS);
}

TEST_F(CuObjTokenManagerTest, GeneratePutTokenNullPtrThrows) {
    CuObjTokenManager mgr;
    EXPECT_THROW(mgr.generatePutToken(nullptr, kBufferSize), std::runtime_error);
}

TEST_F(CuObjTokenManagerTest, GenerateGetTokenZeroSizeThrows) {
    CuObjTokenManager mgr;
    EXPECT_THROW(mgr.generateGetToken(buffer_, 0), std::runtime_error);
}

// --- Memory registration tests (require cuObject library, not RDMA NIC) ---

TEST_F(CuObjTokenManagerTest, RegisterAndDeregisterSystemMemory) {
    CuObjTokenManager mgr;
    if (!mgr.isConnected()) {
        GTEST_SKIP() << "cuObject client not connected (no RDMA hardware)";
    }

    // Register system (host) memory — may fail if cuMemObjGetDescriptor
    // does not support plain posix_memalign'd buffers on this system.
    cuObjErr_t rc = mgr.registerMemory(buffer_, kBufferSize);
    if (rc != CU_OBJ_SUCCESS) {
        GTEST_SKIP() << "cuMemObjGetDescriptor does not accept system memory on this platform";
    }

    // Deregister — should match the registration.
    rc = mgr.deregisterMemory(buffer_);
    EXPECT_EQ(rc, CU_OBJ_SUCCESS);
}

TEST_F(CuObjTokenManagerTest, RegisterMultipleRegions) {
    CuObjTokenManager mgr;
    if (!mgr.isConnected()) {
        GTEST_SKIP() << "cuObject client not connected (no RDMA hardware)";
    }

    // Simulate the NIXL pattern: register multiple pages from a contiguous
    // buffer, each at its own address and size.
    void *page0 = buffer_;
    void *page1 = static_cast<char *>(buffer_) + kPageSize;
    void *page2 = static_cast<char *>(buffer_) + 2 * kPageSize;

    cuObjErr_t rc = mgr.registerMemory(page0, kPageSize);
    if (rc != CU_OBJ_SUCCESS) {
        GTEST_SKIP() << "cuMemObjGetDescriptor does not accept system memory on this platform";
    }
    EXPECT_EQ(mgr.registerMemory(page1, kPageSize), CU_OBJ_SUCCESS);
    EXPECT_EQ(mgr.registerMemory(page2, kPageSize), CU_OBJ_SUCCESS);

    // Deregister in reverse order — each is independent.
    EXPECT_EQ(mgr.deregisterMemory(page2), CU_OBJ_SUCCESS);
    EXPECT_EQ(mgr.deregisterMemory(page1), CU_OBJ_SUCCESS);
    EXPECT_EQ(mgr.deregisterMemory(page0), CU_OBJ_SUCCESS);
}

TEST_F(CuObjTokenManagerTest, RegisterDeregisterIdempotent) {
    CuObjTokenManager mgr;
    if (!mgr.isConnected()) {
        GTEST_SKIP() << "cuObject client not connected (no RDMA hardware)";
    }

    // Register and deregister the same region twice — simulates page reuse.
    cuObjErr_t rc = mgr.registerMemory(buffer_, kBufferSize);
    if (rc != CU_OBJ_SUCCESS) {
        GTEST_SKIP() << "cuMemObjGetDescriptor does not accept system memory on this platform";
    }
    EXPECT_EQ(mgr.deregisterMemory(buffer_), CU_OBJ_SUCCESS);

    // Re-register the same memory (page recycled by the allocator).
    EXPECT_EQ(mgr.registerMemory(buffer_, kBufferSize), CU_OBJ_SUCCESS);
    EXPECT_EQ(mgr.deregisterMemory(buffer_), CU_OBJ_SUCCESS);
}

// --- Token generation tests (require RDMA hardware) ---

TEST_F(CuObjTokenManagerTest, GeneratePutTokenForRegisteredMemory) {
    CuObjTokenManager mgr;
    if (!mgr.isConnected()) {
        GTEST_SKIP() << "cuObject client not connected (no RDMA hardware)";
    }

    cuObjErr_t rc = mgr.registerMemory(buffer_, kBufferSize);
    if (rc != CU_OBJ_SUCCESS) {
        GTEST_SKIP() << "cuMemObjGetDescriptor failed (system memory may not be RDMA-capable)";
    }

    // Generate a PUT token — requires RDMA NIC.
    // If no RDMA hardware, cuMemObjGetRDMAToken will throw.
    try {
        std::string token = mgr.generatePutToken(buffer_, kBufferSize);
        // Token should be non-empty if generation succeeded.
        EXPECT_FALSE(token.empty());
    }
    catch (const std::runtime_error &) {
        // cuMemObjGetRDMAToken failed — expected if no RDMA NIC is present.
        // This is not a test failure; token generation requires real hardware.
    }

    mgr.deregisterMemory(buffer_);
}

TEST_F(CuObjTokenManagerTest, GenerateGetTokenForRegisteredMemory) {
    CuObjTokenManager mgr;
    if (!mgr.isConnected()) {
        GTEST_SKIP() << "cuObject client not connected (no RDMA hardware)";
    }

    cuObjErr_t rc = mgr.registerMemory(buffer_, kBufferSize);
    if (rc != CU_OBJ_SUCCESS) {
        GTEST_SKIP() << "cuMemObjGetDescriptor failed";
    }

    try {
        std::string token = mgr.generateGetToken(buffer_, kBufferSize);
        EXPECT_FALSE(token.empty());
    }
    catch (const std::runtime_error &) {
        // Expected if no RDMA NIC.
    }

    mgr.deregisterMemory(buffer_);
}

TEST_F(CuObjTokenManagerTest, GenerateTokenForUnregisteredMemoryThrows) {
    CuObjTokenManager mgr;
    if (!mgr.isConnected()) {
        GTEST_SKIP() << "cuObject client not connected (no RDMA hardware)";
    }

    // Do NOT register buffer_ — generatePutToken should fail because
    // cuMemObjGetRDMAToken requires prior registration.
    EXPECT_THROW(mgr.generatePutToken(buffer_, kBufferSize), std::runtime_error);
}
