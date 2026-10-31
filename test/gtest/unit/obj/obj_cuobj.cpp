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

/**
 * Unit tests for all cuobjclient based accelerated engine unit tests.
 *
 * This file is only built when the cuobjclient library is available.
 * It exercises the Dell S3-over-RDMA code path and adds the Accel/Dell
 * configurations to the common parameterized test suite.
 */
#include "obj_test_base.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <mutex>

#include "object/rdma/s3_control_plane_http.h"

namespace gtest::obj {

// Accel and Dell test configurations for the common parameterized tests
static const ObjTestConfig accelConfig = {"Accel",
                                          {{"accelerated", "true"}},
                                          "test-accel-agent",
                                          false};
static const ObjTestConfig dellConfig = {"Dell",
                                         {{"accelerated", "true"}, {"type", "dell"}},
                                         "test-dell-agent",
                                         true};

// Parameterized test fixture is defined in obj.cpp; add Accel/Dell instantiations here
INSTANTIATE_TEST_SUITE_P(ObjAccelClientTests,
                         objParamTestFixture,
                         testing::Values(accelConfig, dellConfig),
                         [](const testing::TestParamInfo<ObjTestConfig> &info) {
                             return info.param.name;
                         });

/**
 * Dell ObjectScale accelerated engine test fixture.
 *
 * Exercises the Dell S3-over-RDMA code path by configuring the OBJ engine
 * with accelerated=true and type=dell. Uses mockDellS3Client to simulate
 * RDMA put/get operations via the iDellS3RdmaClient interface without
 * requiring real cuobjclient or ObjectScale infrastructure.
 */
class objDellTestFixture : public objTestBase, public testing::Test {
protected:
    void
    SetUp() override {
        setupEngine("test-dell-agent", {{"accelerated", "true"}, {"type", "dell"}});
    }
};

/** End-to-end RDMA write through the Dell accelerated engine. */
TEST_F(objDellTestFixture, DellRdmaWriteWithDescriptor) {
    testTransferWithSize(NIXL_WRITE, 1024, "-dell-rdma");
}

/** RDMA read at a non-zero object offset; verifies data correctness. */
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
    EXPECT_NE(vram_metadata, nullptr);

    objEngine_->deregisterMem(vram_metadata);
}

/** Multi-descriptor RDMA write with two different buffer sizes. */
TEST_F(objDellTestFixture, DellMultiDescriptorRdmaOperations) {
    testMultiDescriptorWithSizes(NIXL_WRITE, 512, 1024, "-dell-multi");
}

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

    // Test invalid operation type (using a cast to simulate invalid enum)
    nixl_xfer_op_t invalid_op = static_cast<nixl_xfer_op_t>(999);
    nixl_status_t status = objEngine_->prepXfer(
        invalid_op, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    // Should fail due to invalid operation
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** prepXfer rejects OBJ_SEG as the local descriptor segment type. */
TEST_F(objDellTestFixture, DellEngineInvalidLocalMemoryType) {
    // Test that Dell engine rejects invalid local memory type in prepXfer
    nixlBlobDesc local_desc, remote_desc;
    nixlBackendMD *local_metadata = nullptr, *remote_metadata = nullptr;

    // Register both as OBJ_SEG
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

    // Should fail due to invalid local memory type
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** prepXfer rejects a non-OBJ_SEG remote descriptor segment type. */
TEST_F(objDellTestFixture, DellEngineInvalidRemoteMemoryType) {
    // Test that Dell engine rejects non-OBJ_SEG remote memory type in prepXfer
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

/** prepXfer rejects mismatched local/remote descriptor list sizes. */
TEST_F(objDellTestFixture, DellEngineMismatchedDescriptorCounts) {
    // Test that Dell engine rejects mismatched local/remote descriptor counts
    std::vector<char> test_buffer1(512);
    std::vector<char> test_buffer2(512);

    nixlBlobDesc local_desc1, local_desc2;
    local_desc1.addr = reinterpret_cast<uintptr_t>(test_buffer1.data());
    local_desc1.len = test_buffer1.size();
    local_desc1.devId = 1;
    local_desc2.addr = reinterpret_cast<uintptr_t>(test_buffer2.data());
    local_desc2.len = test_buffer2.size();
    local_desc2.devId = 2;
    nixlBackendMD *local_metadata1 = nullptr, *local_metadata2 = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc1, DRAM_SEG, local_metadata1), NIXL_SUCCESS);
    ASSERT_EQ(objEngine_->registerMem(local_desc2, DRAM_SEG, local_metadata2), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 3;
    remote_desc.metaInfo = "mismatch-test-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    // Add 2 local descriptors but only 1 remote descriptor
    nixlMetaDesc local_meta_desc1(local_desc1.addr, local_desc1.len, local_desc1.devId);
    nixlMetaDesc local_meta_desc2(local_desc2.addr, local_desc2.len, local_desc2.devId);
    nixlMetaDesc remote_meta_desc(0, 512, remote_desc.devId);
    local_descs.addDesc(local_meta_desc1);
    local_descs.addDesc(local_meta_desc2);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    objEngine_->deregisterMem(local_metadata1);
    objEngine_->deregisterMem(local_metadata2);
    objEngine_->deregisterMem(remote_metadata);
}

/** prepXfer fails when the remote devId has no registered OBJ_SEG mapping. */
TEST_F(objDellTestFixture, DellUnregisteredRemoteDevId) {
    // Test that Dell engine rejects transfers when remote devId is not registered
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

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** registerMem rejects unsupported segment types (BLK_SEG, FILE_SEG). */
TEST_F(objDellTestFixture, DellRegisterMemUnsupportedType) {
    // Test that Dell engine rejects unsupported memory types
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

/** registerMem rejects DRAM/VRAM buffers exceeding CUOBJ_MAX_MEMORY_REG_SIZE (4 GiB). */
TEST_F(objDellTestFixture, DellRegisterMemExceedsMaxSize) {
    // Test that registerMem rejects DRAM/VRAM buffers larger than CUOBJ_MAX_MEMORY_REG_SIZE (4
    // GiB). This constant is defined in cuobjclient.h; keep in sync if it changes.
    constexpr size_t kMaxMemRegSize = 4ULL * 1024 * 1024 * 1024;
    std::vector<char> dummy(1);

    nixlBlobDesc mem_desc;
    mem_desc.addr = reinterpret_cast<uintptr_t>(dummy.data());
    mem_desc.len = kMaxMemRegSize + 1;
    mem_desc.devId = 1;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, DRAM_SEG, metadata);
    EXPECT_EQ(status, NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(metadata, nullptr);

    // Same check for VRAM_SEG
    metadata = nullptr;
    status = objEngine_->registerMem(mem_desc, VRAM_SEG, metadata);
    EXPECT_EQ(status, NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(metadata, nullptr);
}

/** prepXfer handles zero-length descriptors without crashing. */
TEST_F(objDellTestFixture, DellEngineZeroSizePrep) {
    // Test that prepXfer accepts zero-size descriptors without crashing.
    // Note: both putObjectRdmaAsync and getObjectRdmaAsync in client.cpp
    // reject data_len==0, so a zero-size transfer would fail at postXfer
    // in production. This test only verifies prepXfer handles the edge case.
    mockS3Client_->setSimulateSuccess(true);

    nixlBlobDesc local_desc;
    std::vector<char> dummy(1);
    local_desc.addr = reinterpret_cast<uintptr_t>(dummy.data());
    local_desc.len = 1;
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    // registerMem calls into cuMemObjGetDescriptor which requires a non-zero length buffer
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

/** Simulated RDMA write failure propagates error status through checkXfer. */
TEST_F(objDellTestFixture, DellWriteTransferFailure) {
    testTransferFailure(NIXL_WRITE, 1024, "-dell-fail-write");
}

/** Simulated RDMA read failure propagates error status through checkXfer. */
TEST_F(objDellTestFixture, DellReadTransferFailure) {
    testTransferFailure(NIXL_READ, 1024, "-dell-fail-read");
}

/** Multi-descriptor RDMA read with data-content verification. */
TEST_F(objDellTestFixture, DellMultiDescriptorReadVerifyData) {
    testMultiDescriptorWithSizes(NIXL_READ, 512, 256, "-dell-multi-read");
}

/** Agent name mismatch between prepXfer and engine logs a warning but succeeds. */
TEST_F(objDellTestFixture, DellAgentMismatchWarning) {
    // Test that agent mismatch produces a warning but does not fail
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
    // Test that OBJ_SEG with empty metaInfo auto-generates key from devId
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
    // The auto-generated key should be "42" (from devId)
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
    // Test that deregistering OBJ_SEG cleans up the devId-to-key mapping
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

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    // Should fail because devId 10 mapping was removed
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    objEngine_->deregisterMem(local_metadata);
}

/** checkXfer returns NIXL_IN_PROG before mock callbacks are executed. */
TEST_F(objDellTestFixture, DellCheckXferBeforeExecAsync) {
    // Test checkXfer returns NIXL_IN_PROG before callbacks are executed
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

/** registerMem rejects zero-length DRAM/VRAM buffers with NIXL_ERR_BACKEND. */
TEST_F(objDellTestFixture, DellRegisterMemZeroLengthBuffer) {
    // Test that cuMemObjGetDescriptor fails when given a zero-length buffer
    std::vector<char> dummy(1);

    nixlBlobDesc mem_desc;
    mem_desc.addr = reinterpret_cast<uintptr_t>(dummy.data());
    mem_desc.len = 0;
    mem_desc.devId = 1;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, DRAM_SEG, metadata);
    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(metadata, nullptr);

    // Same check for VRAM_SEG
    metadata = nullptr;
    status = objEngine_->registerMem(mem_desc, VRAM_SEG, metadata);
    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(metadata, nullptr);
}

/** prepXfer fails when the local buffer address was never registered with cuObj. */
TEST_F(objDellTestFixture, DellPrepXferUnregisteredAddress) {
    // Test that cuObjPut fails when the local address was never registered with cuObj
    std::vector<char> registered_buffer(1024);
    std::vector<char> unregistered_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(registered_buffer.data());
    local_desc.len = registered_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "unregistered-addr-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    // Use unregistered_buffer's address - not registered with cuMemObjGetDescriptor
    nixlMetaDesc local_meta_desc(reinterpret_cast<uintptr_t>(unregistered_buffer.data()),
                                 unregistered_buffer.size(),
                                 local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, unregistered_buffer.size(), remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(handle, nullptr);

    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/** prepXfer fails when the local address exceeds the 16 MiB RDMA descriptor coverage limit. */
TEST_F(objDellTestFixture, DellPrepXferLargeMemoryOffset) {
    // Test that cuObjPut fails when the local address is more than 16 MB from the
    // base address of the registered memory region.
    // This is a cuObject library constraint on RDMA descriptor coverage.
    constexpr size_t k16MiB = 16ULL * 1024 * 1024;
    constexpr size_t kBufSize = k16MiB + 4096;
    std::vector<char> large_buffer(kBufSize);

    nixlBlobDesc local_desc;
    local_desc.addr = reinterpret_cast<uintptr_t>(large_buffer.data());
    local_desc.len = large_buffer.size();
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "large-offset-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    // Use an address > 16 MiB from the registered base
    uintptr_t offset_addr = reinterpret_cast<uintptr_t>(large_buffer.data()) + k16MiB + 1;
    nixlMetaDesc local_meta_desc(offset_addr, 1024, local_desc.devId);
    nixlMetaDesc remote_meta_desc(0, 1024, remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = objEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(handle, nullptr);

    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

/**
 * Answers every request with 400 and records its Authorization header, so a
 * test can see which credentials the S3 RDMA control plane signed with.
 */
class authRecordingServer {
public:
    authRecordingServer() {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bind(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
        listen(fd_, 4);
        socklen_t len = sizeof(addr);
        getsockname(fd_, reinterpret_cast<sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);
        thread_ = std::thread([this] { serve(); });
    }

    ~authRecordingServer() {
        shutdown(fd_, SHUT_RDWR);
        close(fd_);
        thread_.join();
    }

    [[nodiscard]] std::string
    endpoint() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    [[nodiscard]] std::vector<std::string>
    authorizations() {
        const std::lock_guard<std::mutex> lock(mutex_);
        return authorizations_;
    }

private:
    void
    serve() {
        for (;;) {
            const int conn = accept(fd_, nullptr, nullptr);
            if (conn < 0) {
                return;
            }
            std::string request;
            char buf[4096];
            while (request.find("\r\n\r\n") == std::string::npos) {
                const ssize_t n = recv(conn, buf, sizeof(buf), 0);
                if (n <= 0) {
                    break;
                }
                request.append(buf, n);
            }
            record(request);
            const std::string response =
                "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(conn, response.data(), response.size(), MSG_NOSIGNAL);
            close(conn);
        }
    }

    void
    record(const std::string &request) {
        std::string lower = request;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        const size_t start = lower.find("\r\nauthorization:");
        if (start == std::string::npos) {
            return;
        }
        const size_t end = request.find("\r\n", start + 2);
        const std::lock_guard<std::mutex> lock(mutex_);
        authorizations_.push_back(request.substr(start + 2, end - start - 2));
    }

    int fd_ = -1;
    uint16_t port_ = 0;
    std::thread thread_;
    std::mutex mutex_;
    std::vector<std::string> authorizations_;
};

/**
 * Sets AWS credential environment variables for one test and restores them,
 * keeping the default chain away from credential files and instance metadata.
 */
class scopedAwsEnv {
public:
    scopedAwsEnv() {
        for (const char *name : kNames) {
            const char *value = std::getenv(name);
            saved_.emplace_back(name, value ? std::optional<std::string>(value) : std::nullopt);
            unsetenv(name);
        }
        setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/credentials", 1);
        setenv("AWS_CONFIG_FILE", "/nonexistent/config", 1);
        setenv("AWS_EC2_METADATA_DISABLED", "true", 1);
    }

    ~scopedAwsEnv() {
        for (const auto &[name, value] : saved_) {
            if (value) {
                setenv(name, value->c_str(), 1);
            } else {
                unsetenv(name);
            }
        }
    }

    void
    setKeys(const char *access_key, const char *secret_key) {
        setenv("AWS_ACCESS_KEY_ID", access_key, 1);
        setenv("AWS_SECRET_ACCESS_KEY", secret_key, 1);
    }

private:
    static constexpr const char *kNames[] = {"AWS_ACCESS_KEY_ID",
                                             "AWS_SECRET_ACCESS_KEY",
                                             "AWS_SESSION_TOKEN",
                                             "AWS_PROFILE",
                                             "AWS_SHARED_CREDENTIALS_FILE",
                                             "AWS_CONFIG_FILE",
                                             "AWS_EC2_METADATA_DISABLED"};
    std::vector<std::pair<const char *, std::optional<std::string>>> saved_;
};

static void
sendRdmaGet(nixl_obj_rdma::S3RdmaControlPlane &cp) {
    nixl_obj_rdma::S3RdmaClientCtx ctx;
    ctx.bucket = "bucket";
    ctx.object = "key";
    EXPECT_LT(cp.rdmaGet(ctx, "token", 16, 0), 0);
}

TEST(objRdmaControlPlaneTest, ChainCredentialsAreReadForEachRequest) {
    scopedAwsEnv env;
    env.setKeys("AKIDFIRST", "secret-first");
    authRecordingServer server;
    nixl_b_params_t params = {{"bucket", "bucket"}, {"endpoint_override", server.endpoint()}};
    nixl_obj_rdma::S3RdmaControlPlane cp(&params);
    ASSERT_TRUE(cp.valid());

    sendRdmaGet(cp);
    env.setKeys("AKIDSECOND", "secret-second");
    sendRdmaGet(cp);

    const auto auth = server.authorizations();
    ASSERT_EQ(auth.size(), 2u);
    EXPECT_NE(auth[0].find("Credential=AKIDFIRST/"), std::string::npos) << auth[0];
    EXPECT_NE(auth[1].find("Credential=AKIDSECOND/"), std::string::npos) << auth[1];
}

TEST(objRdmaControlPlaneTest, ExplicitCredentialsStayFixed) {
    scopedAwsEnv env;
    env.setKeys("AKIDENV", "secret-env");
    authRecordingServer server;
    nixl_b_params_t params = {{"bucket", "bucket"},
                              {"endpoint_override", server.endpoint()},
                              {"access_key", "AKIDPARAM"},
                              {"secret_key", "secret-param"}};
    nixl_obj_rdma::S3RdmaControlPlane cp(&params);
    ASSERT_TRUE(cp.valid());

    env.setKeys("AKIDROTATED", "secret-rotated");
    sendRdmaGet(cp);

    const auto auth = server.authorizations();
    ASSERT_EQ(auth.size(), 1u);
    EXPECT_NE(auth[0].find("Credential=AKIDPARAM/"), std::string::npos) << auth[0];
}

TEST(objRdmaControlPlaneTest, MissingCredentialsSendNothing) {
    scopedAwsEnv env;
    authRecordingServer server;
    nixl_b_params_t params = {{"bucket", "bucket"}, {"endpoint_override", server.endpoint()}};
    nixl_obj_rdma::S3RdmaControlPlane cp(&params);
    EXPECT_FALSE(cp.valid());

    sendRdmaGet(cp);
    EXPECT_TRUE(server.authorizations().empty());
}

} // namespace gtest::obj
