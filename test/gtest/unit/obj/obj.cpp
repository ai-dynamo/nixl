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
#include "obj_test_base.h"

namespace gtest::obj {
/**
 * Object Plugin Unit Tests
 *
 * Test suites use parameterized testing to reduce code duplication:
 * - ObjClientTests: Runs common tests for all client types (Standard S3, S3 CRT, S3 Accel)
 * - Specialized tests: Client-specific tests (e.g., CRT threshold testing)
 *
 * All tests use a mockS3Client to simulate S3 operations without requiring AWS credentials.
 */

// Non-parameterized fixture for specialized tests
class objTestFixture : public objTestBase, public testing::Test {
protected:
    void
    SetUp() override {
        setupEngine("test-agent");
    }
};

// Test configurations
static const ObjTestConfig standardConfig = {"Standard", {}, "test-standard-agent"};
static const ObjTestConfig crtConfig = {"CRT", {{"crtMinLimit", "1024"}}, "test-crt-agent"};

// Parameterized tests - run for all client types
TEST_P(objParamTestFixture, EngineInitialization) {
    ASSERT_NE(objEngine_, nullptr);
    EXPECT_EQ(objEngine_->getType(), "OBJ");
    EXPECT_TRUE(objEngine_->supportsLocal());
    EXPECT_FALSE(objEngine_->supportsRemote());
    EXPECT_FALSE(objEngine_->supportsNotif());
    EXPECT_TRUE(mockS3Client_->hasExecutor());
}

TEST_P(objParamTestFixture, GetSupportedMems) {
    auto supported_mems = objEngine_->getSupportedMems();

    // Check expected number of supported memory types based on configuration
    const auto &config = GetParam();
    if (config.supportsVram) {
        EXPECT_EQ(supported_mems.size(), 3); // OBJ_SEG, DRAM_SEG, VRAM_SEG
    } else {
        EXPECT_EQ(supported_mems.size(), 2); // OBJ_SEG, DRAM_SEG
    }

    EXPECT_TRUE(std::find(supported_mems.begin(), supported_mems.end(), OBJ_SEG) !=
                supported_mems.end());
    EXPECT_TRUE(std::find(supported_mems.begin(), supported_mems.end(), DRAM_SEG) !=
                supported_mems.end());

    // Dell configuration should also support VRAM_SEG
    if (config.supportsVram) {
        EXPECT_TRUE(std::find(supported_mems.begin(), supported_mems.end(), VRAM_SEG) !=
                    supported_mems.end());
    }
}

TEST_P(objParamTestFixture, RegisterMemoryObjSeg) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 42;
    mem_desc.metaInfo = "test-object-key";

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, OBJ_SEG, metadata);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(metadata, nullptr);

    status = objEngine_->deregisterMem(metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_P(objParamTestFixture, RegisterMemoryObjSegWithoutKey) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 99;
    mem_desc.metaInfo = ""; // Empty key - engine will generate a key

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, OBJ_SEG, metadata);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(metadata, nullptr);

    status = objEngine_->deregisterMem(metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_P(objParamTestFixture, RegisterMemoryDramSeg) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 123;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, DRAM_SEG, metadata);

    EXPECT_EQ(status, NIXL_SUCCESS);
    if (GetParam().supportsVram) {
        EXPECT_NE(metadata, nullptr);
    } else {
        EXPECT_EQ(metadata, nullptr);
    }

    if (metadata) {
        status = objEngine_->deregisterMem(metadata);
        EXPECT_EQ(status, NIXL_SUCCESS);
    }
}

TEST_P(objParamTestFixture, RegisterMemoryVramSeg) {
    if (!GetParam().supportsVram) {
        GTEST_SKIP() << "Test requires VRAM support";
    }
    nixlBlobDesc mem_desc;
    mem_desc.devId = 123;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = objEngine_->registerMem(mem_desc, VRAM_SEG, metadata);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(metadata, nullptr);

    status = objEngine_->deregisterMem(metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_P(objParamTestFixture, NullHandlePostXfer) {
    nixlBackendReqH *handle = nullptr;
    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);
    nixl_status_t status =
        objEngine_->postXfer(NIXL_WRITE, local_descs, remote_descs, "", handle, nullptr);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST_P(objParamTestFixture, NullHandleCheckXfer) {
    nixl_status_t status = objEngine_->checkXfer(nullptr);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST_P(objParamTestFixture, NullHandleReleaseReqH) {
    nixl_status_t status = objEngine_->releaseReqH(nullptr);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

/** Verify that deregisterMem(nullptr) returns NIXL_SUCCESS (no-op). */
TEST_P(objParamTestFixture, DeregisterMemNull) {
    nixl_status_t status = objEngine_->deregisterMem(nullptr);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_P(objParamTestFixture, WriteTransfer) {
    testTransferWithSize(NIXL_WRITE, 1024, "-" + GetParam().name);
}

TEST_P(objParamTestFixture, ReadTransfer) {
    testTransferWithSize(NIXL_READ, 1024, "-" + GetParam().name);
}

TEST_P(objParamTestFixture, MultiDescriptorWrite) {
    testMultiDescriptorWithSizes(NIXL_WRITE, 1024, 1024, "-" + GetParam().name);
}

TEST_P(objParamTestFixture, MultiDescriptorRead) {
    testMultiDescriptorWithSizes(NIXL_READ, 1024, 1024, "-" + GetParam().name);
}

TEST_P(objParamTestFixture, TransferFailureHandling) {
    testTransferFailure(NIXL_WRITE, 1024, "-" + GetParam().name);
}

TEST_P(objParamTestFixture, CheckObjectExists) {
    std::string suffix = "-" + GetParam().name;
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-key-1" + suffix));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-key-2" + suffix));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-key-3" + suffix));
    std::vector<nixl_query_resp_t> resp;
    objEngine_->queryMem(descs, resp);

    EXPECT_EQ(resp.size(), 3);
    EXPECT_EQ(resp[0].has_value(), true);
    EXPECT_EQ(resp[1].has_value(), true);
    EXPECT_EQ(resp[2].has_value(), true);

    EXPECT_EQ(mockS3Client_->getCheckedKeys().size(), 3);
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("test-key-1" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("test-key-2" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("test-key-3" + suffix));
}

// Instantiate parameterized tests for standard client configurations.
// Accel and Dell configurations are instantiated in obj_dell.cpp (built only
// when the cuobjclient library is available).
INSTANTIATE_TEST_SUITE_P(ObjClientTests,
                         objParamTestFixture,
                         testing::Values(standardConfig, crtConfig),
                         [](const testing::TestParamInfo<ObjTestConfig> &info) {
                             return info.param.name;
                         });

// Specialized tests for non-parameterized fixture
TEST_F(objTestFixture, CancelTransfer) {
    mockS3Client_->setSimulateSuccess(true);

    nixlBlobDesc local_desc, remote_desc;
    local_desc.devId = 1;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-cancel-key";

    nixlBackendMD *local_metadata = nullptr;
    nixlBackendMD *remote_metadata = nullptr;

    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    std::vector<char> test_buffer(1024);
    nixlMetaDesc local_meta_desc(
        reinterpret_cast<uintptr_t>(test_buffer.data()), test_buffer.size(), 1);
    local_descs.addDesc(local_meta_desc);

    nixlMetaDesc remote_meta_desc(0, test_buffer.size(), 2);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;

    ASSERT_EQ(objEngine_->prepXfer(
                  NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
              NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    nixl_status_t status = objEngine_->postXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_IN_PROG);
    EXPECT_EQ(mockS3Client_->getPendingCount(), 1);

    status = objEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Cancel the transfer before completion by releasing the handle
    // This simulates the cancellation behavior from nixlAgent::releaseXferReq
    status = objEngine_->releaseReqH(handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    mockS3Client_->execAsync();

    // After cancellation/release, we can't check the transfer status anymore
    // as the handle has been released. This verifies that cancelling pending
    // async tasks is handled correctly by properly cleaning up resources.
    status = objEngine_->deregisterMem(local_metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
    status = objEngine_->deregisterMem(remote_metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(objTestFixture, ReadFromOffset) {
    mockS3Client_->setSimulateSuccess(true);

    std::vector<char> test_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-offset-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    const size_t offset = 256;
    const size_t length = 512;
    nixlMetaDesc local_meta_desc(
        reinterpret_cast<uintptr_t>(test_buffer.data()), length, local_desc.devId);
    nixlMetaDesc remote_meta_desc(offset, length, remote_desc.devId);
    local_descs.addDesc(local_meta_desc);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(objEngine_->prepXfer(
                  NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
              NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    nixl_status_t status = objEngine_->postXfer(
        NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_IN_PROG);
    EXPECT_EQ(mockS3Client_->getPendingCount(), 1);
    status = objEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    mockS3Client_->execAsync();
    status = objEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(test_buffer[0], 'A' + (offset % 26));

    objEngine_->releaseReqH(handle);
    objEngine_->deregisterMem(local_metadata);
    objEngine_->deregisterMem(remote_metadata);
}

// CRT-specific tests for threshold behavior.
// crtMinLimit is set to 5 MiB (the S3 minimum part size) so that partSize is
// not clamped by the CRT SDK and MPU is properly exercised for objects above
// the threshold.
class objCrtTestFixture : public objTestBase, public testing::Test {
protected:
    static constexpr size_t kCrtMinLimit = 5242880; // 5 MiB

    void
    SetUp() override {
        setupEngine("test-crt-agent", {{"crtMinLimit", std::to_string(kCrtMinLimit)}});
    }
};

TEST_F(objCrtTestFixture, TransferAboveThreshold) {
    // 6 MiB: above the 5 MiB CRT threshold, triggers MPU (two parts: 5 MiB + 1 MiB)
    testTransferWithSize(NIXL_WRITE, 6291456, "-crt-above");
}

TEST_F(objCrtTestFixture, TransferBelowThreshold) {
    // 1 MiB: below the 5 MiB CRT threshold, uses standard S3 client
    testTransferWithSize(NIXL_READ, 1048576, "-crt-below");
}

TEST_F(objCrtTestFixture, MixedSizeThreshold) {
    // Mixed: 1 MiB (standard client) + 6 MiB (CRT client via MPU)
    testMultiDescriptorWithSizes(NIXL_WRITE, 1048576, 6291456, "-crt-mixed");
}

} // namespace gtest::obj
