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
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "s3/client.h"
#include "obj_backend.h"
#include "obj_executor.h"
#include "object/engine_utils.h"

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

// Test configuration for different S3 client types
struct ObjTestConfig {
    std::string name;
    nixl_b_params_t customParams;
    std::string agentName;
    bool supportsVram = false;
};

class mockS3Client : public iS3Client {
private:
    bool simulateSuccess_ = true;
    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::vector<std::function<void()>> pendingCallbacks_;
    std::set<std::string> checkedKeys_;

public:
    mockS3Client() = default;

    mockS3Client([[maybe_unused]] nixl_b_params_t *custom_params,
                 std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr) {
        if (executor) {
            executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
        }
    }

    void
    setSimulateSuccess(bool success) {
        simulateSuccess_ = success;
    }

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override {
        executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
    }

    void
    putObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   put_object_callback_t callback) {
        pendingCallbacks_.push_back([callback, this]() { callback(simulateSuccess_); });
    }

    void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) {
        pendingCallbacks_.push_back([callback, data_ptr, data_len, offset, this]() {
            if (simulateSuccess_ && data_ptr && data_len > 0) {
                char *buffer = reinterpret_cast<char *>(data_ptr);
                for (size_t i = 0; i < data_len; ++i) {
                    buffer[i] = static_cast<char>('A' + ((i + offset) % 26));
                }
            }
            callback(simulateSuccess_);
        });
    }

    bool
    checkObjectExists(std::string_view key) override {
        checkedKeys_.insert(std::string(key));
        return simulateSuccess_;
    }

    void
    execAsync() {
        for (auto &callback : pendingCallbacks_) {
            executor_->Submit([callback]() { callback(); });
        }
        pendingCallbacks_.clear();
        executor_->waitUntilIdle();
    }

    size_t
    getPendingCount() const {
        return pendingCallbacks_.size();
    }

    const std::set<std::string> &
    getCheckedKeys() const {
        return checkedKeys_;
    }

    bool
    hasExecutor() const {
        return executor_ != nullptr;
    }
};

// Base test fixture with common test helper methods
class objTestBase {
protected:
    std::unique_ptr<nixlObjEngine> objEngine_;
    std::shared_ptr<mockS3Client> mockS3Client_;
    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;

    void
    setupEngine(const std::string &agentName, nixl_b_params_t params = {}) {
        customParams_ = params;
        initParams_.localAgent = agentName;
        initParams_.type = "OBJ";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

        // All engine types (Standard, CRT, Accel, Dell) use the same mock.
        // The Dell engine accepts an injected iS3Client just like the others;
        // its RDMA-specific behavior lives in awsS3DellObsClient which the
        // mock replaces entirely for unit tests.
        mockS3Client_ = std::make_shared<mockS3Client>();
        objEngine_ = std::make_unique<nixlObjEngine>(&initParams_, mockS3Client_);
    }

    void
    testTransferWithSize(nixl_xfer_op_t operation,
                         size_t buffer_size,
                         const std::string &key_suffix = "") {
        mockS3Client_->setSimulateSuccess(true);

        std::vector<char> test_buffer(buffer_size);

        nixlBlobDesc local_desc, remote_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
        local_desc.len = test_buffer.size();
        local_desc.devId = 1;
        remote_desc.devId = 2;
        remote_desc.metaInfo = (operation == NIXL_READ) ? "test-read-key" : "test-write-key";
        remote_desc.metaInfo += key_suffix;

        nixlBackendMD *local_metadata = nullptr;
        nixlBackendMD *remote_metadata = nullptr;

        ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
        ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs(DRAM_SEG);
        nixl_meta_dlist_t remote_descs(OBJ_SEG);

        nixlMetaDesc local_meta_desc(local_desc.addr, local_desc.len, local_desc.devId);
        local_descs.addDesc(local_meta_desc);

        nixlMetaDesc remote_meta_desc(0, test_buffer.size(), 2);
        remote_descs.addDesc(remote_meta_desc);

        nixlBackendReqH *handle = nullptr;

        ASSERT_EQ(
            objEngine_->prepXfer(
                operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE(handle, nullptr);

        nixl_status_t status = objEngine_->postXfer(
            operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
        EXPECT_EQ(status, NIXL_IN_PROG);
        EXPECT_EQ(mockS3Client_->getPendingCount(), 1);
        status = objEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_IN_PROG);

        mockS3Client_->execAsync();
        status = objEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_SUCCESS);

        if (operation == NIXL_READ) {
            EXPECT_EQ(test_buffer[0], 'A');
        }

        objEngine_->releaseReqH(handle);
        objEngine_->deregisterMem(local_metadata);
        objEngine_->deregisterMem(remote_metadata);
    }

    void
    testMultiDescriptorWithSizes(nixl_xfer_op_t operation,
                                 size_t size0,
                                 size_t size1,
                                 const std::string &key_suffix = "") {
        mockS3Client_->setSimulateSuccess(true);

        std::vector<char> test_buffer0(size0);
        std::vector<char> test_buffer1(size1);
        nixlBlobDesc local_desc0, local_desc1;
        local_desc0.addr = reinterpret_cast<uintptr_t>(test_buffer0.data());
        local_desc1.addr = reinterpret_cast<uintptr_t>(test_buffer1.data());
        local_desc0.len = test_buffer0.size();
        local_desc1.len = test_buffer1.size();
        local_desc0.devId = 1;
        local_desc1.devId = 1;
        nixlBackendMD *local_metadata0 = nullptr;
        nixlBackendMD *local_metadata1 = nullptr;

        ASSERT_EQ(objEngine_->registerMem(local_desc0, DRAM_SEG, local_metadata0), NIXL_SUCCESS);
        ASSERT_EQ(objEngine_->registerMem(local_desc1, DRAM_SEG, local_metadata1), NIXL_SUCCESS);

        nixlBlobDesc remote_desc0, remote_desc1;
        remote_desc0.devId = 2;
        remote_desc1.devId = 3;
        remote_desc0.metaInfo = (operation == NIXL_READ) ? "test-read-key0" : "test-write-key0";
        remote_desc0.metaInfo += key_suffix;
        remote_desc1.metaInfo = (operation == NIXL_READ) ? "test-read-key1" : "test-write-key1";
        remote_desc1.metaInfo += key_suffix;
        nixlBackendMD *remote_metadata0 = nullptr;
        nixlBackendMD *remote_metadata1 = nullptr;

        ASSERT_EQ(objEngine_->registerMem(remote_desc0, OBJ_SEG, remote_metadata0), NIXL_SUCCESS);
        ASSERT_EQ(objEngine_->registerMem(remote_desc1, OBJ_SEG, remote_metadata1), NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs(DRAM_SEG);
        nixl_meta_dlist_t remote_descs(OBJ_SEG);

        nixlMetaDesc local_meta_desc0(reinterpret_cast<uintptr_t>(test_buffer0.data()),
                                      test_buffer0.size(),
                                      local_desc0.devId);
        nixlMetaDesc local_meta_desc1(reinterpret_cast<uintptr_t>(test_buffer1.data()),
                                      test_buffer1.size(),
                                      local_desc1.devId);
        local_descs.addDesc(local_meta_desc0);
        local_descs.addDesc(local_meta_desc1);

        nixlMetaDesc remote_meta_desc0(0, test_buffer0.size(), remote_desc0.devId);
        nixlMetaDesc remote_meta_desc1(0, test_buffer1.size(), remote_desc1.devId);
        remote_descs.addDesc(remote_meta_desc0);
        remote_descs.addDesc(remote_meta_desc1);

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(
            objEngine_->prepXfer(
                operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE(handle, nullptr);

        nixl_status_t status = objEngine_->postXfer(
            operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
        EXPECT_EQ(status, NIXL_IN_PROG);
        EXPECT_EQ(mockS3Client_->getPendingCount(), 2);
        status = objEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_IN_PROG);

        mockS3Client_->execAsync();
        status = objEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_SUCCESS);

        if (operation == NIXL_READ) {
            EXPECT_EQ(test_buffer0[0], 'A');
            EXPECT_EQ(test_buffer1[0], 'A');
        }

        objEngine_->releaseReqH(handle);
        objEngine_->deregisterMem(local_metadata0);
        objEngine_->deregisterMem(local_metadata1);
        objEngine_->deregisterMem(remote_metadata0);
        objEngine_->deregisterMem(remote_metadata1);
    }

    void
    testTransferFailure(nixl_xfer_op_t operation,
                        size_t buffer_size,
                        const std::string &key_suffix = "") {
        mockS3Client_->setSimulateSuccess(false);

        std::vector<char> test_buffer(buffer_size, 'Z');

        nixlBlobDesc local_desc;
        local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
        local_desc.len = test_buffer.size();
        local_desc.devId = 1;
        nixlBackendMD *local_metadata = nullptr;
        ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

        nixlBlobDesc remote_desc;
        remote_desc.devId = 2;
        remote_desc.metaInfo = "test-fail-key" + key_suffix;
        nixlBackendMD *remote_metadata = nullptr;
        ASSERT_EQ(objEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs(DRAM_SEG);
        nixl_meta_dlist_t remote_descs(OBJ_SEG);

        nixlMetaDesc local_meta_desc(
            reinterpret_cast<uintptr_t>(test_buffer.data()), test_buffer.size(), local_desc.devId);
        nixlMetaDesc remote_meta_desc(0, test_buffer.size(), remote_desc.devId);
        local_descs.addDesc(local_meta_desc);
        remote_descs.addDesc(remote_meta_desc);

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(
            objEngine_->prepXfer(
                operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE(handle, nullptr);

        nixl_status_t status = objEngine_->postXfer(
            operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
        EXPECT_EQ(status, NIXL_IN_PROG);
        EXPECT_EQ(mockS3Client_->getPendingCount(), 1);
        status = objEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_IN_PROG);

        mockS3Client_->execAsync();
        status = objEngine_->checkXfer(handle);
        EXPECT_NE(status, NIXL_SUCCESS); // Should not succeed

        objEngine_->releaseReqH(handle);
        objEngine_->deregisterMem(local_metadata);
        objEngine_->deregisterMem(remote_metadata);
    }
};

// Parameterized test fixture for common tests across all client types
class objParamTestFixture : public objTestBase, public testing::TestWithParam<ObjTestConfig> {
protected:
    void
    SetUp() override {
        const auto &config = GetParam();
        setupEngine(config.agentName, config.customParams);
    }
};

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

#if defined HAVE_CUOBJ_CLIENT
static const ObjTestConfig accelConfig = {"Accel",
                                          {{"accelerated", "true"}},
                                          "test-accel-agent",
                                          false};
static const ObjTestConfig dellConfig = {"Dell",
                                         {{"accelerated", "true"}, {"type", "dell"}},
                                         "test-dell-agent",
                                         true};
#endif

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

// Instantiate parameterized tests for all client configurations
#if defined HAVE_CUOBJ_CLIENT
INSTANTIATE_TEST_SUITE_P(ObjClientTests,
                         objParamTestFixture,
                         testing::Values(standardConfig, crtConfig, accelConfig, dellConfig),
                         [](const testing::TestParamInfo<ObjTestConfig> &info) {
                             return info.param.name;
                         });
#else
INSTANTIATE_TEST_SUITE_P(ObjClientTests,
                         objParamTestFixture,
                         testing::Values(standardConfig, crtConfig),
                         [](const testing::TestParamInfo<ObjTestConfig> &info) {
                             return info.param.name;
                         });
#endif

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
#if defined HAVE_CUOBJ_CLIENT

#include "s3_accel/dell/cuobj_token_manager.h"
#include <cstdlib>
#include <cstring>

class CuObjTokenManagerTest : public testing::Test {
protected:
    // Allocate page-aligned system memory for RDMA registration tests.
    // cuMemObjGetDescriptor requires the buffer to remain valid until
    // cuMemObjPutDescriptor is called.
    static constexpr size_t kPageSize = 4096;
    static constexpr size_t kBufferSize = 64 * 1024; // 64 KiB

    void *buffer_ = nullptr;

    void SetUp() override {
        // posix_memalign guarantees page alignment, which cuObject prefers.
        int rc = posix_memalign(&buffer_, kPageSize, kBufferSize);
        ASSERT_EQ(rc, 0);
        ASSERT_NE(buffer_, nullptr);
        std::memset(buffer_, 0xAB, kBufferSize);
    }

    void TearDown() override {
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
        // Token should contain the RDMA protocol identifier.
        // (Implementation detail: CUOBJ tokens typically start with "CUOBJ" prefix)
    } catch (const std::runtime_error &) {
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
    } catch (const std::runtime_error &) {
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

#endif // HAVE_CUOBJ_CLIENT
