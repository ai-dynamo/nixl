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
#include <algorithm>
#include <vector>
#include <functional>

#include "s3/client.h"
#include "obj_backend.h"
#include "obj_executor.h"
#include "object/engine_utils.h"
#include "s3_accel/dell/rdma_interface.h"

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
                   put_object_callback_t callback) override {
        pendingCallbacks_.push_back([callback, this]() { callback(simulateSuccess_); });
    }

    void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) override {
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

protected:
    // Make pendingCallbacks_ accessible to derived classes
    std::vector<std::function<void()>> &
    getPendingCallbacks() {
        return pendingCallbacks_;
    }

    // Make simulateSuccess_ accessible to derived classes
    bool
    getSimulateSuccess() const {
        return simulateSuccess_;
    }
};

// Dell-specific mock S3 client with RDMA support
class mockDellS3Client : public mockS3Client, public iDellS3RdmaClient {
public:
    mockDellS3Client() = default;

    mockDellS3Client([[maybe_unused]] nixl_b_params_t *custom_params,
                     std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr)
        : mockS3Client(custom_params, executor) {}

    // Dell-specific RDMA methods
    void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback) override {
        if (rdma_desc.empty()) {
            getPendingCallbacks().push_back([callback]() { callback(false); });
        } else {
            getPendingCallbacks().push_back([callback, this]() { callback(getSimulateSuccess()); });
        }
    }

    void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback) override {
        if (rdma_desc.empty()) {
            getPendingCallbacks().push_back([callback]() { callback(false); });
        } else {
            getPendingCallbacks().push_back([callback, data_ptr, data_len, offset, this]() {
                if (getSimulateSuccess() && data_ptr && data_len > 0) {
                    char *buffer = reinterpret_cast<char *>(data_ptr);
                    for (size_t i = 0; i < data_len; ++i) {
                        buffer[i] = static_cast<char>('A' + ((i + offset) % 26));
                    }
                }
                callback(getSimulateSuccess());
            });
        }
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

        // Use appropriate mock client based on configuration
        if (isDellOBSRequested(&customParams_)) {
            mockS3Client_ = std::make_shared<mockDellS3Client>();
        } else {
            mockS3Client_ = std::make_shared<mockS3Client>();
        }
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

#if defined HAVE_CUOBJ_CLIENT
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

#endif // HAVE_CUOBJ_CLIENT

} // namespace gtest::obj
