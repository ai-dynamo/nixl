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
#include <chrono>
#include <functional>
#include <map>
#include <thread>

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
    bool simulateError_ = false;
    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::vector<std::function<void()>> pendingCallbacks_;
    std::set<std::string> checkedKeys_;
    std::map<std::string, bool> keyOutcomes_;
    std::map<std::string, std::chrono::milliseconds> keyDelays_;
    std::set<std::string> keyErrors_;

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
    setSimulateError(bool error) {
        simulateError_ = error;
    }

    void
    setKeyOutcome(const std::string &key, bool success) {
        keyOutcomes_[key] = success;
    }

    void
    setKeyDelay(const std::string &key, std::chrono::milliseconds delay) {
        keyDelays_[key] = delay;
    }

    void
    setKeyError(const std::string &key) {
        keyErrors_.insert(key);
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

    void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override {
        std::string key_str(key);
        checkedKeys_.insert(key_str);

        // Per-key error overrides global simulateError_
        bool simulate_error = keyErrors_.count(key_str) > 0 || simulateError_;

        // Per-key outcome overrides global simulateSuccess_
        auto outcome_it = keyOutcomes_.find(key_str);
        bool success = (outcome_it != keyOutcomes_.end()) ? outcome_it->second : simulateSuccess_;

        // Per-key delay (defaults to no delay)
        std::chrono::milliseconds delay{0};
        auto delay_it = keyDelays_.find(key_str);
        if (delay_it != keyDelays_.end()) {
            delay = delay_it->second;
        }

        if (executor_) {
            executor_->Submit([callback, success, simulate_error, delay]() {
                if (delay.count() > 0) {
                    std::this_thread::sleep_for(delay);
                }
                callback(simulate_error ? std::optional<bool>(std::nullopt) :
                                          std::optional<bool>(success));
            });
        } else {
            if (delay.count() > 0) {
                std::this_thread::sleep_for(delay);
            }
            callback(simulate_error ? std::optional<bool>(std::nullopt) :
                                      std::optional<bool>(success));
        }
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
                       put_object_callback_t callback) {
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
                       get_object_callback_t callback) {
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

// Mock that invokes the checkObjectExistsAsync callback twice to test
// the exact-once guard in engine_impl.cpp's queryMem.
class doubleCallbackMockS3Client : public iS3Client {
private:
    std::shared_ptr<asioThreadPoolExecutor> executor_;

public:
    doubleCallbackMockS3Client() = default;

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override {
        executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
    }

    void
    putObjectAsync(std::string_view, uintptr_t, size_t, size_t, put_object_callback_t callback)
        override {
        callback(true);
    }

    void
    getObjectAsync(std::string_view, uintptr_t, size_t, size_t, get_object_callback_t callback)
        override {
        callback(true);
    }

    void
    checkObjectExistsAsync(std::string_view, check_object_callback_t callback) override {
        if (executor_) {
            executor_->Submit([callback]() {
                callback(true); // First invocation
                callback(true); // Second invocation — should be a no-op
            });
        } else {
            callback(true);
            callback(true);
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
    nixl_status_t status = objEngine_->queryMem(descs, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    EXPECT_EQ(resp.size(), 3);
    EXPECT_EQ(resp[0].has_value(), true);
    EXPECT_EQ(resp[1].has_value(), true);
    EXPECT_EQ(resp[2].has_value(), true);

    EXPECT_EQ(mockS3Client_->getCheckedKeys().size(), 3);
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("test-key-1" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("test-key-2" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("test-key-3" + suffix));
}

TEST_P(objParamTestFixture, CheckObjectExistsAsyncOrdering) {
    std::string suffix = "-" + GetParam().name;

    // Single combined queryMem call with per-key outcomes and staggered delays
    // so responses complete out of order, exercising the slot-mapping logic.
    nixl_reg_dlist_t combined_descs(OBJ_SEG);
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-1" + suffix)); // exists
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-2" + suffix)); // missing
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-3" + suffix)); // exists
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-4" + suffix)); // missing
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-5" + suffix)); // exists

    // Drive per-key outcomes so exist/missing alternate
    mockS3Client_->setKeyOutcome("async-key-1" + suffix, true);
    mockS3Client_->setKeyOutcome("async-key-2" + suffix, false);
    mockS3Client_->setKeyOutcome("async-key-3" + suffix, true);
    mockS3Client_->setKeyOutcome("async-key-4" + suffix, false);
    mockS3Client_->setKeyOutcome("async-key-5" + suffix, true);

    // Stagger delays: earlier keys complete later to force out-of-order completion
    mockS3Client_->setKeyDelay("async-key-1" + suffix, std::chrono::milliseconds(50));
    mockS3Client_->setKeyDelay("async-key-2" + suffix, std::chrono::milliseconds(40));
    mockS3Client_->setKeyDelay("async-key-3" + suffix, std::chrono::milliseconds(30));
    mockS3Client_->setKeyDelay("async-key-4" + suffix, std::chrono::milliseconds(20));
    mockS3Client_->setKeyDelay("async-key-5" + suffix, std::chrono::milliseconds(10));

    std::vector<nixl_query_resp_t> combined_resp;
    nixl_status_t status = objEngine_->queryMem(combined_descs, combined_resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    ASSERT_EQ(combined_resp.size(), 5);

    // Assert each response corresponds to the descriptor at that index
    EXPECT_TRUE(combined_resp[0].has_value()) << "async-key-1 should exist";
    EXPECT_FALSE(combined_resp[1].has_value()) << "async-key-2 should not exist";
    EXPECT_TRUE(combined_resp[2].has_value()) << "async-key-3 should exist";
    EXPECT_FALSE(combined_resp[3].has_value()) << "async-key-4 should not exist";
    EXPECT_TRUE(combined_resp[4].has_value()) << "async-key-5 should exist";

    // Verify all keys were checked
    EXPECT_EQ(mockS3Client_->getCheckedKeys().size(), 5);
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("async-key-1" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("async-key-2" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("async-key-3" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("async-key-4" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("async-key-5" + suffix));
}

TEST_P(objParamTestFixture, CheckObjectExistsAsyncFailure) {
    mockS3Client_->setSimulateSuccess(false);

    std::string suffix = "-" + GetParam().name;
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "fail-key-1" + suffix));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "fail-key-2" + suffix));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    EXPECT_EQ(resp.size(), 2);

    // When simulateSuccess is false, objects should appear as non-existent
    EXPECT_EQ(resp[0].has_value(), false);
    EXPECT_EQ(resp[1].has_value(), false);

    EXPECT_EQ(mockS3Client_->getCheckedKeys().size(), 2);
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("fail-key-1" + suffix));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("fail-key-2" + suffix));
}

TEST_P(objParamTestFixture, CheckObjectExistsAsyncEmptyList) {
    nixl_reg_dlist_t descs(OBJ_SEG);
    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    EXPECT_EQ(resp.size(), 0);
    EXPECT_EQ(mockS3Client_->getCheckedKeys().size(), 0);
}

TEST_P(objParamTestFixture, CheckObjectExistsAsyncRequestError) {
    // Simulate a transient error (e.g. 5xx / auth failure) that should
    // propagate as NIXL_ERR_BACKEND rather than being treated as "not found".
    mockS3Client_->setSimulateError(true);

    std::string suffix = "-" + GetParam().name;
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "error-key-1" + suffix));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "error-key-2" + suffix));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs, resp);

    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(mockS3Client_->getCheckedKeys().size(), 2);
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

// ---------------------------------------------------------------------------
// Exact-once callback guard tests
// ---------------------------------------------------------------------------

// Fixture that injects the double-callback mock so that every
// checkObjectExistsAsync invocation fires the callback twice.
class objDoubleCallbackFixture : public testing::Test {
protected:
    std::unique_ptr<nixlObjEngine> objEngine_;
    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;

    void
    SetUp() override {
        initParams_.localAgent = "test-double-callback";
        initParams_.type = "OBJ";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

        auto mockClient = std::make_shared<doubleCallbackMockS3Client>();
        objEngine_ = std::make_unique<nixlObjEngine>(&initParams_, mockClient);
    }
};

TEST_F(objDoubleCallbackFixture, QueryMemDuplicateCallback) {
    // Verify that a duplicate callback invocation from the SDK doesn't cause
    // exceptions or corrupt results.  The exact-once guard in engine_impl.cpp
    // (completed->exchange(true)) should make the second invocation a no-op.
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "dup-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "dup-key-2"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "dup-key-3"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    EXPECT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_TRUE(resp[2].has_value());
}

// ---------------------------------------------------------------------------
// queryMem robustness tests
// ---------------------------------------------------------------------------

TEST_F(objTestFixture, QueryMemClearsStaleResp) {
    // First queryMem: 3 keys, all exist
    mockS3Client_->setSimulateSuccess(true);

    nixl_reg_dlist_t descs1(OBJ_SEG);
    descs1.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-1"));
    descs1.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-2"));
    descs1.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-3"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs1, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_TRUE(resp[2].has_value());

    // Second queryMem: 2 keys, both missing — reuses the same resp vector.
    // resp.assign() must clear the 3 stale entries from the first call.
    mockS3Client_->setSimulateSuccess(false);

    nixl_reg_dlist_t descs2(OBJ_SEG);
    descs2.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-4"));
    descs2.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-5"));

    status = objEngine_->queryMem(descs2, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // resp must reflect the second call only: 2 items, both not found
    EXPECT_EQ(resp.size(), 2);
    EXPECT_FALSE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
}

TEST_F(objTestFixture, QueryMemMixedPerKeyErrors) {
    // Mix of outcomes in a single queryMem call: some keys exist, some are
    // missing, and one returns an error (std::nullopt).  The error should
    // cause NIXL_ERR_BACKEND while the other slots are still populated.
    mockS3Client_->setKeyOutcome("mix-key-1", true); // exists
    mockS3Client_->setKeyOutcome("mix-key-2", false); // missing
    mockS3Client_->setKeyError("mix-key-3"); // transient error

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "mix-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "mix-key-2"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "mix-key-3"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs, resp);

    // Should fail because mix-key-3 errored
    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    // Non-error slots should still carry the correct values
    ASSERT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].has_value()) << "mix-key-1 should exist";
    EXPECT_FALSE(resp[1].has_value()) << "mix-key-2 should not exist";
    EXPECT_FALSE(resp[2].has_value()) << "mix-key-3 errored, should be nullopt";

    // All 3 keys should still have been checked
    EXPECT_EQ(mockS3Client_->getCheckedKeys().size(), 3);
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("mix-key-1"));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("mix-key-2"));
    EXPECT_TRUE(mockS3Client_->getCheckedKeys().count("mix-key-3"));
}

// ---------------------------------------------------------------------------
// Exception path tests
// ---------------------------------------------------------------------------

// Mock that throws an exception during checkObjectExistsAsync to test the
// exception handling path in engine_impl.cpp's queryMem.
class exceptionThrowingMockS3Client : public iS3Client {
private:
    std::shared_ptr<asioThreadPoolExecutor> executor_;
    int throw_after_calls_ = 0;
    int call_count_ = 0;

public:
    exceptionThrowingMockS3Client() = default;

    exceptionThrowingMockS3Client(
        [[maybe_unused]] nixl_b_params_t *custom_params,
        std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr,
        int throw_after_calls = 0)
        : throw_after_calls_(throw_after_calls) {
        if (executor) {
            executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
        }
    }

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override {
        executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
    }

    void
    putObjectAsync(std::string_view, uintptr_t, size_t, size_t, put_object_callback_t callback)
        override {
        callback(true);
    }

    void
    getObjectAsync(std::string_view, uintptr_t, size_t, size_t, get_object_callback_t callback)
        override {
        callback(true);
    }

    void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override {
        call_count_++;
        if (call_count_ > throw_after_calls_) {
            throw std::runtime_error("Simulated exception in checkObjectExistsAsync");
        }

        if (executor_) {
            executor_->Submit([callback]() { callback(true); });
        } else {
            callback(true);
        }
    }
};

// Fixture for exception path tests
class objExceptionFixture : public testing::Test {
protected:
    std::unique_ptr<nixlObjEngine> objEngine_;
    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;

    void
    SetUp() override {
        initParams_.localAgent = "test-exception";
        initParams_.type = "OBJ";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    }
};

TEST_F(objExceptionFixture, QueryMemExceptionDuringLaunch) {
    // Test that exceptions during async request launch are handled gracefully.
    // The mock throws after N calls, triggering the catch block in queryMem.
    // This verifies the fix for the use-after-free race: the catch block should
    // wait for all in-flight callbacks to complete before returning.
    auto mockClient = std::make_shared<exceptionThrowingMockS3Client>(&customParams_, nullptr, 2);
    objEngine_ = std::make_unique<nixlObjEngine>(&initParams_, mockClient);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-2"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-3"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-4"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs, resp);

    // Should return error due to exception during launch
    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    // Response vector should be sized correctly (all slots initialized to nullopt)
    EXPECT_EQ(resp.size(), 4);
}

TEST_F(objExceptionFixture, QueryMemExceptionOnFirstCall) {
    // Test exception on the very first async request launch.
    auto mockClient = std::make_shared<exceptionThrowingMockS3Client>(&customParams_, nullptr, 0);
    objEngine_ = std::make_unique<nixlObjEngine>(&initParams_, mockClient);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-immediate-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-immediate-2"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = objEngine_->queryMem(descs, resp);

    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(resp.size(), 2);
}

} // namespace gtest::obj
