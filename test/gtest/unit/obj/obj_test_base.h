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
#ifndef NIXL_TEST_GTEST_UNIT_OBJ_OBJ_TEST_BASE_H
#define NIXL_TEST_GTEST_UNIT_OBJ_OBJ_TEST_BASE_H

#include <gtest/gtest.h>
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include <memory>
#include <string>
#include <algorithm>
#include <vector>
#include <functional>
#include <chrono>
#include <thread>
#include <map>
#include <set>
#include <optional>
#include <span>

#include "s3/client.h"
#include "obj_backend.h"
#include "obj_executor.h"
#include "mock_client_registry.h"

namespace gtest::obj {

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
    std::vector<std::string> requestedKeys_;
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

    /** @brief Record the WRITE key and defer its configured result until execAsync(). */
    void
    putObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   put_object_callback_t callback) override {
        std::string key_str(key);
        requestedKeys_.push_back(key_str);

        // Per-key error overrides global simulateError_
        bool simulate_error = keyErrors_.count(key_str) > 0 || simulateError_;

        // Per-key outcome overrides global simulateSuccess_
        bool success = simulateSuccess_;
        auto outcome_it = keyOutcomes_.find(key_str);
        if (outcome_it != keyOutcomes_.end()) {
            success = outcome_it->second;
        }

        // Per-key delay (defaults to no delay)
        std::chrono::milliseconds delay{0};
        auto delay_it = keyDelays_.find(key_str);
        if (delay_it != keyDelays_.end()) {
            delay = delay_it->second;
        }

        pendingCallbacks_.push_back(
            /** @brief Apply the configured delay and report the mock WRITE result. */
            [callback, success, simulate_error, delay]() {
                if (delay.count() > 0) {
                    std::this_thread::sleep_for(delay);
                }
                callback(simulate_error ? false : success);
            });
    }

    /** @brief Record the READ key and defer its offset-dependent payload and result. */
    void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) override {
        std::string key_str(key);
        requestedKeys_.push_back(key_str);

        // Per-key error overrides global simulateError_
        bool simulate_error = keyErrors_.count(key_str) > 0 || simulateError_;

        // Per-key outcome overrides global simulateSuccess_
        bool success = simulateSuccess_;
        auto outcome_it = keyOutcomes_.find(key_str);
        if (outcome_it != keyOutcomes_.end()) {
            success = outcome_it->second;
        }

        // Per-key delay (defaults to no delay)
        std::chrono::milliseconds delay{0};
        auto delay_it = keyDelays_.find(key_str);
        if (delay_it != keyDelays_.end()) {
            delay = delay_it->second;
        }

        pendingCallbacks_.push_back(
            /** @brief Fill successful READs with the object pattern, then report completion. */
            [callback, data_ptr, data_len, offset, success, simulate_error, delay]() {
                if (delay.count() > 0) {
                    std::this_thread::sleep_for(delay);
                }
                bool final_success = simulate_error ? false : success;
                if (final_success && data_ptr && data_len > 0) {
                    char *buffer = reinterpret_cast<char *>(data_ptr);
                    for (size_t i = 0; i < data_len; ++i) {
                        buffer[i] = static_cast<char>('A' + ((i + offset) % 26));
                    }
                }
                callback(final_success);
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
        if (executor_) {
            for (auto &callback : pendingCallbacks_) {
                executor_->Submit([callback]() { callback(); });
            }
            pendingCallbacks_.clear();
            executor_->waitUntilIdle();
        } else {
            for (auto &callback : pendingCallbacks_) {
                callback();
            }
            pendingCallbacks_.clear();
        }
    }

    size_t
    getPendingCount() const {
        return pendingCallbacks_.size();
    }

    /** @brief Return all READ and WRITE keys in submission order, including duplicates. */
    const std::vector<std::string> &
    getRequestedKeys() const {
        return requestedKeys_;
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

    /**
     * @brief Create a test engine with the registered mock and an optional CRT mock.
     * @param agentName Local agent name used by transfer preparation.
     * @param params Backend settings that select the engine and CRT threshold.
     * @param crt_client Separate CRT mock for observing descriptor routing.
     */
    void
    setupEngine(const std::string &agentName,
                nixl_b_params_t params = {},
                std::shared_ptr<mockS3Client> crt_client = nullptr) {
        customParams_ = params;
        initParams_.localAgent = agentName;
        initParams_.type = "OBJ";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

        auto type_it = customParams_.find("type");
        std::string type = (type_it != customParams_.end()) ? type_it->second : "";
        auto mock = mockClientRegistry::instance().create(type);
        if (!type.empty() && !mock) {
            throw std::runtime_error("No registered mock for engine type '" + type + "'");
        }
        mockS3Client_ = mock ? mock : std::make_shared<mockS3Client>();
        objEngine_ = std::make_unique<nixlObjEngine>(&initParams_, mockS3Client_, crt_client);
    }

    /**
     * @brief Check every byte against the mock's repeating alphabet pattern.
     * @param buffer Destination bytes that the READ must fill.
     * @param offset Starting byte offset in the mock object.
     */
    static void
    expectReadPattern(std::span<const char> buffer, size_t offset = 0) {
        for (size_t i = 0; i < buffer.size(); ++i) {
            ASSERT_EQ(buffer[i], static_cast<char>('A' + ((i + offset) % 26)))
                << "Unexpected read data at byte " << i;
        }
    }

    /** @brief Complete one transfer and verify the full payload for a READ. */
    void
    testTransferWithSize(nixl_xfer_op_t operation,
                         size_t buffer_size,
                         const std::string &key_suffix = "") {
        mockS3Client_->setSimulateSuccess(true);

        std::vector<char> test_buffer(buffer_size);

        nixlBlobDesc local_desc = {}, remote_desc = {};
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
            expectReadPattern(test_buffer);
        }

        objEngine_->releaseReqH(handle);
        objEngine_->deregisterMem(local_metadata);
        objEngine_->deregisterMem(remote_metadata);
    }

    /** @brief Complete two descriptors and verify each READ buffer independently. */
    void
    testMultiDescriptorWithSizes(nixl_xfer_op_t operation,
                                 size_t size0,
                                 size_t size1,
                                 const std::string &key_suffix = "") {
        mockS3Client_->setSimulateSuccess(true);

        std::vector<char> test_buffer0(size0);
        std::vector<char> test_buffer1(size1);
        nixlBlobDesc local_desc0 = {}, local_desc1 = {};
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

        nixlBlobDesc remote_desc0 = {}, remote_desc1 = {};
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
            expectReadPattern(test_buffer0);
            expectReadPattern(test_buffer1);
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

        nixlBlobDesc local_desc = {};
        local_desc.addr = reinterpret_cast<uintptr_t>(test_buffer.data());
        local_desc.len = test_buffer.size();
        local_desc.devId = 1;
        nixlBackendMD *local_metadata = nullptr;
        ASSERT_EQ(objEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

        nixlBlobDesc remote_desc = {};
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
        int max_polls = 100;
        int poll_count = 0;
        do {
            status = objEngine_->checkXfer(handle);
            poll_count++;
        } while (status == NIXL_IN_PROG && poll_count < max_polls);
        EXPECT_NE(status, NIXL_IN_PROG) << "Polling timed out waiting for terminal status";
        EXPECT_NE(status, NIXL_SUCCESS); // Should not succeed

        // Verify failed read did not mutate destination buffer
        if (operation == NIXL_READ) {
            EXPECT_TRUE(std::all_of(
                test_buffer.begin(), test_buffer.end(), [](char c) { return c == 'Z'; }));
        }

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

} // namespace gtest::obj

#endif // NIXL_TEST_GTEST_UNIT_OBJ_OBJ_TEST_BASE_H
