/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Microsoft Corporation.
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
#include <asio.hpp>

#include "azure_blob_client.h"
#include "azure_blob_backend.h"

namespace gtest::azure_blob {

class mockBlobClient : public iBlobClient {
private:
    bool simulateSuccess_ = true;
    bool simulateError_ = false;
    std::shared_ptr<asio::thread_pool> executor_;
    std::vector<std::function<void()>> pendingCallbacks_;
    std::set<std::string> checkedKeys_;
    std::map<std::string, bool> keyOutcomes_;
    std::map<std::string, std::chrono::milliseconds> keyDelays_;
    std::set<std::string> keyErrors_;

public:
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
    setExecutor(std::shared_ptr<asio::thread_pool> executor) override {
        executor_ = executor;
    }

    void
    putBlobAsync(std::string_view key,
                 uintptr_t data_ptr,
                 size_t data_len,
                 size_t offset,
                 put_blob_callback_t callback) override {
        pendingCallbacks_.push_back([callback, this]() { callback(simulateSuccess_); });
    }

    void
    getBlobAsync(std::string_view key,
                 uintptr_t data_ptr,
                 size_t data_len,
                 size_t offset,
                 get_blob_callback_t callback) override {
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
    checkBlobExistsAsync(std::string_view key, check_blob_callback_t callback) override {
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

        asio::post(*executor_, [callback, success, simulate_error, delay]() {
            if (delay.count() > 0) {
                std::this_thread::sleep_for(delay);
            }
            callback(simulate_error ? std::optional<bool>(std::nullopt) :
                                      std::optional<bool>(success));
        });
    }

    void
    execAsync() {
        for (auto &callback : pendingCallbacks_) {
            asio::post(*executor_, [callback]() { callback(); });
        }
        pendingCallbacks_.clear();
        executor_->wait();
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

// Mock that fires the checkBlobExistsAsync callback twice to exercise
// the exact-once guard in azure_blob_backend.cpp's queryMem.
class doubleCallbackMockBlobClient : public iBlobClient {
private:
    std::shared_ptr<asio::thread_pool> executor_;

public:
    void
    setExecutor(std::shared_ptr<asio::thread_pool> executor) override {
        executor_ = executor;
    }

    void
    putBlobAsync(std::string_view, uintptr_t, size_t, size_t, put_blob_callback_t callback)
        override {
        callback(true);
    }

    void
    getBlobAsync(std::string_view, uintptr_t, size_t, size_t, get_blob_callback_t callback)
        override {
        callback(true);
    }

    void
    checkBlobExistsAsync(std::string_view, check_blob_callback_t callback) override {
        asio::post(*executor_, [callback]() {
            callback(std::optional<bool>(true)); // First invocation
            callback(std::optional<bool>(true)); // Second invocation - should be a no-op
        });
    }
};

// Mock that throws an exception during checkBlobExistsAsync to test the
// exception handling path in azure_blob_backend.cpp's queryMem.
class exceptionThrowingMockBlobClient : public iBlobClient {
private:
    std::shared_ptr<asio::thread_pool> executor_;
    int throw_after_calls_ = 0;
    int call_count_ = 0;

public:
    explicit exceptionThrowingMockBlobClient(int throw_after_calls = 0)
        : throw_after_calls_(throw_after_calls) {}

    void
    setExecutor(std::shared_ptr<asio::thread_pool> executor) override {
        executor_ = executor;
    }

    void
    putBlobAsync(std::string_view, uintptr_t, size_t, size_t, put_blob_callback_t callback)
        override {
        callback(true);
    }

    void
    getBlobAsync(std::string_view, uintptr_t, size_t, size_t, get_blob_callback_t callback)
        override {
        callback(true);
    }

    void
    checkBlobExistsAsync(std::string_view, check_blob_callback_t callback) override {
        call_count_++;
        if (call_count_ > throw_after_calls_) {
            throw std::runtime_error("Simulated exception in checkBlobExistsAsync");
        }

        asio::post(*executor_, [callback]() { callback(std::optional<bool>(true)); });
    }
};

void
initTestBlobEngine(const nixlBackendInitParams *init_params,
                   std::shared_ptr<iBlobClient> client,
                   std::unique_ptr<nixlAzureBlobEngine> &out) {
    ASSERT_NE(client, nullptr) << "initTestBlobEngine: blob client must not be null";
    out = std::make_unique<nixlAzureBlobEngine>(init_params, std::move(client));
}

class azureBlobTestFixture : public testing::Test {
protected:
    std::unique_ptr<nixlAzureBlobEngine> blobEngine_;
    std::shared_ptr<mockBlobClient> mockBlobClient_;
    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;

    void
    SetUp() override {
        initParams_.localAgent = "test-agent";
        initParams_.type = "AZURE_BLOB";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

        mockBlobClient_ = std::make_shared<mockBlobClient>();

        // Initialize nixlAzureBlobEngine with the mock IBlobClient
        // The engine will create its own executor and call setExecutor on the mock client
        initTestBlobEngine(&initParams_, mockBlobClient_, blobEngine_);
    }

    void
    testAsyncTransferWithControlledExecution(nixl_xfer_op_t operation) {
        mockBlobClient_->setSimulateSuccess(true);

        nixlBlobDesc local_desc, remote_desc;
        local_desc.devId = 1;
        remote_desc.devId = 2;
        remote_desc.metaInfo = (operation == NIXL_READ) ? "test-read-key" : "test-write-key";

        nixlBackendMD *local_metadata = nullptr;
        nixlBackendMD *remote_metadata = nullptr;

        ASSERT_EQ(blobEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
        ASSERT_EQ(blobEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs(DRAM_SEG);
        nixl_meta_dlist_t remote_descs(OBJ_SEG);

        std::vector<char> test_buffer(1024);

        nixlMetaDesc local_meta_desc(
            reinterpret_cast<uintptr_t>(test_buffer.data()), test_buffer.size(), 1);
        local_descs.addDesc(local_meta_desc);

        nixlMetaDesc remote_meta_desc(0, test_buffer.size(), 2);
        remote_descs.addDesc(remote_meta_desc);

        nixlBackendReqH *handle = nullptr;

        ASSERT_EQ(
            blobEngine_->prepXfer(
                operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE(handle, nullptr);

        nixl_status_t status = blobEngine_->postXfer(
            operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
        EXPECT_EQ(status, NIXL_IN_PROG);
        EXPECT_EQ(mockBlobClient_->getPendingCount(), 1);
        status = blobEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_IN_PROG);

        mockBlobClient_->execAsync();
        status = blobEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_SUCCESS);

        if (operation == NIXL_READ) {
            EXPECT_EQ(test_buffer[0], 'A');
        }

        blobEngine_->releaseReqH(handle);
        blobEngine_->deregisterMem(local_metadata);
        blobEngine_->deregisterMem(remote_metadata);
    }

    void
    testMultiDescriptorTransfer(nixl_xfer_op_t operation) {
        mockBlobClient_->setSimulateSuccess(true);

        std::vector<char> test_buffer0(1024);
        std::vector<char> test_buffer1(1024);
        nixlBlobDesc local_desc0, local_desc1;
        local_desc0.devId = 1;
        local_desc1.devId = 1;
        nixlBackendMD *local_metadata0 = nullptr;
        nixlBackendMD *local_metadata1 = nullptr;

        ASSERT_EQ(blobEngine_->registerMem(local_desc0, DRAM_SEG, local_metadata0), NIXL_SUCCESS);
        ASSERT_EQ(blobEngine_->registerMem(local_desc1, DRAM_SEG, local_metadata1), NIXL_SUCCESS);

        nixlBlobDesc remote_desc0, remote_desc1;
        remote_desc0.devId = 2;
        remote_desc1.devId = 3;
        remote_desc0.metaInfo = (operation == NIXL_READ) ? "test-read-key0" : "test-write-key0";
        remote_desc1.metaInfo = (operation == NIXL_READ) ? "test-read-key1" : "test-write-key1";
        nixlBackendMD *remote_metadata0 = nullptr;
        nixlBackendMD *remote_metadata1 = nullptr;

        ASSERT_EQ(blobEngine_->registerMem(remote_desc0, OBJ_SEG, remote_metadata0), NIXL_SUCCESS);
        ASSERT_EQ(blobEngine_->registerMem(remote_desc1, OBJ_SEG, remote_metadata1), NIXL_SUCCESS);

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
            blobEngine_->prepXfer(
                operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE(handle, nullptr);

        nixl_status_t status = blobEngine_->postXfer(
            operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
        EXPECT_EQ(status, NIXL_IN_PROG);
        EXPECT_EQ(mockBlobClient_->getPendingCount(), 2);
        status = blobEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_IN_PROG);

        mockBlobClient_->execAsync();
        status = blobEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_SUCCESS);

        if (operation == NIXL_READ) {
            EXPECT_EQ(test_buffer0[0], 'A');
            EXPECT_EQ(test_buffer1[0], 'A');
        }

        blobEngine_->releaseReqH(handle);
        blobEngine_->deregisterMem(local_metadata0);
        blobEngine_->deregisterMem(local_metadata1);
        blobEngine_->deregisterMem(remote_metadata0);
        blobEngine_->deregisterMem(remote_metadata1);
    }

    void
    testAsyncTransferFailureIsHandled(nixl_xfer_op_t operation) {
        mockBlobClient_->setSimulateSuccess(false);

        std::vector<char> test_buffer(1024, 'Z');

        nixlBlobDesc local_desc;
        local_desc.devId = 1;
        nixlBackendMD *local_metadata = nullptr;
        ASSERT_EQ(blobEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

        nixlBlobDesc remote_desc;
        remote_desc.devId = 2;
        remote_desc.metaInfo = "test-fail-key";
        nixlBackendMD *remote_metadata = nullptr;
        ASSERT_EQ(blobEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs(DRAM_SEG);
        nixl_meta_dlist_t remote_descs(OBJ_SEG);

        nixlMetaDesc local_meta_desc(
            reinterpret_cast<uintptr_t>(test_buffer.data()), test_buffer.size(), local_desc.devId);
        nixlMetaDesc remote_meta_desc(0, test_buffer.size(), remote_desc.devId);
        local_descs.addDesc(local_meta_desc);
        remote_descs.addDesc(remote_meta_desc);

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(
            blobEngine_->prepXfer(
                operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE(handle, nullptr);

        nixl_status_t status = blobEngine_->postXfer(
            operation, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
        EXPECT_EQ(status, NIXL_IN_PROG);
        EXPECT_EQ(mockBlobClient_->getPendingCount(), 1);
        status = blobEngine_->checkXfer(handle);
        EXPECT_EQ(status, NIXL_IN_PROG);

        mockBlobClient_->execAsync();
        status = blobEngine_->checkXfer(handle);
        EXPECT_NE(status, NIXL_SUCCESS); // Should not succeed

        blobEngine_->releaseReqH(handle);
        blobEngine_->deregisterMem(local_metadata);
        blobEngine_->deregisterMem(remote_metadata);
    }

    void
    testBlobExistence(bool shouldExist) {
        mockBlobClient_->setSimulateSuccess(shouldExist);

        nixl_reg_dlist_t descs(OBJ_SEG);
        descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-key-1"));
        descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-key-2"));
        descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-key-3"));

        std::vector<nixl_query_resp_t> resp;
        blobEngine_->queryMem(descs, resp);

        EXPECT_EQ(resp.size(), 3);
        EXPECT_EQ(resp[0].has_value(), shouldExist);
        EXPECT_EQ(resp[1].has_value(), shouldExist);
        EXPECT_EQ(resp[2].has_value(), shouldExist);

        EXPECT_EQ(mockBlobClient_->getCheckedKeys().size(), 3);
        EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("test-key-1"));
        EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("test-key-2"));
        EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("test-key-3"));
    }
};

TEST_F(azureBlobTestFixture, EngineInitialization) {
    ASSERT_NE(blobEngine_, nullptr);
    EXPECT_EQ(blobEngine_->getType(), "AZURE_BLOB");
    EXPECT_TRUE(blobEngine_->supportsLocal());
    EXPECT_FALSE(blobEngine_->supportsRemote());
    EXPECT_FALSE(blobEngine_->supportsNotif());

    // Verify that the executor was properly set on the mock azure blob client by the engine
    // constructor
    EXPECT_TRUE(mockBlobClient_->hasExecutor());
}

TEST_F(azureBlobTestFixture, GetSupportedMems) {
    auto supported_mems = blobEngine_->getSupportedMems();
    EXPECT_EQ(supported_mems.size(), 2);
    EXPECT_TRUE(std::find(supported_mems.begin(), supported_mems.end(), OBJ_SEG) !=
                supported_mems.end());
    EXPECT_TRUE(std::find(supported_mems.begin(), supported_mems.end(), DRAM_SEG) !=
                supported_mems.end());
}

TEST_F(azureBlobTestFixture, RegisterMemoryObjSeg) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 42;
    mem_desc.metaInfo = "test-blob-name";

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = blobEngine_->registerMem(mem_desc, OBJ_SEG, metadata);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(metadata, nullptr);

    status = blobEngine_->deregisterMem(metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(azureBlobTestFixture, RegisterMemoryObjSegWithoutKey) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 99;
    mem_desc.metaInfo = ""; // Empty name - engine will generate a name for the blob

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = blobEngine_->registerMem(mem_desc, OBJ_SEG, metadata);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_NE(metadata, nullptr);

    status = blobEngine_->deregisterMem(metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(azureBlobTestFixture, RegisterMemoryDramSeg) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 123;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = blobEngine_->registerMem(mem_desc, DRAM_SEG, metadata);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(metadata, nullptr);

    status = blobEngine_->deregisterMem(metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(azureBlobTestFixture, RegisterUnsupportedMemType) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 42;
    mem_desc.metaInfo = "test-blob-name";

    nixlBackendMD *metadata = nullptr;

    // VRAM segment is not supported by Azure Blob backend, which should result in
    // an error when registering it.
    nixl_status_t status = blobEngine_->registerMem(mem_desc, VRAM_SEG, metadata);
    EXPECT_EQ(status, NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(metadata, nullptr);
}

TEST_F(azureBlobTestFixture, CancelTransfer) {
    mockBlobClient_->setSimulateSuccess(true);

    nixlBlobDesc local_desc, remote_desc;
    local_desc.devId = 1;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-cancel-key";

    nixlBackendMD *local_metadata = nullptr;
    nixlBackendMD *remote_metadata = nullptr;

    ASSERT_EQ(blobEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
    ASSERT_EQ(blobEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    std::vector<char> test_buffer(1024);
    nixlMetaDesc local_meta_desc(
        reinterpret_cast<uintptr_t>(test_buffer.data()), test_buffer.size(), 1);
    local_descs.addDesc(local_meta_desc);

    nixlMetaDesc remote_meta_desc(0, test_buffer.size(), 2);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;

    ASSERT_EQ(blobEngine_->prepXfer(
                  NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
              NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    nixl_status_t status = blobEngine_->postXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_IN_PROG);
    EXPECT_EQ(mockBlobClient_->getPendingCount(), 1);

    status = blobEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    // Cancel the transfer before completion by releasing the handle
    // This simulates the cancellation behavior from nixlAgent::releaseXferReq
    status = blobEngine_->releaseReqH(handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    mockBlobClient_->execAsync();

    // After cancellation/release, we can't check the transfer status anymore
    // as the handle has been released. This verifies that cancelling pending
    // async tasks is handled correctly by properly cleaning up resources.
    status = blobEngine_->deregisterMem(local_metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
    status = blobEngine_->deregisterMem(remote_metadata);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(azureBlobTestFixture, ReadFromOffset) {
    mockBlobClient_->setSimulateSuccess(true);

    std::vector<char> test_buffer(1024);

    nixlBlobDesc local_desc;
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ(blobEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-offset-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ(blobEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

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
    ASSERT_EQ(blobEngine_->prepXfer(
                  NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
              NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    nixl_status_t status = blobEngine_->postXfer(
        NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_IN_PROG);
    EXPECT_EQ(mockBlobClient_->getPendingCount(), 1);
    status = blobEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_IN_PROG);

    mockBlobClient_->execAsync();
    status = blobEngine_->checkXfer(handle);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(test_buffer[0], 'A' + (offset % 26));

    blobEngine_->releaseReqH(handle);
    blobEngine_->deregisterMem(local_metadata);
    blobEngine_->deregisterMem(remote_metadata);
}

TEST_F(azureBlobTestFixture, AsyncReadTransferWithControlledExecution) {
    testAsyncTransferWithControlledExecution(NIXL_READ);
}

TEST_F(azureBlobTestFixture, AsyncWriteTransferWithControlledExecution) {
    testAsyncTransferWithControlledExecution(NIXL_WRITE);
}

TEST_F(azureBlobTestFixture, MultiDescriptorWrite) {
    testMultiDescriptorTransfer(NIXL_WRITE);
}

TEST_F(azureBlobTestFixture, MultiDescriptorRead) {
    testMultiDescriptorTransfer(NIXL_READ);
}

TEST_F(azureBlobTestFixture, AsyncReadTransferFailureIsHandled) {
    testAsyncTransferFailureIsHandled(NIXL_READ);
}

TEST_F(azureBlobTestFixture, AsyncWriteTransferFailureIsHandled) {
    testAsyncTransferFailureIsHandled(NIXL_WRITE);
}

TEST_F(azureBlobTestFixture, ValidatesUnsupportedLocalMemTypeForTransfer) {
    mockBlobClient_->setSimulateSuccess(true);

    nixlBlobDesc local_desc, remote_desc;
    local_desc.devId = 1;
    local_desc.metaInfo = "invalid-local-blob";
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-remote-blob";

    nixlBackendMD *local_metadata = nullptr;
    nixlBackendMD *remote_metadata = nullptr;

    // Register local as OBJ_SEG, which is not supported by the Azure Blob backend.
    // This should result in errors when prepping the transfer
    ASSERT_EQ(blobEngine_->registerMem(local_desc, OBJ_SEG, local_metadata), NIXL_SUCCESS);
    ASSERT_EQ(blobEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs(OBJ_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(0, 1024, local_desc.devId);
    local_descs.addDesc(local_meta_desc);

    nixlMetaDesc remote_meta_desc(0, 1024, remote_desc.devId);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;

    nixl_status_t status = blobEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    blobEngine_->deregisterMem(local_metadata);
    blobEngine_->deregisterMem(remote_metadata);
}

TEST_F(azureBlobTestFixture, ValidatesUnsupportedRemoteMemTypeForTransfer) {
    mockBlobClient_->setSimulateSuccess(true);

    nixlBlobDesc local_desc, remote_desc;
    local_desc.devId = 1;
    remote_desc.devId = 2;

    nixlBackendMD *local_metadata = nullptr;
    nixlBackendMD *remote_metadata = nullptr;

    // Register both as DRAM_SEG. Local is valid but remote DRAM is not supported for blob
    // operations which should result in errors when prepping the transfer
    ASSERT_EQ(blobEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
    ASSERT_EQ(blobEngine_->registerMem(remote_desc, DRAM_SEG, remote_metadata), NIXL_SUCCESS);

    std::vector<char> test_buffer(1024);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(DRAM_SEG);

    nixlMetaDesc local_meta_desc(
        reinterpret_cast<uintptr_t>(test_buffer.data()), test_buffer.size(), local_desc.devId);
    local_descs.addDesc(local_meta_desc);

    nixlMetaDesc remote_meta_desc(0, test_buffer.size(), remote_desc.devId);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;

    nixl_status_t status = blobEngine_->prepXfer(
        NIXL_WRITE, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(handle, nullptr);

    blobEngine_->deregisterMem(local_metadata);
    blobEngine_->deregisterMem(remote_metadata);
}

TEST_F(azureBlobTestFixture, ValidatesRemoteRegistrationAtTransferTime) {
    mockBlobClient_->setSimulateSuccess(true);

    nixlBlobDesc local_desc, remote_desc;
    local_desc.devId = 1;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-blob";

    nixlBackendMD *local_metadata = nullptr;
    nixlBackendMD *remote_metadata = nullptr;

    // Register both local and remote memory
    ASSERT_EQ(blobEngine_->registerMem(local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
    ASSERT_EQ(blobEngine_->registerMem(remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    std::vector<char> test_buffer(1024);

    nixl_meta_dlist_t local_descs(DRAM_SEG);
    nixl_meta_dlist_t remote_descs(OBJ_SEG);

    nixlMetaDesc local_meta_desc(
        reinterpret_cast<uintptr_t>(test_buffer.data()), test_buffer.size(), local_desc.devId);
    local_descs.addDesc(local_meta_desc);

    nixlMetaDesc remote_meta_desc(0, test_buffer.size(), remote_desc.devId);
    remote_descs.addDesc(remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(blobEngine_->prepXfer(
                  NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr),
              NIXL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // Deregister the remote memory before attempting transfer. This should cause the transfer
    // to fail when posted.
    ASSERT_EQ(blobEngine_->deregisterMem(remote_metadata), NIXL_SUCCESS);
    nixl_status_t status = blobEngine_->postXfer(
        NIXL_READ, local_descs, remote_descs, initParams_.localAgent, handle, nullptr);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    blobEngine_->deregisterMem(local_metadata);
}

TEST_F(azureBlobTestFixture, CheckBlobExists) {
    testBlobExistence(true);
}

TEST_F(azureBlobTestFixture, CheckBlobDoesNotExist) {
    testBlobExistence(false);
}

TEST_F(azureBlobTestFixture, CheckBlobExistsAsyncOrdering) {
    // Single combined queryMem call with per-key outcomes and staggered delays
    // so responses complete out of order, exercising the slot-mapping logic.
    nixl_reg_dlist_t combined_descs(OBJ_SEG);
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-1")); // exists
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-2")); // missing
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-3")); // exists
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-4")); // missing
    combined_descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "async-key-5")); // exists

    // Drive per-key outcomes so exist/missing alternate
    mockBlobClient_->setKeyOutcome("async-key-1", true);
    mockBlobClient_->setKeyOutcome("async-key-2", false);
    mockBlobClient_->setKeyOutcome("async-key-3", true);
    mockBlobClient_->setKeyOutcome("async-key-4", false);
    mockBlobClient_->setKeyOutcome("async-key-5", true);

    // Stagger delays: earlier keys complete later to force out-of-order completion
    mockBlobClient_->setKeyDelay("async-key-1", std::chrono::milliseconds(50));
    mockBlobClient_->setKeyDelay("async-key-2", std::chrono::milliseconds(40));
    mockBlobClient_->setKeyDelay("async-key-3", std::chrono::milliseconds(30));
    mockBlobClient_->setKeyDelay("async-key-4", std::chrono::milliseconds(20));
    mockBlobClient_->setKeyDelay("async-key-5", std::chrono::milliseconds(10));

    std::vector<nixl_query_resp_t> combined_resp;
    nixl_status_t status = blobEngine_->queryMem(combined_descs, combined_resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    ASSERT_EQ(combined_resp.size(), 5);

    // Assert each response corresponds to the descriptor at that index
    EXPECT_TRUE(combined_resp[0].has_value()) << "async-key-1 should exist";
    EXPECT_FALSE(combined_resp[1].has_value()) << "async-key-2 should not exist";
    EXPECT_TRUE(combined_resp[2].has_value()) << "async-key-3 should exist";
    EXPECT_FALSE(combined_resp[3].has_value()) << "async-key-4 should not exist";
    EXPECT_TRUE(combined_resp[4].has_value()) << "async-key-5 should exist";

    // Verify all keys were checked
    EXPECT_EQ(mockBlobClient_->getCheckedKeys().size(), 5);
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("async-key-1"));
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("async-key-2"));
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("async-key-3"));
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("async-key-4"));
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("async-key-5"));
}

TEST_F(azureBlobTestFixture, CheckBlobExistsAsyncEmptyList) {
    nixl_reg_dlist_t descs(OBJ_SEG);
    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = blobEngine_->queryMem(descs, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    EXPECT_EQ(resp.size(), 0);
    EXPECT_EQ(mockBlobClient_->getCheckedKeys().size(), 0);
}

TEST_F(azureBlobTestFixture, CheckBlobExistsAsyncRequestError) {
    // Simulate a transient error (e.g. 5xx / auth failure) that should
    // propagate as NIXL_ERR_BACKEND rather than being treated as "not found".
    mockBlobClient_->setSimulateError(true);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "error-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "error-key-2"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = blobEngine_->queryMem(descs, resp);

    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(mockBlobClient_->getCheckedKeys().size(), 2);
}

TEST_F(azureBlobTestFixture, QueryMemClearsStaleResp) {
    // First queryMem: 3 keys, all exist
    mockBlobClient_->setSimulateSuccess(true);

    nixl_reg_dlist_t descs1(OBJ_SEG);
    descs1.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-1"));
    descs1.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-2"));
    descs1.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-3"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = blobEngine_->queryMem(descs1, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_TRUE(resp[2].has_value());

    // Second queryMem: 2 keys, both missing - reuses the same resp vector.
    // resp.assign() must clear the 3 stale entries from the first call.
    mockBlobClient_->setSimulateSuccess(false);

    nixl_reg_dlist_t descs2(OBJ_SEG);
    descs2.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-4"));
    descs2.addDesc(nixlBlobDesc(nixlBasicDesc(), "stale-key-5"));

    status = blobEngine_->queryMem(descs2, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // resp must reflect the second call only: 2 items, both not found
    EXPECT_EQ(resp.size(), 2);
    EXPECT_FALSE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
}

TEST_F(azureBlobTestFixture, QueryMemMixedPerKeyErrors) {
    // Mix of outcomes in a single queryMem call: some keys exist, some are
    // missing, and one returns an error (std::nullopt).  The error should
    // cause NIXL_ERR_BACKEND while the other slots are still populated.
    mockBlobClient_->setKeyOutcome("mix-key-1", true); // exists
    mockBlobClient_->setKeyOutcome("mix-key-2", false); // missing
    mockBlobClient_->setKeyError("mix-key-3"); // transient error

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "mix-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "mix-key-2"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "mix-key-3"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = blobEngine_->queryMem(descs, resp);

    // Should fail because mix-key-3 errored
    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    // Non-error slots should still carry the correct values
    ASSERT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].has_value()) << "mix-key-1 should exist";
    EXPECT_FALSE(resp[1].has_value()) << "mix-key-2 should not exist";
    EXPECT_FALSE(resp[2].has_value()) << "mix-key-3 errored, should be nullopt";

    // All 3 keys should still have been checked
    EXPECT_EQ(mockBlobClient_->getCheckedKeys().size(), 3);
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("mix-key-1"));
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("mix-key-2"));
    EXPECT_TRUE(mockBlobClient_->getCheckedKeys().count("mix-key-3"));
}

// ---------------------------------------------------------------------------
// Exact-once callback guard tests
// ---------------------------------------------------------------------------

// Fixture that injects the double-callback mock so that every
// checkBlobExistsAsync invocation fires the callback twice.
class azureBlobDoubleCallbackFixture : public testing::Test {
protected:
    std::unique_ptr<nixlAzureBlobEngine> blobEngine_;
    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;

    void
    SetUp() override {
        initParams_.localAgent = "test-double-callback";
        initParams_.type = "AZURE_BLOB";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

        auto mockClient = std::make_shared<doubleCallbackMockBlobClient>();
        initTestBlobEngine(&initParams_, mockClient, blobEngine_);
    }
};

TEST_F(azureBlobDoubleCallbackFixture, QueryMemDuplicateCallback) {
    // Verify that unintended duplicate callback invocation from the blob client
    // implementation doesn't cause exceptions or corrupt results.
    // The exact-once guard in queryMem (completed->exchange(true)) should make
    // the second invocation a no-op.
    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "dup-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "dup-key-2"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "dup-key-3"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = blobEngine_->queryMem(descs, resp);
    ASSERT_EQ(status, NIXL_SUCCESS);

    EXPECT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_TRUE(resp[2].has_value());
}

// ---------------------------------------------------------------------------
// Exception path tests
// ---------------------------------------------------------------------------

class azureBlobExceptionFixture : public testing::Test {
protected:
    std::unique_ptr<nixlAzureBlobEngine> blobEngine_;
    nixlBackendInitParams initParams_;
    nixl_b_params_t customParams_;

    void
    SetUp() override {
        initParams_.localAgent = "test-exception";
        initParams_.type = "AZURE_BLOB";
        initParams_.customParams = &customParams_;
        initParams_.enableProgTh = false;
        initParams_.pthrDelay = 0;
        initParams_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    }
};

TEST_F(azureBlobExceptionFixture, QueryMemExceptionDuringLaunch) {
    // Test that exceptions during async request launch are handled gracefully.
    // The mock throws after N calls, triggering the catch block in queryMem.
    // This verifies the fix for the use-after-free race: the catch block should
    // wait for all in-flight callbacks to complete before returning.
    auto mockClient = std::make_shared<exceptionThrowingMockBlobClient>(2);
    initTestBlobEngine(&initParams_, mockClient, blobEngine_);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-2"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-3"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-key-4"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = blobEngine_->queryMem(descs, resp);

    // Should return error due to exception during launch
    EXPECT_EQ(status, NIXL_ERR_BACKEND);

    // Response vector should be sized correctly (all slots initialized to nullopt)
    EXPECT_EQ(resp.size(), 4);
}

TEST_F(azureBlobExceptionFixture, QueryMemExceptionOnFirstCall) {
    // Test exception on the very first async request launch.
    auto mockClient = std::make_shared<exceptionThrowingMockBlobClient>(0);
    initTestBlobEngine(&initParams_, mockClient, blobEngine_);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-immediate-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "exception-immediate-2"));

    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = blobEngine_->queryMem(descs, resp);

    EXPECT_EQ(status, NIXL_ERR_BACKEND);
    EXPECT_EQ(resp.size(), 2);
}

} // namespace gtest::azure_blob
