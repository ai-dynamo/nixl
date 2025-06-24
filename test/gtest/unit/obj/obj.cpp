/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nixl_types.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "obj_s3_client.h"
#include "obj_backend.h"
#include "obj_executor.h"

namespace gtest::obj {

class MockS3Client : public IS3Client {
private:
    bool simulate_success_ = true;
    std::shared_ptr<AsioThreadPoolExecutor> executor_;
    std::vector<std::function<void()>> pending_callbacks_;

public:
    void
    setSimulateSuccess (bool success) {
        simulate_success_ = success;
    }

    void
    setExecutor (std::shared_ptr<Aws::Utils::Threading::Executor> executor) override {
        executor_ = std::dynamic_pointer_cast<AsioThreadPoolExecutor> (executor);
    }

    void
    PutObjectAsync (std::string_view key,
                    uintptr_t data_ptr,
                    size_t data_len,
                    size_t offset,
                    PutObjectCallback callback) override {
        pending_callbacks_.push_back ([callback, this]() { callback (simulate_success_); });
    }

    void
    GetObjectAsync (std::string_view key,
                    uintptr_t data_ptr,
                    size_t data_len,
                    size_t offset,
                    GetObjectCallback callback) override {
        pending_callbacks_.push_back ([callback, data_ptr, data_len, offset, this]() {
            if (simulate_success_ && data_ptr && data_len > 0) {
                char *buffer = reinterpret_cast<char *> (data_ptr);
                for (size_t i = 0; i < data_len; ++i) {
                    buffer[i] = static_cast<char> ('A' + ((i + offset) % 26));
                }
            }
            callback (simulate_success_);
        });
    }

    void
    execAsync() {
        for (auto &callback : pending_callbacks_) {
            executor_->Submit ([callback]() { callback(); });
        }
        pending_callbacks_.clear();
        executor_->WaitUntilIdle();
    }

    size_t
    getPendingCount() const {
        return pending_callbacks_.size();
    }

    bool
    hasExecutor() const {
        return executor_ != nullptr;
    }
};

class ObjTestFixture : public testing::Test {
protected:
    std::unique_ptr<nixlObjEngine> obj_engine_;
    std::shared_ptr<MockS3Client> mock_s3_client_;
    nixlBackendInitParams init_params_;
    nixl_b_params_t custom_params_;

    void
    SetUp() override {
        init_params_.localAgent = "test-agent";
        init_params_.type = "OBJ";
        init_params_.customParams = &custom_params_;
        init_params_.enableProgTh = false;
        init_params_.pthrDelay = 0;
        init_params_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;

        mock_s3_client_ = std::make_shared<MockS3Client>();

        // Initialize nixlObjEngine with the mock IS3Client
        // The engine will create its own executor and call setExecutor on the mock client
        obj_engine_ = std::make_unique<nixlObjEngine> (&init_params_, mock_s3_client_);
    }

    void
    testAsyncTransferWithControlledExecution (nixl_xfer_op_t operation) {
        mock_s3_client_->setSimulateSuccess (true);

        nixlBlobDesc local_desc, remote_desc;
        local_desc.devId = 1;
        remote_desc.devId = 2;
        remote_desc.metaInfo = (operation == NIXL_READ) ? "test-read-key" : "test-write-key";

        nixlBackendMD *local_metadata = nullptr;
        nixlBackendMD *remote_metadata = nullptr;

        ASSERT_EQ (obj_engine_->registerMem (local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
        ASSERT_EQ (obj_engine_->registerMem (remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs (DRAM_SEG);
        nixl_meta_dlist_t remote_descs (OBJ_SEG, false);

        std::vector<char> test_buffer (1024);

        if (operation == NIXL_WRITE) {
            for (size_t i = 0; i < test_buffer.size(); ++i) {
                test_buffer[i] = static_cast<char> ('X' + (i % 3));
            }
        }

        nixlMetaDesc local_meta_desc (
            reinterpret_cast<uintptr_t> (test_buffer.data()), test_buffer.size(), 1);
        local_descs.addDesc (local_meta_desc);

        nixlMetaDesc remote_meta_desc (0, test_buffer.size(), 2);
        remote_descs.addDesc (remote_meta_desc);

        nixlBackendReqH *handle = nullptr;

        ASSERT_EQ (
            obj_engine_->prepXfer (
                operation, local_descs, remote_descs, init_params_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE (handle, nullptr);

        nixl_status_t status = obj_engine_->postXfer (
            operation, local_descs, remote_descs, init_params_.localAgent, handle, nullptr);
        EXPECT_EQ (status, NIXL_IN_PROG);
        EXPECT_EQ (mock_s3_client_->getPendingCount(), 1);
        status = obj_engine_->checkXfer (handle);
        EXPECT_EQ (status, NIXL_IN_PROG);

        mock_s3_client_->execAsync();
        status = obj_engine_->checkXfer (handle);
        EXPECT_EQ (status, NIXL_SUCCESS);

        if (operation == NIXL_READ) {
            EXPECT_EQ (test_buffer[0], 'A');
        }

        obj_engine_->releaseReqH (handle);
        obj_engine_->deregisterMem (local_metadata);
        obj_engine_->deregisterMem (remote_metadata);
    }

    void
    testMultiDescriptorTransfer (nixl_xfer_op_t operation) {
        mock_s3_client_->setSimulateSuccess (true);

        std::vector<char> test_buffer0 (1024);
        std::vector<char> test_buffer1 (1024);
        if (operation == NIXL_WRITE) {
            for (size_t i = 0; i < test_buffer0.size(); ++i) {
                test_buffer0[i] = static_cast<char> ('X' + (i % 3));
            }
            for (size_t i = 0; i < test_buffer1.size(); ++i) {
                test_buffer1[i] = static_cast<char> ('x' + (i % 3));
            }
        }

        nixlBlobDesc local_desc0, local_desc1;
        local_desc0.devId = 1;
        local_desc1.devId = 1;
        nixlBackendMD *local_metadata0 = nullptr;
        nixlBackendMD *local_metadata1 = nullptr;

        ASSERT_EQ (obj_engine_->registerMem (local_desc0, DRAM_SEG, local_metadata0), NIXL_SUCCESS);
        ASSERT_EQ (obj_engine_->registerMem (local_desc1, DRAM_SEG, local_metadata1), NIXL_SUCCESS);

        nixlBlobDesc remote_desc0, remote_desc1;
        remote_desc0.devId = 2;
        remote_desc1.devId = 3;
        remote_desc0.metaInfo = (operation == NIXL_READ) ? "test-read-key0" : "test-write-key0";
        remote_desc1.metaInfo = (operation == NIXL_READ) ? "test-read-key1" : "test-write-key1";
        nixlBackendMD *remote_metadata0 = nullptr;
        nixlBackendMD *remote_metadata1 = nullptr;

        ASSERT_EQ (obj_engine_->registerMem (remote_desc0, OBJ_SEG, remote_metadata0),
                   NIXL_SUCCESS);
        ASSERT_EQ (obj_engine_->registerMem (remote_desc1, OBJ_SEG, remote_metadata1),
                   NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs (DRAM_SEG, false);
        nixl_meta_dlist_t remote_descs (OBJ_SEG, false);

        nixlMetaDesc local_meta_desc0 (reinterpret_cast<uintptr_t> (test_buffer0.data()),
                                       test_buffer0.size(),
                                       local_desc0.devId);
        nixlMetaDesc local_meta_desc1 (reinterpret_cast<uintptr_t> (test_buffer1.data()),
                                       test_buffer1.size(),
                                       local_desc1.devId);
        local_descs.addDesc (local_meta_desc0);
        local_descs.addDesc (local_meta_desc1);

        nixlMetaDesc remote_meta_desc0 (0, test_buffer0.size(), remote_desc0.devId);
        nixlMetaDesc remote_meta_desc1 (0, test_buffer1.size(), remote_desc1.devId);
        remote_descs.addDesc (remote_meta_desc0);
        remote_descs.addDesc (remote_meta_desc1);

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ (
            obj_engine_->prepXfer (
                operation, local_descs, remote_descs, init_params_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE (handle, nullptr);

        nixl_status_t status = obj_engine_->postXfer (
            operation, local_descs, remote_descs, init_params_.localAgent, handle, nullptr);
        EXPECT_EQ (status, NIXL_IN_PROG);
        EXPECT_EQ (mock_s3_client_->getPendingCount(), 2);
        status = obj_engine_->checkXfer (handle);
        EXPECT_EQ (status, NIXL_IN_PROG);

        mock_s3_client_->execAsync();
        status = obj_engine_->checkXfer (handle);
        EXPECT_EQ (status, NIXL_SUCCESS);

        if (operation == NIXL_READ) {
            EXPECT_EQ (test_buffer0[0], 'A');
            EXPECT_EQ (test_buffer1[0], 'A');
        }

        obj_engine_->releaseReqH (handle);
        obj_engine_->deregisterMem (local_metadata0);
        obj_engine_->deregisterMem (local_metadata1);
        obj_engine_->deregisterMem (remote_metadata0);
        obj_engine_->deregisterMem (remote_metadata1);
    }

    void
    testAsyncTransferFailureIsHandled (nixl_xfer_op_t operation) {
        mock_s3_client_->setSimulateSuccess (false);

        std::vector<char> test_buffer (1024, 'Z');

        nixlBlobDesc local_desc;
        local_desc.devId = 1;
        nixlBackendMD *local_metadata = nullptr;
        ASSERT_EQ (obj_engine_->registerMem (local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

        nixlBlobDesc remote_desc;
        remote_desc.devId = 2;
        remote_desc.metaInfo = "test-fail-key";
        nixlBackendMD *remote_metadata = nullptr;
        ASSERT_EQ (obj_engine_->registerMem (remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

        nixl_meta_dlist_t local_descs (DRAM_SEG, false);
        nixl_meta_dlist_t remote_descs (OBJ_SEG, false);

        nixlMetaDesc local_meta_desc (
            reinterpret_cast<uintptr_t> (test_buffer.data()), test_buffer.size(), local_desc.devId);
        nixlMetaDesc remote_meta_desc (0, test_buffer.size(), remote_desc.devId);
        local_descs.addDesc (local_meta_desc);
        remote_descs.addDesc (remote_meta_desc);

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ (
            obj_engine_->prepXfer (
                operation, local_descs, remote_descs, init_params_.localAgent, handle, nullptr),
            NIXL_SUCCESS);
        ASSERT_NE (handle, nullptr);

        nixl_status_t status = obj_engine_->postXfer (
            operation, local_descs, remote_descs, init_params_.localAgent, handle, nullptr);
        EXPECT_EQ (status, NIXL_IN_PROG);
        EXPECT_EQ (mock_s3_client_->getPendingCount(), 1);
        status = obj_engine_->checkXfer (handle);
        EXPECT_EQ (status, NIXL_IN_PROG);

        mock_s3_client_->execAsync();
        status = obj_engine_->checkXfer (handle);
        EXPECT_NE (status, NIXL_SUCCESS); // Should not succeed

        obj_engine_->releaseReqH (handle);
        obj_engine_->deregisterMem (local_metadata);
        obj_engine_->deregisterMem (remote_metadata);
    }
};

TEST_F (ObjTestFixture, EngineInitialization) {
    ASSERT_NE (obj_engine_, nullptr);
    EXPECT_EQ (obj_engine_->getType(), "OBJ");
    EXPECT_TRUE (obj_engine_->supportsLocal());
    EXPECT_FALSE (obj_engine_->supportsRemote());
    EXPECT_FALSE (obj_engine_->supportsNotif());
    EXPECT_FALSE (obj_engine_->supportsProgTh());

    // Verify that the executor was properly set on the mock S3 client by the engine constructor
    EXPECT_TRUE (mock_s3_client_->hasExecutor());
}

TEST_F (ObjTestFixture, GetSupportedMems) {
    auto supported_mems = obj_engine_->getSupportedMems();
    EXPECT_EQ (supported_mems.size(), 2);
    EXPECT_TRUE (std::find (supported_mems.begin(), supported_mems.end(), OBJ_SEG) !=
                 supported_mems.end());
    EXPECT_TRUE (std::find (supported_mems.begin(), supported_mems.end(), DRAM_SEG) !=
                 supported_mems.end());
}

TEST_F (ObjTestFixture, RegisterMemoryObjSeg) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 42;
    mem_desc.metaInfo = "test-object-key";

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = obj_engine_->registerMem (mem_desc, OBJ_SEG, metadata);

    EXPECT_EQ (status, NIXL_SUCCESS);
    EXPECT_NE (metadata, nullptr);

    status = obj_engine_->deregisterMem (metadata);
    EXPECT_EQ (status, NIXL_SUCCESS);
}

TEST_F (ObjTestFixture, RegisterMemoryObjSegWithoutKey) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 99;
    mem_desc.metaInfo = ""; // Empty key - engine will generate a key

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = obj_engine_->registerMem (mem_desc, OBJ_SEG, metadata);

    EXPECT_EQ (status, NIXL_SUCCESS);
    EXPECT_NE (metadata, nullptr);

    status = obj_engine_->deregisterMem (metadata);
    EXPECT_EQ (status, NIXL_SUCCESS);
}

TEST_F (ObjTestFixture, RegisterMemoryDramSeg) {
    nixlBlobDesc mem_desc;
    mem_desc.devId = 123;

    nixlBackendMD *metadata = nullptr;
    nixl_status_t status = obj_engine_->registerMem (mem_desc, DRAM_SEG, metadata);

    EXPECT_EQ (status, NIXL_SUCCESS);
    EXPECT_EQ (metadata, nullptr);

    status = obj_engine_->deregisterMem (metadata);
    EXPECT_EQ (status, NIXL_SUCCESS);
}

TEST_F (ObjTestFixture, CancelTransfer) {
    mock_s3_client_->setSimulateSuccess (true);

    nixlBlobDesc local_desc, remote_desc;
    local_desc.devId = 1;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-cancel-key";

    nixlBackendMD *local_metadata = nullptr;
    nixlBackendMD *remote_metadata = nullptr;

    ASSERT_EQ (obj_engine_->registerMem (local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);
    ASSERT_EQ (obj_engine_->registerMem (remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs (DRAM_SEG, false);
    nixl_meta_dlist_t remote_descs (OBJ_SEG, false);

    std::vector<char> test_buffer (1024);
    for (size_t i = 0; i < test_buffer.size(); ++i) {
        test_buffer[i] = static_cast<char> ('T' + (i % 3));
    }

    nixlMetaDesc local_meta_desc (
        reinterpret_cast<uintptr_t> (test_buffer.data()), test_buffer.size(), 1);
    local_descs.addDesc (local_meta_desc);

    nixlMetaDesc remote_meta_desc (0, test_buffer.size(), 2);
    remote_descs.addDesc (remote_meta_desc);

    nixlBackendReqH *handle = nullptr;

    ASSERT_EQ (obj_engine_->prepXfer (
                   NIXL_WRITE, local_descs, remote_descs, init_params_.localAgent, handle, nullptr),
               NIXL_SUCCESS);
    ASSERT_NE (handle, nullptr);

    nixl_status_t status = obj_engine_->postXfer (
        NIXL_WRITE, local_descs, remote_descs, init_params_.localAgent, handle, nullptr);
    EXPECT_EQ (status, NIXL_IN_PROG);
    EXPECT_EQ (mock_s3_client_->getPendingCount(), 1);

    status = obj_engine_->checkXfer (handle);
    EXPECT_EQ (status, NIXL_IN_PROG);

    // Cancel the transfer before completion by releasing the handle
    // This simulates the cancellation behavior from nixlAgent::releaseXferReq
    status = obj_engine_->releaseReqH (handle);
    EXPECT_EQ (status, NIXL_SUCCESS);
    mock_s3_client_->execAsync();

    // After cancellation/release, we can't check the transfer status anymore
    // as the handle has been released. This verifies that cancelling pending
    // async tasks is handled correctly by properly cleaning up resources.
    status = obj_engine_->deregisterMem (local_metadata);
    EXPECT_EQ (status, NIXL_SUCCESS);
    status = obj_engine_->deregisterMem (remote_metadata);
    EXPECT_EQ (status, NIXL_SUCCESS);
}

TEST_F (ObjTestFixture, ReadFromOffset) {
    mock_s3_client_->setSimulateSuccess (true);

    std::vector<char> test_buffer (1024);

    nixlBlobDesc local_desc;
    local_desc.devId = 1;
    nixlBackendMD *local_metadata = nullptr;
    ASSERT_EQ (obj_engine_->registerMem (local_desc, DRAM_SEG, local_metadata), NIXL_SUCCESS);

    nixlBlobDesc remote_desc;
    remote_desc.devId = 2;
    remote_desc.metaInfo = "test-offset-key";
    nixlBackendMD *remote_metadata = nullptr;
    ASSERT_EQ (obj_engine_->registerMem (remote_desc, OBJ_SEG, remote_metadata), NIXL_SUCCESS);

    nixl_meta_dlist_t local_descs (DRAM_SEG, false);
    nixl_meta_dlist_t remote_descs (OBJ_SEG, false);

    const size_t offset = 256;
    const size_t length = 512;
    nixlMetaDesc local_meta_desc (
        reinterpret_cast<uintptr_t> (test_buffer.data()), length, local_desc.devId);
    nixlMetaDesc remote_meta_desc (offset, length, remote_desc.devId);
    local_descs.addDesc (local_meta_desc);
    remote_descs.addDesc (remote_meta_desc);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ (obj_engine_->prepXfer (
                   NIXL_READ, local_descs, remote_descs, init_params_.localAgent, handle, nullptr),
               NIXL_SUCCESS);
    ASSERT_NE (handle, nullptr);

    nixl_status_t status = obj_engine_->postXfer (
        NIXL_READ, local_descs, remote_descs, init_params_.localAgent, handle, nullptr);
    EXPECT_EQ (status, NIXL_IN_PROG);
    EXPECT_EQ (mock_s3_client_->getPendingCount(), 1);
    status = obj_engine_->checkXfer (handle);
    EXPECT_EQ (status, NIXL_IN_PROG);

    mock_s3_client_->execAsync();
    status = obj_engine_->checkXfer (handle);
    EXPECT_EQ (status, NIXL_SUCCESS);
    EXPECT_EQ (test_buffer[0], 'A' + (offset % 26));

    obj_engine_->releaseReqH (handle);
    obj_engine_->deregisterMem (local_metadata);
    obj_engine_->deregisterMem (remote_metadata);
}

TEST_F (ObjTestFixture, AsyncReadTransferWithControlledExecution) {
    testAsyncTransferWithControlledExecution (NIXL_READ);
}

TEST_F (ObjTestFixture, AsyncWriteTransferWithControlledExecution) {
    testAsyncTransferWithControlledExecution (NIXL_WRITE);
}

TEST_F (ObjTestFixture, MultiDescriptorWrite) {
    testMultiDescriptorTransfer (NIXL_WRITE);
}

TEST_F (ObjTestFixture, MultiDescriptorRead) {
    testMultiDescriptorTransfer (NIXL_READ);
}

TEST_F (ObjTestFixture, AsyncReadTransferFailureIsHandled) {
    testAsyncTransferFailureIsHandled (NIXL_READ);
}

TEST_F (ObjTestFixture, AsyncWriteTransferFailureIsHandled) {
    testAsyncTransferFailureIsHandled (NIXL_WRITE);
}

} // namespace gtest::obj
