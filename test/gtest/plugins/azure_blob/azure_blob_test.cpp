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
#include <cstdlib>
#include <chrono>
#include <azure/storage/blobs.hpp>
#include <azure/identity/default_azure_credential.hpp>

#include "../plugins_common.h"
#include "../transfer_handler.h"
#include "azure_blob/azure_blob_backend.h"

namespace gtest::plugins::azure_blob {
/**
 * @note To run Azure plugin tests, the AZURE_STORAGE_ACCOUNT_URL environment variable must be set
 * in order to interact with the Azure Storage account used for testing.
 * The tests will automatically create unique Azure Blob containers for each test case and delete
 * them afterwards.
 */

nixl_b_params_t azure_blob_params;
const std::string local_agent_name = "Agent1";
const nixlBackendInitParams azure_blob_test_params = {.localAgent = local_agent_name,
                                                      .type = "AZURE_BLOB",
                                                      .customParams = &azure_blob_params,
                                                      .enableProgTh = false,
                                                      .pthrDelay = 0,
                                                      .syncMode =
                                                          nixl_thread_sync_t::NIXL_THREAD_SYNC_RW};

class setupAzureBlobTestFixture : public setupBackendTestFixture {
protected:
    void
    SetUp() override {
        const char *account_url = std::getenv("AZURE_STORAGE_ACCOUNT_URL");
        ASSERT_NE(account_url, nullptr) << "AZURE_STORAGE_ACCOUNT_URL environment variable must be "
                                           "set to run Azure Blob plugin tests";
        ASSERT_NE(account_url[0], '\0')
            << "AZURE_STORAGE_ACCOUNT_URL environment variable is empty";
        setupTestContainer(account_url);
        localBackendEngine_ = std::make_shared<nixlAzureBlobEngine>(&GetParam());
        setupBackendTestFixture::SetUp();
    }

    void
    TearDown() {
        cleanupTestContainer();
    }

private:
    std::shared_ptr<Azure::Storage::Blobs::BlobContainerClient> container_client_;

    void
    setupTestContainer(const char *account_url) {
        auto now = std::chrono::system_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        auto test_container_name = "nixl-azure-blob-test-" + std::to_string(timestamp);

        auto service_client = std::make_unique<Azure::Storage::Blobs::BlobServiceClient>(
            account_url, std::make_shared<Azure::Identity::DefaultAzureCredential>());

        container_client_ = std::make_shared<Azure::Storage::Blobs::BlobContainerClient>(
            service_client->GetBlobContainerClient(test_container_name));
        container_client_->Create();
        azure_blob_params["container_name"] = test_container_name;
    }

    void
    cleanupTestContainer() {
        if (container_client_) {
            container_client_->Delete();
        }
    }
};

TEST_P(setupAzureBlobTestFixture, XferTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, 1);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupAzureBlobTestFixture, XferMultiBufsTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, 3);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);
    transfer.resetLocalMem();
    transfer.testTransfer(NIXL_READ);
    transfer.checkLocalMem();
}

TEST_P(setupAzureBlobTestFixture, queryMemTest) {
    transferHandler<DRAM_SEG, OBJ_SEG> transfer(
        localBackendEngine_, localBackendEngine_, local_agent_name, local_agent_name, false, 3);
    transfer.setLocalMem();
    transfer.testTransfer(NIXL_WRITE);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-obj-key-0"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-obj-key-1"));
    descs.addDesc(nixlBlobDesc(nixlBasicDesc(), "test-obj-key-nonexistent"));
    std::vector<nixl_query_resp_t> resp;
    localBackendEngine_->queryMem(descs, resp);

    EXPECT_EQ(resp.size(), 3);
    EXPECT_EQ(resp[0].has_value(), true);
    EXPECT_EQ(resp[1].has_value(), true);
    EXPECT_EQ(resp[2].has_value(), false);
}

INSTANTIATE_TEST_SUITE_P(AzureBlobTests,
                         setupAzureBlobTestFixture,
                         testing::Values(azure_blob_test_params));

} // namespace gtest::plugins::azure_blob
