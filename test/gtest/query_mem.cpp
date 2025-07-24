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
#include <fstream>
#include <filesystem>
#include <optional>
#include "file/file_utils.h"
#include "nixl.h"
#include "nixl_descriptors.h"

class QueryMemTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        // Create temporary test files
        dir_path = "./files_for_query";
        if (!(std::filesystem::exists(dir_path) && std::filesystem::is_directory(dir_path)))
            std::filesystem::create_directory(dir_path);

        test_file1 = dir_path + "/test_query_mem_1.txt";
        test_file2 = dir_path + "/test_query_mem_2.txt";
        non_existent_file = "./non_existent_file.txt";

        // Create test file 1
        {
            std::ofstream file(test_file1);
            file << "Test content for file 1" << std::endl;
            file.close();
        }

        // Create test file 2
        {
            std::ofstream file(test_file2);
            file << "Test content for file 2" << std::endl;
            file.close();
        }
    }

    void
    TearDown() override {
        // Clean up test files
        std::filesystem::remove(test_file1);
        std::filesystem::remove(test_file2);
    }

    std::string dir_path;
    std::string test_file1;
    std::string test_file2;
    std::string non_existent_file;
};

TEST_F(QueryMemTest, QueryMemWithExistingFiles) {
    // Create agent
    nixlAgentConfig cfg(false, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
    nixlAgent agent("test_agent", cfg);

    // Create backend
    nixlBackendH *backend_handle = nullptr;
    nixl_b_params_t params;
    auto status = agent.createBackend("POSIX", params, backend_handle);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(backend_handle, nullptr);

    // Create descriptor list with existing files
    nixlDescList<nixlBlobDesc> descs(FILE_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 0, test_file1));
    descs.addDesc(nixlBlobDesc(0, 0, 0, test_file2));

    // Create extra params with backend
    nixl_opt_args_t extra_params;
    extra_params.backends = {backend_handle};

    std::vector<nixl_query_resp_t> resp;
    status = agent.queryMem(descs, resp, &extra_params);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 2);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_TRUE(resp[0].value().find("size") != resp[0].value().end());
    EXPECT_TRUE(std::stoi(resp[0].value()["size"]) == 24);
    EXPECT_TRUE(resp[1].value().find("size") != resp[1].value().end());
    EXPECT_TRUE(std::stoi(resp[1].value()["size"]) == 24);
}

TEST_F(QueryMemTest, QueryMemWithMixedFiles) {
    // Create agent
    nixlAgentConfig cfg(false, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
    nixlAgent agent("test_agent", cfg);

    // Create backend
    nixlBackendH *backend_handle = nullptr;
    nixl_b_params_t params;
    auto status = agent.createBackend("POSIX", params, backend_handle);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(backend_handle, nullptr);

    // Create descriptor list with existing and non-existing files
    nixlDescList<nixlBlobDesc> descs(FILE_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 0, test_file1));
    descs.addDesc(nixlBlobDesc(0, 0, 0, non_existent_file));
    descs.addDesc(nixlBlobDesc(0, 0, 0, test_file2));

    // Create extra params with backend
    nixl_opt_args_t extra_params;
    extra_params.backends = {backend_handle};

    std::vector<nixl_query_resp_t> resp;
    status = agent.queryMem(descs, resp, &extra_params);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].has_value()); // test_file1 exists
    EXPECT_FALSE(resp[1].has_value()); // non_existent_file doesn't exist
    EXPECT_TRUE(resp[2].has_value()); // test_file2 exists
}

TEST_F(QueryMemTest, QueryMemWithEmptyDescriptors) {
    // Create agent
    nixlAgentConfig cfg(false, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
    nixlAgent agent("test_agent", cfg);

    // Create backend
    nixlBackendH *backend_handle = nullptr;
    nixl_b_params_t params;
    auto status = agent.createBackend("POSIX", params, backend_handle);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(backend_handle, nullptr);

    // Create empty descriptor list
    nixlDescList<nixlBlobDesc> descs(FILE_SEG);

    // Create extra params with backend
    nixl_opt_args_t extra_params;
    extra_params.backends = {backend_handle};

    std::vector<nixl_query_resp_t> resp;
    status = agent.queryMem(descs, resp, &extra_params);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 0);
}

TEST_F(QueryMemTest, QueryMemWithEmptyFilenames) {
    // Create agent
    nixlAgentConfig cfg(false, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
    nixlAgent agent("test_agent", cfg);

    // Create backend
    nixlBackendH *backend_handle = nullptr;
    nixl_b_params_t params;
    auto status = agent.createBackend("POSIX", params, backend_handle);
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_NE(backend_handle, nullptr);

    // Create descriptor list with empty filenames
    nixlDescList<nixlBlobDesc> descs(FILE_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 0, ""));
    descs.addDesc(nixlBlobDesc(0, 0, 0, ""));

    // Create extra params with backend
    nixl_opt_args_t extra_params;
    extra_params.backends = {backend_handle};

    std::vector<nixl_query_resp_t> resp;
    status = agent.queryMem(descs, resp, &extra_params);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 2);
    EXPECT_FALSE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
}

TEST_F(QueryMemTest, QueryMemDirectTest) {
    // Test the queryMem functionality directly using the file utilities
    // This bypasses the agent/backend complexity

    // Create descriptor list with existing files
    nixlDescList<nixlBlobDesc> descs(FILE_SEG);
    descs.addDesc(nixlBlobDesc(0, 0, 0, test_file1));
    descs.addDesc(nixlBlobDesc(0, 0, 0, test_file2));

    // Extract metadata from descriptors which are file names
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i) {
        metadata[i] = descs[i].metaInfo;
    }

    // Use the file utilities directly
    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = nixl::queryFileInfoList(metadata, resp);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 2);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_TRUE(resp[0].value().find("size") != resp[0].value().end());
    EXPECT_TRUE(resp[1].value().find("size") != resp[1].value().end());
}
