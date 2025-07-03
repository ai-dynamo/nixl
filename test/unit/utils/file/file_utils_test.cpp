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
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include "file/file_utils.h"

namespace {

class FileUtilsTest : public testing::Test {
protected:
    const std::string test_file = "/tmp/nixl_test_file";
    const std::string test_file_rd = "RD:/tmp/nixl_test_file";
    const std::string test_file_wr = "WR:/tmp/nixl_test_file";
    const std::string test_file_rw = "RW:/tmp/nixl_test_file";
    int fd;

    void SetUp() override {
        // Ensure test file doesn't exist at start
        std::filesystem::remove(test_file);
    }

    void TearDown() override {
        // Clean up any test files
        std::filesystem::remove(test_file);
    }

    // Helper to write test data to file
    void writeTestData(int fd, const std::string& data) {
        ASSERT_EQ(write(fd, data.c_str(), data.length()), static_cast<ssize_t>(data.length()));
    }

    // Helper to read and verify test data
    std::string readTestData(int fd, size_t length) {
        std::string buffer(length, '\0');
        EXPECT_EQ(read(fd, &buffer[0], length), static_cast<ssize_t>(length));
        return buffer;
    }
};

TEST_F(FileUtilsTest, CreateAndCloseFile) {
    nixl_status_t status = nixlFileUtils::openFile(test_file, nixlFileMode::CREATE, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);

    status = nixlFileUtils::closeFile(fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(FileUtilsTest, FileExists) {
    // First verify file doesn't exist
    nixl_status_t status = nixlFileUtils::fileExists(test_file);
    EXPECT_EQ(status, NIXL_ERR_NOT_FOUND);

    // Create file
    status = nixlFileUtils::openFile(test_file, nixlFileMode::CREATE, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Verify file exists
    status = nixlFileUtils::fileExists(test_file);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(FileUtilsTest, UnlinkFile) {
    // Create file
    nixl_status_t status = nixlFileUtils::openFile(test_file, nixlFileMode::CREATE, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Verify file exists
    status = nixlFileUtils::fileExists(test_file);
    EXPECT_EQ(status, NIXL_SUCCESS);

    // Delete file
    status = nixlFileUtils::unlinkFile(test_file);
    EXPECT_EQ(status, NIXL_SUCCESS);

    // Verify file no longer exists
    status = nixlFileUtils::fileExists(test_file);
    EXPECT_EQ(status, NIXL_ERR_NOT_FOUND);
}

TEST_F(FileUtilsTest, OpenNonExistentFile) {
    nixl_status_t status = nixlFileUtils::openFile(test_file, nixlFileMode::READ_ONLY, fd);
    EXPECT_EQ(status, NIXL_ERR_NOT_FOUND);
}

TEST_F(FileUtilsTest, FileModesTest) {
    // Test CREATE mode
    nixl_status_t status = nixlFileUtils::openFile(test_file, nixlFileMode::CREATE, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Test READ_ONLY mode
    status = nixlFileUtils::openFile(test_file, nixlFileMode::READ_ONLY, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Test WRITE_ONLY mode
    status = nixlFileUtils::openFile(test_file, nixlFileMode::WRITE_ONLY, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Test READ_WRITE mode
    status = nixlFileUtils::openFile(test_file, nixlFileMode::READ_WRITE, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);
}

TEST_F(FileUtilsTest, PrefixedPathReadOnly) {
    // First create a file
    nixl_status_t status = nixlFileUtils::openFile(test_file, nixlFileMode::CREATE, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Now try to open it with RD: prefix
    status = nixlFileUtils::openPrefixedFile(test_file_rd, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);
}

TEST_F(FileUtilsTest, PrefixedPathWriteOnly) {
    // Try to create/open with WR: prefix
    nixl_status_t status = nixlFileUtils::openPrefixedFile(test_file_wr, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Verify file was created
    status = nixlFileUtils::fileExists(test_file);
    EXPECT_EQ(status, NIXL_SUCCESS);
}

TEST_F(FileUtilsTest, PrefixedPathReadWrite) {
    // Try to create/open with RW: prefix
    nixl_status_t status = nixlFileUtils::openPrefixedFile(test_file_rw, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Try to open again with RW: prefix
    status = nixlFileUtils::openPrefixedFile(test_file_rw, fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);
}

TEST_F(FileUtilsTest, PrefixedPathInvalidPrefix) {
    // Try with invalid prefix
    nixl_status_t status = nixlFileUtils::openPrefixedFile("XX:/tmp/file", fd);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    // Try with no prefix
    status = nixlFileUtils::openPrefixedFile("/tmp/file", fd);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

    // Try with empty string
    status = nixlFileUtils::openPrefixedFile("", fd);
    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST_F(FileUtilsTest, PrefixedPathCaseInsensitive) {
    // Try lowercase prefix
    nixl_status_t status = nixlFileUtils::openPrefixedFile("wr:/tmp/nixl_test_file", fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);

    // Try mixed case prefix
    status = nixlFileUtils::openPrefixedFile("Rw:/tmp/nixl_test_file", fd);
    EXPECT_EQ(status, NIXL_SUCCESS);
    nixlFileUtils::closeFile(fd);
}

TEST_F(FileUtilsTest, PrefixedPathFileOperations) {
    const std::string test_data = "Hello NIXL!";
    nixl_status_t status;

    // Write data using WR: prefix
    status = nixlFileUtils::openPrefixedFile(test_file_wr, fd);
    ASSERT_EQ(status, NIXL_SUCCESS);
    writeTestData(fd, test_data);
    nixlFileUtils::closeFile(fd);

    // Read data using RD: prefix
    status = nixlFileUtils::openPrefixedFile(test_file_rd, fd);
    ASSERT_EQ(status, NIXL_SUCCESS);
    std::string read_data = readTestData(fd, test_data.length());
    EXPECT_EQ(read_data, test_data);
    nixlFileUtils::closeFile(fd);

    // Modify data using RW: prefix
    status = nixlFileUtils::openPrefixedFile(test_file_rw, fd);
    ASSERT_EQ(status, NIXL_SUCCESS);

    // Seek to end and write more data
    ASSERT_NE(lseek(fd, 0, SEEK_END), -1);
    const std::string more_data = " More data!";
    writeTestData(fd, more_data);

    // Seek back to start and read all data
    ASSERT_NE(lseek(fd, 0, SEEK_SET), -1);
    std::string full_data = readTestData(fd, test_data.length() + more_data.length());
    EXPECT_EQ(full_data, test_data + more_data);

    nixlFileUtils::closeFile(fd);

    // Verify file size
    size_t file_size;
    status = nixlFileUtils::openPrefixedFile(test_file_rd, fd);
    ASSERT_EQ(status, NIXL_SUCCESS);
    status = nixlFileUtils::getFileSize(fd, file_size);
    ASSERT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(file_size, test_data.length() + more_data.length());
    nixlFileUtils::closeFile(fd);

    // Test write-only mode restrictions
    status = nixlFileUtils::openPrefixedFile(test_file_wr, fd);
    ASSERT_EQ(status, NIXL_SUCCESS);
    char buf[1];
    EXPECT_EQ(read(fd, buf, 1), -1); // Should fail as file is write-only
    nixlFileUtils::closeFile(fd);

    // Test read-only mode restrictions
    status = nixlFileUtils::openPrefixedFile(test_file_rd, fd);
    ASSERT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(write(fd, "x", 1), -1); // Should fail as file is read-only
    nixlFileUtils::closeFile(fd);
}

} // namespace
