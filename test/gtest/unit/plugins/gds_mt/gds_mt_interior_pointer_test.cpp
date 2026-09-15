/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "nixl.h"

namespace {

constexpr size_t kAllocationSize = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kInteriorOffset = 4ULL * 1024ULL * 1024ULL;
constexpr size_t kPayloadSize = 8ULL * 1024ULL * 1024ULL;
constexpr uint8_t kInitialValue = 0x5a;
constexpr uint8_t kClearedValue = 0xcd;
constexpr std::chrono::seconds kTransferTimeout(120);

static_assert(kInteriorOffset + kPayloadSize < kAllocationSize);

class GdsMtInteriorPointerTest : public ::testing::Test {
protected:
    GdsMtInteriorPointerTest()
        : agent_("GDSMTInteriorPointerTest", nixlAgentConfig(true)),
          gpu_registration_(VRAM_SEG),
          file_registration_(FILE_SEG) {}

    void
    SetUp() override {
        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "An NVIDIA GPU is required for the GDS_MT integration test";
        }

        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        nixl_b_params_t backend_params;
        backend_params["thread_count"] = "4";
        ASSERT_EQ(agent_.createBackend("GDS_MT", backend_params, backend_), NIXL_SUCCESS);
    }

    void
    TearDown() override {
        if (file_registered_) {
            EXPECT_EQ(agent_.deregisterMem(file_registration_), NIXL_SUCCESS);
        }
        if (gpu_registered_) {
            EXPECT_EQ(agent_.deregisterMem(gpu_registration_), NIXL_SUCCESS);
        }
        if (file_fd_ >= 0) {
            close(file_fd_);
            file_fd_ = -1;
        }
        if (gpu_buffer_ != nullptr) {
            EXPECT_EQ(cudaFree(gpu_buffer_), cudaSuccess);
            gpu_buffer_ = nullptr;
        }
        if (!file_path_.empty()) {
            std::error_code error;
            std::filesystem::remove(file_path_, error);
        }
    }

    nixl_status_t
    waitForTransfer(nixlXferReqH *request) {
        nixl_status_t status = agent_.postXferReq(request);
        if (status < 0) {
            return status;
        }

        const auto deadline = std::chrono::steady_clock::now() + kTransferTimeout;
        while (status == NIXL_IN_PROG) {
            status = agent_.getXferStatus(request);
            if (status == NIXL_IN_PROG && std::chrono::steady_clock::now() >= deadline) {
                return NIXL_ERR_BACKEND;
            }
        }
        return status;
    }

    static nixlBlobDesc
    makeGpuDescriptor(void *buffer, size_t length) {
        return nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer), length, 0);
    }

    nixlAgent agent_;
    nixlBackendH *backend_ = nullptr;
    nixl_reg_dlist_t gpu_registration_;
    nixl_reg_dlist_t file_registration_;
    bool gpu_registered_ = false;
    bool file_registered_ = false;
    void *gpu_buffer_ = nullptr;
    int file_fd_ = -1;
    std::filesystem::path file_path_;
};

TEST_F(GdsMtInteriorPointerTest, WritesAndReadsUsingInteriorPointer) {
    const char *test_directory = std::getenv("NIXL_GDS_TEST_DIR");
    ASSERT_NE(test_directory, nullptr);
    ASSERT_TRUE(std::filesystem::is_directory(test_directory));

    file_path_ = std::filesystem::path(test_directory) / "gds_mt_interior_pointer_test.bin";
    file_fd_ = open(file_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
    ASSERT_GE(file_fd_, 0);
    ASSERT_EQ(ftruncate(file_fd_, kPayloadSize), 0);

    ASSERT_EQ(cudaMalloc(&gpu_buffer_, kAllocationSize), cudaSuccess);

    std::vector<uint8_t> initial_contents(kAllocationSize, kInitialValue);
    std::vector<uint8_t> expected_contents(kAllocationSize, kClearedValue);
    for (size_t index = 0; index < kPayloadSize; ++index) {
        const uint8_t pattern_value = static_cast<uint8_t>((index * 17 + 3) & 0xff);
        initial_contents[kInteriorOffset + index] = pattern_value;
        expected_contents[kInteriorOffset + index] = pattern_value;
    }

    ASSERT_EQ(
        cudaMemcpy(
            gpu_buffer_, initial_contents.data(), initial_contents.size(), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const nixlBlobDesc full_gpu_desc = makeGpuDescriptor(gpu_buffer_, kAllocationSize);
    gpu_registration_.addDesc(full_gpu_desc);
    ASSERT_EQ(agent_.registerMem(gpu_registration_), NIXL_SUCCESS);
    gpu_registered_ = true;

    const nixlBlobDesc file_desc(0, kPayloadSize, file_fd_);
    file_registration_.addDesc(file_desc);
    ASSERT_EQ(agent_.registerMem(file_registration_), NIXL_SUCCESS);
    file_registered_ = true;

    nixl_reg_dlist_t write_gpu(VRAM_SEG);
    nixl_reg_dlist_t write_file(FILE_SEG);
    write_gpu.addDesc(
        makeGpuDescriptor(static_cast<uint8_t *>(gpu_buffer_) + kInteriorOffset, kPayloadSize));
    write_file.addDesc(file_desc);

    nixlXferReqH *write_request = nullptr;
    ASSERT_EQ(agent_.createXferReq(NIXL_WRITE,
                                   write_gpu.trim(),
                                   write_file.trim(),
                                   "GDSMTInteriorPointerTest",
                                   write_request),
              NIXL_SUCCESS);

    const auto write_start = std::chrono::steady_clock::now();
    const nixl_status_t write_status = waitForTransfer(write_request);
    const auto write_latency = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - write_start);
    agent_.releaseXferReq(write_request);
    ASSERT_EQ(write_status, NIXL_SUCCESS);
    ASSERT_EQ(fsync(file_fd_), 0);

    nixl_reg_dlist_t read_gpu(VRAM_SEG);
    nixl_reg_dlist_t read_file(FILE_SEG);
    read_gpu.addDesc(
        makeGpuDescriptor(static_cast<uint8_t *>(gpu_buffer_) + kInteriorOffset, kPayloadSize));
    read_file.addDesc(file_desc);

    ASSERT_EQ(cudaMemset(gpu_buffer_, kClearedValue, kAllocationSize), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    nixlXferReqH *read_request = nullptr;
    ASSERT_EQ(
        agent_.createXferReq(
            NIXL_READ, read_gpu.trim(), read_file.trim(), "GDSMTInteriorPointerTest", read_request),
        NIXL_SUCCESS);

    const auto read_start = std::chrono::steady_clock::now();
    const nixl_status_t read_status = waitForTransfer(read_request);
    const auto read_latency = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - read_start);
    agent_.releaseXferReq(read_request);
    ASSERT_EQ(read_status, NIXL_SUCCESS);

    std::vector<uint8_t> actual_contents(kAllocationSize);
    ASSERT_EQ(
        cudaMemcpy(
            actual_contents.data(), gpu_buffer_, actual_contents.size(), cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_TRUE(std::all_of(actual_contents.begin(),
                            actual_contents.begin() + kInteriorOffset,
                            [](uint8_t value) { return value == kClearedValue; }));
    EXPECT_TRUE(std::equal(actual_contents.begin() + kInteriorOffset,
                           actual_contents.begin() + kInteriorOffset + kPayloadSize,
                           expected_contents.begin() + kInteriorOffset));
    EXPECT_TRUE(std::all_of(actual_contents.begin() + kInteriorOffset + kPayloadSize,
                            actual_contents.end(),
                            [](uint8_t value) { return value == kClearedValue; }));

    RecordProperty("allocation_size_bytes", std::to_string(kAllocationSize));
    RecordProperty("interior_offset_bytes", std::to_string(kInteriorOffset));
    RecordProperty("payload_size_bytes", std::to_string(kPayloadSize));
    RecordProperty("write_latency_us", std::to_string(write_latency.count()));
    RecordProperty("read_latency_us", std::to_string(read_latency.count()));
}

} // namespace
