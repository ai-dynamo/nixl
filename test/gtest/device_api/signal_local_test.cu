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

#include "agent.h"
#include "kernel.h"
#include "params.h"
#include "mem_type_array.h"

namespace {
constexpr size_t default_num_threads = 32;
constexpr uint32_t test_pattern1 = 0xDEADBEEF;
constexpr uint32_t test_pattern2 = 0xCAFEBABE;
constexpr uint64_t test_signal_increment = 42;

const std::vector<nixl_gpu_level_t> levels = {
    nixl_gpu_level_t::BLOCK,
    nixl_gpu_level_t::WARP,
    nixl_gpu_level_t::THREAD,
};
} // namespace

namespace nixl::device_api {
class signalLocalTest : public testing::TestWithParam<nixl_gpu_level_t> {
protected:
    signalLocalTest()
        : agent_("local_agent"),
          signalBuffer_(agent_.getGpuSignalSize(), VRAM_SEG) {}

    [[nodiscard]] nixl_status_t
    writeAndVerify(uint64_t value, size_t num_threads = default_num_threads);

private:
    [[nodiscard]] nixl_status_t
    write(uint64_t value, size_t num_threads);

    [[nodiscard]] nixl_status_t
    verify(uint64_t expected_value, size_t num_threads);

    const agent agent_;
    memTypeArray<uint8_t> signalBuffer_;
};

nixl_status_t
signalLocalTest::writeAndVerify(uint64_t value, size_t num_threads) {
    cudaMemset(signalBuffer_.get(), 0, signalBuffer_.size());
    const nixl_status_t status = write(value, num_threads);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    return verify(value, num_threads);
}

nixl_status_t
signalLocalTest::write(uint64_t value, size_t num_threads) {
    kernelParams kernel_params(operation_t::SIGNAL_WRITE, GetParam(), num_threads);
    kernel_params.signalWrite.signalAddr = signalBuffer_.get();
    kernel_params.signalWrite.value = value;
    return launchKernel(kernel_params);
}

nixl_status_t
signalLocalTest::verify(uint64_t expected_value, size_t num_threads) {
    kernelParams kernel_params(operation_t::SIGNAL_WAIT, GetParam(), num_threads);

    kernel_params.signalWait.signalAddr = signalBuffer_.get();
    kernel_params.signalWait.expectedValue = expected_value;

    return launchKernel(kernel_params);
}

TEST_P(signalLocalTest, WriteRead) {
    EXPECT_EQ(writeAndVerify(test_pattern1), NIXL_SUCCESS);
}

TEST_P(signalLocalTest, MultipleWrites) {
    const std::vector<uint64_t> test_values{test_pattern1, test_pattern2, test_signal_increment};
    for (const auto &test_value : test_values) {
        EXPECT_EQ(writeAndVerify(test_value), NIXL_SUCCESS);
    }
}

TEST_P(signalLocalTest, SingleThread) {
    EXPECT_EQ(writeAndVerify(test_pattern1, 1), NIXL_SUCCESS);
}

TEST_P(signalLocalTest, ZeroValue) {
    EXPECT_EQ(writeAndVerify(0), NIXL_SUCCESS);
}

TEST_P(signalLocalTest, MaxValue) {
    EXPECT_EQ(writeAndVerify(UINT64_MAX), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         signalLocalTest,
                         testing::ValuesIn(levels),
                         paramsLevelToString);
} // namespace nixl::device_api
