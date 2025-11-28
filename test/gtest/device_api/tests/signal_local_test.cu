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

#include "common/device_test_base.cuh"

namespace gtest::nixl::gpu::signal_local {

namespace {

class SignalLocalTest : public DeviceApiTestBase<nixl_gpu_level_t> {
protected:
    std::vector<MemBuffer> signal_buffers_;

    void setupLocalSignal(size_t &signal_size) {
        nixl_opt_args_t extra_params = {.backends = {backendHandles_[senderAgent]}};
        nixl_status_t status =
            getAgent(senderAgent).getGpuSignalSize(signal_size, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        signal_buffers_.clear();
        signal_buffers_.emplace_back(signal_size, VRAM_SEG);

        cudaMemset(signal_buffers_[0].get(), 0, signal_size);
    }

    void* getSignalBuffer() {
        return signal_buffers_.empty() ? nullptr : signal_buffers_[0].get();
    }

    void writeSignal(void *signal_addr, uint64_t value, size_t num_threads) {
        NixlDeviceKernelParams params = {};
        params.operation = NixlDeviceOperation::SIGNAL_WRITE;
        params.level = GetParam();
        params.numThreads = num_threads;
        params.numBlocks = 1;
        params.numIters = 1;

        params.signalWrite.signalAddr = signal_addr;
        params.signalWrite.value = value;

        const auto result = launchNixlDeviceKernel(params);
        ASSERT_EQ(result.status, NIXL_SUCCESS);
    }

    void readAndVerifySignal(const void *signal_addr, uint64_t expected_value, size_t num_threads) {
        CudaArray<uint64_t> result_ptr(1);

        NixlDeviceKernelParams params = {};
        params.operation = NixlDeviceOperation::SIGNAL_READ;
        params.level = GetParam();
        params.numThreads = num_threads;
        params.numBlocks = 1;
        params.numIters = 1;

        params.signalRead.signalAddr = signal_addr;
        params.signalRead.expectedValue = expected_value;
        params.signalRead.resultPtr = result_ptr.get();

        const auto result = launchNixlDeviceKernel(params);
        ASSERT_EQ(result.status, NIXL_SUCCESS);

        uint64_t actual_value = 0;
        result_ptr.copyToHost(&actual_value, 1);
        ASSERT_EQ(actual_value, expected_value);
    }
};

} // namespace

TEST_P(SignalLocalTest, WriteRead) {
    size_t signal_size;
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal(signal_size));

    constexpr uint64_t test_value = testPattern1;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), test_value, defaultNumThreads));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), test_value, defaultNumThreads));
}

TEST_P(SignalLocalTest, MultipleWrites) {
    size_t signal_size;
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal(signal_size));

    const std::vector<uint64_t> test_values = {testPattern1, testPattern2, testSignalIncrement};

    for (const auto value : test_values) {
        ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), value, defaultNumThreads));
        ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), value, defaultNumThreads));
    }
}

TEST_P(SignalLocalTest, SingleThread) {
    size_t signal_size;
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal(signal_size));

    constexpr uint64_t test_value = testPattern1;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), test_value, 1));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), test_value, 1));
}

TEST_P(SignalLocalTest, ZeroValue) {
    size_t signal_size;
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal(signal_size));

    constexpr uint64_t zero_value = 0;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), zero_value, defaultNumThreads));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), zero_value, defaultNumThreads));
}

TEST_P(SignalLocalTest, MaxValue) {
    size_t signal_size;
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal(signal_size));

    constexpr uint64_t max_value = UINT64_MAX;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), max_value, defaultNumThreads));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), max_value, defaultNumThreads));
}

} // namespace gtest::nixl::gpu::signal_local

using gtest::nixl::gpu::signal_local::SignalLocalTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi, SignalLocalTest,
                         testing::ValuesIn(SignalLocalTest::getTestLevels()),
                         TestNameGenerator::level);
