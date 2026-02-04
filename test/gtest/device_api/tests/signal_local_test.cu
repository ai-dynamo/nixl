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

#include "common/device_test_base.cuh"
#include "common/test_array.h"

namespace nixl::test::device_api {
class signalLocalTest : public test<nixl_gpu_level_t> {
protected:
    void
    setupLocalSignal() {
        nixl_opt_args_t extra_params = {.backends = {getBackendHandle(senderAgent)}};
        size_t signal_size = 0;
        nixl_status_t status = getAgent(senderAgent).getGpuSignalSize(signal_size, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        signalBuffers_.clear();
        signalBuffers_.emplace_back(signal_size, VRAM_SEG);

        cudaMemset(signalBuffers_[0].get(), 0, signal_size);
    }

    void *
    getSignalBuffer() {
        return signalBuffers_.empty() ? nullptr : signalBuffers_[0].get();
    }

    void
    writeSignal(void *signal_addr, uint64_t value, size_t num_threads) {
        kernelParams params;
        params.operation = operation_t::SIGNAL_WRITE;
        params.level = GetParam();
        params.numThreads = num_threads;
        params.numBlocks = 1;
        params.numIters = 1;

        params.signalWrite.signalAddr = signal_addr;
        params.signalWrite.value = value;

        const nixl_status_t status = launchKernel(params);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

    void
    readAndVerifySignal(const void *signal_addr, uint64_t expected_value, size_t num_threads) {
        kernelParams params;
        params.operation = operation_t::SIGNAL_WAIT;
        params.level = GetParam();
        params.numThreads = num_threads;
        params.numBlocks = 1;
        params.numIters = 1;

        params.signalWait.signalAddr = signal_addr;
        params.signalWait.expectedValue = expected_value;

        const nixl_status_t status = launchKernel(params);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

private:
    std::vector<testArray<uint8_t>> signalBuffers_;
};

TEST_P(signalLocalTest, WriteRead) {
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal());

    constexpr uint64_t test_value = testPattern1;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), test_value, defaultNumThreads));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), test_value, defaultNumThreads));
}

TEST_P(signalLocalTest, MultipleWrites) {
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal());

    const std::vector<uint64_t> test_values = {testPattern1, testPattern2, testSignalIncrement};

    for (const auto &value : test_values) {
        ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), value, defaultNumThreads));
        ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), value, defaultNumThreads));
    }
}

TEST_P(signalLocalTest, SingleThread) {
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal());

    constexpr uint64_t test_value = testPattern1;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), test_value, 1));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), test_value, 1));
}

TEST_P(signalLocalTest, ZeroValue) {
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal());

    constexpr uint64_t zero_value = 0;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), zero_value, defaultNumThreads));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), zero_value, defaultNumThreads));
}

TEST_P(signalLocalTest, MaxValue) {
    ASSERT_NO_FATAL_FAILURE(setupLocalSignal());

    constexpr uint64_t max_value = UINT64_MAX;

    ASSERT_NO_FATAL_FAILURE(writeSignal(getSignalBuffer(), max_value, defaultNumThreads));
    ASSERT_NO_FATAL_FAILURE(readAndVerifySignal(getSignalBuffer(), max_value, defaultNumThreads));
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         signalLocalTest,
                         testing::ValuesIn(signalLocalTest::getLevels()),
                         testNameGenerator::level);
} // namespace nixl::test::device_api
