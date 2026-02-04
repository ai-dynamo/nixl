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

namespace nixl::test::device_api {

class writeTest : public test<testParams> {
protected:
    void
    runWrite(const testSetupData &setup_data, size_t num_iters, uint64_t signal_inc) {
        kernelParams params;
        params.operation = operation_t::WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = setup_data.gpuReqHandle;

        applySendMode<writeTest>(params, getSendMode());

        params.write.signalInc = signal_inc;

        const nixl_status_t status = launchKernel(params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel execution failed with status: " << status;
    }
};

TEST_P(writeTest, Basic) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);
    const nixl_mem_t dst_mem_type = getDstMemType();

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, srcMemType, dst_mem_type, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runWrite(setup_data, defaultNumIters, testSignalIncrement));
    EXPECT_TRUE(verifyTestData(sizes, setup_data));
}

TEST_P(writeTest, WithoutSignal) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);
    const nixl_mem_t dst_mem_type = getDstMemType();
    constexpr uint64_t signal_inc = 0;

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, srcMemType, dst_mem_type, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runWrite(setup_data, 1000, signal_inc));
    EXPECT_TRUE(verifyTestData(sizes, setup_data));
}

TEST_P(writeTest, SignalOnly) {
    const std::vector<size_t> sizes;
    const nixl_mem_t dst_mem_type = getDstMemType();

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, srcMemType, dst_mem_type, setup_data));

    ASSERT_NO_FATAL_FAILURE(runWrite(setup_data, 1000, testSignalIncrement));
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         writeTest,
                         testing::ValuesIn(writeTest::getPartialWriteDeviceTestParams()),
                         testNameGenerator::device);
} // namespace nixl::test::device_api
