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

namespace gtest::nixl::gpu::write {

namespace {

    class writeTest : public deviceApiTestBase<device_test_params_t> {
    protected:
        void
        runWrite(const testSetupData &setup_data, size_t num_iters, uint64_t signal_inc) {
            nixlDeviceKernelParams params;
            params.operation = nixl_device_operation_t::WRITE;
            params.level = getLevel();
            params.numThreads = defaultNumThreads;
            params.numBlocks = 1;
            params.numIters = num_iters;
            params.reqHandle = setup_data.gpuReqHandle;

            applySendMode<writeTest>(params, getSendMode());

            params.write.signalInc = signal_inc;

            const nixl_status_t status = launchNixlDeviceKernel(params);
            ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel execution failed with status: " << status;
        }
    };

} // namespace

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

} // namespace gtest::nixl::gpu::write

using gtest::nixl::gpu::write::writeTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         writeTest,
                         testing::ValuesIn(writeTest::getPartialWriteDeviceTestParams()),
                         testNameGenerator::device);
