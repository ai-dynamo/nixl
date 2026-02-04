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

#include <cstring>

namespace nixl::test::device_api {

using single_write_params_t = testParams;

class singleWriteTest : public test<single_write_params_t> {
protected:
    void
    runTest(testSetupData &setup_data, size_t size, size_t num_iters) {
        kernelParams params;
        params.operation = operation_t::SINGLE_WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = setup_data.gpuReqHandle;
        params.singleWrite = {0, 0, 0, size};

        applySendMode<singleWriteTest>(params, getSendMode());

        launchAndCheckKernel(params);
    }
};

TEST_P(singleWriteTest, Basic) {
    constexpr size_t size = defaultBufferSize;
    constexpr size_t count = defaultBufferCount;
    const nixl_mem_t dst_mem_type = getDstMemType();

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWriteTest(size, count, srcMemType, dst_mem_type, setup_data));

    auto *src = reinterpret_cast<uint32_t *>(setup_data.srcBuffers[0].get());
    constexpr uint32_t pattern = testPattern2;
    cudaMemset(src, 0, size);
    cudaMemcpy(src, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    ASSERT_NO_FATAL_FAILURE(runTest(setup_data, size, defaultNumIters));

    uint32_t dst = 0;
    if (dst_mem_type == DRAM_SEG) {
        std::memcpy(&dst, setup_data.dstBuffers[0].get(), sizeof(uint32_t));
    } else {
        cudaMemcpy(&dst,
                   reinterpret_cast<uint32_t *>(setup_data.dstBuffers[0].get()),
                   sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
    }
    ASSERT_EQ(dst, pattern);
}

// TODO: Create separate multi-worker test with dedicated fixture that creates 32 workers

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         singleWriteTest,
                         testing::ValuesIn(singleWriteTest::getDeviceTestParams()),
                         testNameGenerator::device);
} // namespace nixl::test::device_api
