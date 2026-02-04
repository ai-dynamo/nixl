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
#include "common/device_kernels.cuh"
#include "common/test_array.h"
#include <algorithm>

namespace nixl::test::device_api {
class partialWriteTest : public test<testParams> {
protected:
    void
    runPartialWrite(const testSetupData &setup_data,
                    const std::vector<size_t> &sizes,
                    size_t num_iters,
                    uint64_t signal_inc) {
        const size_t data_buf_count = sizes.size();

        std::vector<unsigned> indices_host(data_buf_count);
        std::vector<size_t> local_offsets_host(data_buf_count, 0);
        std::vector<size_t> remote_offsets_host(data_buf_count, 0);

        for (size_t i = 0; i < data_buf_count; ++i) {
            indices_host[i] = static_cast<unsigned>(i);
        }

        testArray<unsigned> indices_gpu(data_buf_count);
        testArray<size_t> sizes_gpu(data_buf_count);
        testArray<size_t> local_offsets_gpu(data_buf_count);
        testArray<size_t> remote_offsets_gpu(data_buf_count);

        indices_gpu.copyFromHost(indices_host);
        sizes_gpu.copyFromHost(sizes);
        local_offsets_gpu.copyFromHost(local_offsets_host);
        remote_offsets_gpu.copyFromHost(remote_offsets_host);

        const unsigned signal_desc_index = static_cast<unsigned>(setup_data.dstBuffers.size() - 1);
        constexpr size_t signal_offset = 0;

        kernelParams params;
        params.operation = operation_t::PARTIAL_WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = setup_data.gpuReqHandle;

        applySendMode<partialWriteTest>(params, getSendMode());

        params.partialWrite.count = data_buf_count;
        params.partialWrite.descIndices = indices_gpu.get();
        params.partialWrite.sizes = sizes_gpu.get();
        params.partialWrite.localOffsets = local_offsets_gpu.get();
        params.partialWrite.remoteOffsets = remote_offsets_gpu.get();
        params.partialWrite.signalDescIndex = signal_desc_index;
        params.partialWrite.signalInc = signal_inc;
        params.partialWrite.signalOffset = signal_offset;

        const nixl_status_t status = launchKernel(params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Kernel execution failed with status: " << status;
    }
};

TEST_P(partialWriteTest, Basic) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);
    const nixl_mem_t dst_mem_type = getDstMemType();

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, srcMemType, dst_mem_type, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(
        runPartialWrite(setup_data, sizes, defaultNumIters, testSignalIncrement));
    EXPECT_TRUE(verifyTestData(sizes, setup_data));
}

TEST_P(partialWriteTest, WithoutSignal) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);
    const nixl_mem_t dst_mem_type = getDstMemType();
    constexpr uint64_t signal_inc = 0;

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, srcMemType, dst_mem_type, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runPartialWrite(setup_data, sizes, defaultNumIters, signal_inc));
    EXPECT_TRUE(verifyTestData(sizes, setup_data));
}

TEST_P(partialWriteTest, SignalOnly) {
    const std::vector<size_t> sizes;
    const nixl_mem_t dst_mem_type = getDstMemType();

    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, srcMemType, dst_mem_type, setup_data));

    ASSERT_NO_FATAL_FAILURE(
        runPartialWrite(setup_data, sizes, defaultNumIters, testSignalIncrement));
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         partialWriteTest,
                         testing::ValuesIn(partialWriteTest::getPartialWriteDeviceTestParams()),
                         testNameGenerator::device);
} // namespace nixl::test::device_api
