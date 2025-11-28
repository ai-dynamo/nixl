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

namespace gtest::nixl::gpu::full_write {

namespace {

class FullWriteTest : public DeviceApiTestBase<DeviceTestParams> {
protected:
    void setupFullWrite(const std::vector<size_t> &sizes, TestSetupData &data) {
        constexpr nixl_mem_t mem_type = VRAM_SEG;
        const size_t data_buf_count = sizes.size();

        for (size_t i = 0; i < data_buf_count; ++i) {
            data.srcBuffers.emplace_back(sizes[i], mem_type);
            data.dstBuffers.emplace_back(sizes[i], mem_type);
        }

        nixl_opt_args_t signal_params = {.backends = {backendHandles_[receiverAgent]}};
        size_t signal_size;
        nixl_status_t status =
            getAgent(receiverAgent).getGpuSignalSize(signal_size, &signal_params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "getGpuSignalSize failed";

        // Add dummy signal buffer
        // TODO: Remove after implementing new createGpuXferReq API
        data.srcBuffers.emplace_back(signal_size, mem_type);
        data.dstBuffers.emplace_back(signal_size, mem_type);

        registerMem(getAgent(senderAgent), data.srcBuffers, mem_type);
        registerMem(getAgent(receiverAgent), data.dstBuffers, mem_type);

        std::vector<MemBuffer> signal_only = {data.dstBuffers.back()};
        auto signal_desc_list = makeDescList<nixlBlobDesc>(signal_only, mem_type);
        status = getAgent(receiverAgent).prepGpuSignal(signal_desc_list, &signal_params);
        ASSERT_EQ(status, NIXL_SUCCESS) << "prepGpuSignal failed";

        ASSERT_NO_FATAL_FAILURE(exchangeMD(senderAgent, receiverAgent));

        createXferRequest(data.srcBuffers, data.dstBuffers, mem_type,
                         data.xferReq, data.gpuReqHandle);
    }

    void runFullWrite(const TestSetupData &setup_data,
                     size_t num_iters,
                     uint64_t signal_inc) {
        constexpr unsigned channel_id = defaultChannelId;

        NixlDeviceKernelParams params = {};
        params.operation = NixlDeviceOperation::FULL_WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = setup_data.gpuReqHandle;

        applySendMode(params, getSendMode());

        params.fullWrite.signalInc = signal_inc;
        params.fullWrite.channelId = channel_id;

        const auto result = launchNixlDeviceKernel(params);
        ASSERT_EQ(result.status, NIXL_SUCCESS)
            << "Kernel execution failed with status: " << result.status;
    }
};

} // namespace

TEST_P(FullWriteTest, Basic) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupFullWrite(sizes, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runFullWrite(setup_data, defaultNumIters,
                                        testSignalIncrement));
    ASSERT_NO_FATAL_FAILURE(verifyTestData(sizes, setup_data));
}

TEST_P(FullWriteTest, WithoutSignal) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);
    constexpr uint64_t signal_inc = 0;

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupFullWrite(sizes, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runFullWrite(setup_data, 1000, signal_inc));
    ASSERT_NO_FATAL_FAILURE(verifyTestData(sizes, setup_data));
}

TEST_P(FullWriteTest, SignalOnly) {
    const std::vector<size_t> sizes;

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupFullWrite(sizes, setup_data));

    ASSERT_NO_FATAL_FAILURE(runFullWrite(setup_data, 1000,
                                        testSignalIncrement));
}

} // namespace gtest::nixl::gpu::full_write

using gtest::nixl::gpu::full_write::FullWriteTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         FullWriteTest,
                         testing::ValuesIn(FullWriteTest::getPartialWriteDeviceTestParams()),
                         TestNameGenerator::device);
