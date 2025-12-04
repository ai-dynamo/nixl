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
#include "common/device_kernels.cuh"
#include <algorithm>

namespace gtest::nixl::gpu::partial_write {

namespace {

class PartialWriteTest : public DeviceApiTestBase<DeviceTestParams> {
protected:
    void setupPartialWrite(const std::vector<size_t> &sizes, TestSetupData &data) {
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

    void runPartialWrite(const TestSetupData &setup_data,
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

        CudaArray<unsigned> indices_gpu(data_buf_count);
        CudaArray<size_t> sizes_gpu(data_buf_count);
        CudaArray<size_t> local_offsets_gpu(data_buf_count);
        CudaArray<size_t> remote_offsets_gpu(data_buf_count);

        indices_gpu.copyFromHost(indices_host);
        sizes_gpu.copyFromHost(sizes);
        local_offsets_gpu.copyFromHost(local_offsets_host);
        remote_offsets_gpu.copyFromHost(remote_offsets_host);

        const unsigned signal_desc_index = static_cast<unsigned>(setup_data.dstBuffers.size() - 1);
        constexpr size_t signal_offset = 0;

        NixlDeviceKernelParams params = {};
        params.operation = NixlDeviceOperation::PARTIAL_WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = setup_data.gpuReqHandle;

        applySendMode(params, getSendMode());

        params.partialWrite.count = data_buf_count;
        params.partialWrite.descIndices = indices_gpu.get();
        params.partialWrite.sizes = sizes_gpu.get();
        params.partialWrite.localOffsets = local_offsets_gpu.get();
        params.partialWrite.remoteOffsets = remote_offsets_gpu.get();
        params.partialWrite.signalDescIndex = signal_desc_index;
        params.partialWrite.signalInc = signal_inc;
        params.partialWrite.signalOffset = signal_offset;
        params.partialWrite.channelId = defaultChannelId;

        const auto result = launchNixlDeviceKernel(params);
        ASSERT_EQ(result.status, NIXL_SUCCESS)
            << "Kernel execution failed with status: " << result.status;
    }
};

} // namespace

TEST_P(PartialWriteTest, Basic) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupPartialWrite(sizes, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runPartialWrite(setup_data, sizes, defaultNumIters,
                                            testSignalIncrement));
    ASSERT_NO_FATAL_FAILURE(verifyTestData(sizes, setup_data));
}

TEST_P(PartialWriteTest, WithoutSignal) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);
    constexpr uint64_t signal_inc = 0;

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupPartialWrite(sizes, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runPartialWrite(setup_data, sizes, defaultNumIters,
                                            signal_inc));
    ASSERT_NO_FATAL_FAILURE(verifyTestData(sizes, setup_data));
}

TEST_P(PartialWriteTest, SignalOnly) {
    const std::vector<size_t> sizes;

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupPartialWrite(sizes, setup_data));

    ASSERT_NO_FATAL_FAILURE(runPartialWrite(setup_data, sizes, defaultNumIters,
                                            testSignalIncrement));
}

} // namespace gtest::nixl::gpu::partial_write

using gtest::nixl::gpu::partial_write::PartialWriteTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         PartialWriteTest,
                         testing::ValuesIn(PartialWriteTest::getPartialWriteDeviceTestParams()),
                         TestNameGenerator::device);
