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

namespace gtest::nixl::gpu::single_write {

namespace {

class SingleWriteTest : public DeviceApiTestBase<DeviceTestParams> {
protected:
    void runTest(TestSetupData &data, size_t size, size_t num_iters) {
        NixlDeviceKernelParams params = {};
        params.operation = NixlDeviceOperation::SINGLE_WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = data.gpuReqHandle;
        params.singleWrite = {0, 0, 0, size, defaultChannelId};

        applySendMode(params, getSendMode());

        launchAndCheckKernel(params);
    }
};

} // namespace

TEST_P(SingleWriteTest, Basic) {
    constexpr size_t size = defaultBufferSize;
    constexpr size_t count = defaultBufferCount;

    TestSetupData data;
    auto guard = data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWriteTest(size, count, VRAM_SEG, data));

    auto *src = static_cast<uint32_t *>(data.srcBuffers[0].get());
    constexpr uint32_t pattern = testPattern2;
    cudaMemset(src, 0, size);
    cudaMemcpy(src, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);

    ASSERT_NO_FATAL_FAILURE(runTest(data, size, defaultNumIters));

    uint32_t dst;
    cudaMemcpy(&dst, static_cast<uint32_t *>(data.dstBuffers[0].get()),
               sizeof(uint32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(dst, pattern);
}

// Test with multiple workers using custom worker_id parameter
TEST_P(SingleWriteTest, MultipleWorkers) {
    constexpr size_t size = 4096;
    constexpr size_t num_iters = 100;

    std::vector<std::vector<MemBuffer>> src_buffers(numUcxWorkers);
    std::vector<std::vector<MemBuffer>> dst_buffers(numUcxWorkers);
    std::vector<std::vector<uint32_t>> patterns(numUcxWorkers);

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        createRegisteredMem(getAgent(senderAgent), size, 1, VRAM_SEG, src_buffers[worker_id]);
        createRegisteredMem(getAgent(receiverAgent), size, 1, VRAM_SEG, dst_buffers[worker_id]);

        constexpr size_t num_elements = size / sizeof(uint32_t);
        patterns[worker_id].resize(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            patterns[worker_id][i] = 0xDEAD0000 | static_cast<uint32_t>(worker_id);
        }
        cudaMemcpy(src_buffers[worker_id][0].get(), patterns[worker_id].data(),
                   size, cudaMemcpyHostToDevice);
    }

    exchangeMD(senderAgent, receiverAgent);

    std::vector<nixlXferReqH *> xfer_reqs(numUcxWorkers);
    std::vector<nixlGpuXferReqH> gpu_req_handles(numUcxWorkers);

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        const std::string custom_param = "worker_id=" + std::to_string(worker_id);
        createXferRequest(src_buffers[worker_id], dst_buffers[worker_id], VRAM_SEG,
                         xfer_reqs[worker_id], gpu_req_handles[worker_id], custom_param);
    }

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        NixlDeviceKernelParams params = {};
        params.operation = NixlDeviceOperation::SINGLE_WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = gpu_req_handles[worker_id];
        params.singleWrite = {0, 0, 0, size, defaultChannelId};

        applySendMode(params, getSendMode());

        const auto result = launchNixlDeviceKernel(params);
        ASSERT_EQ(result.status, NIXL_SUCCESS)
            << "Kernel launch failed for worker " << worker_id;
    }

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        std::vector<uint32_t> received(size / sizeof(uint32_t));
        cudaMemcpy(received.data(), dst_buffers[worker_id][0].get(),
                   size, cudaMemcpyDeviceToHost);

        ASSERT_EQ(received, patterns[worker_id])
            << "Worker " << worker_id << " data verification failed";
    }

    for (size_t worker_id = 0; worker_id < numUcxWorkers; worker_id++) {
        getAgent(senderAgent).releaseGpuXferReq(gpu_req_handles[worker_id]);
        nixl_status_t status = getAgent(senderAgent).releaseXferReq(xfer_reqs[worker_id]);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

    invalidateMD();
}

} // namespace gtest::nixl::gpu::single_write

using gtest::nixl::gpu::single_write::SingleWriteTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi, SingleWriteTest,
                         testing::ValuesIn(SingleWriteTest::getDeviceTestParams()),
                         TestNameGenerator::device);
