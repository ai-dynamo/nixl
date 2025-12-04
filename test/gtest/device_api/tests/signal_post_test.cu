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
#include <random>

namespace gtest::nixl::gpu::signal_post {

namespace {

class SignalPostTest : public DeviceApiTestBase<DeviceTestParams> {
protected:
    void setupSignalPost(TestSetupData &data) {
        nixl_opt_args_t extra_params = {.backends = {backendHandles_[receiverAgent]}};
        size_t signal_size;
        nixl_status_t status =
            getAgent(receiverAgent).getGpuSignalSize(signal_size, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        createRegisteredMem(getAgent(receiverAgent), signal_size, 1, VRAM_SEG, data.dstBuffers);

        auto signal_desc_list = makeDescList<nixlBlobDesc>(data.dstBuffers, VRAM_SEG);
        status = getAgent(receiverAgent).prepGpuSignal(signal_desc_list, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        createRegisteredMem(getAgent(senderAgent), signal_size, 1, VRAM_SEG, data.srcBuffers);

        ASSERT_NO_FATAL_FAILURE(exchangeMD(senderAgent, receiverAgent));

        createXferRequest(data.srcBuffers, data.dstBuffers, VRAM_SEG,
                         data.xferReq, data.gpuReqHandle);
    }

    void runSignalPost(TestSetupData &setup_data, size_t num_iters,
                      unsigned index, uint64_t signal_inc, unsigned channel_id) {
        constexpr size_t signal_offset = 0;

        NixlDeviceKernelParams post_params = {};
        post_params.operation = NixlDeviceOperation::SIGNAL_POST;
        post_params.level = getLevel();
        post_params.numThreads = defaultNumThreads;
        post_params.numBlocks = 1;
        post_params.numIters = num_iters;
        post_params.reqHandle = setup_data.gpuReqHandle;

        applySendMode(post_params, getSendMode());

        post_params.signalPost.signalDescIndex = index;
        post_params.signalPost.signalInc = signal_inc;
        post_params.signalPost.signalOffset = signal_offset;
        post_params.signalPost.channelId = channel_id;

        auto result = launchNixlDeviceKernel(post_params);
        ASSERT_EQ(result.status, NIXL_SUCCESS);
    }

    void verifySignal(TestSetupData &setup_data, uint64_t expected_value) {
        CudaArray<uint64_t> result_ptr(1);

        NixlDeviceKernelParams read_params = {};
        read_params.operation = NixlDeviceOperation::SIGNAL_READ;
        read_params.level = getLevel();
        read_params.numThreads = defaultNumThreads;
        read_params.numBlocks = 1;
        read_params.numIters = 1;

        read_params.signalRead.signalAddr = setup_data.dstBuffers[0].get();
        read_params.signalRead.expectedValue = expected_value;
        read_params.signalRead.resultPtr = result_ptr.get();

        auto result = launchNixlDeviceKernel(read_params);
        ASSERT_EQ(result.status, NIXL_SUCCESS);

        uint64_t signal_value = 0;
        result_ptr.copyToHost(&signal_value, 1);
        ASSERT_EQ(signal_value, expected_value);
    }
};

} // namespace

TEST_P(SignalPostTest, Basic) {
#ifndef HAVE_UCX_GPU_DEVICE_API
    GTEST_SKIP() << "UCX GPU device API not available, skipping test";
#else
    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupSignalPost(setup_data));

    constexpr size_t num_iters = defaultNumIters;
    constexpr unsigned index = 0;
    constexpr uint64_t signal_inc = testSignalIncrement;
    const uint64_t expected_value = signal_inc * num_iters;

    ASSERT_NO_FATAL_FAILURE(runSignalPost(setup_data, num_iters, index,
                                          signal_inc, defaultChannelId));

    ASSERT_NO_FATAL_FAILURE(verifySignal(setup_data, expected_value));
#endif
}

} // namespace gtest::nixl::gpu::signal_post

using gtest::nixl::gpu::signal_post::SignalPostTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi, SignalPostTest,
                         testing::ValuesIn(SignalPostTest::getDeviceTestParams()),
                         TestNameGenerator::device);
