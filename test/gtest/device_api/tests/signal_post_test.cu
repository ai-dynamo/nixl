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
#include <random>

namespace nixl::test::device_api {
class signalPostTest : public test<testParams> {
protected:
    size_t
    getNumOpsMultiplier() const {
        switch (getLevel()) {
        case nixl_gpu_level_t::THREAD:
            return defaultNumThreads;
        case nixl_gpu_level_t::WARP:
            return defaultNumThreads / 32;
        case nixl_gpu_level_t::BLOCK:
        default:
            return 1;
        }
    }

    void
    setupSignalPost(testSetupData &data) {
        nixl_opt_args_t extra_params = {.backends = {getBackendHandle(receiverAgent)}};
        size_t signal_size;
        nixl_status_t status = getAgent(receiverAgent).getGpuSignalSize(signal_size, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        const nixl_mem_t dst_mem_type = getDstMemType();
        createRegisteredMem(getAgent(receiverAgent),
                            signal_size,
                            defaultBufferAllocCount,
                            dst_mem_type,
                            data.dstBuffers);

        auto signal_desc_list = makeDescList<nixlBlobDesc>(data.dstBuffers, dst_mem_type);
        status = getAgent(receiverAgent).prepGpuSignal(signal_desc_list, &extra_params);
        ASSERT_EQ(status, NIXL_SUCCESS);

        createRegisteredMem(getAgent(senderAgent),
                            signal_size,
                            defaultBufferAllocCount,
                            srcMemType,
                            data.srcBuffers);

        ASSERT_NO_FATAL_FAILURE(exchangeMD(senderAgent, receiverAgent));

        createXferRequest(data.srcBuffers,
                          srcMemType,
                          data.dstBuffers,
                          dst_mem_type,
                          data.xferReq,
                          data.gpuReqHandle);
    }

    void
    runSignalPost(testSetupData &setup_data,
                  size_t num_iters,
                  unsigned index,
                  uint64_t signal_inc) {
        constexpr size_t signal_offset = 0;

        kernelParams post_params;
        post_params.operation = operation_t::SIGNAL_POST;
        post_params.level = getLevel();
        post_params.numThreads = defaultNumThreads;
        post_params.numBlocks = 1;
        post_params.numIters = num_iters;
        post_params.reqHandle = setup_data.gpuReqHandle;

        applySendMode<signalPostTest>(post_params, getSendMode());

        post_params.signalPost.signalDescIndex = index;
        post_params.signalPost.signalInc = signal_inc;
        post_params.signalPost.signalOffset = signal_offset;

        const nixl_status_t status = launchKernel(post_params);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

    [[nodiscard]] nixl_status_t
    verifySignal(testSetupData &setup_data, uint64_t expected_value) {
        kernelParams read_params;
        read_params.operation = operation_t::SIGNAL_WAIT;
        read_params.level = getLevel();
        read_params.numThreads = defaultNumThreads;
        read_params.numBlocks = 1;
        read_params.numIters = 1;

        read_params.signalWait.signalAddr = setup_data.dstBuffers[0].get();
        read_params.signalWait.expectedValue = expected_value;

        return launchKernel(read_params);
    }
};

TEST_P(signalPostTest, Basic) {
    testSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupSignalPost(setup_data));

    constexpr size_t num_iters = defaultNumIters;
    constexpr unsigned index = 0;
    constexpr uint64_t signal_inc = testSignalIncrement;
    const uint64_t expected_value = signal_inc * num_iters * getNumOpsMultiplier();

    ASSERT_NO_FATAL_FAILURE(runSignalPost(setup_data, num_iters, index, signal_inc));

    EXPECT_EQ(verifySignal(setup_data, expected_value), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         signalPostTest,
                         testing::ValuesIn(signalPostTest::getDeviceTestParams()),
                         testNameGenerator::device);
} // namespace nixl::test::device_api
