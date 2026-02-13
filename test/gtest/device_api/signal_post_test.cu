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

#include "p2p_test.h"

#include "kernel.h"

namespace {
constexpr size_t warp_size = 32;
constexpr uint64_t signal_increment = 42;
constexpr size_t signal_offset = 0;
constexpr unsigned signal_index = 0;
} // namespace

namespace nixl::device_api {
class signalPostTest : public p2pTest {
protected:
    signalPostTest() : p2pTest() {}

    [[nodiscard]] nixl_status_t
    postSignal();

    [[nodiscard]] nixl_status_t
    verifySignal();

private:
    [[nodiscard]] uint64_t
    expectedSignalValue() const noexcept {
        return signal_increment * numOpsMultiplier();
    }

    [[nodiscard]] size_t
    numOpsMultiplier() const noexcept {
        switch (getLevel()) {
        case nixl_gpu_level_t::THREAD:
            return warp_size;
        default:
            return 1;
        }
    }
};

[[nodiscard]] nixl_status_t
signalPostTest::postSignal() {
    kernelParams post_params(
        operation_t::SIGNAL_POST, getLevel(), GetParam().mode, createGpuXferReq());
    post_params.signalPost.signalDescIndex = signal_index;
    post_params.signalPost.signalInc = signal_increment;
    post_params.signalPost.signalOffset = signal_offset;
    return launchKernel(post_params);
}

nixl_status_t
signalPostTest::verifySignal() {
    kernelParams read_params(operation_t::SIGNAL_WAIT, getLevel());
    read_params.signalWait.signalAddr = dstBuffer(0).get();
    read_params.signalWait.expectedValue = expectedSignalValue();
    return launchKernel(read_params);
}

TEST_P(signalPostTest, Basic) {
    ASSERT_NO_THROW(addSignalBuffers());
    ASSERT_EQ(postSignal(), NIXL_SUCCESS);
    EXPECT_EQ(verifySignal(), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         signalPostTest,
                         testing::ValuesIn(paramsWithBlockLevel()),
                         paramsInfoToString);
} // namespace nixl::device_api
