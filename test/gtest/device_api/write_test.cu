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

#include "partial_write_test.h"

#include "kernel.h"

namespace nixl::device_api {
class writeTest : public partialWriteTest {
protected:
    writeTest() : partialWriteTest() {}

    [[nodiscard]] nixl_status_t
    run(uint64_t signal_inc = signalIncrement_) override;
};

nixl_status_t
writeTest::run(uint64_t signal_inc) {
    kernelParams params(operation_t::WRITE, getLevel(), GetParam().mode, createGpuXferReq());
    params.write.signalInc = signal_inc;
    return launchKernel(params);
}

TEST_P(writeTest, Basic) {
    ASSERT_NO_THROW(addDataBuffers());
    ASSERT_NO_THROW(addSignalBuffers());
    ASSERT_NO_THROW(setSrcBuffers());
    ASSERT_EQ(run(), NIXL_SUCCESS);
    EXPECT_TRUE(verifyDstBuffers());
}

TEST_P(writeTest, WithoutSignal) {
    ASSERT_NO_THROW(addDataBuffers());
    ASSERT_NO_THROW(addSignalBuffers());
    ASSERT_NO_THROW(setSrcBuffers());
    ASSERT_EQ(run(0), NIXL_SUCCESS);
    EXPECT_TRUE(verifyDstBuffers());
}

TEST_P(writeTest, SignalOnly) {
    ASSERT_NO_THROW(addSignalBuffers());
    EXPECT_EQ(run(), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         writeTest,
                         testing::ValuesIn(paramsWithoutBlockLevel()),
                         paramsInfoToString);
} // namespace nixl::device_api
