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
constexpr size_t buffer_size = 128;
constexpr uint32_t test_pattern = 0xDEADBEEF;

const std::vector<size_t> sizes = {buffer_size};
} // namespace

namespace nixl::device_api {
class singleWriteTest : public p2pTest {
protected:
    singleWriteTest() : p2pTest(sizes) {}

    void
    setSrcBuffers();

    [[nodiscard]] nixl_status_t
    run();

    [[nodiscard]] bool
    verifyDstBuffers() const;
};

void
singleWriteTest::setSrcBuffers() {
    auto src = reinterpret_cast<uint32_t *>(srcBuffer(0).get());
    constexpr uint32_t pattern = test_pattern;
    cudaMemset(src, 0, buffer_size);
    cudaMemcpy(src, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);
}

nixl_status_t
singleWriteTest::run() {
    kernelParams params(operation_t::SINGLE_WRITE, getLevel(), GetParam().mode, createGpuXferReq());
    params.singleWrite = {0, 0, 0, buffer_size};
    return launchKernel(params);
}

bool
singleWriteTest::verifyDstBuffers() const {
    uint32_t dst = 0;
    if (dstBuffer(0).memType() == DRAM_SEG) {
        std::memcpy(&dst, dstBuffer(0).get(), sizeof(dst));
    } else {
        cudaMemcpy(&dst,
                   reinterpret_cast<uint32_t *>(dstBuffer(0).get()),
                   sizeof(dst),
                   cudaMemcpyDeviceToHost);
    }
    return dst == test_pattern;
}

TEST_P(singleWriteTest, Basic) {
    ASSERT_NO_THROW(addDataBuffers());
    ASSERT_NO_THROW(setSrcBuffers());
    ASSERT_EQ(run(), NIXL_SUCCESS);
    EXPECT_TRUE(verifyDstBuffers());
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         singleWriteTest,
                         testing::ValuesIn(paramsWithBlockLevel()),
                         paramsInfoToString);
} // namespace nixl::device_api
