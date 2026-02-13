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

namespace {
[[nodiscard]] std::vector<uint8_t>
generateTestPattern(size_t size, size_t offset) {
    constexpr size_t patternModulo = 256;
    std::vector<uint8_t> pattern(size);
    for (size_t i = 0; i < size; ++i) {
        pattern[i] = static_cast<uint8_t>((offset * patternModulo + i) % patternModulo);
    }
    return pattern;
}

constexpr size_t buffer_size = 128;

const std::vector<size_t> sizes = {buffer_size, buffer_size};
} // namespace

namespace nixl::device_api {
partialWriteTest::partialWriteTest()
    : p2pTest(sizes) {}

void
partialWriteTest::setSrcBuffers() {
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto pattern = generateTestPattern(sizes[i], i);
        srcBuffer(i).copyFromHost(pattern.data(), sizes[i]);
    }
}

nixl_status_t
partialWriteTest::run(uint64_t signal_inc) {
    const size_t data_buf_count = sizes.size();

    std::vector<unsigned> indices_host(data_buf_count);
    std::vector<size_t> local_offsets_host(data_buf_count, 0);
    std::vector<size_t> remote_offsets_host(data_buf_count, 0);

    for (size_t i = 0; i < data_buf_count; ++i) {
        indices_host[i] = static_cast<unsigned>(i);
    }

    memTypeArray<unsigned> indices_gpu(data_buf_count);
    memTypeArray<size_t> sizes_gpu(data_buf_count);
    memTypeArray<size_t> local_offsets_gpu(data_buf_count);
    memTypeArray<size_t> remote_offsets_gpu(data_buf_count);

    indices_gpu.copyFromHost(indices_host);
    sizes_gpu.copyFromHost(sizes);
    local_offsets_gpu.copyFromHost(local_offsets_host);
    remote_offsets_gpu.copyFromHost(remote_offsets_host);

    const auto signal_desc_index = static_cast<unsigned>(dataBufferCount());
    constexpr size_t signal_offset = 0;

    kernelParams params(operation_t::PARTIAL_WRITE, getLevel(), GetParam().mode, createGpuXferReq());

    params.partialWrite.count = data_buf_count;
    params.partialWrite.descIndices = indices_gpu.get();
    params.partialWrite.sizes = sizes_gpu.get();
    params.partialWrite.localOffsets = local_offsets_gpu.get();
    params.partialWrite.remoteOffsets = remote_offsets_gpu.get();
    params.partialWrite.signalDescIndex = signal_desc_index;
    params.partialWrite.signalInc = signal_inc;
    params.partialWrite.signalOffset = signal_offset;

    return launchKernel(params);
}

bool
partialWriteTest::verifyDstBuffers() const {
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto expected_pattern = generateTestPattern(sizes[i], i);
        std::vector<uint8_t> received_data(sizes[i]);
        dstBuffer(i).copyToHost(received_data.data(), sizes[i]);
        if (received_data != expected_pattern) return false;
    }

    return true;
}

TEST_P(partialWriteTest, Basic) {
    ASSERT_NO_THROW(addDataBuffers());
    ASSERT_NO_THROW(addSignalBuffers());
    ASSERT_NO_THROW(setSrcBuffers());
    ASSERT_EQ(run(), NIXL_SUCCESS);
    EXPECT_TRUE(verifyDstBuffers());
}

TEST_P(partialWriteTest, WithoutSignal) {
    ASSERT_NO_THROW(addDataBuffers());
    ASSERT_NO_THROW(addSignalBuffers());
    ASSERT_NO_THROW(setSrcBuffers());
    ASSERT_EQ(run(0), NIXL_SUCCESS);
    EXPECT_TRUE(verifyDstBuffers());
}

TEST_P(partialWriteTest, SignalOnly) {
    ASSERT_NO_THROW(addDataBuffers());
    ASSERT_NO_THROW(addSignalBuffers());
    ASSERT_NO_THROW(setSrcBuffers());
    ASSERT_EQ(run(), NIXL_SUCCESS);
    EXPECT_TRUE(verifyDstBuffers());
}

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         partialWriteTest,
                         testing::ValuesIn(paramsWithoutBlockLevel()),
                         paramsInfoToString);
} // namespace nixl::device_api
