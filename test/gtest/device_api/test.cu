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

#include "agent.h"
#include "mem_buffer.h"

#include "nixl.h"
#include "nixl_device.cuh"

#include <gtest/gtest.h>

namespace {
constexpr size_t max_polls{1000000};
constexpr uint64_t test_value{0x1234567890ABCDEFULL};
constexpr size_t size{sizeof(test_value)};
constexpr size_t half_size{size / 2};
const std::string sender_agent_name{"sender"};
const std::string receiver_agent_name{"receiver"};

__device__ void
waitCompletion(nixl_status_t status, nixlGpuXferStatusH &xfer_status) {
    size_t polls = 0;
    while ((status == NIXL_IN_PROG) && (polls++ < max_polls)) {
        status = nixlGpuGetXferStatus(xfer_status);
    }
    assert(status == NIXL_SUCCESS);
}

__global__ void
atomicAddKernel(void *counter_mvh) {
    nixlMemViewElem counter{counter_mvh, 0, 0};
    nixlGpuXferStatusH xfer_status;
    nixl_status_t status = nixlAtomicAdd(test_value, counter, 0, 0, &xfer_status);
    waitCompletion(status, xfer_status);
}

__device__ void
putAndWait(const nixlMemViewElem &src,
           const nixlMemViewElem &dst,
           size_t size,
           unsigned channel_id = 0) {
    nixlGpuXferStatusH xfer_status;
    nixl_status_t status = nixlPut(src, dst, size, channel_id, 0, &xfer_status);
    waitCompletion(status, xfer_status);
}

__global__ void
putOffsetKernel(void *src_mvh, void *dst_mvh) {
    assert(threadIdx.x == 0);
    nixlMemViewElem src{src_mvh, 0, 0};
    nixlMemViewElem dst{dst_mvh, 0, 0};
    const nixl_status_t status = nixlPut(src, dst, half_size, 0, nixl_gpu_flags::defer);
    assert(status == NIXL_IN_PROG);

    src.offset = half_size;
    dst.offset = half_size;
    putAndWait(src, dst, half_size);
}

__global__ void
putChannelKernel(void *src_mvh, void *dst_mvh) {
    assert(threadIdx.x < 2);
    const size_t dst_index = threadIdx.x;
    const unsigned channel_id = threadIdx.x;
    const nixlMemViewElem src{src_mvh, 0, 0};
    const nixlMemViewElem dst{dst_mvh, dst_index, 0};
    putAndWait(src, dst, size, channel_id);
}
} // namespace

namespace nixl::gpu {
class deviceApiTest : public testing::Test {
protected:
    void
    SetUp() override {
        int count;
        if (cudaGetDeviceCount(&count) != cudaSuccess) {
            FAIL() << "Failed to get CUDA device count";
        }
        if (count < 1) {
            GTEST_SKIP() << "No CUDA-capable GPU is available";
        }
        if (cudaSetDevice(0) != cudaSuccess) {
            FAIL() << "Failed to set CUDA device 0";
        }

        senderAgent_ = std::make_unique<agent>(sender_agent_name);
        receiverAgent_ = std::make_unique<agent>(receiver_agent_name);
    }

    [[nodiscard]] void *
    prepRemoteMemView(const std::vector<memBuffer> &mbs) {
        receiverAgent_->registerMem(mbs);
        senderAgent_->loadRemoteMD(receiverAgent_->getLocalMD());
        return senderAgent_->prepRemoteMemView(mbs);
    }

    [[nodiscard]] void *
    prepLocalMemView(const std::vector<memBuffer> &mbs) {
        return senderAgent_->regAndPrepLocalMemView(mbs);
    }

private:
    std::unique_ptr<agent> senderAgent_;
    std::unique_ptr<agent> receiverAgent_;
};

TEST_F(deviceApiTest, atomicAdd) {
    std::vector<memBuffer> counters;
    counters.emplace_back(std::move(memBuffer{size}));

    void *counter_mvh = prepRemoteMemView(counters);

    atomicAddKernel<<<1, 1>>>(counter_mvh);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(counters[0], &test_value);
}

TEST_F(deviceApiTest, putOffset) {
    std::vector<memBuffer> src_mbs;
    src_mbs.emplace_back(std::move(memBuffer{size, &test_value}));
    std::vector<memBuffer> dst_mbs;
    dst_mbs.emplace_back(std::move(memBuffer{size}));

    void *src_mvh = prepLocalMemView(src_mbs);
    void *dst_mvh = prepRemoteMemView(dst_mbs);

    putOffsetKernel<<<1, 1>>>(src_mvh, dst_mvh);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(src_mbs[0], dst_mbs[0]);
}

TEST_F(deviceApiTest, putChannel) {
    std::vector<memBuffer> src_mbs;
    src_mbs.emplace_back(std::move(memBuffer{size, &test_value}));
    std::vector<memBuffer> dst_mbs;
    dst_mbs.emplace_back(std::move(memBuffer{size}));
    dst_mbs.emplace_back(std::move(memBuffer{size}));

    void *src_mvh = prepLocalMemView(src_mbs);
    void *dst_mvh = prepRemoteMemView(dst_mbs);

    putChannelKernel<<<1, 2>>>(src_mvh, dst_mvh);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(src_mbs[0], dst_mbs[0]);
    EXPECT_EQ(src_mbs[0], dst_mbs[1]);
}
} // namespace nixl::gpu
