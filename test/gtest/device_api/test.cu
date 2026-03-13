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

#include <string_view>

namespace {
constexpr size_t max_polls{1000000};
constexpr uint64_t test_value{0x1234567890ABCDEFULL};
constexpr size_t size{sizeof(test_value)};
constexpr size_t half_size{size / 2};
constexpr std::string_view sender_agent_name{"sender"};
constexpr std::string_view receiver_agent_name{"receiver"};

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
    const nixlMemViewElem src{src_mvh, 0, 0};
    const nixlMemViewElem dst{dst_mvh, dst_index, 0};
    putAndWait(src, dst, size, dst_index);
}
} // namespace

namespace nixl::gpu {
class deviceApiTest : public testing::Test {};

TEST_F(deviceApiTest, atomicAdd) {
    memBuffer counter{size};

    agent sender_agent{std::string{sender_agent_name}};
    agent receiver_agent{std::string{receiver_agent_name}};

    receiver_agent.registerMem(counter);
    sender_agent.loadRemoteMD(receiver_agent.getLocalMD());

    void *counter_mvh = sender_agent.prepRemoteMemView(counter);

    atomicAddKernel<<<1, 1>>>(counter_mvh);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(counter, &test_value);
}

TEST_F(deviceApiTest, putOffset) {
    memBuffer src_mb{size, &test_value};
    memBuffer dst_mb{size};

    agent sender_agent{std::string{sender_agent_name}};
    agent receiver_agent{std::string{receiver_agent_name}};

    void *src_mvh = sender_agent.regAndPrepLocalMemView(src_mb);

    receiver_agent.registerMem(dst_mb);

    sender_agent.loadRemoteMD(receiver_agent.getLocalMD());
    void *dst_mvh = sender_agent.prepRemoteMemView(dst_mb);

    putOffsetKernel<<<1, 1>>>(src_mvh, dst_mvh);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(src_mb, dst_mb);
}

TEST_F(deviceApiTest, putChannel) {
    memBuffer src_mb{size, &test_value};
    std::vector<memBuffer> dst_mbs;
    dst_mbs.emplace_back(std::move(memBuffer{size}));
    dst_mbs.emplace_back(std::move(memBuffer{size}));

    agent sender_agent{std::string{sender_agent_name}};
    agent receiver_agent{std::string{receiver_agent_name}};

    void *src_mvh = sender_agent.regAndPrepLocalMemView(src_mb);

    for (const auto &dst_mb : dst_mbs) {
        receiver_agent.registerMem(dst_mb);
    }

    sender_agent.loadRemoteMD(receiver_agent.getLocalMD());
    void *dst_mvh = sender_agent.prepRemoteMemView(dst_mbs);

    putChannelKernel<<<1, 2>>>(src_mvh, dst_mvh);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(src_mb, dst_mbs[0]);
    EXPECT_EQ(src_mb, dst_mbs[1]);
}
} // namespace nixl::gpu
