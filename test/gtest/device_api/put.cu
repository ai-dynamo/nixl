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

#include "agent.cuh"
#include "cuda_ptr.cuh"

#include "nixl.h"
#include "nixl_device.cuh"

#include <gtest/gtest.h>

namespace {
constexpr std::string_view sender_agent_name{"sender"};
constexpr std::string_view receiver_agent_name1{"receiver1"};
constexpr std::string_view receiver_agent_name2{"receiver2"};

__global__ void
putKernelOffset(void *src_mvh, void *dst_mvh, nixl_status_t *status) {
    nixlMemViewElem src{src_mvh, 0, 0};
    nixlMemViewElem dst{dst_mvh, 0, 0};
    *status = nixlPut(src, dst, sizeof(uint64_t) / 2, 0, nixl_gpu_flags::defer);
    if (*status != NIXL_IN_PROG) {
        return;
    }

    src.offset = sizeof(uint64_t) / 2;
    dst.offset = sizeof(uint64_t) / 2;
    nixlGpuXferStatusH xfer_status;
    *status = nixlPut(src, dst, sizeof(uint64_t) / 2, 0, 0, &xfer_status);
    while (*status == NIXL_IN_PROG) {
        *status = nixlGpuGetXferStatus(xfer_status);
    }
}

__global__ void
putKernelChannel(void *src_mvh, void *dst_mvh, nixl_status_t *status) {
    nixlMemViewElem src{src_mvh, 0, 0};
    nixlMemViewElem dst{dst_mvh, 0, 0};
    nixlGpuXferStatusH xfer_status;
    *status = nixlPut(src, dst, sizeof(uint64_t), 0, 0, &xfer_status);
    while (*status == NIXL_IN_PROG) {
        *status = nixlGpuGetXferStatus(xfer_status);
    }

    if (*status != NIXL_SUCCESS) {
        return;
    }

    src.index = 1;
    dst.index = 1;
    *status = nixlPut(src, dst, sizeof(uint64_t), 1, 0, &xfer_status);
    while (*status == NIXL_IN_PROG) {
        *status = nixlGpuGetXferStatus(xfer_status);
    }
}

__global__ void
putKernelRemote(void *src_mvh, void *dst_mvh, nixl_status_t *status) {
    nixlMemViewElem src{src_mvh, 0, 0};
    nixlMemViewElem dst{dst_mvh, 0, 0};
    *status = nixlPut(src, dst, sizeof(uint64_t), 0, nixl_gpu_flags::defer);
    if (*status != NIXL_IN_PROG) {
        return;
    }

    src.index = 1;
    dst.index = 1;
    nixlGpuXferStatusH xfer_status;
    *status = nixlPut(src, dst, sizeof(uint64_t), 0, 0, &xfer_status);
    while (*status == NIXL_IN_PROG) {
        *status = nixlGpuGetXferStatus(xfer_status);
    }
}
} // namespace

namespace nixl::gpu {
class putSingleTest : public testing::Test {};

TEST_F(putSingleTest, offset) {
    cudaPtr<uint64_t> src_cuda_ptr;
    const uint64_t init_value = std::rand();
    ASSERT_EQ(cudaMemcpy(src_cuda_ptr.get(), &init_value, sizeof(uint64_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    cudaPtr<uint64_t> dst_cuda_ptr;

    agent sender_agent{std::string(sender_agent_name)};
    agent receiver_agent{std::string(receiver_agent_name1)};

    sender_agent.registerMem(src_cuda_ptr);
    receiver_agent.registerMem(dst_cuda_ptr);

    sender_agent.loadRemoteMD(receiver_agent.getLocalMD());

    void *src_mvh = sender_agent.prepLocalMemView(src_cuda_ptr);
    void *dst_mvh = sender_agent.prepRemoteMemView(dst_cuda_ptr, std::string(receiver_agent_name1));
    cudaPtr<nixl_status_t> status(true);

    putKernelOffset<<<1, 1>>>(src_mvh, dst_mvh, status.get());

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(*status.get(), NIXL_SUCCESS);
    EXPECT_EQ(src_cuda_ptr, dst_cuda_ptr);
}

TEST_F(putSingleTest, channel) {
    std::vector<cudaPtr<uint64_t>> src_cuda_ptrs(2);
    for (const auto &src_cuda_ptr : src_cuda_ptrs) {
        const uint64_t init_value = std::rand();
        ASSERT_EQ(
            cudaMemcpy(src_cuda_ptr.get(), &init_value, sizeof(uint64_t), cudaMemcpyHostToDevice),
            cudaSuccess);
    }
    std::vector<cudaPtr<uint64_t>> dst_cuda_ptrs(2);

    agent sender_agent{std::string(sender_agent_name)};
    agent receiver_agent{std::string(receiver_agent_name1)};

    for (const auto &src_cuda_ptr : src_cuda_ptrs) {
        sender_agent.registerMem(src_cuda_ptr);
    }
    for (const auto &dst_cuda_ptr : dst_cuda_ptrs) {
        receiver_agent.registerMem(dst_cuda_ptr);
    }

    sender_agent.loadRemoteMD(receiver_agent.getLocalMD());

    void *src_mvh = sender_agent.prepLocalMemView(src_cuda_ptrs);
    void *dst_mvh =
        sender_agent.prepRemoteMemView(dst_cuda_ptrs, std::string(receiver_agent_name1));
    cudaPtr<nixl_status_t> status(true);

    putKernelChannel<<<1, 1>>>(src_mvh, dst_mvh, status.get());

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(*status.get(), NIXL_SUCCESS);
    EXPECT_EQ(src_cuda_ptrs, dst_cuda_ptrs);
}

TEST_F(putSingleTest, remote) {
    std::vector<cudaPtr<uint64_t>> src_cuda_ptrs(2);
    for (const auto &src_cuda_ptr : src_cuda_ptrs) {
        const uint64_t init_value = std::rand();
        ASSERT_EQ(
            cudaMemcpy(src_cuda_ptr.get(), &init_value, sizeof(uint64_t), cudaMemcpyHostToDevice),
            cudaSuccess);
    }
    std::vector<cudaPtr<uint64_t>> dst_cuda_ptrs(2);

    agent sender_agent{std::string(sender_agent_name)};
    agent receiver_agent1{std::string(receiver_agent_name1)};
    agent receiver_agent2{std::string(receiver_agent_name2)};

    for (const auto &src_cuda_ptr : src_cuda_ptrs) {
        sender_agent.registerMem(src_cuda_ptr);
    }
    receiver_agent1.registerMem(dst_cuda_ptrs[0]);
    receiver_agent2.registerMem(dst_cuda_ptrs[1]);

    sender_agent.loadRemoteMD(receiver_agent1.getLocalMD());
    sender_agent.loadRemoteMD(receiver_agent2.getLocalMD());

    void *src_mvh = sender_agent.prepLocalMemView(src_cuda_ptrs);
    std::vector<std::string> remote_agent_names = {std::string(receiver_agent_name1),
                                                   std::string(receiver_agent_name2)};
    void *dst_mvh = sender_agent.prepRemoteMemView(dst_cuda_ptrs, remote_agent_names);
    cudaPtr<nixl_status_t> status(true);

    putKernelRemote<<<1, 1>>>(src_mvh, dst_mvh, status.get());

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(*status.get(), NIXL_SUCCESS);
    EXPECT_EQ(src_cuda_ptrs, dst_cuda_ptrs);
}
} // namespace nixl::gpu
