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

#include "nixl_device.cuh"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {
constexpr std::string_view sender_agent_name{"sender"};
constexpr std::string_view receiver_agent_name{"receiver"};
constexpr uint64_t value = 42;

__global__ void
atomicAddKernel(uint64_t value, void *counter_mvh, nixl_status_t *status) {
    nixlMemViewElem counter{counter_mvh, 0, 0};
    nixlGpuXferStatusH xfer_status;
    *status = nixlAtomicAdd(value, counter, 0, 0, &xfer_status);
    while (*status == NIXL_IN_PROG) {
        *status = nixlGpuGetXferStatus(xfer_status);
    }
}
} // namespace

namespace nixl::gpu {
class atomicAddTest : public testing::Test {};

TEST_F(atomicAddTest, single) {
    cudaPtr<uint64_t> counter;

    agent sender_agent{std::string(sender_agent_name)};
    agent receiver_agent{std::string(receiver_agent_name)};

    receiver_agent.registerMem(counter);
    sender_agent.loadRemoteMD(receiver_agent.getLocalMD());

    void *counter_mvh = sender_agent.prepCounterMemView(counter, std::string(receiver_agent_name));

    cudaPtr<nixl_status_t> status(true);

    atomicAddKernel<<<1, 1>>>(value, counter_mvh, status.get());

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(*status.get(), NIXL_SUCCESS);
    EXPECT_EQ(*counter, value);
}
} // namespace nixl::gpu
