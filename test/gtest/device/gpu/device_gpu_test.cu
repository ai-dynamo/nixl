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

#include "device_gpu_agent.h"
#include "common.h"
#include "device_gpu_mem_buffer.h"

#include <nixl.h>
#include <gpu/nixl_device.cuh>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace {
constexpr size_t max_polls{1000000};
constexpr uint64_t test_value{0x1234567890ABCDEFULL};
constexpr size_t size{sizeof(test_value)};
constexpr size_t half_size{size / 2};
const std::string sender_agent_name{"sender"};
const std::string receiver_agent_name{"receiver"};
constexpr std::chrono::seconds data_arrival_timeout{5};
} // namespace

namespace nixl::gpu {
__device__ void
waitPosted(nixl_status_t status, xferStatusH &xfer_status) {
    size_t polls = 0;
    while ((status == NIXL_IN_PROG) && (polls++ < max_polls)) {
        status = getXferStatus(xfer_status);
    }
    assert(status == NIXL_SUCCESS);
}

__device__ void
waitPostedWithSystemFence(nixl_status_t status, xferStatusH &xfer_status) {
    waitPosted(status, xfer_status);
    __threadfence_system();
}

__global__ void
atomicAddKernel(void *counter_mvh) {
    memViewElem counter{counter_mvh, 0, 0};
    xferStatusH xfer_status;
    nixl_status_t status = atomicAdd(test_value, counter, 0, 0, &xfer_status);
    waitPostedWithSystemFence(status, xfer_status);
}

__device__ void
putAndWait(const memViewElem &src, const memViewElem &dst, size_t size, unsigned channel_id = 0) {
    xferStatusH xfer_status;
    nixl_status_t status = put(src, dst, size, channel_id, 0, &xfer_status);
    waitPostedWithSystemFence(status, xfer_status);
}

__global__ void
putOffsetKernel(void *src_mvh, void *dst_mvh) {
    assert(threadIdx.x == 0);
    memViewElem src{src_mvh, 0, 0};
    memViewElem dst{dst_mvh, 0, 0};
    const nixl_status_t status = put(src, dst, half_size, 0, flags::defer);
    assert(status == NIXL_IN_PROG);

    src.offset = half_size;
    dst.offset = half_size;
    putAndWait(src, dst, half_size);
}

__global__ void
putChannelKernel(void *src_mvh, void *dst_mvh) {
    assert(threadIdx.x == 0);
    const memViewElem src{src_mvh, 0, 0};
    for (unsigned channel_id = 0; channel_id < 2; ++channel_id) {
        const memViewElem dst{dst_mvh, channel_id, 0};
        putAndWait(src, dst, size, channel_id);
    }
}

void
waitForData(const memBuffer &actual, const memBuffer &expected) {
    const auto deadline = std::chrono::steady_clock::now() + data_arrival_timeout;
    while (!(actual == expected) && (std::chrono::steady_clock::now() < deadline)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

class test : public testing::TestWithParam<bool> {
protected:
    void
    SetUp() override {
#ifndef HAVE_UCX_GPU_DEVICE_API
        if (!isProxy()) {
            GTEST_SKIP() << "UCX direct GPU API is unavailable";
        }
#endif
        int count;
        if ((cudaGetDeviceCount(&count) != cudaSuccess) || (count < 1)) {
            GTEST_SKIP() << "No CUDA-capable GPU is available";
        }
        if (cudaSetDevice(0) != cudaSuccess) {
            FAIL() << "Failed to set CUDA device 0";
        }

        lig_ = std::make_unique<gtest::LogIgnoreGuard>(
            "IB device\\(s\\) were detected, but accelerated IB support was not found");

        senderAgent_ = std::make_unique<agent>(sender_agent_name, isProxy());
        receiverAgent_ = std::make_unique<agent>(receiver_agent_name, isProxy());
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
    [[nodiscard]] bool
    isProxy() const {
        return GetParam();
    }

    std::unique_ptr<gtest::LogIgnoreGuard> lig_;
    std::unique_ptr<agent> senderAgent_;
    std::unique_ptr<agent> receiverAgent_;
};

TEST_P(test, atomicAdd) {
    std::vector<memBuffer> counters;
    counters.emplace_back(std::move(memBuffer{size}));

    void *counter_mvh = prepRemoteMemView(counters);

    atomicAddKernel<<<1, 1>>>(counter_mvh);
    // Launch errors are reported only here, not by cudaDeviceSynchronize().
    const cudaError_t launch_status = cudaGetLastError();
    ASSERT_EQ(launch_status, cudaSuccess) << cudaGetErrorString(launch_status);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const memBuffer expected{size, &test_value};
    waitForData(counters[0], expected);
    EXPECT_EQ(counters[0], expected);
}

TEST_P(test, putOffset) {
    std::vector<memBuffer> src_mbs;
    src_mbs.emplace_back(std::move(memBuffer{size, &test_value}));
    std::vector<memBuffer> dst_mbs;
    dst_mbs.emplace_back(std::move(memBuffer{size}));

    void *src_mvh = prepLocalMemView(src_mbs);
    void *dst_mvh = prepRemoteMemView(dst_mbs);

    putOffsetKernel<<<1, 1>>>(src_mvh, dst_mvh);
    // Launch errors are reported only here, not by cudaDeviceSynchronize().
    const cudaError_t launch_status = cudaGetLastError();
    ASSERT_EQ(launch_status, cudaSuccess) << cudaGetErrorString(launch_status);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    waitForData(dst_mbs[0], src_mbs[0]);
    EXPECT_EQ(src_mbs[0], dst_mbs[0]);
}

TEST_P(test, putChannel) {
    std::vector<memBuffer> src_mbs;
    src_mbs.emplace_back(std::move(memBuffer{size, &test_value}));
    std::vector<memBuffer> dst_mbs;
    dst_mbs.emplace_back(std::move(memBuffer{size}));
    dst_mbs.emplace_back(std::move(memBuffer{size}));

    void *src_mvh = prepLocalMemView(src_mbs);
    void *dst_mvh = prepRemoteMemView(dst_mbs);

    putChannelKernel<<<1, 1>>>(src_mvh, dst_mvh);
    // Launch errors are reported only here, not by cudaDeviceSynchronize().
    const cudaError_t launch_status = cudaGetLastError();
    ASSERT_EQ(launch_status, cudaSuccess) << cudaGetErrorString(launch_status);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    waitForData(dst_mbs[0], src_mbs[0]);
    waitForData(dst_mbs[1], src_mbs[0]);
    EXPECT_EQ(src_mbs[0], dst_mbs[0]);
    EXPECT_EQ(src_mbs[0], dst_mbs[1]);
}

INSTANTIATE_TEST_SUITE_P(deviceGpu,
                         test,
                         testing::Bool(),
                         [](const testing::TestParamInfo<bool> &info) {
                             return info.param ? "PROXY" : "UCX";
                         });
} // namespace nixl::gpu
