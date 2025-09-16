/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include "nixl.h"

#include <cuda_runtime.h>
#include <nixl_device.cuh>

namespace gtest::nixl::gpu {

class ucxDeviceApi : public ::testing::Test {};

template<nixl_gpu_level_t level>
__global__ void dummyKernel() {
    const nixlGpuSignal signal { 1, 0x1000 };
    nixlGpuXferStatusH status;

    [[maybe_unused]] auto result1 = nixlGpuPostSingleWriteXferReq<level>(nullptr, 0, nullptr, 0, 0);
    [[maybe_unused]] auto result2 = nixlGpuPostSignalXferReq<level>(nullptr, 0, signal);
    [[maybe_unused]] auto result3 = nixlGpuPostPartialWriteXferReq<level>(nullptr, 0, nullptr, nullptr, nullptr, nullptr, signal);
    [[maybe_unused]] auto result4 = nixlGpuPostWriteXferReq<level>(nullptr, nullptr, nullptr, nullptr, signal);
    [[maybe_unused]] auto result5 = nixlGpuGetXferStatus<level>(status);
    [[maybe_unused]] auto result6 = nixlGpuReadSignal<level>(nullptr);
}

TEST_F(ucxDeviceApi, compilationTest) {
    dummyKernel<nixl_gpu_level_t::THREAD><<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

} // namespace gtest::nixl::gpu
