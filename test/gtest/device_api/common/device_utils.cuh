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

#ifndef NIXL_DEVICE_UTILS_CUH
#define NIXL_DEVICE_UTILS_CUH

#include <cuda_runtime.h>
#include <nixl.h>
#include <nixl_device.cuh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <tuple>
#include <iostream>
#include "cuda_array.h"
#include "mem_buffer.h"
#include "device_kernels.cuh"

enum class SendMode {
    NODELAY_WITH_REQ,
    NODELAY_WITHOUT_REQ,
    WITHOUT_NODELAY_WITHOUT_REQ,
};

inline std::string_view getSendModeStr(SendMode mode) {
    switch (mode) {
    case SendMode::NODELAY_WITH_REQ:
        return "nodelay_with_req";
    case SendMode::NODELAY_WITHOUT_REQ:
        return "nodelay_without_req";
    case SendMode::WITHOUT_NODELAY_WITHOUT_REQ:
        return "without_nodelay_without_req";
    default:
        return "unknown";
    }
}

inline void applySendMode(NixlDeviceKernelParams &params, SendMode mode) {
    switch (mode) {
    case SendMode::NODELAY_WITH_REQ:
        params.withNoDelay = true;
        params.withRequest = true;
        break;
    case SendMode::NODELAY_WITHOUT_REQ:
        params.withNoDelay = true;
        params.withRequest = false;
        break;
    case SendMode::WITHOUT_NODELAY_WITHOUT_REQ:
        params.withNoDelay = false;
        params.withRequest = false;
        break;
    }
}

using DeviceTestParams = std::tuple<nixl_gpu_level_t, SendMode>;

inline std::string_view getGpuLevelStr(nixl_gpu_level_t level) {
    switch (level) {
    case nixl_gpu_level_t::WARP:
        return "WARP";
    case nixl_gpu_level_t::BLOCK:
        return "BLOCK";
    case nixl_gpu_level_t::THREAD:
        return "THREAD";
    default:
        return "UNKNOWN";
    }
}

struct TestNameGenerator {
    static std::string device(const testing::TestParamInfo<DeviceTestParams> &info) {
        const auto level = std::get<0>(info.param);
        const auto mode = std::get<1>(info.param);
        return std::string("UCX_") + std::string(getGpuLevelStr(level)) + "_" + std::string(getSendModeStr(mode));
    }

    static std::string level(const testing::TestParamInfo<nixl_gpu_level_t> &info) {
        return std::string("UCX_") + std::string(getGpuLevelStr(info.param));
    }
};

[[nodiscard]] inline nixl_status_t checkCudaErrors() {
    const cudaError_t sync_error = cudaDeviceSynchronize();
    const cudaError_t launch_error = cudaGetLastError();

    if (sync_error != cudaSuccess || launch_error != cudaSuccess) {
        if (sync_error != cudaSuccess) {
            std::cerr << "CUDA synchronization error: " << cudaGetErrorString(sync_error) << "\n";
        }
        if (launch_error != cudaSuccess) {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(launch_error) << "\n";
        }
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

#endif // NIXL_DEVICE_UTILS_CUH
