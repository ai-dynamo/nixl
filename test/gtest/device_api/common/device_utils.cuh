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

#ifndef TEST_GTEST_DEVICE_API_COMMON_DEVICE_UTILS_CUH
#define TEST_GTEST_DEVICE_API_COMMON_DEVICE_UTILS_CUH

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
#include "device_kernels.cuh"

namespace nixl::test::device_api {

enum class send_mode_t {
    NODELAY_WITH_REQ,
    NODELAY_WITHOUT_REQ,
    WITHOUT_NODELAY_WITHOUT_REQ,
    MULTI_CHANNEL,
};

inline std::string_view
getSendModeStr(send_mode_t mode) {
    switch (mode) {
    case send_mode_t::NODELAY_WITH_REQ:
        return "nodelay_with_req";
    case send_mode_t::NODELAY_WITHOUT_REQ:
        return "nodelay_without_req";
    case send_mode_t::WITHOUT_NODELAY_WITHOUT_REQ:
        return "without_nodelay_without_req";
    case send_mode_t::MULTI_CHANNEL:
        return "multi_channel";
    default:
        return "unknown";
    }
}

template<typename test>
inline void
applySendMode(kernelParams &params, send_mode_t mode) {
    switch (mode) {
    case send_mode_t::MULTI_CHANNEL:
        params.numChannels = test::multiChannelCount;
    case send_mode_t::NODELAY_WITH_REQ:
        params.withNoDelay = true;
        params.withRequest = true;
        break;
    case send_mode_t::NODELAY_WITHOUT_REQ:
        params.withNoDelay = true;
        params.withRequest = false;
        break;
    case send_mode_t::WITHOUT_NODELAY_WITHOUT_REQ:
        params.withNoDelay = false;
        params.withRequest = false;
        break;
    }
}

struct testParams {
    nixl_gpu_level_t level;
    send_mode_t mode;
    nixl_mem_t mem_type;

    testParams(nixl_gpu_level_t l, send_mode_t m, nixl_mem_t mt)
        : level(l),
          mode(m),
          mem_type(mt) {}
};

inline std::string_view
getGpuLevelStr(nixl_gpu_level_t level) {
    switch (level) {
    case nixl_gpu_level_t::WARP:
        return "warp";
    case nixl_gpu_level_t::BLOCK:
        return "block";
    case nixl_gpu_level_t::THREAD:
        return "thread";
    default:
        return "unknown";
    }
}

inline std::string_view
getMemTypeStr(nixl_mem_t mem_type) {
    switch (mem_type) {
    case VRAM_SEG:
        return "dst_vram";
    case DRAM_SEG:
        return "dst_dram";
    default:
        return "unknown";
    }
}

struct testNameGenerator {
    static std::string
    device(const testing::TestParamInfo<testParams> &info) {
        const auto &p = info.param;
        return std::string(getMemTypeStr(p.mem_type)) + "_" + std::string(getGpuLevelStr(p.level)) +
            "_" + std::string(getSendModeStr(p.mode));
    }

    static std::string
    level(const testing::TestParamInfo<nixl_gpu_level_t> &info) {
        return std::string(getGpuLevelStr(info.param));
    }
};

[[nodiscard]] inline nixl_status_t
checkCudaErrors() {
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(launch_error) << "\n";
        return NIXL_ERR_BACKEND;
    }

    const cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        std::cerr << "CUDA synchronization error: " << cudaGetErrorString(sync_error) << "\n";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}
} // namespace nixl::test::device_api
#endif // NIXL_DEVICE_UTILS_CUH
