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

#ifndef _DEVICE_TEST_BASE_CUH
#define _DEVICE_TEST_BASE_CUH

#include <gtest/gtest.h>
#include "nixl.h"
#include "common.h"
#include <nixl_device.cuh>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <functional>
#include <type_traits>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <absl/strings/str_format.h>

#include "mem_buffer.cuh"
#include "cuda_ptr.cuh"
#include "device_utils.cuh"

class DeviceApiTestBase : public testing::TestWithParam<nixl_gpu_level_t> {
public:
    static const char *
    getGpuXferLevelStr(nixl_gpu_level_t level);

    static const std::vector<nixl_gpu_level_t>
    getTestLevels() {
        static const std::vector<nixl_gpu_level_t> testLevels = {
            nixl_gpu_level_t::BLOCK,
            nixl_gpu_level_t::WARP,
            nixl_gpu_level_t::THREAD,
        };
        return testLevels;
    }

    static constexpr const char *notifMsg = "notification";

protected:
    static nixlAgentConfig
    getConfig();
    nixl_b_params_t
    getBackendParams();
    void
    SetUp() override;
    void
    TearDown() override;

    template<typename Desc>
    nixlDescList<Desc>
    makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type) {
        nixlDescList<Desc> desc_list(mem_type);
        for (const auto &buffer : buffers) {
            desc_list.addDesc(Desc(buffer, buffer.getSize(), uint64_t(devId)));
        }
        return desc_list;
    }

    void
    registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type);
    void
    completeWireup(size_t from_agent, size_t to_agent);
    void
    exchangeMD(size_t from_agent, size_t to_agent);
    void
    invalidateMD();

    void
    createRegisteredMem(nixlAgent &agent,
                        size_t size,
                        size_t count,
                        nixl_mem_t mem_type,
                        std::vector<MemBuffer> &out);

    nixlAgent &
    getAgent(size_t idx);
    std::string
    getAgentName(size_t idx);

    void
    initTiming(unsigned long long **start_time_ptr, unsigned long long **end_time_ptr);
    void
    getTiming(unsigned long long *start_time_ptr,
              unsigned long long *end_time_ptr,
              unsigned long long &start_time_cpu,
              unsigned long long &end_time_cpu);

    template<typename KernelFunc>
    nixl_status_t
    dispatchKernelByLevel(nixl_gpu_level_t level, KernelFunc kernel_func) {
        switch (level) {
        case nixl_gpu_level_t::BLOCK:
            return kernel_func(std::integral_constant<nixl_gpu_level_t, nixl_gpu_level_t::BLOCK>{});
        case nixl_gpu_level_t::WARP:
            return kernel_func(std::integral_constant<nixl_gpu_level_t, nixl_gpu_level_t::WARP>{});
        case nixl_gpu_level_t::THREAD:
            return kernel_func(
                std::integral_constant<nixl_gpu_level_t, nixl_gpu_level_t::THREAD>{});
        default:
            ADD_FAILURE() << "Unknown level: " << static_cast<int>(level);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

protected:
    static constexpr size_t senderAgent = 0;
    static constexpr size_t receiverAgent = 1;

    std::vector<nixlBackendH *> backend_handles;

private:
    static constexpr uint64_t devId = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
};

#endif // _DEVICE_TEST_BASE_CUH
