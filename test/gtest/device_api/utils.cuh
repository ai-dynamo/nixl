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

#ifndef _DEVICE_API_UTILS_CUH
#define _DEVICE_API_UTILS_CUH

#include <gtest/gtest.h>
#include "nixl.h"
#include "common.h"
#include <nixl_device.cuh>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <optional>
#include <thread>
#include <chrono>
#include <functional>
#include <absl/strings/str_format.h>

#define MAX_THREADS 1024
#define UCS_NSEC_PER_SEC 1000000000ul
#define NS_TO_SEC(ns) ((ns) * 1.0 / (UCS_NSEC_PER_SEC))

static const std::string NOTIF_MSG = "notification";

class MemBuffer : std::shared_ptr<void> {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type) :
        std::shared_ptr<void>(allocate(size, mem_type),
                            [mem_type](void *ptr) {
                                release(ptr, mem_type);
                            }),
        size(size)
    {
    }

    operator uintptr_t() const
    {
        return reinterpret_cast<uintptr_t>(get());
    }

    operator void*() const
    {
        return get();
    }

    operator const void*() const
    {
        return get();
    }

    size_t getSize() const
    {
        return size;
    }

private:
    static void *allocate(size_t size, nixl_mem_t mem_type)
    {
        void *ptr;
        return cudaSuccess == cudaMalloc(&ptr, size)? ptr : nullptr;
    }

    static void release(void *ptr, nixl_mem_t mem_type)
    {
        cudaFree(ptr);
    }

    size_t size;
};

namespace gtest {
namespace gpu {

static const std::vector<nixl_gpu_level_t> _test_levels = {
    nixl_gpu_level_t::BLOCK,
    nixl_gpu_level_t::WARP,
    nixl_gpu_level_t::THREAD,
};

const char *GetGpuXferLevelStr(nixl_gpu_level_t level);

void initTiming(unsigned long long **start_time_ptr, unsigned long long **end_time_ptr);
void getTiming(unsigned long long *start_time_ptr,
               unsigned long long *end_time_ptr,
               unsigned long long &start_time_cpu,
               unsigned long long &end_time_cpu);
void logResults(size_t size,
                size_t count,
                size_t num_iters,
                unsigned long long start_time_cpu,
                unsigned long long end_time_cpu);

} // namespace gpu
} // namespace gtest

__device__ inline unsigned long long GetTimeNs() {
    unsigned long long globaltimer;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

template<nixl_gpu_level_t level>
__device__ constexpr size_t GetReqIdx() {
    switch (level) {
    case nixl_gpu_level_t::THREAD:
        return threadIdx.x;
    case nixl_gpu_level_t::WARP:
        return threadIdx.x / warpSize;
    case nixl_gpu_level_t::BLOCK:
        return 0;
    default:
        return 0;
    }
}

class DeviceApiTestBase : public testing::TestWithParam<nixl_gpu_level_t> {
protected:
    static nixlAgentConfig getConfig();
    nixl_b_params_t getBackendParams();
    void SetUp() override;
    void TearDown() override;

    template<typename Desc>
    nixlDescList<Desc> makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type);

    void registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type);
    void completeWireup(size_t from_agent, size_t to_agent);
    void exchangeMD(size_t from_agent, size_t to_agent);
    void invalidateMD();

    void createRegisteredMem(nixlAgent &agent,
                            size_t size,
                            size_t count,
                            nixl_mem_t mem_type,
                            std::vector<MemBuffer> &out);

    nixlAgent &getAgent(size_t idx);
    std::string getAgentName(size_t idx);

protected:
    static constexpr size_t SENDER_AGENT = 0;
    static constexpr size_t RECEIVER_AGENT = 1;

private:
    static constexpr uint64_t DEV_ID = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
    std::vector<nixlBackendH *> backend_handles;
};

#endif // _DEVICE_API_UTILS_CUH
