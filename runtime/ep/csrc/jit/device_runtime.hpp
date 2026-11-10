// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <nixl_ep_v2/common/exception.cuh>

#include "../utils/lazy_init.hpp"

namespace nixl_ep_v2::jit {

class DeviceRuntime {
    int64_t cached_clock_rate = 0;
    std::shared_ptr<cudaDeviceProp> cached_prop;

    std::shared_ptr<cudaDeviceProp> get_prop() {
        if (cached_prop == nullptr) {
            int device_idx;
            cudaDeviceProp prop;
            CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
            CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_idx));
            cached_prop = std::make_shared<cudaDeviceProp>(prop);
        }
        return cached_prop;
    }

public:
    int64_t get_clock_rate() {
        if (cached_clock_rate == 0) {
            // NOTES: we should convert kHz into Hz
            int device_idx, rate;
            CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
            CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, device_idx));
            cached_clock_rate = static_cast<int64_t>(rate) * 1000ll;
        }
        return cached_clock_rate;
    }

    int get_num_smem_bytes() {
        return static_cast<int>(get_prop()->sharedMemPerBlockOptin);
    }

    int get_num_sms() {
        return get_prop()->multiProcessorCount;
    }

    int get_max_warps_per_block() {
        const auto prop = get_prop();
        return prop->maxThreadsPerBlock / prop->warpSize;
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch() {
        const auto [major, minor] = get_arch_pair();
        if (major == 10 and minor != 1)
            return "100f";
        return std::to_string(major * 10 + minor) + "a";
    }
};

static auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });

} // namespace nixl_ep_v2::jit
