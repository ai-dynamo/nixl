/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch(const bool& number_only = false,
                         const bool& support_arch_family = false) {
        const auto [major, minor] = get_arch_pair();
        if (major == 10 and minor != 1) {
            if (number_only)
                return "100";
            return support_arch_family ? "100f" : "100a";
        }
        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");
    }

    int get_arch_major() {
        return get_arch_pair().first;
    }
};

static auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });

} // namespace nixl_ep_v2::jit
