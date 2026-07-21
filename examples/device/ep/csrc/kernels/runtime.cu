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

#include <atomic>
#include <vector>
#include <cstring>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
#include <nixl_device.cuh>
#endif

namespace nixl_ep {

namespace {

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    std::atomic<uint64_t> active_proxy_context_owner{0};
#endif

} // namespace

cudaError_t
publish_proxy_context(void *context, uint64_t owner_id) {
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    if (context == nullptr || owner_id == 0) {
        return cudaErrorInvalidValue;
    }

    bool acquired_owner = false;
    uint64_t expected_owner = 0;
    if (active_proxy_context_owner.compare_exchange_strong(
            expected_owner, owner_id, std::memory_order_acq_rel)) {
        acquired_owner = true;
    } else if (expected_owner != owner_id) {
        return cudaErrorInvalidValue;
    }

    const cudaError_t status =
        nixlProxyPublishContext(static_cast<nixlProxyDeviceContextData *>(context));
    if (status != cudaSuccess && acquired_owner) {
        active_proxy_context_owner.store(0, std::memory_order_release);
    }
    return status;
#else
    (void)context;
    (void)owner_id;
    return cudaSuccess;
#endif
}

cudaError_t
clear_proxy_context(uint64_t owner_id) {
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    if (owner_id == 0 || active_proxy_context_owner.load(std::memory_order_acquire) != owner_id) {
        return cudaErrorInvalidValue;
    }

    const cudaError_t status = nixlProxyClearContext();
    if (status == cudaSuccess) {
        uint64_t expected_owner = owner_id;
        active_proxy_context_owner.compare_exchange_strong(
            expected_owner, 0, std::memory_order_acq_rel);
    }
    return status;
#else
    (void)owner_id;
    return cudaSuccess;
#endif
}

namespace intranode {

    template<int kNumRanks>
    __global__ void
    barrier(int **barrier_signal_ptrs, int rank, uint64_t timeout_cycles) {
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank, timeout_cycles);
    }

    void
    barrier(int **barrier_signal_ptrs,
            int rank,
            int num_nvl_ranks,
            uint64_t timeout_cycles,
            cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank, timeout_cycles); \
    break

        SETUP_LAUNCH_CONFIG(1, 32, stream);
        SWITCH_NVL_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
    }

} // namespace intranode

} // namespace nixl_ep
