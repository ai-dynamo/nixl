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
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_LEVEL_UTILS_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_LEVEL_UTILS_CUH

#include <cooperative_groups.h>

#include <gpu/device_types.cuh>

namespace nixl::gpu::impl {

/** This thread's index among the threads that make one call at `level`. */
template<level_t level>
__device__ __forceinline__ uint32_t
laneId() {
    uint32_t lane_id = 0;
    if constexpr (level == level_t::WARP) {
        lane_id = threadIdx.x % warpSize;
    } else if constexpr (level == level_t::BLOCK) {
        lane_id = threadIdx.x;
    } else if constexpr (level == level_t::GRID) {
        lane_id = threadIdx.x + blockIdx.x * blockDim.x;
    }
    return lane_id;
}

/** Whether this thread acts for all the threads that make one call at `level`. */
template<level_t level>
__device__ __forceinline__ bool
isLeader() {
    return laneId<level>() == 0;
}

/** Barrier across the threads that make one call at `level`; a grid needs a cooperative launch. */
template<level_t level>
__device__ __forceinline__ void
sync() {
    if constexpr (level == level_t::WARP) {
        __syncwarp();
    } else if constexpr (level == level_t::BLOCK) {
        __syncthreads();
    } else if constexpr (level == level_t::GRID) {
        cooperative_groups::this_grid().sync();
    }
}

/** A barrier that also hands every thread the leader's status; only a barrier at grid level. */
template<level_t level>
__device__ __forceinline__ nixl_status_t
broadcastStatus(nixl_status_t status) {
    if constexpr (level == level_t::WARP) {
        status = static_cast<nixl_status_t>(__shfl_sync(0xffffffff, static_cast<int>(status), 0));
    } else if constexpr (level == level_t::BLOCK) {
        __shared__ nixl_status_t s_status;
        if (threadIdx.x == 0) {
            s_status = status;
        }
        __syncthreads();
        status = s_status;
        __syncthreads();
    } else if constexpr (level == level_t::GRID) {
        sync<level>();
    }
    return status;
}

} // namespace nixl::gpu::impl

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_LEVEL_UTILS_CUH
