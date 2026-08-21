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
#ifndef NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
#define NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH

#include <cooperative_groups.h>
#include <cuda/atomic>

#include "../common/nixl_device_types.cuh"
#include <device_proxy/proxy_protocol.h>

// Overlay struct written into nixlGpuXferStatusH::storage by enqueue()
// and read back by pollXferStatus().  Must fit within the 64-byte opaque blob.
struct ProxyXferStatus {
    nixlProxyCompletionSlot *slot; // device pointer to the channel's nixlProxyCompletionSlot
    uint64_t opIdx;
};

static_assert(sizeof(ProxyXferStatus) <= nixl_gpu_xfer_status_payload_size,
              "ProxyXferStatus must fit in the transfer-status payload");

__device__ __forceinline__ uint32_t
proxyMemViewIdFromHandle(nixlMemViewH mvh) {
    return static_cast<const nixlProxyDeviceMemView *>(mvh)->proxyMemViewId;
}

static_assert(sizeof(*nixlProxyWorkRing{}.producerIdx) == 8,
              "producer_idx must be 64-bit to avoid wrap-around false completions");
static_assert(sizeof(*nixlProxyWorkRing{}.consumerIdx) == 8,
              "consumer_idx must be 64-bit to match producer_idx");
static_assert(sizeof(*nixlProxyWorkRing{}.consumerIdxCache) == 8,
              "consumer_idx_cache must be 64-bit to match producer_idx");
static_assert(sizeof(nixlProxyCompletionSlot::completedIdx) == 8,
              "completed_idx must be 64-bit to match producer_idx");

template<nixl_gpu_level_t level>
__device__ inline void
nixlProxyExecInit(uint32_t &lane_id) {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        lane_id = 0;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        lane_id = threadIdx.x % warpSize;
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        lane_id = threadIdx.x;
    } else if constexpr (level == nixl_gpu_level_t::GRID) {
        lane_id = threadIdx.x + blockIdx.x * blockDim.x;
    }
}

template<nixl_gpu_level_t level>
__device__ inline void
nixlProxySync() {
    if constexpr (level == nixl_gpu_level_t::WARP) {
        __syncwarp();
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        __syncthreads();
    } else if constexpr (level == nixl_gpu_level_t::GRID) {
        cooperative_groups::this_grid().sync();
    }
}

struct ProxyDeviceContext : nixlProxyDeviceContextData {

    __device__ __forceinline__ size_t
    channelIndex(uint32_t peer_index, uint32_t channel_id) const {
        return static_cast<size_t>(channel_id) * maxPeers + peer_index;
    }

    __device__ inline nixl_status_t
    enqueue(nixlProxySubmission submission, nixlGpuXferStatusH *xfer_status = nullptr) const {
        if (submission.dstIndex >= maxPeers || numChannels == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        submission.channelId = static_cast<uint16_t>(submission.channelId % numChannels);

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> shut(*shutdownWord);
        if (shut.load(cuda::memory_order_relaxed) ==
            static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)) {
            return NIXL_ERR_BACKEND;
        }

        nixlProxyChannelView &channel_view =
            channels[channelIndex(submission.dstIndex, submission.channelId)];
        if (channel_view.workRing == nullptr || channel_view.completionSlot == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
        nixlProxyWorkRing *ring = channel_view.workRing;

        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> producer_idx(*ring->producerIdx);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> cons(*ring->consumerIdx);

        // Atomically claim a unique slot in the ring.
        const uint64_t ticket = producer_idx.fetch_add(1, cuda::memory_order_relaxed);

        // Fast path: use the device cache. Refresh from host only if the ring
        // appears full, since mapped-host loads are much slower than HBM loads.
        uint64_t cached_consumer_idx = *ring->consumerIdxCache;
        while (ticket - cached_consumer_idx >= ring->depth) {
            cached_consumer_idx = cons.load(cuda::memory_order_acquire);
            *ring->consumerIdxCache = cached_consumer_idx;

            if (shut.load(cuda::memory_order_relaxed) ==
                static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)) {
                return NIXL_ERR_BACKEND;
            }
        }

        const uint64_t submission_op_idx = ticket + 1;
        const uint32_t slot = static_cast<uint32_t>(ticket % ring->depth);

        // Signal this slot is ready for the consumer.  The release
        // guarantees the record write above is visible before the
        // consumer reads op_idx via an acquire load. op_idx == 0 means empty.
        submission.opIdx = 0;
        ring->records[slot] = submission;

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> record_op_idx(
            ring->records[slot].opIdx);
        record_op_idx.store(submission_op_idx, cuda::memory_order_release);

        if (xfer_status != nullptr) {
            ProxyXferStatus pxs{channel_view.completionSlot, submission_op_idx};
            memcpy(xfer_status->storage, &pxs, sizeof(ProxyXferStatus));
        }

        return NIXL_IN_PROG;
    }

    // Poll the completion slot recorded by enqueue().
    //
    // The completion slot implements collapsed-CQ semantics:
    // - completed_idx > op_idx  => this op completed earlier, so it succeeded
    // - completed_idx == op_idx => next_status is this op's terminal status
    // - completed_idx < op_idx  => this op is still pending, unless an earlier
    //                              completion published a terminal error and
    //                              latched the channel
    __device__ inline static nixl_status_t
    pollXferStatus(const nixlGpuXferStatusH &xfer_status) {
        const ProxyXferStatus *pxs = reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);
        if (pxs->slot == nullptr) {
            return NIXL_ERR_BACKEND;
        }

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> comp_idx(pxs->slot->completedIdx);

        const uint64_t completed_idx = comp_idx.load(cuda::memory_order_acquire);
        if (completed_idx > pxs->opIdx) {
            return NIXL_SUCCESS;
        }
        const nixl_status_t current_status = pxs->slot->nextStatus;
        if (completed_idx == pxs->opIdx) {
            return current_status;
        }
        if (current_status < 0) {
            return current_status;
        }

        return NIXL_IN_PROG;
    }
};

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
