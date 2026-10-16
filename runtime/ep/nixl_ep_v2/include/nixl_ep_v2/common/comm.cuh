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

#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/topology.cuh>

namespace nixl_ep_v2::elastic::comm {

static constexpr int64_t kNumOneSecCycles = 2000000000;  // An approximation of the GPU clock at 2000 MHz

// Some reserved tags
static constexpr int kKernelBarrierTag = 1;
static constexpr int kCombineTag0 = 4;
static constexpr int kCombineTag1 = 5;
static constexpr int kHybridDispatchTag0 = 6;
static constexpr int kHybridDispatchTag1 = 7;
static constexpr int kHybridCombineTag0 = 8;

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const bool& condition, const func_t& func,
                                              int64_t start_clock = 0) {
    // User may share a start clock for multiple waits
    if (start_clock == 0)
        start_clock = clock64();

    while (condition) {
        const bool timeout = clock64() - start_clock >= kNumTimeoutCycles;
        if (func(timeout))
            break;

        if (timeout) {
            // Wait another 1 second to let all threads print information and trap
            start_clock = clock64();
            while (clock64() - start_clock < kNumOneSecCycles) {}
            ptx::trap();
        }
    }
}

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const func_t& func, const int64_t& start_clock = 0) {
    timeout_while<kNumTimeoutCycles, func_t>(true, func, start_clock);
}

// Strided wait: `func(i, is_last_check)` for every `i` of this thread's stride, one timeout budget for the loop
template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while_strided(const int& begin, const int& end, const int& stride, const func_t& func) {
    const auto start_clock = clock64();
    for (int i = begin; i < end; i += stride)
        timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) { return func(i, is_last_check); }, start_clock);
}

// Grid barrier over the active blocks. `barrier_ptr[0]` counts the blocks that have arrived in the current
// round, `barrier_ptr[1]` is the round number; both live in the zero-initialised workspace. Each block adds one
// to the counter; the last block to arrive resets the counter and increments the round number; the others spin
// on the round number until it changes. Two words, because a waiting block cannot spin on the counter: the
// reset for the next round would look like "not everyone has arrived yet".
//
// Memory ordering at GPU scope, as with the cooperative-groups grid sync this replaces: the leading
// __syncthreads() orders the whole block's writes before thread 0's arrival; the arrival is an acq_rel atomic,
// so the last arriver has observed every block's writes when it releases the round number; every waiter
// acquires the round number, and the trailing __syncthreads() extends what thread 0 observed to its block.
template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void active_grid_sync(unsigned* barrier_ptr, const int& num_active_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        // Read the round number before arriving: the last arriver may increment it right after our arrival
        const auto round = ptx::ld_acquire_gpu(barrier_ptr + 1);
        const auto arrived = ptx::atom_add_acq_rel_gpu(barrier_ptr, 1u);
        if (arrived + 1 == static_cast<unsigned>(num_active_blocks)) {
            // Everyone has arrived and nobody reads the counter: reset it, then publish the new round
            ptx::st_relaxed_gpu(barrier_ptr, 0u);
            ptx::st_release_gpu(barrier_ptr + 1, round + 1);
        } else {
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                if (ptx::ld_acquire_gpu(barrier_ptr + 1) != round)
                    return true;
                if (is_last_check)
                    printf("NIXL EP V2 adaptive grid barrier timeout, block: %d, active blocks: %d, round: %u, arrived: %u\n",
                           static_cast<int>(blockIdx.x), num_active_blocks, round, ptx::ld_acquire_gpu(barrier_ptr));
                return false;
            });
        }
    }
    __syncthreads();
}

// The channel a warp issues on. Channel 0 serves the notify warps (reserved for them when `kWithNotifyWarps`) and a one-channel context;
// the other channels are striped over the launch blocks, a block's warps cycling through its stripe, or wrap around when the blocks outnumber them
template <int kNumLaunchSMs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_channel_idx(const int& sm_idx, const int& channel_in_sm_idx,
                                               const int& num_channels, const bool& is_notify_warp = false) {
    if (num_channels == 1 or is_notify_warp)
        return 0;
    constexpr int kChannelStartIdx = static_cast<int>(kWithNotifyWarps);
    const int num_available_channels = num_channels - kChannelStartIdx;
    if (kNumLaunchSMs <= num_available_channels) {
        const int num_channels_in_sm = (num_available_channels / kNumLaunchSMs) +
                                       (sm_idx < (num_available_channels % kNumLaunchSMs));
        return kChannelStartIdx + sm_idx + (channel_in_sm_idx % num_channels_in_sm) * kNumLaunchSMs;
    }
    return kChannelStartIdx + (sm_idx * kNumChannelsPerSM + channel_in_sm_idx) % num_available_channels;
}

// One leg of the per-peer barrier, run by one block: thread `t` arrives on peer position `t` of the leg (a scale-up
// position of our domain, or a domain at our position) by writing our arrival count into the peer's word for us, then
// waits until our word for the peer reaches the same count. Masked peers are skipped; nothing is reset on a mask change (a connect restarts a pair's words, host side).
template <int kNumThreads, int64_t kNumTimeoutCycles, int kTag, int kNumMaxScaleupRanks>
__device__ __forceinline__ void peer_barrier_leg_rt(const nixl::NIXLTransport& transport,
                                                    const layout::WorkspaceLayout& workspace,
                                                    const topology::EPTopology& topo,
                                                    const bool& scaleout_leg, const int& sm_idx, const int& thread_idx) {
    // This barrier only uses 1 SM
    if (sm_idx > 0)
        return;

    const int my_rid = topology::rank_id(topo.scaleout_rank_idx, topo.scaleup_rank_idx, kNumMaxScaleupRanks);
    const int num_ranks = scaleout_leg ? topo.num_scaleout_ranks : topo.num_scaleup_ranks;
    EP_DEVICE_ASSERT(num_ranks <= kNumThreads and "Insufficient threads");
    if (thread_idx < num_ranks) {
        const bool live = scaleout_leg ? topo.is_scaleout_live(thread_idx) : topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, thread_idx);
        const int rid = scaleout_leg ?
            topology::rank_id(thread_idx, topo.scaleup_rank_idx, kNumMaxScaleupRanks) :
            topology::rank_id(topo.scaleout_rank_idx, thread_idx, kNumMaxScaleupRanks);
        if (live and rid != my_rid) {
            // Expect one more arrival from the peer
            const auto expected_ptr = workspace.get_peer_barrier_expected_ptr(rid);
            const int64_t expected = *expected_ptr + 1;
            *expected_ptr = expected;

            // Arrive on the peer with a release write of the count (both sides count the same barriers, see
            // `put_word_rel`); arrival k uses word and staging slot k % 2 so that two in-flight writes never share a word
            const int parity = static_cast<int>(expected & 1);
            transport.put_word_rel(workspace.get_peer_barrier_signal_ptr(my_rid, parity),
                                   workspace.get_peer_barrier_staging_ptr(rid, parity), expected, rid);

            // Wait for the peer's arrival
            const auto signal_ptr = workspace.get_peer_barrier_signal_ptr(rid, parity);
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);
                if (signal >= expected)
                    return true;
                if (is_last_check) {
                    printf("NIXL EP V2 adaptive peer barrier timeout, tag: %d, leg: %s, scale-out: %d/%d, scale-up: %d/%d, "
                           "peer position: %d (id %d), signal: %lld, expected: %lld\n", kTag, scaleout_leg ? "scale-out" : "scale-up",
                           topo.scaleout_rank_idx, topo.num_scaleout_ranks, topo.scaleup_rank_idx, topo.num_scaleup_ranks, thread_idx, rid, static_cast<long long>(signal), static_cast<long long>(expected));
                }
                return false;
            });
        }
    }
    __syncthreads();
}

// No kernel waits for an RDMA completion (the puts carry no status handle): a put has landed once its receiver has
// consumed it, and every receiver consumes all puts of a launch before its next barrier with the sender (dispatch
// records and count words inside the receiving dispatch, combine rows in the reduce epilogue, stream-ordered before
// the next launch). A rank past a start barrier or a `barrier_impl` on `comm_stream` may therefore rewrite its send
// buffers; a peer may be disconnected once it has passed that barrier too (its arrival word is not its exit).
// The adaptive kernels' GPU barrier: scale-up inside the NVLink domain, scale-out over the rail peers, in parallel
// when both are needed (SM 0 does scale-up, SM 1 does scale-out). Every grid-wide sync counts active blocks only.
template <int kNumThreads, int64_t kNumTimeoutCycles, int kTag, int kNumMaxScaleupRanks,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true>
__device__ __forceinline__ void gpu_barrier_rt(const nixl::NIXLTransport& transport,
                                               const layout::WorkspaceLayout& workspace,
                                               const topology::EPTopology& topo,
                                               const int& sm_idx, const int& thread_idx,
                                               bool do_scaleout = true, bool do_scaleup = true) {
    const auto grid_barrier_ptr = workspace.get_active_grid_barrier_ptr();
    const auto& num_active_sms = topo.num_active_sms;

    // A general TMA store wait to prevent proxy memory issues
    if constexpr (kFlushStores) {
        ptx::tma_store_commit();
        ptx::tma_store_wait();
        __syncwarp();
    }

    // All the active SMs should wait
    if constexpr (kSyncAtStart) {
        active_grid_sync<kNumTimeoutCycles>(grid_barrier_ptr, num_active_sms);
        EP_PROFILE_STAMP(workspace, 9);
    } else {
        EP_STATIC_ASSERT(not kFlushStores, "No data to be flushed");
    }

    do_scaleout &= topo.num_live_scaleout_ranks() > 1;
    do_scaleup &= topo.scaleup_mask.popc() > 1;

    // Barrier arrivals go on channel 0
    const nixl::NIXLTransport barrier_transport(transport.ctx, 0);
    if (do_scaleup and do_scaleout) {
        EP_DEVICE_ASSERT(num_active_sms >= 2 and "At least 2 SMs for a hybrid barrier");
        if (sm_idx == 0) {
            peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, kTag, kNumMaxScaleupRanks>(
                barrier_transport, workspace, topo, false, 0, thread_idx);
        } else {
            peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, kTag, kNumMaxScaleupRanks>(
                barrier_transport, workspace, topo, true, sm_idx - 1, thread_idx);
        }
    } else if (do_scaleup) {
        peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, kTag, kNumMaxScaleupRanks>(
            barrier_transport, workspace, topo, false, sm_idx, thread_idx);
    } else if (do_scaleout) {
        peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, kTag, kNumMaxScaleupRanks>(
            barrier_transport, workspace, topo, true, sm_idx, thread_idx);
    }

    // All the active SMs should wait
    if constexpr (kSyncAtEnd)
        active_grid_sync<kNumTimeoutCycles>(grid_barrier_ptr, num_active_sms);
}

}  // namespace nixl_ep_v2::elastic::comm
