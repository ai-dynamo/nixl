// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

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

// A status-bearing atomic retires this peer/channel's earlier RDMA operations as well as the atomic itself.
template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void red_add_rel_and_wait(const nixl::NIXLTransport& transport, int64_t* word,
                                                     const int64_t value, const int world_rank, const int64_t start_clock = 0) {
    nixlGpuXferStatusH completion;
    const bool rdma = transport.get_p2p_ptr(word, world_rank) == nullptr;
    transport.red_add_rel(word, value, world_rank, rdma ? &completion : nullptr);
    if (rdma)
        timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            const auto status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(completion);
            if (status == NIXL_SUCCESS)
                return true;
            if (status != NIXL_IN_PROG or is_last_check) {
                printf("NIXL EP v2 completion %s, world rank %d, peer %d, channel %u, status %d; "
                       "all old-world peers must remain connected until the drain completes\n",
                       status == NIXL_IN_PROG ? "timeout" : "error", transport.ctx.world_rank, world_rank,
                       transport.channel, static_cast<int>(status));
                if (status != NIXL_IN_PROG)
                    ptx::trap();
            }
            return false;
        }, start_clock);
}

// The channel a warp issues on: channel 0 for the notify warps and a one-channel context, the others striped over the active blocks
template <int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_channel_idx(const int& sm_idx, const int& channel_in_sm_idx,
                                               const int& num_channels, const int& num_active_sms, const bool& is_notify_warp = false) {
    // Only one channel
    if (num_channels == 1)
        return 0;

    // The notify warp always use 1 SM and 1 channel
    if (is_notify_warp)
        return 0;

    // Data channels
    constexpr int kChannelStartIdx = static_cast<int>(kWithNotifyWarps);
    const int num_available_channels = num_channels - kChannelStartIdx;
    if (num_active_sms <= num_available_channels) {
        // A single SM uses an entire channel
        // e.g., 3 SMs and 10 channels
        // SM 0: 0 3 6 9
        // SM 1: 1 4 7
        // SM 2: 2 5 8
        const int num_channels_in_sm = (num_available_channels / num_active_sms) + (sm_idx < (num_available_channels % num_active_sms));
        return kChannelStartIdx + sm_idx + (channel_in_sm_idx % num_channels_in_sm) * num_active_sms;
    } else {
        // All SMs share all channels
        const auto global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return kChannelStartIdx + (global_channel_idx % num_available_channels);
    }
}

// Active-block grid barrier: word 0 counts arrivals; word 1 is the generation that waiters poll.
// The last acq_rel arrival observes all blocks' writes, resets the counter and releases the next generation.
// Waiters acquire that generation; the surrounding __syncthreads propagate ordering to/from each block.
// Both words start at zero. Polling the generation avoids confusing a reset counter with incomplete arrivals.
template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void active_grid_sync(unsigned* barrier_ptr, const int& num_active_sms) {
    __syncthreads();
    if (threadIdx.x == 0) {
        // Read the round number before arriving: the last arriver may increment it right after our arrival
        const auto round = ptx::ld_acquire_gpu(barrier_ptr + 1);
        const auto arrived = ptx::atom_add_acq_rel_gpu(barrier_ptr, 1u);
        if (arrived + 1 == static_cast<unsigned>(num_active_sms)) {
            // Everyone has arrived and nobody reads the counter: reset it, then publish the new round
            ptx::st_relaxed_gpu(barrier_ptr, 0u);
            ptx::st_release_gpu(barrier_ptr + 1, round + 1);
        } else {
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                if (ptx::ld_acquire_gpu(barrier_ptr + 1) != round)
                    return true;
                if (is_last_check)
                    printf("NIXL EP v2 adaptive grid barrier timeout, block: %d, active blocks: %d, round: %u, arrived: %u\n",
                           static_cast<int>(blockIdx.x), num_active_sms, round, ptx::ld_acquire_gpu(barrier_ptr));
                return false;
            });
        }
    }
    __syncthreads();
}

enum class Leg { kScaleup, kScaleout };

// One leg of the per-peer barrier, run by the calling block: thread `t` arrives on peer position `t` of the leg (a scale-up
// position of our domain, or a domain at our position) by incrementing the peer's arrival word for us, then
// waits until our word for the peer reaches the same count. Masked peers are skipped; nothing is reset on a mask change (a connect restarts a pair's words, host side).
template <int kNumThreads, int64_t kNumTimeoutCycles, int kTag, int kNumMaxScaleupRanks,
          bool kWaitForLocalCompletion = false>
__device__ __forceinline__ void peer_barrier_leg_rt(const nixl::NIXLTransport& transport,
                                                    const layout::WorkspaceLayout& workspace,
                                                    const topology::EPTopology& topo,
                                                    const Leg& leg, const int& thread_idx) {
    const bool scaleout_leg = leg == Leg::kScaleout;
    const int my_world_rank = transport.ctx.world_rank;
    const int num_ranks = scaleout_leg ? topo.num_scaleout_ranks : topo.num_scaleup_ranks;
    EP_STATIC_ASSERT(kNumMaxScaleupRanks <= kNumThreads and topology::kNumMaxScaleoutRanksCap <= kNumThreads, "Insufficient threads for one arrival per peer");
    if (thread_idx < num_ranks) {
        const bool active = scaleout_leg ? topo.is_scaleout_active(thread_idx) : topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, thread_idx);
        const int world_rank = scaleout_leg ?
            topology::scaleout_peer<kNumMaxScaleupRanks>(topo, thread_idx) :
            topology::scaleup_peer<kNumMaxScaleupRanks>(topo, thread_idx);
        if (active and world_rank != my_world_rank) {
            // Expect one more arrival from the peer
            const auto expected_ptr = workspace.get_peer_barrier_expected_ptr(world_rank);
            const int64_t expected = *expected_ptr + 1;
            *expected_ptr = expected;

            if constexpr (kWaitForLocalCompletion)
                red_add_rel_and_wait<kNumTimeoutCycles>(transport, workspace.get_peer_barrier_signal_ptr(my_world_rank), 1, world_rank);
            else
                transport.red_add_rel(workspace.get_peer_barrier_signal_ptr(my_world_rank), int64_t(1), world_rank);

            // Wait for the peer's arrival
            const auto signal_ptr = workspace.get_peer_barrier_signal_ptr(world_rank);
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);
                if (signal >= expected)
                    return true;
                if (is_last_check) {
                    printf("NIXL EP v2 adaptive peer barrier timeout, tag: %d, leg: %s, scale-out: %d/%d, scale-up: %d/%d, "
                           "peer position: %d (world rank %d), signal: %lld, expected: %lld\n", kTag, scaleout_leg ? "scale-out" : "scale-up",
                           topo.scaleout_rank_idx, topo.num_scaleout_ranks, topo.scaleup_rank_idx, topo.num_scaleup_ranks, thread_idx, world_rank, static_cast<long long>(signal), static_cast<long long>(expected));
                }
                return false;
            });
        }
    }
    __syncthreads();
}

// Active blocks synchronize; SM 0 barriers over NVLink and SM 1 over rail peers when both legs are needed.
// Each receiver consumes dispatch records/counts and combine rows (including its stream-ordered epilogue)
// before the next entry barrier, which permits sender-buffer reuse without local RDMA completion waits.
// Retiring peers additionally requires the standalone barrier to drain every RDMA channel.
template <int kNumThreads, int64_t kNumTimeoutCycles, int kTag, int kNumMaxScaleupRanks,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true>
__device__ __forceinline__ void gpu_barrier_rt(const nixl::NIXLDeviceContext& ctx,
                                               const layout::WorkspaceLayout& workspace,
                                               const topology::EPTopology& topo,
                                               const int& sm_idx, const int& thread_idx,
                                               bool do_scaleout = true) {
    const auto grid_barrier_ptr = workspace.get_active_grid_barrier_ptr();
    const auto& num_active_sms = topo.num_active_sms;

    // A general TMA store wait to prevent proxy memory issues
    if constexpr (kFlushStores) {
        ptx::tma_store_commit();
        ptx::tma_store_wait();
        __syncwarp();
    }

    // All the SMs should wait
    if constexpr (kSyncAtStart) {
        active_grid_sync<kNumTimeoutCycles>(grid_barrier_ptr, num_active_sms);
    } else {
        EP_STATIC_ASSERT(not kFlushStores, "No data to be flushed");
    }

    do_scaleout &= topo.num_active_scaleout_ranks() > 1;
    const bool do_scaleup = topo.num_active_scaleup_ranks() > 1;

    // Barrier arrivals go on channel 0
    const nixl::NIXLTransport barrier_transport(ctx, 0);
    const auto leg = [&](const Leg& which) {
        peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, kTag, kNumMaxScaleupRanks>(
            barrier_transport, workspace, topo, which, thread_idx);
    };
    if (do_scaleup and do_scaleout) {
        if (sm_idx == 0)
            leg(Leg::kScaleup);
        else if (sm_idx == 1)
            leg(Leg::kScaleout);
    } else if (sm_idx == 0 and do_scaleup) {
        leg(Leg::kScaleup);
    } else if (sm_idx == 0 and do_scaleout) {
        leg(Leg::kScaleout);
    }

    // All the SMs should wait
    if constexpr (kSyncAtEnd)
        active_grid_sync<kNumTimeoutCycles>(grid_barrier_ptr, num_active_sms);
}

}  // namespace nixl_ep_v2::elastic::comm
