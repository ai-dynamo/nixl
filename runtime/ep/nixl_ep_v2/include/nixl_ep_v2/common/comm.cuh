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
static constexpr int kHybridDispatchTag0 = 6;
static constexpr int kHybridDispatchTag1 = 7;
static constexpr int kHybridCombineTag0 = 8;
static constexpr int kDirectDispatchTag = 9;
static constexpr int kDirectCombineTag = 10;

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

// ---- Direct mode's per-peer signalling (the dispatch and combine kernels without zero copy) ----
// Instead of the two full-team barriers of `gpu_barrier_rt`, each launch advances a pair's expectation word `e` by exactly 2. A
// peer's READY is the value e + 1 in its signal word `kPeerReadyWord` (its receive buffers are free: every earlier kernel of its
// stream has completed, as the direct kernels are never launched as PDL secondaries; SM 0 publishes it at kernel entry), its
// ARRIVED is e + 2 in word `kPeerArrivedWord` (all its writes into our buffers are done; the grid's last block sends it to each
// peer after acquiring that peer's READY). Values only grow, so a peer that is ahead satisfies a wait too, and the hybrid kernels'
// parity-indexed legs above (+1 per leg, word = count & 1) interleave with this scheme safely: every write exceeds the pair's
// earlier values and every wait is "signal >= threshold". Every remote write of a launch sits behind its writer's own acquire of
// the destination's READY (`peer_ready_acquire_rt`). Masked peers are skipped; a connect restarts a pair's words (host side), as
// for the barrier legs
// Enumerators, not namespace-scope `static constexpr int`s: the waits and the layout accessors take the word by `const int&`, which
// ODR-uses a variable (undefined in device code) but only materialises a temporary from an enumerator
enum PeerWord : int { kPeerReadyWord = 0, kPeerArrivedWord = 1 };

// Wait (acquire) until peer `rid` at position `pos` has written at least `threshold` into our copy of its `word`
template <int64_t kNumTimeoutCycles, int kTag>
__device__ __forceinline__ void peer_word_wait_rt(const layout::WorkspaceLayout& workspace, const topology::EPTopology& topo,
                                                  const int& pos, const int& rid, const int& word, const int64_t& threshold) {
    const auto signal_ptr = workspace.get_peer_barrier_signal_ptr(rid, word);
    timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
        const auto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);
        if (signal >= threshold)
            return true;
        if (is_last_check) {
            printf("NIXL EP V2 adaptive peer %s timeout, tag: %d, scale-out: %d/%d, scale-up: %d/%d, block: %d, warp: %d, "
                   "peer position: %d (id %d), signal: %lld, expected: %lld\n", word == kPeerReadyWord ? "ready" : "arrived", kTag,
                   topo.scaleout_rank_idx, topo.num_scaleout_ranks, topo.scaleup_rank_idx, topo.num_scaleup_ranks,
                   static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x / 32), pos, rid,
                   static_cast<long long>(signal), static_cast<long long>(threshold));
        }
        return false;
    });
}

// SM 0's thread `pos` publishes our READY (e + 1) to live peer position `pos`
template <int kNumThreads, int kNumMaxScaleupRanks>
__device__ __forceinline__ void peer_ready_publish_rt(const nixl::NIXLTransport& transport, const layout::WorkspaceLayout& workspace,
                                                      const topology::EPTopology& topo, const int& sm_idx, const int& thread_idx) {
    if (sm_idx > 0)
        return;
    const int my_rid = topology::rank_id(topo.scaleout_rank_idx, topo.scaleup_rank_idx, kNumMaxScaleupRanks);
    for (int pos = thread_idx; pos < topo.num_scaleup_ranks; pos += kNumThreads) {
        if (pos == topo.scaleup_rank_idx or not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, pos))
            continue;
        const int rid = topology::rank_id(topo.scaleout_rank_idx, pos, kNumMaxScaleupRanks);
        transport.put_word_rel(workspace.get_peer_barrier_signal_ptr(my_rid, kPeerReadyWord),
                               workspace.get_peer_barrier_staging_ptr(rid, kPeerReadyWord),
                               *workspace.get_peer_barrier_expected_ptr(rid) + 1, rid);
    }
}

// The positions whose READY a thread has acquired in this launch, one bit per position of the declared maximum (the shape of
// `topology::ScaleupMask`): a thread acquires each destination once, before its first remote write to it
template <int kNumMaxScaleupRanks>
struct AcquiredReadyMask {
    static constexpr int kNumWords = (kNumMaxScaleupRanks + 63) / 64;
    uint64_t words[kNumWords] = {};

    // Marks `pos`, a position of the topology (in [0, kNumMaxScaleupRanks)); true when it was not yet marked (a predicate chain, not a
    // dynamic index, as `ScaleupMask::test`)
    __device__ __forceinline__ bool set(const int& pos) {
        const uint64_t bit = 1ull << (pos & 63);
        bool was_clear = false;
        #pragma unroll
        for (int i = 0; i < kNumWords; ++ i) {
            if ((pos >> 6) == i) {
                was_clear = (words[i] & bit) == 0;
                words[i] |= bit;
            }
        }
        return was_clear;
    }
};

// Acquire live peer position `pos`'s READY of this launch, once per thread (a thread's own position needs none)
template <int64_t kNumTimeoutCycles, int kTag, int kNumMaxScaleupRanks>
__device__ __forceinline__ void peer_ready_acquire_rt(AcquiredReadyMask<kNumMaxScaleupRanks>& acquired, const layout::WorkspaceLayout& workspace,
                                                      const topology::EPTopology& topo, const int& pos) {
    if (pos == topo.scaleup_rank_idx or not acquired.set(pos))
        return;
    const int rid = topology::rank_id(topo.scaleout_rank_idx, pos, kNumMaxScaleupRanks);
    peer_word_wait_rt<kNumTimeoutCycles, kTag>(workspace, topo, pos, rid, kPeerReadyWord, *workspace.get_peer_barrier_expected_ptr(rid) + 1);
}

// The block's end flush and grid arrival: its bulk stores complete, the block arrives on the grid counter (acq_rel: the last
// arriver has observed every block's completed stores), and the result reaches every thread through a reducing CTA barrier (a
// synchronizing operation of the memory model), so every lane that then releases a peer word or zeroes a counter is ordered after
// that acquire. The counter is back to zero for the next launch (no round number: unlike `active_grid_sync`, nothing waits for
// the arrival). `kBarrierIndex` is a named barrier the block's threads may reach from different code locations
template <int kNumThreads, int kBarrierIndex>
__device__ __forceinline__ bool active_grid_arrive(unsigned* grid_barrier_ptr, const int& num_active_blocks) {
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    ptx::cta_barrier<kNumThreads>(kBarrierIndex);
    bool arrived_last = false;
    if (threadIdx.x == 0) {
        arrived_last = ptx::atom_add_acq_rel_gpu(grid_barrier_ptr, 1u) + 1 == static_cast<unsigned>(num_active_blocks);
        if (arrived_last)
            ptx::st_relaxed_gpu(grid_barrier_ptr, 0u);
    }
    return ptx::cta_barrier_or<kNumThreads>(kBarrierIndex, arrived_last);
}

// The last block's exchange, run by one warp: for every live peer, acquire its READY (it has entered this launch, so our
// ARRIVED cannot precede its entry) and release our ARRIVED (e + 2); then wait for every peer's ARRIVED and advance the pair's
// expectation by exactly 2 (this block is the only writer: every other block read the word before arriving)
template <int64_t kNumTimeoutCycles, int kTag, int kNumMaxScaleupRanks>
__device__ __forceinline__ void peer_arrived_exchange_rt(const nixl::NIXLTransport& transport, const layout::WorkspaceLayout& workspace,
                                                         const topology::EPTopology& topo, const int& lane_idx) {
    const int my_rid = topology::rank_id(topo.scaleout_rank_idx, topo.scaleup_rank_idx, kNumMaxScaleupRanks);
    const auto live_peer_rid = [&](const int& pos) {   // -1 for our own or a masked position
        return pos == topo.scaleup_rank_idx or not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, pos) ?
            -1 : topology::rank_id(topo.scaleout_rank_idx, pos, kNumMaxScaleupRanks);
    };
    for (int pos = lane_idx; pos < topo.num_scaleup_ranks; pos += 32) {
        const int rid = live_peer_rid(pos);
        if (rid < 0)
            continue;
        const int64_t expected = *workspace.get_peer_barrier_expected_ptr(rid);
        peer_word_wait_rt<kNumTimeoutCycles, kTag>(workspace, topo, pos, rid, kPeerReadyWord, expected + 1);
        transport.put_word_rel(workspace.get_peer_barrier_signal_ptr(my_rid, kPeerArrivedWord),
                               workspace.get_peer_barrier_staging_ptr(rid, kPeerArrivedWord), expected + 2, rid);
    }
    for (int pos = lane_idx; pos < topo.num_scaleup_ranks; pos += 32) {
        const int rid = live_peer_rid(pos);
        if (rid < 0)
            continue;
        const auto expected_ptr = workspace.get_peer_barrier_expected_ptr(rid);
        const int64_t expected = *expected_ptr;
        peer_word_wait_rt<kNumTimeoutCycles, kTag>(workspace, topo, pos, rid, kPeerArrivedWord, expected + 2);
        *expected_ptr = expected + 2;
    }
    __syncwarp();
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
