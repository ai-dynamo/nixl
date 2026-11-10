// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <nixl_ep_v2/common/comm.cuh>
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/math.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>
#include <nixl_ep_v2/impls/adaptive/combine_policy.cuh>


namespace nixl_ep_v2::elastic::adaptive {

// Dispatch on a fixed grid; `EPTopology` supplies rank decomposition and the active block count.
// Elastic on the NIXL transport: the active ranks are the mask product of `topology.cuh`; masked peers are not counted, written, signalled or waited for
template <bool kDoCPUSync,
          int kNumLaunchSMs,
          int kNumNotifyWarps, int kNumChannelsPerSM,
          int kNumMaxScaleoutRanks, int kNumMaxScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumExpertsPerRank, int kNumTopk, int kExpertAlignment,
          int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumMaxScaleupRanks, 32),
          int kNumScaleoutWarps = kNumChannelsPerSM,
          int kNumForwardWarps = kNumChannelsPerSM,
          int kScaleoutUpdateInterval = 6,
          int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,
          int kNumMaxRanks = kNumMaxScaleoutRanks * kNumMaxScaleupRanks,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,
          int kNumForwardThreads = kNumForwardWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
adaptive_dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* token_metadata_at_forward,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    void* buffer, void* send_buffer,
    void* mapped_host_workspace,
    const topology::EPTopology* topology_ptr,
    const int num_max_tokens_per_rank) {
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    EP_STATIC_ASSERT(kNumMaxScaleoutRanks <= topology::kNumMaxScaleoutRanksCap, "Too many scale-out ranks");
    EP_STATIC_ASSERT(kNumMaxScaleupRanks <= topology::kNumMaxScaleupRanksCap, "Too many scale-up ranks");
    EP_STATIC_ASSERT(kNumMaxScaleoutRanks <= kNumNotifyThreads, "kNumMaxScaleoutRanks must be less than kNumNotifyThreads");
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumChannelsPerSM and kNumForwardWarps == kNumChannelsPerSM, "The channel indices assume one scale-out and one forward warp per channel");
    EP_STATIC_ASSERT(kNumLaunchSMs % 2 == 0, "The adaptive grid is launched in clusters of two");
    EP_STATIC_ASSERT(kNumMaxScaleupRanks <= kNumThreads, "Insufficient threads to clean the sender counters");
    EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for top-k selections");

    // Topology: uniform per block, read once
    const auto topo = topology::load(topology_ptr);
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);

    // Surplus blocks of the fixed grid release their SM at once (before any barrier or mbarrier init)
    if (sm_idx >= topo.num_active_sms)
        return;
    topo.check_launch();

    // The transport context comes from the topology descriptor (`EPTopology`), not from the launch
    const auto ctx = nixl::load_context(topo.transport);
    EP_PROFILE_STAMP(ctx.local_base, DispatchEntry);

    // Runtime topology-derived values
    const int num_scaleout_ranks = topo.num_scaleout_ranks, num_scaleup_ranks = topo.num_scaleup_ranks;
    const int scaleout_rank_idx = topo.scaleout_rank_idx, scaleup_rank_idx = topo.scaleup_rank_idx;
    const int num_active_sms = topo.num_active_sms;
    const int num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    const int num_active_scaleup_ranks = topo.num_active_scaleup_ranks();
    const int num_experts_per_rank = kNumExpertsPerRank;
    const int num_experts = num_ranks * kNumExpertsPerRank;
    const int num_experts_per_scaleout = num_scaleup_ranks * kNumExpertsPerRank;
    const int num_channels = kNumChannelsPerSM * num_active_sms;
    const int num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);

    // Direct mode (MNNVL): one flat NVLink team = exactly one active NVLink domain, whatever its index (the extent
    // `num_scaleout_ranks` may exceed 1; ranks, experts and tokens keep their global numbering, NVLink peers and
    // per-position slots are addressed by the position `rank % num_scaleup_ranks`); the data warps write straight
    // into the peers' scale-up buffers instead of staging through the scale-out send buffer and the forward warps
    const bool is_direct = topo.mode == topology::kModeDirect;

    // Utils
    // NOTES: a warp is a channel (several channels may share one transport channel, see `comm::get_channel_idx`)
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto rank_idx = scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;

    // A rank's position inside its domain (from its rank index)
    const auto position_of = [=](const int& rank) { return rank % num_scaleup_ranks; };

    // Whether the rank owning a (global) expert is active; a selection of a dead expert is dropped
    const auto is_expert_owner_active = [=](const int& expert_idx) { return topology::is_rank_active<kNumMaxScaleupRanks>(topo, expert_idx / num_experts_per_rank); };

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(ctx.local_base);
    const auto ready_atomic_buff = workspace_layout.get_scaleout_channel_signaled_tail_ptr();
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace);

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    // The notify region is sized for the compile-time maxima
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = layout::get_num_notify_smem_bytes(kNumMaxRanks, kNumMaxRanks * kNumExpertsPerRank, kNumNotifyThreads);
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // The data warp's channel: SM-major for the hybrid roles (one scale-out and one forward warp per channel); the direct path numbers its own
    const int channel_in_sm_idx = (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM;
    const int channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;

    // Transport handle. Each warp is a channel, spread over the active blocks and the context's channel count (`comm::get_channel_idx`)
    const auto transport = nixl::NIXLTransport(ctx, comm::get_channel_idx<kNumChannelsPerSM, true>(
        sm_idx, channel_in_sm_idx, ctx.num_channels, num_active_sms, warp_idx < kNumNotifyWarps));

    // Global parallel barriers for scale-out subteam and scale-up subteam
    comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kHybridDispatchTag0, kNumMaxScaleupRanks, false, false, true>(
        ctx, workspace_layout, topo, sm_idx, thread_idx);
    EP_PROFILE_STAMP(ctx.local_base, DispatchEntryBarrier);

    // The golden layout during the whole process for both scale-out and forward warps
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumScaleoutWarps + kNumForwardWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);

    // All the buffers
    // The scale-out send buffer (the RDMA sources) is a region of its own: the NIC reads it after this kernel has ended; the next
    // dispatch rewrites it only past its start barrier, which every receiver reaches after consuming this launch's records (see `comm::gpu_barrier_rt`)
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank, buffer);
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, 1, num_max_tokens_per_rank, send_buffer);
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, num_scaleout_ranks, num_channels * num_max_tokens_per_channel, scaleup_buffer.get_buffer_end_ptr());

    // Init TMA for scale-out and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // Preload next token
    const auto preload_next_token = [&](const int& token_idx) {
        if (token_idx >= num_tokens)
            return;

        // Issue TMA load
        const auto token_i64_idx = static_cast<int64_t>(token_idx);
        if (ptx::elect_one_sync()) {
            ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                             mbarrier_ptr, kNumHiddenBytes);
        }
        __syncwarp();

        // Issue SF `cp.async`
        if constexpr (kNumSFPacks > 0) {
            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
            const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
            const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

            constexpr auto kNumFullIters = kNumSFPacks / 32;
            #pragma unroll
            for (int k = 0; k < kNumFullIters; ++ k) {
                ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                 smem_dst_ptr + k * 32 + lane_idx);
            }
            if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                 smem_dst_ptr + kNumFullIters * 32 + lane_idx);
            }
            ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
            __syncwarp();
        }
    };

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // Assign shared memory
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + num_ranks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * num_active_sms) {
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0 and not is_expert_owner_active(dst_expert_idx))
                dst_expert_idx = -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / num_experts_per_rank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        EP_PROFILE_STAMP(ctx.local_base, DispatchCountLocal);

        // Do full-grid reduction
        for (int i = thread_idx; i < num_ranks + num_experts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0
        if (sm_idx == 0) {
            // Reduce all SM's count
            // Wait all active SMs' arrival
            for (int i = thread_idx; i < num_ranks + num_experts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == num_active_sms) {
                        // Encode and write into the send buffer (hybrid) or shared memory (direct)
                        const auto encoded = math::encode_decode_positive<int>(status & 0xffffffffll);
                        if (is_direct) {
                            rank_expert_count[i] = encoded;
                        } else {
                            int* dst_ptr;
                            if (i < num_ranks) {
                                dst_ptr = workspace_layout.get_scaleout_rank_count_ptr<true>(
                                    i / num_scaleup_ranks, i % num_scaleup_ranks);
                            } else {
                                const int expert_idx = i - num_ranks;
                                dst_ptr = workspace_layout.get_scaleout_expert_count_ptr<true>(
                                    expert_idx / num_experts_per_scaleout, expert_idx % num_experts_per_scaleout);
                            }
                            *dst_ptr = encoded;
                        }

                        // Clean for the next usage
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("NIXL EP v2 adaptive notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "slot: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, num_scaleout_ranks, scaleup_rank_idx, num_scaleup_ranks, i,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), num_active_sms);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            EP_PROFILE_STAMP(ctx.local_base, DispatchCountReduction);
            if (is_direct) {
                // Direct mode uses a single-stage exchange over the flat NVLink team. Every
                // rank writes its (encoded) per-rank and per-expert counts into its own slots of every peer,
                // waits for all peers' slots, and reduces the expert counts locally. No release atomics.
                // Rank counts travel in position space (the slot is the writer's position; the value is the count
                // destined to our domain's rank at position `i`), expert counts in global space (the slot is the
                // writer's rank index times the slots per rank plus the expert): the one active domain need not be domain 0
                // Issue scaleup rank count writes to peers
                for (int i = thread_idx; i < num_scaleup_ranks; i += kNumNotifyThreads) {
                    if (not topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, i))
                        continue;
                    const auto dst_rank_counter = workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx;
                    transport.put_value<int64_t>(dst_rank_counter, static_cast<int64_t>(rank_count[scaleout_rank_idx * num_scaleup_ranks + i]),
                                                 topology::scaleup_peer<kNumMaxScaleupRanks>(topo, i));
                }
                __syncwarp();

                // Issue scaleup expert count writes to peers (per-element NVLink stores)
                for (int i = thread_idx; i < num_experts; i += kNumNotifyThreads) {
                    const auto dst_rank = i / num_experts_per_rank;
                    if (not topology::is_rank_active<kNumMaxScaleupRanks>(topo, dst_rank))
                        continue;
                    const auto idx = num_experts_per_rank * rank_idx + (i % num_experts_per_rank);
                    transport.put_value<int64_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), topology::scaleup_peer<kNumMaxScaleupRanks>(topo, position_of(dst_rank)));
                }

                // This is necessary, as the waited results will rewrite the shared memory
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

                // Wait for rank and expert count: `num_scaleup_ranks` rank slots (writer = position `i`), then the
                // `num_experts` expert slots (writer = the rank index owning expert `i - num_scaleup_ranks`)
                const auto start_clock = clock64();
                for (int i = thread_idx; i < num_scaleup_ranks + num_experts; i += kNumNotifyThreads) {
                    const bool is_rank_slot = i < num_scaleup_ranks;
                    const int expert_slot = i - num_scaleup_ranks;
                    const bool active = is_rank_slot ? topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, i)
                                                     : is_expert_owner_active(expert_slot);
                    int* smem_dst = is_rank_slot ? rank_count + i : expert_count + expert_slot;
                    if (not active) {
                        *smem_dst = 0;
                        continue;
                    }
                    const auto slot_ptr = workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks);
                    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                        // NOTES: the global memory type has 64 bits
                        const auto count = static_cast<int>(ptx::ld_volatile<int64_t>(slot_ptr));
                        const auto decoded = math::encode_decode_positive(count);
                        if (math::is_decoded_positive_ready(decoded)) {
                            *slot_ptr = 0;
                            *smem_dst = decoded;
                            return true;
                        }

                        if (is_last_check) {
                            printf("NIXL EP v2 adaptive notify (direct) timeout, rank: %d, slot: %d, wait %s: %d, count: %d\n",
                                   rank_idx, i, i < num_scaleup_ranks ? "position" : "rank",
                                   i < num_scaleup_ranks ? i : (i - num_scaleup_ranks) / num_experts_per_rank, decoded);
                        }
                        return false;
                    }, start_clock);
                }
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

                // Reduce expert count and add stats
                for (int i = thread_idx; i < num_experts_per_rank; i += kNumNotifyThreads) {
                    int sum = 0;
                    for (int j = 0; j < num_ranks; ++ j)
                        sum += expert_count[j * num_experts_per_rank + i];

                    // Write unaligned count before aligning
                    num_unaligned_recv_tokens_per_expert[i] = sum;

                    expert_count[i] = math::align(sum, kExpertAlignment);

                    // Update statistics counters
                    if (cumulative_local_expert_recv_stats != nullptr)
                        atomicAdd(cumulative_local_expert_recv_stats + i, sum);
                }
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

                // Write host workspace: the `num_scaleup_ranks` position-space rank counts, then the local expert counts
                // (what the host reads back under `topo.num_scaleup_ranks` rank slots)
                if constexpr (kDoCPUSync) {
                    for (int i = thread_idx; i < num_scaleup_ranks + num_experts_per_rank; i += kNumNotifyThreads) {
                        *host_workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks) =
                            math::encode_decode_positive(i < num_scaleup_ranks ? rank_count[i] : expert_count[i - num_scaleup_ranks]);
                    }
                    __syncwarp();
                }
            } else {
                // Hybrid mode: two-stage exchange, scale-out over the rail peers then scale-up over NVLink
                // Issue rank and expert counts separately.
                if (topo.is_scaleout_active(thread_idx)) {
                    const auto dst_id = topology::scaleout_peer<kNumMaxScaleupRanks>(topo, thread_idx);
                    transport.put_words(
                        workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
                        workspace_layout.get_scaleout_rank_count_ptr<true>(thread_idx),
                        num_scaleup_ranks * sizeof(int), dst_id, false);
                    transport.put_words(
                        workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
                        workspace_layout.get_scaleout_expert_count_ptr<true>(thread_idx),
                        num_experts_per_scaleout * sizeof(int), dst_id);
                }
                __syncwarp();

                // Util functions to get metadata from scale-out peers
                // NOTES: each count word is polled on its own until it reads as ready: a 4-byte word lands whole (see `put_words`)
                const auto recv_and_reduce = [=](const auto& get_ptr_func, const int& slot, const bool& is_expert_reduction = false) -> int {
                    int count = 0;
                    for (int j = 0; j < num_scaleout_ranks; ++ j) {
                        // Masked domains send nothing
                        if (not topo.is_scaleout_active(j))
                            continue;
                        const auto ptr = get_ptr_func(j);
                        int decoded;
                        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){
                            decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));
                            if (math::is_decoded_positive_ready(decoded))
                                return true;

                            if (is_last_check) {
                                printf("NIXL EP v2 adaptive notify (scale-out %s reduction) timeout, "
                                       "scale-out: %d, scale-up: %d, "
                                       "slot: %d, wait scale-out: %d, decoded: %d\n",
                                       is_expert_reduction ? "expert" : "rank",
                                       scaleout_rank_idx, scaleup_rank_idx, slot, j,
                                       decoded);
                            }
                            return false;
                        });

                        // Add and clean for next usages
                        count += decoded, *ptr = 0;
                    }
                    return count;
                };

                // Write into all scale-up peers' rank-level counters
                for (int i = thread_idx; i < num_scaleup_ranks; i += kNumNotifyThreads) {
                    // Wait scale-out arrival and reduce (a masked position's words too: the active domains send an encoded zero for it)
                    const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                        return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
                    }, i);
                    if (not topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, i))
                        continue;

                    // Write into the remote scale-up peer
                    const int64_t counter = (static_cast<int64_t>(num_active_scaleup_ranks) << 32ll) | count;
                    transport.put_value<int64_t>(
                        workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
                        counter, topology::scaleup_peer<kNumMaxScaleupRanks>(topo, i));
                }
                __syncwarp();

                // Atomic add into all scale-up peers' expert-level counters
                for (int i = thread_idx; i < num_experts_per_scaleout; i += kNumNotifyThreads) {
                    // Wait scale-out arrival and reduce
                    const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                        return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                    }, i, true);

                    // Write into the remote scale-up peer
                    const int64_t counter = (1ll << 32ll) | count;
                    const auto dst_scaleup_rank_idx = i / num_experts_per_rank;
                    if (not topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, dst_scaleup_rank_idx))
                        continue;
                    const auto expert_idx_in_dst_rank = i % num_experts_per_rank;
                    transport.red_add_rel<int64_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
                        counter, topology::scaleup_peer<kNumMaxScaleupRanks>(topo, dst_scaleup_rank_idx));
                }
                // There are shared memory reads above, a barrier is necessary
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

                // NOTES: from now on, the `rank` and `expert`s size change into the local size
                expert_count = rank_expert_count + num_scaleup_ranks;

                // Wait local counters to be ready
                // NOTES: here we only care the prefix sum by scale-up peers (used for later epilogue), not all ranks
                // Strided over the notify threads: a small world can have more counters than the notify warps have threads
                comm::timeout_while_strided<kNumTimeoutCycles>(thread_idx, num_scaleup_ranks + num_experts_per_rank, kNumNotifyThreads,
                    [&](const int& i, const bool& is_last_check) {
                    if (i < num_scaleup_ranks and not topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, i)) {
                        rank_expert_count[i] = 0;
                        if constexpr (kDoCPUSync)
                            *host_workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks) = math::encode_decode_positive(0);
                        return true;
                    }
                    const auto slot_ptr = workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks);
                    const auto status = ptx::ld_volatile<int64_t>(slot_ptr);
                    if ((status >> 32ull) == num_active_scaleup_ranks) {
                        // Clean GPU workspace and write into host workspace
                        const auto count = static_cast<int>(status & 0xffffffffll);
                        const auto aligned_count = math::align<int>(
                            count, i < num_scaleup_ranks ? 1 : kExpertAlignment);

                        *slot_ptr = 0;
                        if constexpr (kDoCPUSync) {
                            *host_workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks) =
                                math::encode_decode_positive(aligned_count);
                        }

                        // Update statistics counters
                        if (cumulative_local_expert_recv_stats != nullptr and i >= num_scaleup_ranks)
                            atomicAdd(cumulative_local_expert_recv_stats + (i - num_scaleup_ranks), count);

                        // Write unaligned count before aligning
                        if (i >= num_scaleup_ranks)
                            num_unaligned_recv_tokens_per_expert[i - num_scaleup_ranks] = count;

                        // Save for later prefix sum calculation
                        rank_expert_count[i] = aligned_count;
                        return true;
                    }

                    if (is_last_check) {
                        printf("NIXL EP v2 adaptive notify (scale-up reduction) timeout, "
                               "scale-out: %d/%d, scale-up: %d/%d, "
                               "slot: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, num_scaleout_ranks, scaleup_rank_idx, num_scaleup_ranks, i,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), num_active_scaleup_ranks);
                    }
                    return false;
                });
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
            }  // hybrid mode

            EP_PROFILE_STAMP(ctx.local_base, DispatchCountExchange);

            // Do prefix sum by the warps of the first SM
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            // NOTES: in direct mode the rank counts are in position space (the first `num_scaleup_ranks` entries of
            // `rank_count`) and `expert_count` still points at `rank_expert_count + num_ranks`
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // Store into global memory
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    psum = ptx::exchange(sum, 31);
                }
                return psum;
            };
            if (warp_idx == 0) {
                // Inclusive prefix sum
                // The total goes into the fixed slot `[kNumMaxScaleupRanks]`, which the epilogue and the combine read whatever the active ranks
                const auto total = do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, num_scaleup_ranks, 0);
                if (lane_idx == 0)
                    psum_num_recv_tokens_per_scaleup_rank[kNumMaxScaleupRanks] = total;
                EP_PROFILE_STAMP(ctx.local_base, DispatchCountPrefix);
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, num_experts_per_rank, 1);
                EP_PROFILE_STAMP(ctx.local_base, DispatchCountPrefix);
            }
        }
    } else if (is_direct) {
        // Direct mode: every data warp (scale-out and forward warps alike) is a channel that TMA-stores its
        // tokens straight into the destination peers' scale-up buffers over NVLink; linked-list slots stay unused.
        constexpr int kNumDataWarps = kNumScaleoutWarps + kNumForwardWarps;
        const int data_warp_idx = warp_idx - kNumNotifyWarps;
        // Warp-major channel numbering sends consecutive tokens to different SMs, so a
        // small batch uses every SM's write bandwidth instead of piling onto SM 0 (the hybrid roles number
        // channels SM-major because their per-channel buffers and tails are laid out that way)
        const int num_data_channels = kNumDataWarps * num_active_sms;
        const int channel_idx = data_warp_idx * num_active_sms + sm_idx;

        // My region in every peer's scale-up buffer (`[num_scaleup_ranks][...]`, indexed by my position; the slots beyond
        // `num_max_tokens_per_rank` that an extent above 1 adds stay unused in direct mode)
        const auto recv_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Iterate all tokens
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += num_data_channels) {
            // Wait TMA store arrivals
            ptx::tma_store_wait();
            __syncwarp();

            // Issue data TMA and SF `cp.async`
            preload_next_token(token_idx);

            // Load top-k indices and weights
            int stored_dst_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                // The destination rank is global; its NVLink peer and its per-position slots are its position
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / num_experts_per_rank : -1;
                if (stored_dst_rank_idx >= 0 and not topology::is_rank_active<kNumMaxScaleupRanks>(topo, stored_dst_rank_idx))
                    stored_dst_rank_idx = -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            // Fence metadata writes here; SF copies are fenced after their completion wait below
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * num_max_tokens_per_rank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
            if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
                stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + position_of(stored_dst_rank_idx), 1);
            __syncwarp();

            // Wait TMA load arrival
            // NOTES: this arrive must be after the `ptx::cp_async_mbarrier_arrive`
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                // Make completed SF cp.async writes visible to the TMA stores
                ptx::tma_store_fence();
            }
            __syncwarp();

            // Issue TMA NVLink stores
            if (stored_dst_slot_idx >= 0) {
                const auto dst_ptr = transport.get_p2p_ptr(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                                                   topology::scaleup_peer<kNumMaxScaleupRanks>(topo, position_of(stored_dst_rank_idx)));
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            }
            ptx::tma_store_commit();
            __syncwarp();
        }
        EP_PROFILE_STAMP(ctx.local_base, DispatchSendSubmitted);
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer(channel_idx, num_max_tokens_per_channel);

        // One packed word per channel/source domain: finish flag and cumulative ready count.
        int stored_scaleout_tail = 0, stored_old_scaleout_tail = 0;
        const bool use_incremental_combine = combine_policy::use_incremental(topo, num_tokens);
        auto* channel_ready_atomic_buff = ready_atomic_buff +
            channel_idx * topology::kNumMaxScaleoutRanksCap + scaleout_rank_idx;
        const auto update_scaleout_tail = [&](const bool& finish_flag = false) {
            if (topo.is_scaleout_active(lane_idx) and
                (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag)) {
                const auto signaled_tail = math::pack2<int, int64_t>(
                    combine_policy::dispatch_finish_flag(finish_flag, use_incremental_combine), stored_scaleout_tail);
                const auto old_signaled_tail = math::pack2<int, int64_t>(0, stored_old_scaleout_tail);
                // Same-peer/channel atomics order the deferred payload puts; local stores were TMA-waited.
                transport.red_add_rel(channel_ready_atomic_buff, signaled_tail - old_signaled_tail,
                                      topology::scaleout_peer<kNumMaxScaleupRanks>(topo, lane_idx));
                stored_old_scaleout_tail = stored_scaleout_tail;
            }
            __syncwarp();
        };

        // Iterate all tokens
        preload_next_token(channel_idx);
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += num_channels) {
            // Load top-k indices and weights
            int stored_dst_scaleout_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 and is_expert_owner_active(dst_expert_idx) ?
                    dst_expert_idx / num_experts_per_scaleout : -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * num_max_tokens_per_rank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
            const auto stored_old_slot_idx = ptx::exchange(
                stored_scaleout_tail, stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            if (ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and stored_dst_scaleout_rank_idx >= 0)
                stored_dst_slot_idx = stored_old_slot_idx;

            // Update scale-out tail
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);
            stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // Our own domain's record is stored into our receive slot; another domain's goes through the send buffer and an RDMA put
            const bool needs_rdma = stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx;
            const bool own_domain_row = stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx;
            const bool any_rdma = ptx::any(needs_rdma);

            // Wait TMA arrival and store into the send buffer when any lane goes by RDMA
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                // Make completed SF cp.async writes visible to the TMA stores
                ptx::tma_store_fence();
            }
            __syncwarp();
            // `any_rdma` is a full-warp vote: consume it in warp-uniform control flow, never inside the elected lane's block alone
            if (any_rdma) {
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                                      tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                }
                __syncwarp();
            }

            // Own-domain rows: store the payload before publishing completion
            // Local rank can be bypassed
            if (own_domain_row) {
                ptx::tma_store_1d(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            }
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            __syncwarp();

            // Preload the next token (overlapping with the RDMA puts)
            preload_next_token(token_idx + num_channels);

            // Issue payload writes; the following completion atomic rings their deferred doorbells.
            if (needs_rdma) {
                transport.rdma_put(
                    scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                    scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                    tma_buffer.get_num_bytes<false>(),
                    topology::scaleout_peer<kNumMaxScaleupRanks>(topo, stored_dst_scaleout_rank_idx), false);
            }
            __syncwarp();
            update_scaleout_tail();
        }

        // Publish the last partial batch and finish even when this channel sent no records to a domain.
        update_scaleout_tail(true);
        EP_PROFILE_STAMP(ctx.local_base, DispatchSendSubmitted);
    } else {
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer(channel_idx, num_max_tokens_per_channel);
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Shape of `token_metadata_at_forward`: `[num_channels, num_scaleout_ranks * num_max_tokens_per_channel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        const int num_tokens_in_linked_list = num_max_tokens_per_channel * num_scaleout_ranks + 1;
        token_metadata_at_forward += channel_idx * (num_tokens_in_linked_list * kNumForwardMetadataDims);

        // Transform linked list index
        const auto transform_linked_list_idx = [=](const int& idx) {
            return channel_idx * (num_tokens_in_linked_list * num_scaleup_ranks) +
                idx * num_scaleup_ranks + scaleup_rank_idx;
        };

        // Forward tokens from scale-out ranks
        int num_tokens_processed = 0;
        int stored_scaleout_old_tail_idx = 0;
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};
        const bool lane_tracks_active_domain = topo.is_scaleout_active(lane_idx);
        int stored_finish_flag = not lane_tracks_active_domain;
        int stored_scaleout_tail_idx = 0;
        int recv_scaleout_rank_idx = channel_idx % num_scaleout_ranks;
        uint32_t wip_mask;
        while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {
            // Pick next rank in round-robin
            const auto offset = (recv_scaleout_rank_idx + 1) % num_scaleout_ranks;
            const auto hi_mask = (wip_mask >> offset) << offset;
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);

            // Wait for this rank to have data (or finish)
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                const uint32_t arrived_or_finished =
                    stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))
                    return true;

                // Timeout
                if (is_last_check) {
                    if (lane_tracks_active_domain) {
                        printf("NIXL EP v2 adaptive dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                               "channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d)\n",
                               scaleout_rank_idx, scaleup_rank_idx,
                               channel_idx, lane_idx, stored_scaleout_old_tail_idx,
                               stored_finish_flag, stored_scaleout_tail_idx);
                    }
                    return false;
                }

                // Read new signaled tails
                if (lane_tracks_active_domain) {
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                        ready_atomic_buff + channel_idx * topology::kNumMaxScaleoutRanksCap + lane_idx);
                    math::unpack2<int, int64_t>(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
                }
                __syncwarp();
                return false;
            });

            // Process one chunk from the current rank
            const auto start_slot_idx = ptx::exchange(stored_scaleout_old_tail_idx, recv_scaleout_rank_idx);
            const auto end_slot_idx = std::min(
                ptx::exchange(stored_scaleout_tail_idx, recv_scaleout_rank_idx),
                start_slot_idx + kNumSlotsPerForwardChunk
            );
            if (lane_idx == recv_scaleout_rank_idx)
                stored_scaleout_old_tail_idx = end_slot_idx;

            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);
            for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {
                const auto token_buffer = recv_buffer.get_token_buffer(slot_idx);

                // Wait TMA arrival
                ptx::tma_store_wait();
                __syncwarp();

                // TMA load into shared memory
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), token_buffer.get_base_ptr(),
                                     mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                }
                __syncwarp();

                // Read top-k indices
                int stored_dst_scaleup_rank_idx = -1;
                auto dst_expert_idx = lane_idx < kNumTopk ? tma_buffer.get_topk_idx_ptr()[lane_idx] : -1;
                dst_expert_idx -= scaleout_rank_idx * num_experts_per_scaleout;
                stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < num_experts_per_scaleout ?
                    dst_expert_idx / num_experts_per_rank : -1;
                if (stored_dst_scaleup_rank_idx >= 0 and not topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, stored_dst_scaleup_rank_idx))
                    stored_dst_scaleup_rank_idx = -1;

                // Write the per-scaleup channel index for this token
                int linked_list_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;
                    const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;
                    const auto exchanged = ptx::exchange(
                        stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                    linked_list_idx = valid ? exchanged : linked_list_idx;
                }
                if (lane_idx < kNumTopk) {
                    tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Deduplicate for scale-up ranks
                int stored_dst_slot_idx = -1;
                if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                __syncwarp();

                // Issue TMAs
                if (stored_dst_slot_idx >= 0) {
                    const auto dst_ptr = transport.get_p2p_ptr(
                        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        topology::scaleup_peer<kNumMaxScaleupRanks>(topo, stored_dst_scaleup_rank_idx));
                    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // Add per-scale-up counter: one 32-bit mask per lane group of positions (lane `l` of group `j` is position `j * 32 + l`)
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto group_mask = ptx::gather_position_group(stored_dst_scaleup_rank_idx, j);
                    stored_scaleup_send_counters[j] += (group_mask >> lane_idx) & 1;
                }

                // Record metadata at forward
                const auto metadata_ptr = token_metadata_at_forward +
                    num_tokens_processed * kNumForwardMetadataDims;

                // Source token index and last token index flag
                if (ptx::elect_one_sync()) {
                    metadata_ptr[0] = tma_buffer.get_src_token_global_idx_ptr()[0];
                    metadata_ptr[1] = slot_idx == (end_slot_idx - 1);
                }

                // Second, original top-k indices and destination slots
                if (lane_idx < kNumTopk) {
                    metadata_ptr[2 + lane_idx] = stored_dst_scaleup_rank_idx;
                    metadata_ptr[2 + kNumTopk + lane_idx] = stored_dst_slot_idx;
                }
                num_tokens_processed += 1;
                __syncwarp();
            }
        }

        // All records from every active source domain have been consumed; reset for the next invocation
        if (lane_tracks_active_domain)
            ptx::st_relaxed_sys(ready_atomic_buff + channel_idx * topology::kNumMaxScaleoutRanksCap + lane_idx, int64_t(0));
        __syncwarp();

        // Every source publishes its policy even for empty sends; retain it in the reserved final row.
        const auto incremental_combine_domains = ptx::gather(
            lane_tracks_active_domain and combine_policy::finish_uses_incremental(stored_finish_flag));
        if (ptx::elect_one_sync()) {
            token_metadata_at_forward[num_tokens_processed * kNumForwardMetadataDims] = -1;
            token_metadata_at_forward[(num_tokens_in_linked_list - 1) * kNumForwardMetadataDims + 1] =
                static_cast<int>(incremental_combine_domains);
        }
        __syncwarp();

        // Update linked list's ending position
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
            if (const auto j = i * 32 + lane_idx; topology::is_scaleup_active<kNumMaxScaleupRanks>(topo, j)) {
                const auto dst_ptr = transport.get_p2p_ptr(tail_ptr, topology::scaleup_peer<kNumMaxScaleupRanks>(topo, j));
                ptx::st_relaxed_sys(dst_ptr, transform_linked_list_idx(stored_scaleup_send_counters[i]));
            }
        }
        __syncwarp();
        EP_PROFILE_STAMP(ctx.local_base, DispatchForwardSubmitted);
    }

    // Scale-up barrier to ensure data arrival
    // As scale-out tokens have already been consumed by forwarders, no need to do scale-out barrier again
    comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kHybridDispatchTag1, kNumMaxScaleupRanks, true, true, false>(
        ctx, workspace_layout, topo, sm_idx, thread_idx, /* do not scale-out */ false);
    EP_PROFILE_STAMP(ctx.local_base, DispatchFinalBarrier);

    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Clean scale-up counters
    // All scale-out counters should be cleaned before
    if (sm_idx == 0 and thread_idx < num_scaleup_ranks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace nixl_ep_v2::elastic::adaptive
