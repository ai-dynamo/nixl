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

#include <nixl_ep_v2/common/comm.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/math.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>
#include <nixl_ep_v2/impls/combine_utils.cuh>

namespace nixl_ep_v2::elastic::adaptive {

// DeepSeek's hybrid combine on a fixed grid; rank decomposition and active block count come from the topology descriptor (`EPTopology`)
// Elastic on the NIXL transport: masked positions have no linked list and get no tails, masked domains are not sent to (see `topology.cuh`)
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumLaunchSMs,
          int kNumChannelsPerSM,
          int kNumMaxScaleoutRanks, int kNumMaxScaleupRanks,
          int kHidden,
          int kNumExpertsPerRank, int kNumTopk,
          int64_t kNumTimeoutCycles,
          // Arrival protocol of the scale-out rows: the flags inside the records (`TokenLayout::with_flags`) instead of the
          // completion counters; the host's `payload_flags`
          bool kPayloadFlags = false,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumMaxScaleupRanks, 32),
          int kNumScaleupUpdateInterval = 3,
          // RDMA completion batch to a scale-out peer: publish progress after this many newly completed tokens, and ring the
          // doorbell after this many sends without progress (the forward warps); with payload flags the doorbell interval of
          // the rows to a domain
          int kNumScaleoutUpdateInterval = kPayloadFlags ? 6 : 3,
          int kNumScaleupWarps = kNumChannelsPerSM,
          int kNumForwardWarps = kNumChannelsPerSM,
          int kNumWarps = kNumScaleupWarps + kNumForwardWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
adaptive_combine_impl(nv_bfloat16* x,
                      float* topk_weights,
                      int* src_metadata,
                      int* psum_num_recv_tokens_per_scaleup_rank,
                      int* token_metadata_at_forward,
                      int* channel_linked_list,
                      void* buffer, void* send_buffer,
                      void* workspace,
                      const topology::EPTopology* topology_ptr,
                      int num_reduced_tokens,
                      const bool dynamic_take,
                      const int num_max_tokens_per_rank) {
    EP_STATIC_ASSERT(kNumMaxScaleoutRanks <= topology::kNumMaxScaleoutRanksCap, "Too many scale-out ranks");
    EP_STATIC_ASSERT(kNumMaxScaleupRanks <= topology::kNumMaxScaleupRanksCap, "Too many scale-up ranks");
    EP_STATIC_ASSERT(kNumLaunchSMs % 2 == 0, "The adaptive grid is launched in clusters of two");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // Topology: uniform per block, read once
    const auto topo = topology::load(topology_ptr);

    // Surplus blocks of the fixed grid release their SM at once
    if (sm_idx >= topo.num_active_sms)
        return;

    // The transport context comes from the topology descriptor (`EPTopology`)
    const auto ctx = nixl::load_context(topo.transport);

    const int num_scaleout_ranks = topo.num_scaleout_ranks, num_scaleup_ranks = topo.num_scaleup_ranks;
    const int scaleout_rank_idx = topo.scaleout_rank_idx, scaleup_rank_idx = topo.scaleup_rank_idx;
    const int num_channels = kNumChannelsPerSM * topo.num_active_sms;
    const int num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
    EP_DEVICE_ASSERT(num_scaleout_ranks <= kNumMaxScaleoutRanks and num_scaleup_ranks <= kNumMaxScaleupRanks);
    EP_DEVICE_ASSERT(topo.is_participant() and "A rank outside the live world must not launch");

    // Dense rank ids of our peers
    const auto scaleup_peer_id = [=](const int& k) { return topology::rank_id(scaleout_rank_idx, k, kNumMaxScaleupRanks); };
    const auto scaleout_peer_id = [=](const int& d) { return topology::rank_id(d, scaleup_rank_idx, kNumMaxScaleupRanks); };

    // Layout decisions (compile-time in the hybrid kernel)
    const bool use_scaleout_rank_layout = kAllowMultipleReduction and num_scaleout_ranks <= kNumTopk;
    const bool use_scaleup_rank_layout = kAllowMultipleReduction and num_scaleup_ranks <= kNumTopk;
    const int num_tokens_in_scaleout_layout = use_scaleout_rank_layout ? num_scaleout_ranks : kNumTopk;
    const int num_tokens_in_scaleup_layout = use_scaleup_rank_layout ? num_scaleup_ranks : kNumTopk;

    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;

    // Combine vector type selection
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);

    // Workspaces (fixed expert slots per rank: the live expert count is the live rank count times the slots)
    const auto workspace_layout = layout::WorkspaceLayout(workspace, num_scaleout_ranks, num_scaleup_ranks,
                                                          num_scaleout_ranks * num_scaleup_ranks * kNumExpertsPerRank);
    const auto ready_atomic_buff = workspace_layout.get_combine_ready_atomic_buff();
    EP_PROFILE_STAMP(workspace_layout, 14);  // combine main: entry

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens < 0)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumMaxScaleupRanks);

    // Token layouts (the communication record: flags inside under `kPayloadFlags`)
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false, nullptr, kPayloadFlags);

    // TMA buffers
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);

    // All the buffer layouts
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank,
        scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, kAllowMultipleReduction ? 1 : kNumTopk, num_channels * (num_scaleout_ranks * num_max_tokens_per_channel),
        send_buffer);

    // Init TMA for scale-up and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // Each warp is a channel
    const auto channel_idx = comm::get_channel_idx<kNumLaunchSMs, kNumChannelsPerSM>(sm_idx, warp_idx % kNumChannelsPerSM, ctx.num_channels);
    const auto transport = nixl::NIXLTransport(ctx, channel_idx);
    constexpr int kEndBarrierIndex = 1;   // the direct kernel's end sequence (`comm::active_grid_arrive`)

    // Global parallel barriers for scale-out subteam and scale-up subteam
    // NOTES: the hybrid roles need a grid sync first, as there are channel scale-up tail cleaning before;
    // the direct path has no such cleaning and starts like DeepSeek's direct kernel (no start sync)
    const bool is_direct = topo.mode == topology::kModeDirect;
    // Direct mode: per-peer ready/arrived signalling instead of the two full-team barriers (`comm::kPeerReadyWord`): SM 0 publishes
    // our READY now, every remote row write below sits behind its writer's acquire of the destination's READY, and the grid's
    // last block exchanges the ARRIVED words at the end
    if (is_direct) {
        comm::peer_ready_publish_rt<kNumThreads, kNumMaxScaleupRanks>(transport, workspace_layout, topo, sm_idx, thread_idx);
    } else {
        // The previous epilogue has finished. Initialize descending progress to num_max_tokens_per_rank before the entry barrier
        // (no counters under payload flags: the records carry their arrival)
        if constexpr (not kPayloadFlags) {
            if (warp_idx >= kNumScaleupWarps and lane_idx < num_scaleout_ranks) {
                const int completion_channel_idx = sm_idx * kNumChannelsPerSM + warp_idx - kNumScaleupWarps;
                ptx::st_relaxed_sys(ready_atomic_buff + completion_channel_idx * num_scaleout_ranks + lane_idx, int64_t{num_max_tokens_per_rank});
            }
        }
        comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kHybridCombineTag0, kNumMaxScaleupRanks, false, true, true>(
            transport, workspace_layout, topo, sm_idx, thread_idx);
    }

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;
    constexpr int kNumRegistersForScaleupWarps = 40;
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;

    // Direct mode (MNNVL, exactly one live NVLink domain, whatever its index): every warp TMA-stores its share of the
    // tokens straight into the source rank's receive rows over NVLink, and the reduce epilogue reads those rows. The
    // rows are the scale-up buffer (`[num_tokens_in_scaleup_layout][num_scaleout_ranks * num_max_tokens_per_rank]` per
    // position: the bound's row stride, whatever the live domain), which is exactly DeepSeek's direct-kernel receive
    // buffer. No forwarding stage, no linked list. A source rank is global
    // (`src_global_token_idx / num_max_tokens_per_rank`); its NVLink peer is its position `rank % num_scaleup_ranks`.
    EP_DEVICE_ASSERT(not is_direct or topo.num_live_scaleout_ranks() == 1);
    if (is_direct) {
        // Rotate the warp index by the rank so that peers spread their writes (as DeepSeek's direct kernel does)
        const auto rotated_warp_idx = (warp_idx + scaleup_rank_idx) % kNumWarps;
        const auto global_warp_idx = rotated_warp_idx * topo.num_active_sms + sm_idx;

        // Expanding send mode must not be backward
        if constexpr (kDoExpandedSend)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // The source rank's address of one of our symmetric pointers (direct mode: every peer reachable with stores)
        const auto peer_ptr = [&](void* ptr, const int& src_rank_idx) {
            const auto dst_ptr = transport.get_p2p_ptr(ptr, scaleup_peer_id(src_rank_idx % num_scaleup_ranks));
            EP_DEVICE_ASSERT(dst_ptr != nullptr and "Direct mode needs every live peer reachable with stores");
            return dst_ptr;
        };

        // Do TMA writes into the remote buffers. Static partition (`dynamic_take` off): one contiguous chunk of
        // ceil(N / W) rows per warp over the W active warps, as before. Dynamic take: chunks of `chunk_rows` rows
        // (a quarter of a warp's share, 1 to kNumMaxChunkRows), the first chunk per warp static, the rest from a
        // rank-wide counter when the previous chunk's stores are committed; the chunk sequence is interleaved over
        // the source-rank regions of the received rows (rows are grouped by source position) and rotated per rank,
        // so a rank's writes spread over all its destinations at every moment and peers start on different ones.
        // Chunks that fit one wave take no atomic. The counter is zero at rest (zeroed by the end sequence below)
        constexpr int kNumMaxChunkRows = 8, kNumChunkWaves = 4;
        const int num_total_warps = topo.num_active_sms * kNumWarps;
        const int num_tokens_per_warp = math::ceil_div(num_reduced_tokens, num_total_warps);
        const int dyn_chunk_rows = max(1, min(kNumMaxChunkRows, num_reduced_tokens / (kNumChunkWaves * num_total_warps)));
        const int dyn_num_chunks = math::ceil_div(num_reduced_tokens, dyn_chunk_rows);
        const bool take_chunks = dynamic_take and dyn_num_chunks > num_total_warps;
        const int chunk_rows = take_chunks ? dyn_chunk_rows : num_tokens_per_warp;
        const int num_chunks = chunk_rows > 0 ? math::ceil_div(num_reduced_tokens, chunk_rows) : 0;
        // Interleaving: the chunk sequence index `c` maps to region (c + rank) mod G, chunk c / G inside it, G = positions;
        // indices past the last chunk (an uneven last region) are skipped, so the sequence runs to G x per-region chunks
        const int num_regions = take_chunks ? num_scaleup_ranks : 1;
        const int chunks_per_region = math::ceil_div(num_chunks, num_regions);
        const int num_sequence = num_regions * chunks_per_region;
        const auto next_chunk = [&]() {
            if (not take_chunks)
                return num_sequence;   // the static form: one chunk per warp
            int next = 0;
            if (lane_idx == 0)
                next = num_total_warps + static_cast<int>(atomicAdd(workspace_layout.get_take_counter_ptr(1), 1u));
            return __shfl_sync(0xffffffff, next, 0);
        };
        // The positions whose READY this warp has acquired (every lane acquires: the lanes that store weights and the elected
        // lane that stores the row are all behind their own acquire)
        comm::AcquiredReadyMask<kNumMaxScaleupRanks> acquired_ready;
        const auto acquire_ready = [&](const int& pos) {
            comm::peer_ready_acquire_rt<kNumTimeoutCycles, comm::kDirectCombineTag, kNumMaxScaleupRanks>(acquired_ready, workspace_layout, topo, pos);
        };
        for (int chunk = global_warp_idx; chunk < num_sequence; chunk = next_chunk()) {
            const int mapped_chunk = take_chunks ?
                ((chunk % num_regions + scaleup_rank_idx) % num_regions) * chunks_per_region + chunk / num_regions : chunk;
            if (mapped_chunk >= num_chunks)
                continue;
            const int token_start_idx = mapped_chunk * chunk_rows;
            const int token_end_idx = min(token_start_idx + chunk_rows, num_reduced_tokens);
            for (int i = token_start_idx; i < token_end_idx; ++ i) {
                // Source token: rank from the global index, master top-k lane from the slot-relative field
                constexpr int kMetadataStride = 2 + kNumTopk;
                const int src_global_token_idx = __ldg(src_metadata + i * kMetadataStride);
                const int src_token_idx = src_global_token_idx % num_max_tokens_per_rank;
                const int src_rank_idx = src_global_token_idx / num_max_tokens_per_rank;
                const int src_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1) % kNumTopk;

                // The destination's NVLink position: every remote store of this row sits behind `acquire_ready(dst_pos)` (all lanes of
                // the warp, uniform); the no-reduce and reduce arms issue it after the row's local work so the wait overlaps it
                const int dst_pos = src_rank_idx % num_scaleup_ranks;

                // Destination row in the source rank's receive rows, addressed over NVLink
                auto master_token_buffer = scaleup_buffer
                    .get_rank_buffer(use_scaleup_rank_layout ? scaleup_rank_idx : src_topk_idx)
                    .get_token_buffer(src_token_idx);
                master_token_buffer.set_base_ptr(peer_ptr(master_token_buffer.get_base_ptr(), src_rank_idx));

                // Hidden requirements
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                int stored_topk_slot_idx = -1;
                if constexpr (kUseExpandedLayout) {
                    if (lane_idx < kNumTopk)
                        stored_topk_slot_idx = __ldg(src_metadata + i * kMetadataStride + (2 + lane_idx));
                    __syncwarp();
                }

                // 3 cases:
                //  - no expand + no reduce, or expand + no reduce
                //  - expand + reduce
                //  - expand + send all
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    int token_idx_in_tensor = i;
                    if constexpr (kUseExpandedLayout)
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // No reduce: the local payload load is issued first, the destination's ready is acquired while it lands,
                    // and only then does the elected lane (behind its own acquire) store the row
                    const bool is_elected = ptx::elect_one_sync();
                    if (is_elected) {
                        const auto load_ptr =
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                    }
                    acquire_ready(dst_pos);
                    if (is_elected) {
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                        ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                        ptx::tma_store_commit();
                    }
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {
                    // Do local reduction
                    // Sort valid top-k indices to front
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
                    combine_reduce<kHiddenVec, kUnrollFactor, 1>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait();
                            __syncwarp();
                        }
                    );
                    ptx::tma_store_fence();
                    __syncwarp();

                    // Issue TMA stores
                    acquire_ready(dst_pos);
                    if (ptx::elect_one_sync()) {
                        ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                        ptx::tma_store_commit();
                    }
                    __syncwarp();
                } else {
                    // No local reduction, send all data (expanded send)
                    acquire_ready(dst_pos);
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        const auto slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (slot_idx >= 0) {
                            const auto src_token_ptr = math::advance_ptr<int4>(x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                            const auto token_buffer = scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx);
                            if (ptx::elect_one_sync()) {
                                ptx::tma_store_wait();
                                ptx::tma_load_1d(tma_buffer.get_base_ptr(), src_token_ptr, mbarrier_ptr, kNumHiddenBytes);
                                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                                ptx::tma_store_1d(peer_ptr(token_buffer.get_base_ptr(), src_rank_idx),
                                                  tma_buffer.get_base_ptr(), kNumHiddenBytes);
                                ptx::tma_store_commit();
                            }
                            __syncwarp();
                        }
                    }
                }

                // Write topk weights
                if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
                    float value = 0;
                    if constexpr (kUseExpandedLayout) {
                        if (stored_topk_slot_idx >= 0)
                            value = __ldg(topk_weights + stored_topk_slot_idx);
                    } else {
                        value = __ldg(topk_weights + (i * kNumTopk + lane_idx));
                    }
                    master_token_buffer.get_topk_weights_ptr()[lane_idx] = value;
                }
                __syncwarp();
            }
        }

        // End sequence: the block's flush and grid arrival; the last block's first warp zeroes the take counter, exchanges the
        // ARRIVED words and triggers the reduce epilogue
        const bool is_last = comm::active_grid_arrive<kNumThreads, kEndBarrierIndex>(workspace_layout.get_active_grid_barrier_ptr(), topo.num_active_sms);
        if (is_last and warp_idx == 0) {
            // Every block has arrived (every take is behind it): the row-chunk take counter back to zero for the next launch
            if (lane_idx == 0)
                *workspace_layout.get_take_counter_ptr(1) = 0u;
            comm::peer_arrived_exchange_rt<kNumTimeoutCycles, comm::kDirectCombineTag, kNumMaxScaleupRanks>(transport, workspace_layout, topo, lane_idx);
            // Trigger the reduce epilogue only now: every peer's row has arrived, so no epilogue block parks on an SM while the
            // peers are still writing (the hybrid combine keeps its implicit trigger at kernel exit)
            cudaTriggerProgrammaticLaunchCompletion();
        }
        EP_PROFILE_STAMP_LAST(workspace_layout, 15);  // combine main: last block done
        return;
    }

    // Different warp roles (hybrid mode)
    if (warp_idx < kNumScaleupWarps) {
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

        // Shift into the right buffer if using rank layout
        if (use_scaleup_rank_layout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Expanding send mode must not be backward
        if constexpr (kDoExpandedSend)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // The scale-up peer's address of one of our symmetric pointers (the scale-up peers are reachable with stores).
        // Per lane (any `k`, divergent callers): the transport's table lookup
        const auto scaleup_peer_ptr_lane = [&](auto* ptr, const int& k) {
            const auto dst_ptr = transport.get_p2p_ptr(ptr, scaleup_peer_id(k));
            EP_DEVICE_ASSERT(dst_ptr != nullptr and "A scale-up peer is not reachable with stores");
            return dst_ptr;
        };
        // The live peers' region bases, loaded once per warp into lane registers (lane `l` of word `j` holds position
        // `j * 32 + l`) and fetched with a shuffle under a warp-uniform `k`: the table load behind `get_p2p_ptr` leaves the per-token path
        unsigned long long stored_scaleup_p2p_base[kNumScaleupRanksPerLane];
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const int k = j * 32 + lane_idx;
            stored_scaleup_p2p_base[j] = (k < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, k)) ?
                reinterpret_cast<unsigned long long>(transport.get_p2p_ptr(static_cast<uint8_t*>(transport.ctx.local_base), scaleup_peer_id(k))) : 0ull;
        }
        const auto scaleup_peer_ptr = [&](auto* ptr, const int& k) {
            unsigned long long base = 0ull;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                const auto word = __shfl_sync(0xffffffff, stored_scaleup_p2p_base[j], k & 31);
                base = (k >> 5) == j ? word : base;
            }
            EP_DEVICE_ASSERT(base != 0ull and "A scale-up peer is not reachable with stores");
            return transport.get_p2p_ptr_from_base(reinterpret_cast<uint8_t*>(base), ptr);
        };

        // Tail issuer
        // `st.release.sys` is pretty slow, so do it by an interval
        int update_counter = 0;
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        const auto update_tails = [&](const bool& finish = false) {
            ++ update_counter;
            if (finish or update_counter == kNumScaleupUpdateInterval) {
                // Wait all TMA stores to finish
                ptx::tma_store_wait();
                __syncwarp();

                // Issue
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    if (const auto j = i * 32 + lane_idx; j < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, j)) {
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                            ptx::st_release_sys(scaleup_peer_ptr_lane(tail_ptr, j), stored_num_tokens_sent[i]);
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                update_counter = 0;
            }
            __syncwarp();
        };

        // Shape of `channel_linked_list`: `[num_channels, num_scaleout_ranks * num_max_tokens_per_channel + 1, num_scaleup_ranks]`
        // Iterate until all scale-up peers finish
        const int num_tokens_in_linked_list = num_scaleout_ranks * num_max_tokens_per_channel + 1;
        // Send in reverse dispatch order; tails publish capacity minus the completed list position.
        int dst_scaleup_rank_idx = channel_idx;
        int stored_ll_idx[kNumScaleupRanksPerLane], stored_token_idx[kNumScaleupRanksPerLane] = {};
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
            const auto j = i * 32 + lane_idx;
            stored_ll_idx[i] = j < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, j) ?
                -__ldg(channel_linked_list +
                       channel_idx * num_tokens_in_linked_list * num_scaleup_ranks +
                       (num_tokens_in_linked_list - 1) * num_scaleup_ranks + j) - 2 : -1;
            stored_token_idx[i] = -1;
        }
        while (true) {
            // Load token indices in the list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                const auto j = i * 32 + lane_idx;
                stored_token_idx[i] = stored_ll_idx[i] >= 0 ?
                    __ldg(channel_linked_list +
                          channel_idx * num_tokens_in_linked_list * num_scaleup_ranks +
                          stored_ll_idx[i] * num_scaleup_ranks + j) : -1;
            }
            __syncwarp();

            // Check whether all ranks are finished
            bool exited = true;
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                exited &= ptx::all(stored_token_idx[i] < 0);
            if (exited)
                break;

            // Process tokens for all ranks together using bitmask to skip inactive ranks
            unsigned wip_mask[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                wip_mask[j] = ptx::gather(stored_token_idx[j] >= 0);
            const auto any_wip = [&]() {
                bool any = false;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    any |= wip_mask[j] != 0;
                return any;
            };
            while (any_wip()) {
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)
                const auto start = (dst_scaleup_rank_idx + 1) % num_scaleup_ranks;
                int next_idx = -1, lowest_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto shift = start - j * 32;  // bits of this word below `start`
                    const auto hi_mask = shift <= 0 ? wip_mask[j] : (shift >= 32 ? 0u : ((wip_mask[j] >> shift) << shift));
                    if (next_idx < 0 and hi_mask != 0)
                        next_idx = j * 32 + ptx::ffs(hi_mask);
                    if (lowest_idx < 0 and wip_mask[j] != 0)
                        lowest_idx = j * 32 + ptx::ffs(wip_mask[j]);
                }
                dst_scaleup_rank_idx = next_idx >= 0 ? next_idx : lowest_idx;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    wip_mask[j] ^= (dst_scaleup_rank_idx >> 5) == j ? (1u << (dst_scaleup_rank_idx & 31)) : 0u;

                // Exchange token index from the owning lane using static partition iteration
                int token_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;
                }
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);

                // Get source metadata and decide the destination buffer
                constexpr int kMetadataStride = 2 + kNumTopk;
                const auto src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride + 0);
                const auto src_token_idx = src_global_token_idx % num_max_tokens_per_rank;
                const auto src_scaleout_rank_idx = src_global_token_idx / (num_max_tokens_per_rank * num_scaleup_ranks);
                auto token_buffer = [&]() {
                    if (use_scaleup_rank_layout) {
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * num_max_tokens_per_rank + src_token_idx);
                    }
                }();
                token_buffer.set_base_ptr(scaleup_peer_ptr(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));

                // Some checks
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                int stored_topk_slot_idx = -1;
                if constexpr (kUseExpandedLayout) {
                    if (lane_idx < kNumTopk)
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    __syncwarp();
                }

                // 3 cases:
                //  - no-expand, expand + no-reduce
                //  - expand + reduce
                //  - expand + send all
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    int token_idx_in_tensor = token_idx;
                    if constexpr (kUseExpandedLayout)
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // Directly load
                    if (ptx::elect_one_sync()) {
                        const auto load_ptr =
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {
                    // Do local reduction
                    // Sort valid top-k indices to front
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
                    combine_reduce<kHiddenVec, kUnrollFactor, 1>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait();
                            __syncwarp();
                        }
                    );
                    ptx::tma_store_fence();
                    __syncwarp();
                } else {
                    // No local reduction, send all data (expanded send)
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (topk_slot_idx < 0)
                            continue;

                        // Resolved by the whole warp (warp-uniform inputs): `scaleup_peer_ptr` shuffles the peer's base, every lane takes part
                        const auto dst_token_buffer = scaleup_buffer
                            .get_rank_buffer(k)
                            .get_token_buffer(src_scaleout_rank_idx * num_max_tokens_per_rank + src_token_idx);
                        const auto dst_ptr = scaleup_peer_ptr(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx);
                        if (ptx::elect_one_sync()) {
                            // Load
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);
                            ptx::tma_store_wait();
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode

                            // Store
                            ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            ptx::tma_store_commit();
                        }
                        __syncwarp();
                    }
                }

                // Write top-k weights (expanded send handled inside the loop above)
                if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
                    float value = 0;
                    if constexpr (kUseExpandedLayout) {
                        if (stored_topk_slot_idx >= 0)
                            value = __ldg(topk_weights + stored_topk_slot_idx);
                    } else {
                        value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));
                    }
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Issue TMA stores into remote scale-up buffer
                // NOTES: `kDoExpandedSend` mode has already issued
                if (not kDoExpandedSend and ptx::elect_one_sync()) {
                    // Wait TMA arrival (only for non-reduced cases)
                    if (no_local_reduce) {
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    if ((j * 32 + lane_idx) == dst_scaleup_rank_idx)
                        stored_num_tokens_sent[j] = num_tokens_in_linked_list - 1 - stored_ll_idx[j];
                __syncwarp();
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            update_tails();

            // Move linked list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                stored_ll_idx[i] -= (stored_token_idx[i] >= 0);
        }

        // Update for the unissued ones
        update_tails(true);
        EP_PROFILE_STAMP_LAST(workspace_layout, 27);   // hybrid combine: the last scale-up warp is done
    } else {
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();

        // Shift into the right buffer
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer(channel_idx, num_scaleout_ranks * num_max_tokens_per_channel);

        // This channel's forwarding metadata (`layout::ForwardMetadataLayout`: one row per forwarded token, then the footer)
        const auto forward_metadata = layout::ForwardMetadataLayout<kNumTopk>{num_scaleout_ranks, num_max_tokens_per_channel};
        token_metadata_at_forward += channel_idx * forward_metadata.num_words_per_channel();

        // Each remote receiver has one progress word per channel and producing domain.
        auto* channel_ready_atomic_buff = ready_atomic_buff + channel_idx * num_scaleout_ranks + scaleout_rank_idx;
        // Lane d keeps destination domain d: the lowest token index published to it, the lowest completed token index (sent,
        // with every higher index of d in this channel sent), the completed tokens not yet published, the sends without
        // progress since the last ring, and the rows of d still to forward (the footer count: it decides d's final publication)
        int stored_last_published_token_idx = num_max_tokens_per_rank, stored_completed_token_idx = num_max_tokens_per_rank;
        int stored_num_unpublished_tokens = 0, stored_num_unrung_sends = 0;
        int stored_num_remaining_tokens = lane_idx < num_scaleout_ranks ?
            __ldg(token_metadata_at_forward + forward_metadata.domain_count_word(lane_idx)) : 0;
        // Payload flags: the rows sent to d so far (the doorbell rings every `kNumScaleoutUpdateInterval`-th and the last one)
        int stored_rows_sent_to_domain = 0;

        // A reduced token goes back to its source domain: our own domain straight into our receive buffer, another domain
        // through the send buffer with an RDMA put. Progress to a domain is its lowest completed token index (descending): the
        // receiver's reduce epilogue reads a token once the word is at or below the token's index, so a token is covered only
        // when every higher index of that domain in this channel was put before it on the same queue
        struct ScaleoutSend { int signal_delta; bool ring_doorbell; };
        // Called once per token to another domain, when the token is selected: publish (a signalled put) once
        // `kNumScaleoutUpdateInterval` tokens completed or on the domain's final row, else ring the doorbell on every
        // `kNumScaleoutUpdateInterval`-th send without progress, so that queued puts leave between publications.
        // Payload flags: no publication; the doorbell rings every `kNumScaleoutUpdateInterval` rows to a domain and on the channel's
        // LAST row to it: every peer's reduce epilogue waits for its last row, so what stays queued behind the last ring is a plain tail
        const auto account_scaleout_send = [&](const int& dst_scaleout_rank_idx, const int& completed_token_idx, const int& num_new_completed_tokens) {
            int signal_delta = 0, ring_doorbell = 0;
            if (lane_idx == dst_scaleout_rank_idx) {
                if constexpr (kPayloadFlags) {
                    const int row_ordinal = ++ stored_rows_sent_to_domain;
                    ring_doorbell = row_ordinal % kNumScaleoutUpdateInterval == 0 or stored_num_remaining_tokens == 1;
                } else {
                    stored_num_unpublished_tokens += num_new_completed_tokens;
                    if (completed_token_idx >= 0)
                        stored_completed_token_idx = min(stored_completed_token_idx, completed_token_idx);
                    if (stored_num_unpublished_tokens >= kNumScaleoutUpdateInterval or stored_num_remaining_tokens == 1) {
                        signal_delta = stored_completed_token_idx - stored_last_published_token_idx;
                        stored_last_published_token_idx = stored_completed_token_idx;
                        ring_doorbell = signal_delta == 0 and num_new_completed_tokens == 0;
                        stored_num_unpublished_tokens = stored_num_unrung_sends = 0;
                    } else if (num_new_completed_tokens == 0 and ++ stored_num_unrung_sends == kNumScaleoutUpdateInterval) {
                        ring_doorbell = 1;
                        stored_num_unrung_sends = 0;
                    }
                }
                -- stored_num_remaining_tokens;
            }
            signal_delta = ptx::exchange(signal_delta, dst_scaleout_rank_idx);
            ring_doorbell = ptx::exchange(ring_doorbell, dst_scaleout_rank_idx);
            return ScaleoutSend{signal_delta, signal_delta != 0 or ring_doorbell != 0};
        };

        // Overlap TMA stores and reduction: the staged token's put goes out once its store completed
        int last_src_scaleout_rank_idx = -1;
        ScaleoutSend last_send = {0, false};
        void* last_recv_token_buffer_ptr = nullptr;
        void* last_send_token_buffer_ptr = nullptr;
        // Profile accumulators of SM 0's first forward warp, slots 40 to 50 (the slot map is in `layout.cuh`)
        [[maybe_unused]] const bool profile_warp = sm_idx == 0 and warp_idx == kNumScaleupWarps;
        const auto flush_last_tma_and_issue_rdma = [&]() {
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {
                EP_PROFILE_ACCUM_BEGIN(profile_t_wait);
                ptx::tma_store_wait();
                EP_PROFILE_ACCUM_END(workspace_layout, 43, profile_t_wait, profile_warp);

                // Issue only if not local rank
                if (last_src_scaleout_rank_idx != scaleout_rank_idx) {
                    EP_PROFILE_ACCUM_BEGIN(profile_t_put);
                    if constexpr (kPayloadFlags) {
                        // Payload flags: a plain put, the doorbell per the ring rule (no signal, no counter)
                        transport.rdma_put(
                            last_recv_token_buffer_ptr,
                            last_send_token_buffer_ptr,
                            token_layout.get_num_bytes<false>(),
                            scaleout_peer_id(last_src_scaleout_rank_idx), last_send.ring_doorbell);
                    } else if (last_send.signal_delta != 0) {
                        transport.rdma_put_and_signal(
                            last_recv_token_buffer_ptr,
                            last_send_token_buffer_ptr,
                            token_layout.get_num_bytes<false>(),
                            channel_ready_atomic_buff,
                            scaleout_peer_id(last_src_scaleout_rank_idx), true, static_cast<uint64_t>(last_send.signal_delta));
                    } else {
                        transport.rdma_put(
                            last_recv_token_buffer_ptr,
                            last_send_token_buffer_ptr,
                            token_layout.get_num_bytes<false>(),
                            scaleout_peer_id(last_src_scaleout_rank_idx), last_send.ring_doorbell);
                    }
                    EP_PROFILE_ACCUM_END(workspace_layout, 40, profile_t_put, profile_warp);
                    // The batches that ring the doorbell, separately (slot 41 time, 42 count)
                    EP_PROFILE_ACCUM_END(workspace_layout, 41, profile_t_put, profile_warp and last_send.ring_doorbell);
                    EP_PROFILE_COUNT(workspace_layout, 42, profile_warp and last_send.ring_doorbell);
                }
            }
            __syncwarp();
            last_src_scaleout_rank_idx = -1;
        };

        // Out-of-order forwarding over a window of `kNumWindowTokens` list positions below `first_pending_token_idx`, the
        // highest position not yet forwarded: a token whose scale-up data arrived is reduced and sent while earlier positions
        // still wait. Bit (position mod kNumWindowTokens) of `forwarded_mask` marks a forwarded position inside the window; the
        // forwarded prefix retires and admits new positions. Rows are visited from the last one down, the scale-up senders' order
        constexpr int kNumWindowTokens = 32;
        uint32_t forwarded_mask = 0;
        EP_STATIC_ASSERT(kNumWindowTokens == 8 * sizeof(forwarded_mask), "The window is one mask word");
        const auto window_bit = [](const int& idx) { return 1u << (idx % kNumWindowTokens); };
        const int last_token_idx = -__ldg(token_metadata_at_forward + forward_metadata.length_word()) - 2;
        int first_pending_token_idx = last_token_idx;

        // The scale-up tails, cached: lane `l` of group `j` holds peer `j * 32 + l`'s (zero at rest, below every threshold).
        // Reloaded with system-scope acquire loads, once per poll: the data of a token whose thresholds the cached tails pass
        // is visible to the warp after the vote, as with a load per token
        int stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        const auto reload_scaleup_tails = [&]() {   // true when any tail moved
            bool changed = false;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                const auto k = j * 32 + lane_idx;
                const int tail = k < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, k) ?
                    ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : 0;
                changed |= tail != stored_cached_scaleup_tail[j];
                stored_cached_scaleup_tail[j] = tail;
            }
            return ptx::any(changed);
        };
        // Lane `k < kNumTopk` checks row `row`'s selection k against the cached tails. Dispatch gives a slot to one selection
        // per peer (experts of one GPU share it); a selection without a slot has no threshold. The shuffles run in every lane
        const auto arrived = [&](const int& row) {
            const bool has_slot = lane_idx < kNumTopk and __ldg(token_metadata_at_forward + forward_metadata.slot_word(row, lane_idx)) >= 0;
            const int peer = has_slot ? __ldg(token_metadata_at_forward + forward_metadata.peer_word(row, lane_idx)) : 0;
            const int required_tail = has_slot ? __ldg(token_metadata_at_forward + forward_metadata.required_tail_word(row, lane_idx)) : 0;
            int tail = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                const int group_tail = ptx::exchange(stored_cached_scaleup_tail[j], peer % 32);
                tail = peer / 32 == j ? group_tail : tail;
            }
            return ptx::all(not has_slot or tail >= required_tail);
        };

        while (first_pending_token_idx >= 0) {
            const int window_end_idx = max(0, first_pending_token_idx - (kNumWindowTokens - 1));
            int i = -1;
            EP_PROFILE_ACCUM_BEGIN(profile_t_row);   // per-row time, split by destination path (slots 47 / 49)
            EP_PROFILE_ACCUM_BEGIN(profile_t_tail);
            // The first poll scans with the cached tails (a position that entered the window may already be covered); after
            // that only a tail that moved can make a token ready, so a poll that saw none scans nothing
            bool scan = true;
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                if (scan) {
                    for (int candidate_idx = first_pending_token_idx; candidate_idx >= window_end_idx; -- candidate_idx) {
                        if ((forwarded_mask & window_bit(candidate_idx)) != 0)
                            continue;
                        if (arrived(candidate_idx)) {
                            i = candidate_idx;
                            return true;
                        }
                    }
                }

                // Nothing is ready: let the staged put leave, then poll the tails
                if constexpr (kAllowMultipleReduction)
                    flush_last_tma_and_issue_rdma();
                scan = reload_scaleup_tails();
                if (is_last_check) {
                    if (lane_idx == 0)
                        printf("NIXL EP V2 combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, pending token: %d, forwarded mask: %08x\n",
                               scaleout_rank_idx, num_scaleout_ranks, scaleup_rank_idx, num_scaleup_ranks,
                               channel_idx, first_pending_token_idx, forwarded_mask);
                    // The pending token's selections: each lane its peer, the tail it needs and the tail it sees
                    if (lane_idx < kNumTopk and __ldg(token_metadata_at_forward + forward_metadata.slot_word(first_pending_token_idx, lane_idx)) >= 0) {
                        const int peer = __ldg(token_metadata_at_forward + forward_metadata.peer_word(first_pending_token_idx, lane_idx));
                        printf("NIXL EP V2 combine (scale-up wait) timeout, channel: %d, pending token: %d, selection %d: peer %d, needs %d, tail %d\n",
                               channel_idx, first_pending_token_idx, lane_idx, peer,
                               __ldg(token_metadata_at_forward + forward_metadata.required_tail_word(first_pending_token_idx, lane_idx)),
                               ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, peer)));
                    }
                }
                return false;
            });
            EP_PROFILE_ACCUM_END(workspace_layout, 44, profile_t_tail, profile_warp and lane_idx == 0);

            const auto src_token_global_idx = __ldg(token_metadata_at_forward + forward_metadata.src_token_word(i));
            const auto src_rank_idx = src_token_global_idx / num_max_tokens_per_rank;
            const auto src_scaleout_rank_idx = src_rank_idx / num_scaleup_ranks;
            const auto src_token_idx = src_token_global_idx % num_max_tokens_per_rank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + forward_metadata.peer_word(i, lane_idx)) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + forward_metadata.slot_word(i, lane_idx)) : -1;

            // The progress this token completes for its domain. The rows of one source domain stand in ascending token order
            // (the dispatch's scale-out senders walk their channel's tokens upward and the forwarder consumes a domain's
            // records in order), and every position above `first_pending_token_idx` is forwarded: so the lowest index of the
            // run of forwarded rows of this domain, from the top of the window down to its first unforwarded row, is the
            // lowest index with every higher one sent. The tokens of the run below the domain's previous completed index are new.
            // Payload flags: no progress to compute, every record validates itself
            const uint32_t next_forwarded_mask = forwarded_mask | window_bit(i);
            int completed_token_idx = -1, num_new_completed_tokens = 0;
            if (not kPayloadFlags and src_scaleout_rank_idx != scaleout_rank_idx) {
                const int previous_completed_token_idx = ptx::exchange(stored_completed_token_idx, src_scaleout_rank_idx);
                for (int candidate_idx = first_pending_token_idx; candidate_idx >= window_end_idx; -- candidate_idx) {
                    const auto candidate_global_idx = __ldg(token_metadata_at_forward + forward_metadata.src_token_word(candidate_idx));
                    if (candidate_global_idx / (num_max_tokens_per_rank * num_scaleup_ranks) != src_scaleout_rank_idx)
                        continue;
                    if ((next_forwarded_mask & window_bit(candidate_idx)) == 0)
                        break;
                    completed_token_idx = candidate_global_idx % num_max_tokens_per_rank;
                    num_new_completed_tokens += completed_token_idx < previous_completed_token_idx;
                }
            }
            const auto send = src_scaleout_rank_idx != scaleout_rank_idx ?
                account_scaleout_send(src_scaleout_rank_idx, completed_token_idx, num_new_completed_tokens) : ScaleoutSend{0, false};

            if constexpr (not kAllowMultipleReduction) {
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                const auto src_slot_idx = src_scaleout_rank_idx * num_max_tokens_per_rank + src_token_idx;
                auto topk_valid_mask = kUseExpandedLayout ?
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                if (ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        if ((topk_valid_mask & (1u << k)) == 0u)
                            continue;

                        // Issue TMA load, and wait
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        // Stage the payload in the send buffer for RDMA, else directly in our receive buffer (payload flags: the flags
                        // go into the staged record first, see `TokenLayout`)
                        const auto recv_buffer_ptr = scaleout_recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx).get_base_ptr();
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr : scaleout_send_buffer.get_rank_buffer(k).get_token_buffer(i).get_base_ptr();
                        if constexpr (kPayloadFlags)
                            tma_buffer.spill_and_set_flags();
                        ptx::tma_store_fence();
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        ptx::tma_store_wait();

                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx) {
                            EP_PROFILE_ACCUM_BEGIN(profile_t_put);
                            if constexpr (kPayloadFlags) {
                                // Payload flags: a plain put, the doorbell per the ring rule on the token's last put
                                transport.rdma_put(
                                    recv_buffer_ptr,
                                    send_buffer_ptr,
                                    token_layout.get_num_bytes<false>(),
                                    scaleout_peer_id(src_scaleout_rank_idx), topk_valid_mask == 0 and send.ring_doorbell);
                            } else if (topk_valid_mask == 0 and send.signal_delta != 0) {
                                transport.rdma_put_and_signal(
                                    recv_buffer_ptr,
                                    send_buffer_ptr,
                                    token_layout.get_num_bytes<false>(),
                                    channel_ready_atomic_buff,
                                    scaleout_peer_id(src_scaleout_rank_idx), true, static_cast<uint64_t>(send.signal_delta));
                            } else {
                                transport.rdma_put(
                                    recv_buffer_ptr,
                                    send_buffer_ptr,
                                    token_layout.get_num_bytes<false>(),
                                    scaleout_peer_id(src_scaleout_rank_idx), topk_valid_mask == 0 and send.ring_doorbell);
                            }
                            EP_PROFILE_ACCUM_END(workspace_layout, 40, profile_t_put, profile_warp);
                            EP_PROFILE_ACCUM_END(workspace_layout, 41, profile_t_put, profile_warp and topk_valid_mask == 0 and send.ring_doorbell);
                            EP_PROFILE_COUNT(workspace_layout, 42, profile_warp and topk_valid_mask == 0 and send.ring_doorbell);
                        }
                    }
                }
                __syncwarp();
            } else {
                // NOTES: we must do deduplicate and only add once from one rank
                auto reduce_valid_mask = ptx::gather(
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // Calculate the source buffer index
                int stored_src_buffer_idx = 0;
                if (use_scaleup_rank_layout) {
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    const auto src_slot_idx = src_scaleout_rank_idx * num_max_tokens_per_rank + src_token_idx;
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }

                // Preprocess top-k indices
                int topk_slot_idx[kNumTopk];
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,
                    [=](const int& idx) {
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                // Do reduce
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>();
                EP_PROFILE_ACCUM_BEGIN(profile_t_reduce);   // slot 45: the reduce, the previous row's flush included
                combine_reduce<kHiddenVec, kUnrollFactor, 1>(
                    lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    /* Get source base */ [=](const int& slot_idx) {
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ [=]() {
                        flush_last_tma_and_issue_rdma();
                    }
                );
                EP_PROFILE_ACCUM_END(workspace_layout, 45, profile_t_reduce, profile_warp and lane_idx == 0);

                // Merge topk weights
                // NOTES: the slot indices must follow the master lane
                stored_src_buffer_idx = ptx::exchange(
                    stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                if (stored_src_scaleup_rank_idx >= 0) {
                    tma_buffer.get_topk_weights_ptr()[lane_idx] =
                        scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                    .get_topk_weights_ptr()[lane_idx];
                }
                ptx::tma_store_fence();
                __syncwarp(); // Necessary to let the leader lane see the writes

                // Assign send and receive buffers
                // NOTES: as we only have 1 destination, we will use "send" as "recv" for local transfer
                int scaleout_recv_buffer_rank_idx;
                if (use_scaleout_rank_layout) {
                    scaleout_recv_buffer_rank_idx = scaleout_rank_idx;
                } else {
                    const int src_topk_idx = ptx::get_master_lane_idx(ptx::gather(stored_src_scaleup_rank_idx >= 0));
                    scaleout_recv_buffer_rank_idx = src_topk_idx;
                }
                const auto recv_token_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_recv_buffer_rank_idx).get_token_buffer(src_token_idx);
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                    recv_token_buffer :
                    scaleout_send_buffer.get_token_buffer(i);

                // Stage the payload in the send buffer for RDMA, else directly in our receive buffer (payload flags: the flags go
                // into the staged record first, see `TokenLayout`)
                if (ptx::elect_one_sync()) {
                    EP_PROFILE_ACCUM_BEGIN(profile_t_stage);   // slot 46: (flag staging +) store issue
                    if constexpr (kPayloadFlags)
                        tma_buffer.spill_and_set_flags();
                    ptx::tma_store_fence();
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                    EP_PROFILE_ACCUM_END(workspace_layout, 46, profile_t_stage, profile_warp);
                }
                __syncwarp();

                // Defer the put until its TMA store completes; its signal may cover other completed rows too (payload flags: the
                // doorbell decision only)
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_send = send;
                last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();
                last_send_token_buffer_ptr = send_token_buffer.get_base_ptr();
                // Profile: the row's time into slot 47 (our own domain) or 49 (RDMA path), its count into slot 48 / 50
                EP_PROFILE_ACCUM_END(workspace_layout, src_scaleout_rank_idx != scaleout_rank_idx ? 49 : 47, profile_t_row, profile_warp and lane_idx == 0);
                EP_PROFILE_COUNT(workspace_layout, src_scaleout_rank_idx != scaleout_rank_idx ? 50 : 48, profile_warp and lane_idx == 0);
            }

            forwarded_mask = next_forwarded_mask;
            // Retire the forwarded prefix: each cleared bit admits the position one window below
            while (first_pending_token_idx >= 0 and (forwarded_mask & window_bit(first_pending_token_idx)) != 0) {
                forwarded_mask &= ~window_bit(first_pending_token_idx);
                -- first_pending_token_idx;
            }
        }

        // Finish the last TMA store and submit its RDMA write if the destination is remote.
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_issue_rdma();

        // Every other domain's rows went out, the last of them with the publication its footer count triggered
        EP_DEVICE_ASSERT((lane_idx >= num_scaleout_ranks or lane_idx == scaleout_rank_idx or stored_num_remaining_tokens == 0) and
                         "A source domain's forwarded-token count does not match its rows");

        // Clean scaleup tails
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const auto k = j * 32 + lane_idx;
            if (k < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, k))
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        __syncwarp();
        EP_PROFILE_STAMP_LAST(workspace_layout, 31);   // hybrid combine: the last forward warp is done
    }

    // No barrier at epilogue
    EP_PROFILE_STAMP_LAST(workspace_layout, 15);  // combine main: last block done
}

}  // namespace nixl_ep_v2::elastic::adaptive
