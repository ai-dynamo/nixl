#pragma once

#include <nixl_ep_v2/common/comm.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/math.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>
#include <nixl_ep_v2/impls/combine_utils.cuh>

// Doorbell interval of the RDMA rows to a domain; overridable at JIT time (`NIXL_EP_JIT_EXTRA_FLAGS=-DEP_COMBINE_RING_INTERVAL=...`)
#ifndef EP_COMBINE_RING_INTERVAL
#define EP_COMBINE_RING_INTERVAL 6
#endif
#if EP_COMBINE_RING_INTERVAL < 1
#error "EP_COMBINE_RING_INTERVAL must be at least 1"
#endif

namespace nixl_ep_v2::elastic::adaptive {

// DeepSeek's hybrid combine on a fixed grid; rank decomposition and active block count come from the topology descriptor (`EPTopology`)
// Elastic on the NIXL transport: masked positions have no linked list and get no tails, masked domains are not sent to (see `topology.cuh`)
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumLaunchSMs,
          int kNumChannelsPerSM,
          int kNumMaxScaleoutRanks, int kNumMaxScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExpertsPerRank, int kNumTopk,
          int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumMaxScaleupRanks, 32),
          int kNumScaleupUpdateInterval = 3,
          int kDoorbellInterval = EP_COMBINE_RING_INTERVAL,
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
                      void* buffer, void* send_buffer, void* workspace,
                      const topology::EPTopology* topology_ptr,
                      int num_reduced_tokens) {
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
    const int num_max_tokens_per_channel = math::ceil_div(kNumMaxTokensPerRank, num_channels);
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
    EP_PROFILE_STAMP(workspace_layout, 14);  // combine main: entry

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens < 0)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumMaxScaleupRanks);

    // Token layouts
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

    // TMA buffers
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);

    // All the buffer layouts
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * kNumMaxTokensPerRank,
        buffer);
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, num_tokens_in_scaleout_layout, kNumMaxTokensPerRank,
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

    // Global parallel barriers for scale-out subteam and scale-up subteam
    // NOTES: the hybrid roles need a grid sync first, as there are channel scale-up tail cleaning before;
    // the direct path has no such cleaning and starts like DeepSeek's direct kernel (no start sync)
    const bool is_direct = topo.mode == topology::kModeDirect;
    if (is_direct) {
        comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kCombineTag0, kNumMaxScaleupRanks, false, false, true>(
            transport, workspace_layout, topo, sm_idx, thread_idx);
    } else {
        comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kHybridCombineTag0, kNumMaxScaleupRanks, false, true, true>(
            transport, workspace_layout, topo, sm_idx, thread_idx);
    }

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;
    constexpr int kNumRegistersForScaleupWarps = 40;
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;

    // Direct mode (MNNVL, one scale-out rank): every warp TMA-stores its share of the tokens straight into the
    // source rank's receive rows over NVLink, and the reduce epilogue reads those rows. The rows are the
    // scale-up buffer (`[num_tokens_in_scaleup_layout][num_max_tokens_per_rank]` with one scale-out rank),
    // which is exactly DeepSeek's direct-kernel receive buffer. No forwarding stage, no linked list.
    EP_DEVICE_ASSERT(not is_direct or num_scaleout_ranks == 1);
    if (is_direct) {
        // Rotate the warp index by the rank so that peers spread their writes (as DeepSeek's direct kernel does)
        const auto rotated_warp_idx = (warp_idx + scaleup_rank_idx) % kNumWarps;
        const auto global_warp_idx = rotated_warp_idx * topo.num_active_sms + sm_idx;

        // Expanding send mode must not be backward
        if constexpr (kDoExpandedSend)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // The source rank's address of one of our symmetric pointers (direct mode: every peer reachable with stores)
        const auto peer_ptr = [&](void* ptr, const int& src_rank_idx) {
            const auto dst_ptr = transport.get_p2p_ptr(ptr, scaleup_peer_id(src_rank_idx));
            EP_DEVICE_ASSERT(dst_ptr != nullptr and "Direct mode needs every live peer reachable with stores");
            return dst_ptr;
        };

        // Do TMA writes into the remote buffers: static partition over the active warps
        const int num_tokens_per_warp = math::ceil_div(num_reduced_tokens, topo.num_active_sms * kNumWarps);
        const int token_start_idx = num_tokens_per_warp * global_warp_idx;
        const int token_end_idx = min(token_start_idx + num_tokens_per_warp, num_reduced_tokens);
        for (int i = token_start_idx; i < token_end_idx; ++ i) {
            // Source token: rank from the global index, master top-k lane from the slot-relative field
            constexpr int kMetadataStride = 2 + kNumTopk;
            const int src_global_token_idx = __ldg(src_metadata + i * kMetadataStride);
            const int src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
            const int src_rank_idx = src_global_token_idx / kNumMaxTokensPerRank;
            const int src_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1) % kNumTopk;

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

                // No reduce
                if (ptx::elect_one_sync()) {
                    const auto load_ptr =
                        math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                    ptx::tma_store_wait();
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
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
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                    ptx::tma_store_commit();
                }
                __syncwarp();
            } else {
                // No local reduction, send all data (expanded send)
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

        // Final barrier to ensure data arrival (the hybrid roles synchronise through their tails instead)
        comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kCombineTag1, kNumMaxScaleupRanks, true, true, false>(
            transport, workspace_layout, topo, sm_idx, thread_idx);
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
        int dst_scaleup_rank_idx = channel_idx;
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_token_idx[i] = -1;
        while (true) {
            // Load token indices in the list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                const auto j = i * 32 + lane_idx;
                stored_token_idx[i] = j < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, j) ?
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
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * num_scaleup_ranks);
                auto token_buffer = [&]() {
                    if (use_scaleup_rank_layout) {
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
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
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
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
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            update_tails();

            // Move linked list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                stored_ll_idx[i] += (stored_token_idx[i] >= 0);
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

        // Shape of `token_metadata_at_forward`: `[num_channels, num_scaleout_ranks * num_max_tokens_per_channel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((num_scaleout_ranks * num_max_tokens_per_channel + 1) * kNumForwardMetadataDims);

        // How many rows this channel sends back to each source domain (lane `d`): the put of the LAST row to a domain
        // rings the doorbell of its deferred puts (one row per replayed token; the list ends at the first negative source index)
        int stored_num_rows_to_domain = 0, stored_rows_sent_to_domain = 0;
        if constexpr (kAllowMultipleReduction) {
            const int num_replay_entries = num_scaleout_ranks * num_max_tokens_per_channel + 1;
            for (int base = 0; base < num_replay_entries; base += 32) {
                const int i = base + lane_idx;
                const auto src_token_global_idx = i < num_replay_entries ?
                    __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims) : -1;
                const auto end_mask = ptx::gather(src_token_global_idx < 0);
                const int num_valid = end_mask != 0 ? ptx::ffs(end_mask) : 32;
                const int d = lane_idx < num_valid ? (src_token_global_idx / kNumMaxTokensPerRank) / num_scaleup_ranks : -1;
                for (int dd = 0; dd < num_scaleout_ranks; ++ dd) {
                    const auto rows_mask = ptx::gather(d == dd);
                    if (lane_idx == dd)
                        stored_num_rows_to_domain += __popc(rows_mask);
                }
                if (num_valid < 32)
                    break;
            }
        }

        // A reduced token goes back to its source domain: our own domain straight into our receive buffer (whole record, flags
        // inside, no release: the reduce epilogue reads it after this kernel), another domain through the send buffer and an RDMA put

        // Overlap TMA stores and reduction
        int last_src_scaleout_rank_idx = -1;
        int last_is_token_last_in_chunk = 0;
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
                    transport.rdma_put(
                        last_recv_token_buffer_ptr,
                        last_send_token_buffer_ptr,
                        token_layout.get_num_bytes<false>(),
                        scaleout_peer_id(last_src_scaleout_rank_idx),
                        last_is_token_last_in_chunk != 0);
                    EP_PROFILE_ACCUM_END(workspace_layout, 40, profile_t_put, profile_warp);
                    // The puts that ring the doorbell, separately (slot 41 time, 42 count)
                    EP_PROFILE_ACCUM_END(workspace_layout, 41, profile_t_put, profile_warp and last_is_token_last_in_chunk != 0);
                    EP_PROFILE_COUNT(workspace_layout, 42, profile_warp and last_is_token_last_in_chunk != 0);
                }
            }
            __syncwarp();
        };

        // Replay the dispatch
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        for (int i = 0; ; ++ i) {
            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / num_scaleup_ranks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                break;
            EP_PROFILE_ACCUM_BEGIN(profile_t_row);   // per-row time, split by destination path (slots 47 / 49)

            // Scaleup rank mask
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                const auto group_mask = ptx::reduce_or(
                    (stored_src_scaleup_rank_idx >= 0 and (stored_src_scaleup_rank_idx >> 5) == j) ?
                    (1u << (stored_src_scaleup_rank_idx & 31)) : 0u);
                stored_is_scaleup_rank_needed[j] = (group_mask >> lane_idx) & 1;
            }

            // Wait all tails to arrive
            EP_PROFILE_ACCUM_BEGIN(profile_t_tail);
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                bool arrived = true;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                if (ptx::all(arrived))
                    return true;

                // Reload cached
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto k = j * 32 + lane_idx;
                    stored_cached_scaleup_tail[j] = k < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, k) ?
                        ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        printf("NIXL EP V2 combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, num_scaleout_ranks, scaleup_rank_idx, num_scaleup_ranks,
                               channel_idx, j * 32 + lane_idx,
                               stored_num_tokens_recv[j],
                               stored_cached_scaleup_tail[j],
                               stored_is_scaleup_rank_needed[j]);
                    }
                }
                return false;
            });
            EP_PROFILE_ACCUM_END(workspace_layout, 44, profile_t_tail, profile_warp and lane_idx == 0);

            // Increase received count
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);

            if constexpr (not kAllowMultipleReduction) {
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
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

                        // Flags into the staged record (see `TokenLayout`), then one TMA store: the send buffer for RDMA, else our receive buffer
                        const auto recv_buffer_ptr = scaleout_recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx).get_base_ptr();
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr : scaleout_send_buffer.get_rank_buffer(k).get_token_buffer(i).get_base_ptr();
                        tma_buffer.spill_and_set_flags();
                        ptx::tma_store_fence();
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        ptx::tma_store_wait();

                        // Issue the RDMA put
                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx) {
                            transport.rdma_put(
                                recv_buffer_ptr,
                                send_buffer_ptr,
                                token_layout.get_num_bytes<false>(),
                                scaleout_peer_id(src_scaleout_rank_idx),
                                topk_valid_mask == 0 and is_token_last_in_chunk);
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
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
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

                // Flags into the staged record (see `TokenLayout`), then one TMA store: the send buffer for RDMA, else our receive buffer
                if (ptx::elect_one_sync()) {
                    EP_PROFILE_ACCUM_BEGIN(profile_t_stage);   // slot 46: flag staging + store issue
                    tma_buffer.spill_and_set_flags();
                    ptx::tma_store_fence();
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                    EP_PROFILE_ACCUM_END(workspace_layout, 46, profile_t_stage, profile_warp);
                }
                __syncwarp();

                // Record RDMA info to issue later. The doorbell rings every `kDoorbellInterval` rows to a domain and on the channel's
                // LAST row to it: every peer's reduce epilogue waits for its last row, so what stays queued behind the last ring is a plain tail
                stored_rows_sent_to_domain += lane_idx == src_scaleout_rank_idx;
                const int row_ordinal = ptx::exchange(stored_rows_sent_to_domain, src_scaleout_rank_idx);
                const bool ring_doorbell = row_ordinal % kDoorbellInterval == 0 or
                                           row_ordinal == ptx::exchange(stored_num_rows_to_domain, src_scaleout_rank_idx);
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_is_token_last_in_chunk = ring_doorbell;
                last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();
                last_send_token_buffer_ptr = send_token_buffer.get_base_ptr();
                // Profile: the row's time into slot 47 (our own domain) or 49 (RDMA path), its count into slot 48 / 50
                EP_PROFILE_ACCUM_END(workspace_layout, src_scaleout_rank_idx != scaleout_rank_idx ? 49 : 47, profile_t_row, profile_warp and lane_idx == 0);
                EP_PROFILE_COUNT(workspace_layout, src_scaleout_rank_idx != scaleout_rank_idx ? 50 : 48, profile_warp and lane_idx == 0);
            }
        }

        // Issue the last RDMA
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_issue_rdma();

        EP_PROFILE_STAMP_LAST(workspace_layout, 31);   // hybrid combine: the last forward warp is done

        // Clean scaleup tails
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const auto k = j * 32 + lane_idx;
            if (k < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, k))
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        __syncwarp();

        // No finish signal: the records carry their arrival flags and the reduce epilogue waits for the rows it reduces; the
        // send buffer is rewritten only past the next start barrier, which a peer reaches after its epilogue (see `comm::gpu_barrier_rt`)
        ptx::tma_store_wait();
        __syncwarp();
    }

    // No barrier at epilogue
    EP_PROFILE_STAMP_LAST(workspace_layout, 15);  // combine main: last block done
}

}  // namespace nixl_ep_v2::elastic::adaptive
