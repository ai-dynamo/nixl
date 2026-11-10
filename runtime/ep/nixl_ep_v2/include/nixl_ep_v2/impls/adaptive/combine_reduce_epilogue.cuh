// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <nixl_ep_v2/common/comm.cuh>
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/topology.cuh>
#include <nixl_ep_v2/impls/adaptive/combine_policy.cuh>

#include <nixl_ep_v2/impls/combine_utils.cuh>


namespace nixl_ep_v2::elastic::adaptive {

// Reduce epilogue of the adaptive combine: given the buffer base, it derives the rows to reduce from the topology descriptor (`EPTopology`)
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps, int kNumChannelsPerSM,
          int kHidden,
          int kNumExpertsPerRank, int kNumTopk,
          int64_t kNumTimeoutCycles,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             topk_idx_t* combined_topk_idx,
                             void* buffer,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const topology::EPTopology* topology_ptr,
                             const int num_max_tokens_per_rank) {
    // Topology
    const auto topo = topology::load(topology_ptr);
#if defined(EP_ADAPTIVE_PROFILE) && EP_ADAPTIVE_PROFILE
    const auto profile_workspace = nixl::load_context(topo.transport).local_base;
#endif
    EP_PROFILE_STAMP(profile_workspace, CombineReduceEntry);
    const int num_scaleout_ranks = topo.num_scaleout_ranks, num_scaleup_ranks = topo.num_scaleup_ranks;
    const int num_channels = kNumChannelsPerSM * topo.num_active_sms;
    const int num_experts_per_rank = kNumExpertsPerRank;
    const int num_tokens_in_scaleup_layout = get_num_tokens_in_layout(kAllowMultipleReduction, num_scaleup_ranks, kNumTopk);

    // Which rows hold the data to reduce:
    //   hybrid: the scale-out receive rows, indexed by the scale-out rank (domain) owning the expert
    //   direct: the scale-up rows (the direct receive buffer), indexed by the position of the rank owning the expert
    //           (see the direct-mode note in `dispatch.cuh`)
    const bool is_direct = topo.mode == topology::kModeDirect;
    const bool use_incremental_combine = combine_policy::use_incremental(topo, num_combined_tokens);
    const int num_reduce_ranks = is_direct ? num_scaleup_ranks : num_scaleout_ranks;
    const bool use_rank_layout = elastic::use_rank_layout(kAllowMultipleReduction, num_reduce_ranks, kNumTopk);
    const int num_tokens_in_layout = get_num_tokens_in_layout(kAllowMultipleReduction, num_reduce_ranks, kNumTopk);

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;   // NOTES: Here we prioritize distributing tasks to different SMs to ensure that the last wave is evenly concentrated on each SM.

    // Load buffers from scale-out or scale-up ranks
    // The scale-out receive buffer sits right after the scale-up buffer
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    const auto scaleup_buffer = layout::BufferLayout<false>(
        comm_token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank, buffer);
    // Direct: the rows are the scale-up buffer the combine kernel wrote (its layout, its row stride); hybrid: the
    // scale-out receive rows after it, one `num_max_tokens_per_rank` row per reduce rank
    const auto comm_buffer = is_direct ? scaleup_buffer : layout::BufferLayout<false>(
        comm_token_layout, num_tokens_in_layout, num_max_tokens_per_rank, scaleup_buffer.get_buffer_end_ptr());

    // Store buffers
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    // Bias layout
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);

    // Main combine waits for final completion at larger batches; small hybrid batches poll their rows below.
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();
    EP_PROFILE_STAMP(profile_workspace, CombineReduceReady);

    int64_t* ready_atomic_buff = nullptr;
    if (use_incremental_combine) {
        const auto ctx = nixl::load_context(topo.transport);
        ready_atomic_buff = layout::WorkspaceLayout(ctx.local_base).get_combine_ready_atomic_buff();
    }

    // Read from buffers and do reduction
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {
        // Preprocess all indices
        // NOTES: the "destination rank" here is the rank owning the expert: its domain (hybrid) or its position (direct)
        int stored_dst_rank_idx = -1, owner = -1;
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        if (lane_idx < kNumTopk) {
            const int dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            owner = dst_expert_idx >= 0 ? dst_expert_idx / num_experts_per_rank : -1;
            const int owner_domain = owner / num_scaleup_ranks, owner_position = owner % num_scaleup_ranks;
            // A selection of a masked rank's expert was dropped at dispatch: no row to reduce, weight zero
            if (owner >= 0 and not topo.is_active(owner_domain, owner_position))
                owner = -1;
            stored_dst_rank_idx = owner >= 0 ? (is_direct ? owner_position : owner_domain) : -1;
        }
        __syncwarp();

        // Sort valid top-k indices to front
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {
                // Activations are never reduced before
                return {false, 0};
            } else if constexpr (not kUseExpandedLayout and not kAllowMultipleReduction) {
                // Without expanded layout and multiple reduction. Should deduplicate on a per-rank basis
                return {true, owner};
            } else {
                // Should deduplicate on a per-rank (for non-hybrid mode) or a per-scale-rank (for hybrid mode) basis
                return {true, stored_dst_rank_idx};
            }
        }();
        auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);

        // One lane per remote producing domain waits for this token's cumulative channel watermark
        if (use_incremental_combine) {
            const bool lane_polls = ptx::deduplicate(stored_dst_rank_idx, lane_idx) and
                                    stored_dst_rank_idx >= 0 and stored_dst_rank_idx != topo.scaleout_rank_idx;
            if (lane_polls) {
                const int channel_idx = token_idx % num_channels;
                const auto completion_ptr = ready_atomic_buff +
                    channel_idx * topology::kNumMaxScaleoutRanksCap + stored_dst_rank_idx;
                comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                    const auto signal = ptx::ld_volatile<int64_t>(completion_ptr);
                    if (signal > token_idx) {
                        ptx::fence_acq_rel_sys();
                        return true;
                    }
                    if (is_last_check) {
                        printf("NIXL EP v2 adaptive combine (reduce epilogue) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, source domain: %d, token: %d, signal: %lld\n",
                               topo.scaleout_rank_idx, num_scaleout_ranks, topo.scaleup_rank_idx, num_scaleup_ranks,
                               channel_idx, stored_dst_rank_idx, token_idx, static_cast<long long>(signal));
                    }
                    return false;
                });
            }
            __syncwarp();
        }

        // Iterate over per-hidden-chunk stage
        // Specialize the gather slot count for the runtime layout.
        using combine_vec_t = int4;
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);
        constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
        const auto reduce_with = [&]<int kNumTokensInLayout>() {
            int topk_slot_idx[kNumTokensInLayout];
            compute_topk_slots(
                topk_slot_idx, reduce_valid_mask,
                [=](const int& idx) {
                    return use_rank_layout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
                }
            );
            combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(
                lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                /* Get source base */ [=](const int& slot_idx) {
                    return static_cast<combine_vec_t*>(
                        comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());
                },
                /* Wait buffer release */ [=]() {
                    ptx::tma_store_wait();
                    __syncwarp();
                },
                /* Bias 0 */ bias_0 == nullptr ?
                    nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()),
                /* Bias 1 */ bias_1 == nullptr ?
                    nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr())
            );
        };
        switch (num_tokens_in_layout) {
            case 1: reduce_with.template operator()<1>(); break;
            case 2: reduce_with.template operator()<2>(); break;
            case 4: reduce_with.template operator()<(kNumTopk >= 4 ? 4 : kNumTopk)>(); break;
            default: reduce_with.template operator()<kNumTopk>(); break;
        }
        ptx::tma_store_fence();
        __syncwarp();

        // Issue TMA copy
        if (ptx::elect_one_sync()) {
            ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),
                              tma_buffer.get_base_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();
    }
    EP_PROFILE_STAMP(profile_workspace, CombineReduceSubmitted);
}

}  // namespace nixl_ep_v2::elastic::adaptive
