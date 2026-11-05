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
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/topology.cuh>

#include <nixl_ep_v2/impls/combine_utils.cuh>


namespace nixl_ep_v2::elastic::adaptive {

// Reduce epilogue of the adaptive combine: given the buffer base, it derives the rows to reduce from the topology descriptor (`EPTopology`)
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExpertsPerRank, int kNumTopk,
          int64_t kNumTimeoutCycles,
          bool kAcquireFlags = false,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* buffer, void* workspace,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const topology::EPTopology* topology_ptr) {
    // Topology
    const auto topo = topology::load(topology_ptr);
    const int num_scaleout_ranks = topo.num_scaleout_ranks, num_scaleup_ranks = topo.num_scaleup_ranks;
    const int num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    // Fixed expert slots per rank: the live expert count follows the live rank count
    const int num_experts_per_rank = kNumExpertsPerRank;
    const bool use_scaleup_rank_layout = kAllowMultipleReduction and num_scaleup_ranks <= kNumTopk;
    const int num_tokens_in_scaleup_layout = use_scaleup_rank_layout ? num_scaleup_ranks : kNumTopk;

    // Which rows hold the data to reduce:
    //   hybrid: the scale-out receive rows, indexed by the scale-out rank (domain) owning the expert
    //   direct: the scale-up rows (the direct receive buffer), indexed by the POSITION of the rank owning the expert
    //           (the one live domain need not be domain 0, so an owner's global rank is not its position)
    const bool is_direct = topo.mode == topology::kModeDirect;
    const int num_reduce_ranks = is_direct ? num_scaleup_ranks : num_scaleout_ranks;
    const bool use_rank_layout = kAllowMultipleReduction and num_reduce_ranks <= kNumTopk;
    const int num_tokens_in_layout = use_rank_layout ? num_reduce_ranks : kNumTopk;

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;   // NOTES: Here we prioritize distributing tasks to different SMs to ensure that the last wave is evenly concentrated on each SM.

    // Load buffers from scale-out or scale-up ranks
    // The scale-out receive buffer sits right after the scale-up buffer
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto scaleup_buffer = layout::BufferLayout<false>(
        comm_token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * kNumMaxTokensPerRank, buffer);
    // Direct: the rows are the scale-up buffer the combine kernel wrote (its layout, its row stride); hybrid: the
    // scale-out receive rows after it, one `kNumMaxTokensPerRank` row per reduce rank
    const auto comm_buffer = is_direct ? scaleup_buffer : layout::BufferLayout<false>(
        comm_token_layout, num_tokens_in_layout, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());

    // Store buffers
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false, nullptr, /* with_flags */ false);
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    // Bias layout
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);

    // Will block until the main combine kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();

    // Profile slots 16/17: epilogue work start (block 0) / last block done
#ifdef EP_ADAPTIVE_PROFILE
    const auto profile_layout = layout::WorkspaceLayout(workspace, num_scaleout_ranks, num_scaleup_ranks, num_ranks * kNumExpertsPerRank);
    EP_PROFILE_STAMP(profile_layout, 16);
#endif

    // Read from buffers and do reduction
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {
        // Preprocess all indices
        // NOTES: the "destination rank" here is the rank owning the expert: its domain (hybrid) or its position (direct)
        int stored_dst_rank_idx = -1, stored_dst_expert_idx = -1;
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        if (lane_idx < kNumTopk) {
            stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            // A selection of a masked rank's expert was dropped at dispatch: no row to reduce, weight zero
            const int owner = stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / num_experts_per_rank : -1;
            if (owner >= 0 and not topo.is_live(owner / num_scaleup_ranks, owner % num_scaleup_ranks))
                stored_dst_expert_idx = -1;
            stored_dst_rank_idx = stored_dst_expert_idx >= 0 ? (is_direct ? owner % num_scaleup_ranks : owner / num_scaleup_ranks) : -1;
        }
        __syncwarp();

        // Sort valid top-k indices to front
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {
                // Activations are never reduced before
                return {false, 0};
            } else if constexpr (not kUseExpandedLayout and not kAllowMultipleReduction) {
                // Without expanded layout and multiple reduction, the forward warps only reduce within a rank,
                // so deduplicate on a per-rank basis
                return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / num_experts_per_rank : -1};
            } else {
                // Otherwise the scale-up domain has already been reduced, so deduplicate on a per-scale-out-rank basis
                return {true, stored_dst_rank_idx};
            }
        }();
        auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);

        // Hybrid: the rows came over the scale-out path with their arrival flags inside (see `TokenLayout`); lane `l` waits
        // for the row it reduces, restores the spills, and zeroes the flags once the row is consumed (below)
        const bool lane_has_row = not is_direct and ((reduce_valid_mask >> lane_idx) & 1);
        const auto lane_row = comm_buffer.get_rank_buffer(lane_has_row ? (use_rank_layout ? stored_dst_rank_idx : lane_idx) : 0)
                                         .get_token_buffer(token_idx);
        if (lane_has_row) {
            int pending_flag_idx = lane_row.num_spill_slots;
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                if (lane_row.template are_flags_set<kAcquireFlags>(pending_flag_idx))
                    return true;
                if (is_last_check) {
                    printf("NIXL EP V2 adaptive combine (reduce epilogue) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                           "token: %d, row: %d\n",
                           topo.scaleout_rank_idx, num_scaleout_ranks, topo.scaleup_rank_idx, num_scaleup_ranks,
                           token_idx, use_rank_layout ? stored_dst_rank_idx : lane_idx);
                }
                return false;
            });
            if constexpr (not kAcquireFlags) {
                // The flags were polled with relaxed loads: the acquire fence before the row is read (the review tip's protocol). The reads
                // that follow use `ld.global.nc`, safe because no L1 line of the row can predate its arrival: the polls bypass L1, the data
                // loads carry `L1::no_allocate`, the one plain load below (the spill slot) lies inside the landed record
                ptx::fence_acq_rel_sys();
            }
            // With acquire polls (`acquire_flags`) no fence is needed: this lane's plain accesses (the spill slots here, the top-k weight
            // below) follow the acquire, the other lanes' reads follow the `__syncwarp`
            lane_row.restore_spills();
        }
        __syncwarp();

        // Iterate over per-hidden-chunk stage
        // The number of rows to gather is a runtime value; pick the instantiation DeepSeek's kernels would use
        // (slot array length and expected valid count), so the reduction loop stays as tight
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
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

        // Write top-k weights
        if (combined_topk_weights != nullptr) {
            const auto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));
            if (lane_idx < kNumTopk) {
                float value = 0;
                if (stored_dst_rank_idx >= 0) {
                    const auto dst_ptr = comm_buffer
                        .get_rank_buffer(use_rank_layout ? stored_dst_rank_idx : master_lane_idx)
                        .get_token_buffer(token_idx).get_topk_weights_ptr() + lane_idx;
                    value = *dst_ptr;
                }
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;
            }
            __syncwarp();
        }

        // The rows are consumed: their flag words back to zero for the next use
        if (lane_has_row)
            lane_row.clear_flags();
        __syncwarp();
    }

#ifdef EP_ADAPTIVE_PROFILE
    ptx::tma_store_wait();
    __syncthreads();
    EP_PROFILE_STAMP_LAST(profile_layout, 17);
#endif
}

}  // namespace nixl_ep_v2::elastic::adaptive
