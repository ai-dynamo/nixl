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
          int kNumSMs, int kNumWarps, int kNumChannelsPerSM,
          int kHidden,
          int kNumExpertsPerRank, int kNumTopk,
          int64_t kNumTimeoutCycles,
          bool kAcquireFlags = false,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* buffer,
                             void* workspace,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const topology::EPTopology* topology_ptr,
                             const int num_max_tokens_per_rank) {
    // Topology
    const auto topo = topology::load(topology_ptr);
    const int num_scaleout_ranks = topo.num_scaleout_ranks, num_scaleup_ranks = topo.num_scaleup_ranks;
    const int num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    const int num_channels = kNumChannelsPerSM * topo.num_active_sms;
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

    // Per-token preprocessing of the top-k indices, warp-collective (every lane calls it, lane `l` handles selection `l`). It reads
    // only `combined_topk_idx` (an input of the combine; no kernel of this launch pair writes it) and the topology descriptor, never
    // the receive rows, so a warp's first token is preprocessed before the dependency wait, off the post-trigger critical path
    // NOTES: the "destination rank" here is the rank owning the expert: its domain (hybrid) or its position (direct)
    struct TokenIndices {
        int stored_dst_rank_idx;
        unsigned reduce_valid_mask;
    };
    const auto preprocess_token = [&](const int& token_idx) -> TokenIndices {
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
        const auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);
        return {stored_dst_rank_idx, reduce_valid_mask};
    };

    // Hidden-dimension chunking (direct only): a work item is one (token, chunk) pair, `kChunkVec` vectors wide, so a token's row
    // is reduced by `kNumChunks` warps on different SMs. Bit-exact with the one-warp row: the slot order and the bypass choice depend
    // only on the token's indices, and every chunk reduces its own disjoint slice. Hybrid keeps one warp per token (its wait on the
    // producing channel's progress word is per row)
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);
    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
    constexpr int kChunkVec = kUnrollFactor * 32;
    constexpr int kNumChunks = kHiddenVec / kChunkVec;
    constexpr int kNumBytesPerChunk = kChunkVec * static_cast<int>(sizeof(combine_vec_t));
    EP_STATIC_ASSERT(kNumChunks * kNumBytesPerChunk == kNumHiddenBytes, "Invalid chunking");
    // Direct, small batch: 512-byte items (`kSmallBatchChunkVec` vectors, 28 per BF16 7168 row) plus one item for the top-k weight
    // write: each lane loads one int4 per source row, so a warp's row loads are one batch (`kBatchRowLoads`), and the weights leave
    // the reduce critical path. The per-item costs (token preprocessing, TMA wait / fence / commit) are paid 4x per byte against
    // the 2 KB chunks, so the split serves at most `kNumMaxSmallBatchTokens` tokens per rank; larger direct batches keep the chunk
    // items above. Same slots, same per-element order. The predicate is block-uniform: `num_combined_tokens` is a kernel argument
    constexpr int kSmallBatchChunkVec = 32;
    constexpr int kNumSmallBatchChunks = kHiddenVec / kSmallBatchChunkVec;
    constexpr int kNumBytesPerSmallBatchChunk = kSmallBatchChunkVec * static_cast<int>(sizeof(combine_vec_t));
    EP_STATIC_ASSERT(kNumSmallBatchChunks * kNumBytesPerSmallBatchChunk == kNumHiddenBytes, "Invalid small-batch chunking");
    constexpr int kNumMaxSmallBatchTokens = 16;
    const bool is_small_batch = is_direct and num_combined_tokens <= kNumMaxSmallBatchTokens;
    const int num_items_per_token = not is_direct ? 1 :
        is_small_batch ? kNumSmallBatchChunks + (combined_topk_weights != nullptr ? 1 : 0) : kNumChunks;
    const int num_bytes_per_chunk = is_small_batch ? kNumBytesPerSmallBatchChunk : kNumBytesPerChunk;
    const int num_work_items = num_combined_tokens * num_items_per_token;

    // The token of a work item: direct items run upward (`num_items_per_token` per token); hybrid rows run from the last token
    // down, the order in which the producing channels' progress words complete (they descend from the highest token index)
    const auto item_token_idx = [&](const int& item_idx) {
        return is_direct ? item_idx / num_items_per_token : num_combined_tokens - 1 - item_idx;
    };

    // The warp's first token, preprocessed before the dependency wait (a warp-uniform condition)
    int cached_token_idx = -1;
    TokenIndices token_indices = {-1, 0u};
    if (global_warp_idx < num_work_items) {
        cached_token_idx = item_token_idx(global_warp_idx);
        token_indices = preprocess_token(cached_token_idx);
    }

    // Wait for the main combine and its local stores; the remote result batches are polled below (hybrid)
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();

    const auto workspace_layout = layout::WorkspaceLayout(workspace, num_scaleout_ranks, num_scaleup_ranks, num_ranks * kNumExpertsPerRank);
    const auto ready_atomic_buff = workspace_layout.get_combine_ready_atomic_buff();

    // Profile slots 16/17: epilogue work start (block 0) / last block done
#ifdef EP_ADAPTIVE_PROFILE
    EP_PROFILE_STAMP(workspace_layout, 16);
#endif

    // Read from buffers and do reduction
    for (int item_idx = global_warp_idx; item_idx < num_work_items; item_idx += kNumWarps * kNumSMs) {
        const int token_idx = item_token_idx(item_idx);
        const int chunk_begin = item_idx % num_items_per_token;   // the item's chunk (direct); 0 for hybrid's whole row
        const bool is_weights_item = is_small_batch and chunk_begin == kNumSmallBatchChunks;
        // Preprocess all indices (the first token was done above)
        if (token_idx != cached_token_idx) {
            token_indices = preprocess_token(token_idx);
            cached_token_idx = token_idx;
        }
        const int stored_dst_rank_idx = token_indices.stored_dst_rank_idx;
        const auto reduce_valid_mask = token_indices.reduce_valid_mask;

        // One lane per remote producing domain waits for this token's cumulative channel watermark.
        if (not is_direct) {
            const bool lane_polls = ptx::deduplicate(stored_dst_rank_idx, lane_idx) and
                                    stored_dst_rank_idx >= 0 and stored_dst_rank_idx != topo.scaleout_rank_idx;
            if (lane_polls) {
                const int channel_idx = token_idx % num_channels;
                const auto completion_ptr = ready_atomic_buff + channel_idx * num_scaleout_ranks + stored_dst_rank_idx;
                comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                    const auto signal = kAcquireFlags ? ptx::ld_acquire_sys<int64_t>(completion_ptr) :
                                                       ptx::ld_relaxed_sys<int64_t>(completion_ptr);
                    if (signal <= token_idx) {
                        if constexpr (not kAcquireFlags)
                            ptx::fence_acq_sys();
                        return true;
                    }
                    if (is_last_check) {
                        printf("NIXL EP V2 combine (reduce epilogue) timeout, scale-out: %d/%d, scale-up: %d/%d, "
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
        // The number of rows to gather is a runtime value; pick the instantiation DeepSeek's kernels would use
        // (slot array length and expected valid count), so the reduction loop stays as tight
        const auto reduce_with = [&]<int kNumTokensInLayout>() {
            int topk_slot_idx[kNumTokensInLayout];
            compute_topk_slots(
                topk_slot_idx, reduce_valid_mask,
                [=](const int& idx) {
                    return use_rank_layout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
                }
            );
            const auto src_ptr = [=](const int& slot_idx) {
                return static_cast<combine_vec_t*>(comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());
            };
            const auto wait_tma_buffer = [=]() {
                ptx::tma_store_wait();
                __syncwarp();
            };
            auto* dst_ptr = static_cast<combine_vec_t*>(tma_buffer.get_base_ptr());
            auto* bias_0_ptr = bias_0 == nullptr ? nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr());
            auto* bias_1_ptr = bias_1 == nullptr ? nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr());
            if (is_small_batch) {
                // This item's 512-byte chunk, one vector per lane, every row load in flight before the row accumulation
                const int vec_offset = chunk_begin * kSmallBatchChunkVec;
                combine_reduce<kSmallBatchChunkVec, 1, kNumTokensInLayout, true>(
                    lane_idx, topk_slot_idx, dst_ptr + vec_offset,
                    [=](const int& slot_idx) { return src_ptr(slot_idx) + vec_offset; }, wait_tma_buffer,
                    bias_0_ptr == nullptr ? nullptr : bias_0_ptr + vec_offset, bias_1_ptr == nullptr ? nullptr : bias_1_ptr + vec_offset);
                return;
            }
            // Direct: this item's chunk; hybrid: the whole row, chunk by chunk through the same instantiation. The staging buffer is
            // released once, before the first chunk's write: the row's TMA store is issued after the loop
            const int chunk_end = is_direct ? chunk_begin + 1 : kNumChunks;
            #pragma unroll 1
            for (int chunk_idx = chunk_begin; chunk_idx < chunk_end; ++ chunk_idx) {
                const int vec_offset = chunk_idx * kChunkVec;
                combine_reduce<kChunkVec, kUnrollFactor, kNumTokensInLayout>(
                    lane_idx, topk_slot_idx, dst_ptr + vec_offset,
                    [=](const int& slot_idx) { return src_ptr(slot_idx) + vec_offset; },
                    [=]() { if (chunk_idx == chunk_begin) wait_tma_buffer(); },
                    bias_0_ptr == nullptr ? nullptr : bias_0_ptr + vec_offset, bias_1_ptr == nullptr ? nullptr : bias_1_ptr + vec_offset);
            }
        };
        if (not is_weights_item) {
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
                const int byte_offset = chunk_begin * num_bytes_per_chunk;
                const int num_bytes = is_direct ? num_bytes_per_chunk : kNumHiddenBytes;
                ptx::tma_store_1d(math::advance_ptr<int8_t>(output_buffer.get_token_buffer(token_idx).get_base_ptr(), byte_offset),
                                  math::advance_ptr<int8_t>(tma_buffer.get_base_ptr(), byte_offset), num_bytes);
                ptx::tma_store_commit();
            }
            __syncwarp();
        }

        // Write top-k weights (small-batch direct: the token's weights item; otherwise the token's first item)
        if (combined_topk_weights != nullptr and (is_small_batch ? is_weights_item : chunk_begin == 0)) {
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
    }

#ifdef EP_ADAPTIVE_PROFILE
    ptx::tma_store_wait();
    __syncthreads();
    EP_PROFILE_STAMP_LAST(workspace_layout, 17);
#endif
}

}  // namespace nixl_ep_v2::elastic::adaptive
