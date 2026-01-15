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

#include <vector>
#include <iostream>
#include "nixl_types.h"
#include "exception.cuh"
#include "configs.cuh"

namespace nixl_ep {

namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_nvl_ranks, cudaStream_t stream);

}  // namespace intranode

struct gpu_nixl_ctx {
    nixlMemoryViewH local_mvh;
    nixlMemoryViewH barrier_mvh;
    nixlMemoryViewH remote_mvh;
    nixlMemoryViewH internode_barrier_mvh; // Remote internode barrier counters (uint64_t per rank)
    int *sync_buffer_ptr; // [src_rank]
    int *sync_count_ptr; // [dst_rank]
    void *rdma_buffer_ptr;
    int max_num_ranks;
    int num_rdma_ranks;
    int rank;

    __device__ inline uint64_t offset_get(uint64_t ptr) {
        return ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr);
    }
    __device__ inline void* p2p_ptr_get(uint64_t dst_ptr, int dst_rank);

    // Internode-specific barrier counters (used only by internode kernels)
    uint64_t *last_barrier_counter;        // Tracks barrier epochs
    uint64_t *local_barrier_counter_ptr;   // Receives barrier signals from remote ranks
};


// Internode runtime
namespace internode {

void *alloc(size_t size, size_t alignment);

void free(void *ptr);

} // namespace internode

// Layout kernels
namespace layout {

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream);

} // namespace layout

// Internode kernels
namespace internode {

int get_source_meta_bytes();

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode,
                     gpu_nixl_ctx nixl_ctx);

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              int scale_token_stride,
              int scale_hidden_stride,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode,
              gpu_nixl_ctx nixl_ctx);

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode,
                   gpu_nixl_ctx nixl_ctx);

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_channels,
             bool low_latency_mode,
             gpu_nixl_ctx nixl_ctx);

} // namespace internode


// EP kernels
namespace ep_kernels {
void clean_buffer(int* clean_0,
                  int num_clean_int_0,
                  int* clean_1,
                  int num_clean_int_1,
                  int rank,
                  int num_ranks,
                  int* mask_buffer,
                  int* sync_buffer,
                  cudaStream_t stream);

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              uint64_t* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              uint64_t* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases,
              nixl_ep::gpu_nixl_ctx nixl_ctx);

void combine(void* combined_x,
             void* rdma_recv_x,
             uint64_t* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer,
             int64_t* combine_wait_recv_cost_stats,
             uint64_t* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy,
             nixl_ep::gpu_nixl_ctx nixl_ctx);

void barrier(gpu_nixl_ctx nixl_ctx, int* mask_buffer_ptr, cudaStream_t stream);

void query_mask_buffer(int* mask_buffer_ptr,
                       int num_ranks,
                       int* output_mask_tensor,
                       cudaStream_t stream);

void update_mask_buffer(int* mask_buffer_ptr,
                        int rank_to_mask,
                        bool mask,
                        cudaStream_t stream);

void clean_mask_buffer(int* mask_buffer_ptr,
                       int num_ranks,
                       cudaStream_t stream);

} // namespace ep_kernels

} // namespace nixl_ep
