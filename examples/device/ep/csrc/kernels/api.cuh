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

// EP kernels
namespace ep_kernels {
struct gpu_nixl_ctx {
    nixlGpuXferReqH *batch_reqs; // [dest_rank]
    nixlGpuXferReqH *remote_barrier_reqs; // [dest_rank]
    int *local_barrier_buffer; // [src_rank]
    int *local_barrier_cnt; // [dst_rank]
    void **rdma_p2p_ptrs; // [num_ranks]
    void *rdma_buffer_ptr;
    int num_ranks;
    int rank;

    /* Double buffering considerations are handled by the caller */
    __device__ inline void *rdma_p2p_ptr_get(uint64_t ptr, int dst_rank) {
        if (rdma_p2p_ptrs[dst_rank] == nullptr)
            return nullptr;

        return (void *)(reinterpret_cast<uint64_t>(rdma_p2p_ptrs[dst_rank]) + batch_offset_get(ptr));
    }

    __device__ inline nixlGpuXferReqH remote_barrier_get(int dest_rank) {
        return remote_barrier_reqs[dest_rank];
    }

    __device__ inline nixlGpuXferReqH batch_get(int dest_rank) {
        return batch_reqs[dest_rank];
    }

    __device__ inline size_t batch_offset_get(uint64_t ptr) {
        return ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr);
    }
};

void clean_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              int rank, int num_ranks, int* mask_buffer, int* sync_buffer,
                              cudaStream_t stream);

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, uint64_t* rdma_recv_count, void* rdma_x,
              const void* x, const topk_idx_t* topk_idx,
              uint64_t* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              void* workspace, int num_device_sms,
              cudaStream_t stream, int phases, ep_kernels::gpu_nixl_ctx nixl_ctx);

void combine(void* combined_x,
             void* rdma_recv_x, uint64_t* rdma_recv_flag, void* rdma_send_x,
             const void* x, const topk_idx_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* mask_buffer,
             int64_t* combine_wait_recv_cost_stats,
             uint64_t* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy, ep_kernels::gpu_nixl_ctx nixl_ctx);

void barrier(ep_kernels::gpu_nixl_ctx nixl_ctx, int* mask_buffer_ptr, int* sync_buffer_ptr, cudaStream_t stream);

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, cudaStream_t stream);

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, cudaStream_t stream);

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream);

} // namespace ep_kernels

} // namespace nixl_ep
