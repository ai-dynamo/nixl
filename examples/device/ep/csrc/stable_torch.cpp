/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "nixl_ep.hpp"

#include <torch/csrc/stable/library.h>

namespace {
nixl_ep::Buffer &
buffer_from_ptr(int64_t buffer_ptr) {
    EP_HOST_ASSERT(buffer_ptr != 0);
    return *reinterpret_cast<nixl_ep::Buffer *>(buffer_ptr);
}

const nixl_ep::Config &
config_from_ptr(int64_t config_ptr) {
    EP_HOST_ASSERT(config_ptr != 0);
    return *reinterpret_cast<const nixl_ep::Config *>(config_ptr);
}

const nixl_ep::EventHandle *
event_handle_from_ptr(int64_t event_handle_ptr) {
    EP_HOST_ASSERT(event_handle_ptr != 0);
    return reinterpret_cast<const nixl_ep::EventHandle *>(event_handle_ptr);
}

void
query_mask_buffer(int64_t buffer_ptr, const torch::stable::Tensor &mask_status) {
    buffer_from_ptr(buffer_ptr).query_mask_buffer(mask_status);
}

torch::stable::Tensor
get_next_combine_buffer(int64_t buffer_ptr, int num_max_dispatch_tokens_per_rank, int hidden) {
    return buffer_from_ptr(buffer_ptr)
        .get_next_combine_buffer(num_max_dispatch_tokens_per_rank, hidden);
}

torch::stable::Tensor
get_local_buffer_tensor(int64_t buffer_ptr,
                        torch::headeronly::ScalarType dtype,
                        int64_t offset,
                        bool use_rdma_buffer) {
    return buffer_from_ptr(buffer_ptr).get_local_buffer_tensor(dtype, offset, use_rdma_buffer);
}

std::tuple<torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           torch::stable::Tensor,
           torch::stable::Tensor,
           int64_t>
get_dispatch_layout(int64_t buffer_ptr,
                    const torch::stable::Tensor &topk_idx,
                    int num_experts,
                    int64_t previous_event_ptr,
                    bool async,
                    bool allocate_on_comm_stream) {
    return buffer_from_ptr(buffer_ptr)
        .get_dispatch_layout(topk_idx,
                             num_experts,
                             event_handle_from_ptr(previous_event_ptr),
                             async,
                             allocate_on_comm_stream);
}

std::tuple<torch::stable::Tensor, int64_t>
combine(int64_t buffer_ptr,
        const torch::stable::Tensor &x,
        const torch::stable::Tensor &topk_idx,
        const torch::stable::Tensor &topk_weights,
        const torch::stable::Tensor &src_info,
        const torch::stable::Tensor &layout_range,
        const std::optional<torch::stable::Tensor> &combine_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        bool use_logfmt,
        bool zero_copy,
        bool async,
        bool return_recv_hook,
        const std::optional<torch::stable::Tensor> &out) {
    return buffer_from_ptr(buffer_ptr)
        .combine(x,
                 topk_idx,
                 topk_weights,
                 src_info,
                 layout_range,
                 combine_wait_recv_cost_stats,
                 num_max_dispatch_tokens_per_rank,
                 use_logfmt,
                 zero_copy,
                 async,
                 return_recv_hook,
                 out);
}

void
combine_recv_hook(int64_t buffer_ptr) {
    buffer_from_ptr(buffer_ptr).run_combine_recv_hook();
}

std::tuple<torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           torch::stable::Tensor,
           torch::stable::Tensor,
           torch::stable::Tensor,
           int64_t>
dispatch(int64_t buffer_ptr,
         const torch::stable::Tensor &x,
         const torch::stable::Tensor &topk_idx,
         const std::optional<torch::stable::Tensor> &cumulative_local_expert_recv_stats,
         const std::optional<torch::stable::Tensor> &dispatch_wait_recv_cost_stats,
         int num_max_dispatch_tokens_per_rank,
         bool use_fp8,
         bool round_scale,
         bool use_ue8m0,
         bool async,
         bool return_recv_hook) {
    return buffer_from_ptr(buffer_ptr)
        .dispatch(x,
                  topk_idx,
                  cumulative_local_expert_recv_stats,
                  dispatch_wait_recv_cost_stats,
                  num_max_dispatch_tokens_per_rank,
                  use_fp8,
                  round_scale,
                  use_ue8m0,
                  async,
                  return_recv_hook);
}

void
dispatch_recv_hook(int64_t buffer_ptr) {
    buffer_from_ptr(buffer_ptr).run_dispatch_recv_hook();
}

std::tuple<torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           std::optional<torch::stable::Tensor>,
           std::optional<torch::stable::Tensor>,
           std::vector<int>,
           torch::stable::Tensor,
           torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           std::optional<torch::stable::Tensor>,
           std::optional<torch::stable::Tensor>,
           int64_t>
ht_dispatch(int64_t buffer_ptr,
            const torch::stable::Tensor &x,
            const std::optional<torch::stable::Tensor> &x_scales,
            const std::optional<torch::stable::Tensor> &topk_idx,
            const std::optional<torch::stable::Tensor> &topk_weights,
            const std::optional<torch::stable::Tensor> &num_tokens_per_rank,
            const std::optional<torch::stable::Tensor> &num_tokens_per_rdma_rank,
            const torch::stable::Tensor &is_token_in_rank,
            const std::optional<torch::stable::Tensor> &num_tokens_per_expert,
            int cached_num_recv_tokens,
            int cached_num_rdma_recv_tokens,
            const std::optional<torch::stable::Tensor> &cached_rdma_channel_prefix_matrix,
            const std::optional<torch::stable::Tensor> &cached_recv_rdma_rank_prefix_sum,
            const std::optional<torch::stable::Tensor> &cached_gbl_channel_prefix_matrix,
            const std::optional<torch::stable::Tensor> &cached_recv_gbl_rank_prefix_sum,
            int expert_alignment,
            int64_t config_ptr,
            int64_t previous_event_ptr,
            bool async,
            bool allocate_on_comm_stream) {
    return buffer_from_ptr(buffer_ptr)
        .ht_dispatch(x,
                     x_scales,
                     topk_idx,
                     topk_weights,
                     num_tokens_per_rank,
                     num_tokens_per_rdma_rank,
                     is_token_in_rank,
                     num_tokens_per_expert,
                     cached_num_recv_tokens,
                     cached_num_rdma_recv_tokens,
                     cached_rdma_channel_prefix_matrix,
                     cached_recv_rdma_rank_prefix_sum,
                     cached_gbl_channel_prefix_matrix,
                     cached_recv_gbl_rank_prefix_sum,
                     expert_alignment,
                     config_from_ptr(config_ptr),
                     event_handle_from_ptr(previous_event_ptr),
                     async,
                     allocate_on_comm_stream);
}

std::tuple<torch::stable::Tensor, std::optional<torch::stable::Tensor>, int64_t>
ht_combine(int64_t buffer_ptr,
           const torch::stable::Tensor &x,
           const std::optional<torch::stable::Tensor> &topk_weights,
           const std::optional<torch::stable::Tensor> &bias_0,
           const std::optional<torch::stable::Tensor> &bias_1,
           const torch::stable::Tensor &src_meta,
           const torch::stable::Tensor &is_combined_token_in_rank,
           const torch::stable::Tensor &rdma_channel_prefix_matrix,
           const torch::stable::Tensor &rdma_rank_prefix_sum,
           const torch::stable::Tensor &gbl_channel_prefix_matrix,
           const torch::stable::Tensor &combined_rdma_head,
           const torch::stable::Tensor &combined_nvl_head,
           int64_t config_ptr,
           int64_t previous_event_ptr,
           bool async,
           bool allocate_on_comm_stream) {
    return buffer_from_ptr(buffer_ptr)
        .ht_combine(x,
                    topk_weights,
                    bias_0,
                    bias_1,
                    src_meta,
                    is_combined_token_in_rank,
                    rdma_channel_prefix_matrix,
                    rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    combined_rdma_head,
                    combined_nvl_head,
                    config_from_ptr(config_ptr),
                    event_handle_from_ptr(previous_event_ptr),
                    async,
                    allocate_on_comm_stream);
}
} // namespace

STABLE_TORCH_LIBRARY(nixl_ep, m) {
    m.def("query_mask_buffer_(int buffer, Tensor mask_status) -> ()");
    m.def("get_next_combine_buffer(int buffer, int num_max_dispatch_tokens_per_rank, int hidden) "
          "-> Tensor");
    m.def("get_local_buffer_tensor(int buffer, ScalarType dtype, int offset, bool use_rdma_buffer) "
          "-> Tensor");
    m.def("get_dispatch_layout(int buffer, Tensor topk_idx, int num_experts, int previous_event, "
          "bool async, bool allocate_on_comm_stream) -> (Tensor, Tensor?, Tensor, Tensor, int)");
    m.def("combine(int buffer, Tensor x, Tensor topk_idx, Tensor topk_weights, Tensor src_info, "
          "Tensor layout_range, Tensor? combine_wait_recv_cost_stats, int "
          "num_max_dispatch_tokens_per_rank, bool use_logfmt, bool zero_copy, bool async, bool "
          "return_recv_hook, Tensor? out) -> (Tensor, int)");
    m.def("combine_recv_hook(int buffer) -> ()");
    m.def("dispatch(int buffer, Tensor x, Tensor topk_idx, Tensor? "
          "cumulative_local_expert_recv_stats, Tensor? dispatch_wait_recv_cost_stats, int "
          "num_max_dispatch_tokens_per_rank, bool use_fp8, bool round_scale, bool use_ue8m0, bool "
          "async, bool return_recv_hook) -> (Tensor, Tensor?, Tensor, Tensor, Tensor, int)");
    m.def("dispatch_recv_hook(int buffer) -> ()");
    m.def("ht_dispatch(int buffer, Tensor x, Tensor? x_scales, Tensor? topk_idx, Tensor? "
          "topk_weights, Tensor? num_tokens_per_rank, Tensor? num_tokens_per_rdma_rank, Tensor "
          "is_token_in_rank, Tensor? num_tokens_per_expert, int cached_num_recv_tokens, int "
          "cached_num_rdma_recv_tokens, Tensor? cached_rdma_channel_prefix_matrix, Tensor? "
          "cached_recv_rdma_rank_prefix_sum, Tensor? cached_gbl_channel_prefix_matrix, Tensor? "
          "cached_recv_gbl_rank_prefix_sum, int expert_alignment, int config, int "
          "previous_event, bool async, bool allocate_on_comm_stream) -> (Tensor, Tensor?, Tensor?, "
          "Tensor?, int[], Tensor, Tensor, Tensor?, Tensor, Tensor?, Tensor, Tensor?, Tensor?, "
          "Tensor?, int)");
    m.def(
        "ht_combine(int buffer, Tensor x, Tensor? topk_weights, Tensor? bias_0, Tensor? bias_1, "
        "Tensor src_meta, Tensor is_combined_token_in_rank, Tensor rdma_channel_prefix_matrix, "
        "Tensor rdma_rank_prefix_sum, Tensor gbl_channel_prefix_matrix, Tensor combined_rdma_head, "
        "Tensor combined_nvl_head, int config, int previous_event, bool async, bool "
        "allocate_on_comm_stream) -> (Tensor, Tensor?, int)");
}

STABLE_TORCH_LIBRARY_IMPL(nixl_ep, CUDA, m) {
    m.impl("query_mask_buffer_", TORCH_BOX(&query_mask_buffer));
    m.impl("get_dispatch_layout", TORCH_BOX(&get_dispatch_layout));
    m.impl("combine", TORCH_BOX(&combine));
    m.impl("dispatch", TORCH_BOX(&dispatch));
    m.impl("ht_dispatch", TORCH_BOX(&ht_dispatch));
    m.impl("ht_combine", TORCH_BOX(&ht_combine));
}

STABLE_TORCH_LIBRARY_IMPL(nixl_ep, CompositeExplicitAutograd, m) {
    m.impl("get_next_combine_buffer", TORCH_BOX(&get_next_combine_buffer));
    m.impl("get_local_buffer_tensor", TORCH_BOX(&get_local_buffer_tensor));
    m.impl("combine_recv_hook", TORCH_BOX(&combine_recv_hook));
    m.impl("dispatch_recv_hook", TORCH_BOX(&dispatch_recv_hook));
}
