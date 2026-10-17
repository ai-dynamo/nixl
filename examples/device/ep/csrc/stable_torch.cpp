/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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

void
query_mask_buffer(int64_t buffer_ptr, const torch::stable::Tensor &mask_status) {
    buffer_from_ptr(buffer_ptr).query_mask_buffer(mask_status);
}

torch::stable::Tensor
get_next_combine_buffer(int64_t buffer_ptr,
                        int num_max_dispatch_tokens_per_rank,
                        int hidden,
                        int rank_bound) {
    return buffer_from_ptr(buffer_ptr)
        .get_next_combine_buffer(num_max_dispatch_tokens_per_rank, hidden, rank_bound);
}

torch::stable::Tensor
get_local_buffer_tensor(int64_t buffer_ptr,
                        torch::headeronly::ScalarType dtype,
                        int64_t offset,
                        bool use_rdma_buffer) {
    return buffer_from_ptr(buffer_ptr).get_local_buffer_tensor(dtype, offset, use_rdma_buffer);
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
         std::optional<int> num_experts,
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
                  num_experts,
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
} // namespace

STABLE_TORCH_LIBRARY(nixl_ep, m) {
    m.def("query_mask_buffer_(int buffer, Tensor(a!) mask_status) -> ()");
    m.def(
        "get_next_combine_buffer(int buffer, int num_max_dispatch_tokens_per_rank, int hidden, "
        "int rank_bound) -> Tensor");
    m.def("get_local_buffer_tensor(int buffer, ScalarType dtype, int offset, bool use_rdma_buffer) "
          "-> Tensor");
    m.def("combine(int buffer, Tensor x, Tensor topk_idx, Tensor topk_weights, Tensor src_info, "
          "Tensor layout_range, Tensor(a!)? combine_wait_recv_cost_stats, int "
          "num_max_dispatch_tokens_per_rank, bool use_logfmt, bool zero_copy, bool async, bool "
          "return_recv_hook, Tensor(b!)? out) -> (Tensor, int)");
    m.def("combine_recv_hook(int buffer) -> ()");
    m.def("dispatch(int buffer, Tensor x, Tensor topk_idx, Tensor(a!)? "
          "cumulative_local_expert_recv_stats, Tensor(b!)? dispatch_wait_recv_cost_stats, int "
          "num_max_dispatch_tokens_per_rank, int? num_experts, bool use_fp8, bool round_scale, "
          "bool use_ue8m0, bool async, bool return_recv_hook) -> (Tensor, Tensor?, Tensor, Tensor, "
          "Tensor, int)");
    m.def("dispatch_recv_hook(int buffer) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(nixl_ep, CUDA, m) {
    m.impl("query_mask_buffer_", TORCH_BOX(&query_mask_buffer));
    m.impl("combine", TORCH_BOX(&combine));
    m.impl("dispatch", TORCH_BOX(&dispatch));
}

STABLE_TORCH_LIBRARY_IMPL(nixl_ep, CompositeExplicitAutograd, m) {
    m.impl("get_next_combine_buffer", TORCH_BOX(&get_next_combine_buffer));
    m.impl("get_local_buffer_tensor", TORCH_BOX(&get_local_buffer_tensor));
    m.impl("combine_recv_hook", TORCH_BOX(&combine_recv_hook));
    m.impl("dispatch_recv_hook", TORCH_BOX(&dispatch_recv_hook));
}
