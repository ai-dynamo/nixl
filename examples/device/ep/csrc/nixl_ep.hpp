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

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include <optional>
#include <tuple>
#include <vector>
#include <string>

#include <memory>
#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

#include "nixl.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME nixl_ep_cpp
#endif

namespace nixl_ep {

struct NixlPeerInfo {
    void* rdma_buffer_ptr;
    int* sync_buffer_ptr;
    int device_id;
    int rank;
};

struct NixlAgentInfo
{
    NixlAgentInfo(std::shared_ptr<nixlAgent> agent, nixlBackendH* backend, int max_num_ranks): agent(agent), backend(backend) {
        wire_up_done.resize(max_num_ranks, false);
        remote_agent_names.resize(max_num_ranks);
    }

    std::shared_ptr<nixlAgent> agent;
    std::string agent_name;
    std::vector<std::string> remote_agent_names;
    nixl_opt_args_t extra_params;
    nixlBackendH* backend;
    std::vector<bool> wire_up_done; // [num_peers]
};

struct Buffer {
private:
    int buffer_idx = 0; // Double buffering index

    // RDMA Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    int *mask_buffer_ptr = nullptr;
    int *sync_buffer_ptr = nullptr;
    int *sync_count_ptr = nullptr;

    // Device info and communication
    int device_id;
    int num_device_sms;
    int rank;
    int num_ranks;
    std::vector<int> remote_ranks; /* global ranks */

    // Stream for communication
    at::cuda::CUDAStream comm_stream;

    // After synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Workspace
    void* workspace = nullptr;

    std::unique_ptr<NixlAgentInfo> nixl_agent_info;
    std::vector<NixlPeerInfo> nixl_peer_info;
    NixlPeerInfo my_peer_info;
    int max_num_ranks;
    int max_experts_per_rank;
    ep_kernels::gpu_nixl_ctx gpu_ctx;

    /* Common private funcs */
    void _nixl_agent_init();
    void _nixl_agents_connect(const std::vector<int>& ranks, const std::vector<nixl_blob_t>& remote_mds = {});
    void _nixl_agents_disconnect(const std::vector<int>& ranks);
    void _nixl_agents_peer_info_gather(std::vector<int>& ranks);
    void _nixl_agents_peer_info_cleanup(const std::vector<int>& ranks);

    void _nixl_ep_init(void);
    void _nixl_ep_memory_views_create(void);
    void _nixl_ep_memory_views_destroy(void);
    void _nixl_ep_destroy(void);

public:
    Buffer(int rank, bool explicitly_destroy);

    void update_memory_buffers(int num_ranks, int max_experts_per_rank, int64_t num_rdma_bytes);

    void connect_ranks(const std::vector<int>& remote_ranks_list, const std::optional<std::vector<nixl_blob_t>>& remote_mds = std::nullopt);

    void disconnect_ranks(const std::vector<int>& remote_ranks_list);

    void init(int num_ranks, int max_experts_per_rank, int64_t num_rdma_bytes);

    ~Buffer() noexcept(false);

    bool is_available() const;

    int get_local_device_id() const;

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset) const;

    torch::Stream get_comm_stream() const;

    void destroy();

    void clean_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                         int num_max_dispatch_tokens_per_rank, int num_experts,
                         bool use_fp8, bool round_scale, bool use_ue8m0,
                         bool async, bool return_recv_hook);

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                        const torch::Tensor& src_info, const torch::Tensor& layout_range,
                        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                        int num_max_dispatch_tokens_per_rank, int num_experts,
                        bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                        const std::optional<torch::Tensor>& out = std::nullopt);

    void barrier();

    torch::Tensor
    get_next_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;

    void update_mask_buffer(int rank_to_mask, bool mask);

    void query_mask_buffer(const torch::Tensor& mask_status);

    void clean_mask_buffer();

    std::string get_local_metadata() const;
};

} // namespace nixl_ep
