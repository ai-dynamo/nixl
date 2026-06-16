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
#include <functional>
#include <optional>
#include <tuple>
#include <vector>
#include <string>

#include <memory>
#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"
#include "vmm.hpp"

#include "nixl.h"

#include <torch/csrc/stable/tensor.h>

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME nixl_ep_cpp
#endif

namespace nixl_ep {

struct NixlPeerInfo {
    void* rdma_buffer_ptr;
    int* sync_buffer_ptr;
    uint64_t* ht_barrier_ptr;
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
    nixl_reg_dlist_t rdma_reg_descs{VRAM_SEG};
    nixl_reg_dlist_t sync_reg_descs{VRAM_SEG};
    nixl_reg_dlist_t sync_count_reg_descs{VRAM_SEG};
    nixl_reg_dlist_t ht_barrier_reg_descs{VRAM_SEG};
    std::vector<bool> wire_up_done; // [num_peers]
};

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    int buffer_idx = 0; // Double buffering index
    bool low_latency_mode = false;
    uint64_t timeout_ms = 30000;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // RDMA Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    int *mask_buffer_ptr = nullptr;
    int *sync_buffer_ptr = nullptr;
    int *sync_count_ptr = nullptr;

    /* Owning VMM allocations (keep raw ptrs above as aliases) */
    std::unique_ptr<vmm_region> m_rdma_alloc;
    std::unique_ptr<vmm_region> m_mask_alloc;
    std::unique_ptr<vmm_region> m_sync_alloc;
    std::unique_ptr<vmm_region> m_sync_count_alloc;
    std::unique_ptr<vmm_region> m_workspace_alloc;

    // Device info and communication
    int device_id;
    int num_device_sms;
    uint64_t timeout_cycles = 0;
    int rank, rdma_rank, nvl_rank;
    int max_num_ranks;
    std::vector<int> remote_ranks; /* global ranks */
    // Host-side active rank state over max_num_ranks. This can differ from
    // the runtime device mask, which kernels may update on faults/timeouts.
    // Host state changes only through explicit control APIs.
    std::vector<bool> active_ranks;
    // Upper bound for active rank ids. Ranks may be sparse;
    // masked holes inside [0, active_rank_bound) are skipped by LL kernels.
    int active_rank_bound = 0;
    int num_rdma_ranks = 0, num_nvl_ranks = 0;
    int num_experts_per_rank = 0;
    cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

    // Stream for communication
    cudaStream_t comm_stream;

    // After synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    std::unique_ptr<NixlAgentInfo> nixl_agent_info;
    std::vector<NixlPeerInfo> nixl_peer_info;
    NixlPeerInfo my_peer_info;
    nixl_ep::gpu_nixl_ctx gpu_ctx;
    nixl_ep::gpu_nixl_ctx* gpu_ctx_ptr = nullptr;
    uint64_t* last_ht_barrier_counter = nullptr;
    uint64_t* local_ht_barrier_counter = nullptr;

    std::function<void()> pending_combine_recv_hook;
    std::function<void()> pending_dispatch_recv_hook;

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
    bool _is_rank_connected(int rank_id) const;
    void set_active_rank_bound(int bound);
    void _refresh_active_rank_bound();

    /* high-throughput mode private funcs */
    void _ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles);

public:
    Buffer(int rank, bool explicitly_destroy, bool low_latency_mode, int timeout_ms);

    void update_memory_buffers(int num_ranks, int num_experts_per_rank, int64_t num_rdma_bytes, int64_t num_nvl_bytes = 0);

    void connect_ranks(const std::vector<int>& remote_ranks_list, const std::optional<std::vector<nixl_blob_t>>& remote_mds = std::nullopt, const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles = {}, bool activate = true);

    void disconnect_ranks(const std::vector<int>& remote_ranks_list);

    void init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes);

    ~Buffer() noexcept;

    bool is_available() const;

    bool is_ht_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    int get_root_rdma_rank(bool global) const;

    int get_local_device_id() const;

    pybind11::bytearray get_local_ipc_handle() const;

    torch::stable::Tensor
    get_local_buffer_tensor(torch::headeronly::ScalarType dtype,
                            int64_t offset,
                            bool use_rdma_buffer) const;

    void destroy();

    std::tuple<torch::stable::Tensor,
               std::optional<torch::stable::Tensor>,
               torch::stable::Tensor,
               torch::stable::Tensor,
               int64_t>
    get_dispatch_layout(const torch::stable::Tensor &topk_idx,
                        int num_experts,
                        const EventHandle *previous_event,
                        bool async,
                        bool allocate_on_comm_stream);

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
    ht_dispatch(const torch::stable::Tensor &x,
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
                const Config &config,
                const EventHandle *previous_event,
                bool async,
                bool allocate_on_comm_stream);

    std::tuple<torch::stable::Tensor, std::optional<torch::stable::Tensor>, int64_t>
    ht_combine(const torch::stable::Tensor &x,
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
               const Config &config,
               const EventHandle *previous_event,
               bool async,
               bool allocate_on_comm_stream);

    std::tuple<torch::stable::Tensor,
               std::optional<torch::stable::Tensor>,
               torch::stable::Tensor,
               torch::stable::Tensor,
               torch::stable::Tensor,
               int64_t>
    dispatch(const torch::stable::Tensor &x,
             const torch::stable::Tensor &topk_idx,
             const std::optional<torch::stable::Tensor> &cumulative_local_expert_recv_stats,
             const std::optional<torch::stable::Tensor> &dispatch_wait_recv_cost_stats,
             int num_max_dispatch_tokens_per_rank,
             bool use_fp8,
             bool round_scale,
             bool use_ue8m0,
             bool async,
             bool return_recv_hook);

    std::tuple<torch::stable::Tensor, int64_t>
    combine(const torch::stable::Tensor &x,
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
            const std::optional<torch::stable::Tensor> &out);

    void
    run_dispatch_recv_hook();

    void
    run_combine_recv_hook();

    void barrier();

    torch::stable::Tensor
    get_next_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden) const;

    void update_mask_buffer(int rank_to_mask, bool mask);

    void
    query_mask_buffer(const torch::stable::Tensor &mask_status) const;

    void clean_mask_buffer();

    std::string get_local_metadata() const;
};

} // namespace nixl_ep
