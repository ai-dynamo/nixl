// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

#include <nixl_ep_v2/common/topology.cuh>

#include "../kernels/backend/nixl.hpp"
#include "../utils/event.hpp"

namespace nixl::runtime::ep::elastic {

class buffer {
public:
    // `(region base, device, world rank, registered bytes)`
    using peer_info_t = std::tuple<uint64_t, int, int, uint64_t>;

    buffer(const int64_t& num_buffer_bytes,
           const bool& allow_multiple_reduction,
           const bool& use_fp8_dispatch,
           const bool& prefer_overlap_with_compute,
           const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
           const bool& explicitly_destroy,
           const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
           const int& world_rank,
           const int& num_nixl_channels, const std::string& agent_name_prefix,
           const int& num_experts_per_rank, const int& num_launch_sms);

    ~buffer() noexcept(false);

    void destroy();

    torch::Stream get_comm_stream() const;

    // ---- Topology descriptor (`EPTopology`) ----
    // Validate the new world before activating prepared views and publishing it to the device.

    // Python-facing: the position mask as 64-bit words, least significant first (missing words are zero)
    void update_topology_masks(const int& mode, const uint32_t& scaleout_mask, const std::vector<uint64_t>& scaleup_mask_words,
                               const std::vector<int>& disconnect_ranks);

    int get_mode() const;

    const topology::EPTopology* get_topology_ptr() const;

    std::vector<int> get_active_ranks() const;

    std::vector<int> get_rank_mapping() const;

    // The masks of the active domains and positions of the committed topology descriptor (`EPTopology`), the position mask as its 64-bit words
    std::tuple<uint32_t, std::vector<uint64_t>> get_topology_masks() const;

    // ---- Peers ----
    // Join: `connect_ranks`, then `set_active_ranks`. Leave: `disconnect_ranks` excludes and retires peers.
    std::string get_local_agent_name() const;

    std::string get_local_metadata() const;

    peer_info_t get_local_peer_info() const;

    void connect_ranks(const std::vector<int>& world_ranks, const std::vector<std::string>& remote_agent_names,
                       const std::vector<peer_info_t>& peer_infos, const std::vector<int>& world_to_slot,
                       const std::vector<std::string>& remote_metadata, const std::string& connection_id,
                       const std::vector<std::string>& remote_connection_ids);

    std::vector<int> get_connected_ranks() const;

    // One isolated pair or graph replay between reads; synchronization and reset are diagnostic-only.
    std::tuple<torch::Tensor, std::vector<std::string>> read_profile() const;

    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync) const;

    torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                    const bool& allocate_on_comm_stream,
                                                    c10::cuda::OptionalCUDAStreamGuard& stream_guard) const;

    void stream_control_before_epilogue(const std::optional<EventHandle>& previous_event_before_epilogue) const;

    std::optional<EventHandle> stream_control_epilogue(const std::vector<std::optional<torch::Tensor>>& tensors,
                                                       const at::cuda::CUDAStream& compute_stream,
                                                       const bool& allocate_on_comm_stream,
                                                       const bool& async_with_compute_stream) const;

    static std::tuple<int64_t, int64_t> get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& num_hidden_bytes, const int& num_sf_packs, const int& num_topk,
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks);

    static std::tuple<int64_t, int64_t> get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const bool& allow_multiple_reduction);

    static int get_max_num_fp8_sf_packs(const int& hidden);

    // Dispatch and combine share RECEIVE at the start and SEND at the end, each sized for the larger requirement.
    // A declared one-domain maximum needs only scale-up RECEIVE storage.
    static std::tuple<int64_t, int64_t> get_buffer_arena_sizes(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                                               const bool& allow_multiple_reduction, const bool& use_fp8_dispatch,
                                                               const int& num_scaleout_ranks, const int& num_scaleup_ranks);

    int64_t get_send_buffer_offset(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk) const;

    // Buffer size for a declared maximum world (the committed extent's sub-layouts are prefixes of the maximum ones)
    static int64_t calculate_buffer_size(const int& max_num_scaleout_ranks, const int& max_num_scaleup_ranks,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk,
                                         const bool& use_fp8_dispatch,
                                         const bool& allow_multiple_reduction);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               int,
               std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor,
               torch::Tensor, torch::Tensor,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const int& num_max_tokens_per_rank,
             const int& expert_alignment,
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf) const;

    std::tuple<torch::Tensor, std::optional<EventHandle>>
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const torch::Tensor& token_metadata_at_forward,
            const torch::Tensor& channel_linked_list,
            const bool& recv_count_is_exact,
            const int& num_max_tokens_per_rank,
            const std::optional<EventHandle>& previous_event,
            const std::optional<EventHandle>& previous_event_before_epilogue,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout) const;

    // The 2 MB unit of the workspace, the GPU buffer and `calculate_buffer_size`
    static constexpr int64_t kNumAlignmentBytes = 2097152;

private:
    topology::EPTopology make_topology(const int& mode, const uint32_t& scaleout_mask, const topology::ScaleupMask& scaleup_mask) const;

    void publish_topology(const topology::EPTopology& next, const at::cuda::CUDAStream& stream);

    // The committed topology descriptor of an active rank: `dispatch` / `combine` refuse a rank outside the committed world
    const topology::EPTopology& active_rank_topology() const;

    // Buffer bytes (excludes workspace)
    // Memory layout: [[Workspace] GPU buffer]
    int64_t num_buffer_bytes;
    void* buffer_ptr;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // Workspace
    // Control words reset at rest except combine readiness; barrier counters retain their epochs.
    void *workspace;
    void* host_workspace;
    torch::Tensor profile_storage;

    // CUDA streams
    at::cuda::CUDAStream comm_stream;

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;

    // Fixed dispatch capacity; both operations derive their source arena offsets from it.
    const bool use_fp8_dispatch;

    // Whether to prefer overlapping communication with compute
    bool prefer_overlap_with_compute;

    // Timeout settings
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;

    // The transport: this rank's NIXL agent, its registered region (workspace + GPU buffer) and the peers' views
    std::shared_ptr<nixl::NIXLEPContext> nixl_context;

    // Capacity bounds; caller-assigned world ranks are independent of topology coordinates.
    int max_num_scaleout_ranks = 0, max_num_scaleup_ranks = 0, max_num_ranks = 0;
    // Physical routing slots per rank; independent of the model's logical expert count.
    int num_experts_per_rank = 0;
    // Fixed number of participating main-kernel blocks for this buffer.
    const int num_launch_sms;
    // Adaptive kernels: device-side topology descriptor (+ host mirror)
    torch::Tensor topology_tensor, rank_mapping_tensor;
    std::vector<int> committed_rank_maps;  // world-to-slot followed by slot-to-world
    topology::EPTopology topology_host;
};

void register_apis(pybind11::module_& m);

}  // namespace nixl::runtime::ep::elastic
