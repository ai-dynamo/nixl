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

#include "nixl_ep.hpp"

#include "cuda_warn.hpp"
#include "kernels/api.cuh"

#include <pybind11/functional.h>

#include <ATen/cuda/CUDADataType.h>
#include <torch/python.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <unordered_set>

#define NIXL_ETCD_WATCH_TIMEOUT std::chrono::microseconds(1000000000) // 1000 seconds

namespace {

void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

uint64_t milliseconds_to_cycles(uint64_t milliseconds, int device_clock_rate_khz) {
    EP_HOST_ASSERT(device_clock_rate_khz > 0);
    return milliseconds * static_cast<uint64_t>(device_clock_rate_khz);
}

} // namespace

namespace nixl_ep {

void Buffer::update_memory_buffers(int num_ranks, int num_experts_per_rank, int64_t num_rdma_bytes)
{
    if (!available) {
        init(num_ranks, num_experts_per_rank, num_rdma_bytes);
        available = true;
    } else {
        throw std::runtime_error("Multiple calls to update_memory_buffers are not supported");
    }
}

Buffer::Buffer(int rank, bool explicitly_destroy, int timeout_ms):
        timeout_ms([timeout_ms] {
            EP_HOST_ASSERT(timeout_ms >= 0);
            return static_cast<uint64_t>(timeout_ms);
        }()),
        rank(rank),
        explicitly_destroy(explicitly_destroy) {}

bool Buffer::_is_rank_connected(int rank_id) const {
    return rank_id == rank or std::find(remote_ranks.begin(), remote_ranks.end(), rank_id) != remote_ranks.end();
}

void Buffer::set_active_rank_bound(int bound) {
    EP_HOST_ASSERT(bound > 0 && "active_rank_bound must be positive");
    active_rank_bound = bound;
}

void Buffer::_refresh_active_rank_bound() {
    int bound = 0;
    for (int rank_id = max_num_ranks - 1; rank_id >= 0; --rank_id) {
        if (active_ranks[rank_id]) {
            bound = rank_id + 1;
            break;
        }
    }
    set_active_rank_bound(bound);
}

int Buffer::get_rank_bound(std::optional<int> num_experts) const {
    if (!num_experts)
        return active_rank_bound;

    EP_HOST_ASSERT(*num_experts % num_experts_per_rank == 0);
    const int active_expert_bound = active_rank_bound * num_experts_per_rank;
    const int max_num_experts = max_num_ranks * num_experts_per_rank;
    EP_HOST_ASSERT(*num_experts >= active_expert_bound and *num_experts <= max_num_experts);
    return *num_experts / num_experts_per_rank;
}

void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_rdma_bytes)
{
    EP_HOST_ASSERT(num_ranks > 0);
    EP_HOST_ASSERT(num_experts_per_rank > 0);

    // Update buffer attributes
    this->max_num_ranks = num_ranks;
    this->num_experts_per_rank = num_experts_per_rank;
    this->num_rdma_bytes = num_rdma_bytes;

    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0);
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks);

    CUDA_CHECK(cudaGetDevice(&device_id));
    comm_stream = cuda_stream::get_from_pool();

    // Get device info
    int device_clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_device_sms, cudaDevAttrMultiProcessorCount, device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&device_clock_rate_khz, cudaDevAttrClockRate, device_id));
    timeout_cycles = milliseconds_to_cycles(timeout_ms, device_clock_rate_khz);
    int denom_sms = std::max(1, num_device_sms / 2);
    auto per_channel_bytes = ceil_div<int64_t>(num_rdma_bytes, denom_sms);
    EP_HOST_ASSERT(per_channel_bytes < std::numeric_limits<int>::max());

    // Create 32 MiB workspace
    m_workspace_alloc = std::make_unique<vmm_region>(NUM_WORKSPACE_BYTES);
    workspace = m_workspace_alloc->ptr();
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    m_rdma_alloc = std::make_unique<vmm_region>(static_cast<size_t>(num_rdma_bytes));
    rdma_buffer_ptr = m_rdma_alloc->ptr();
    CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

    // Allocate and clean shrink buffer
    int num_mask_buffer_bytes = max_num_ranks * sizeof(int);
    m_mask_alloc = std::make_unique<vmm_region>(static_cast<size_t>(num_mask_buffer_bytes));
    mask_buffer_ptr = static_cast<int *>(m_mask_alloc->ptr());
    CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0xff, num_mask_buffer_bytes));
    CUDA_CHECK(cudaMemset(mask_buffer_ptr + rank, 0, sizeof(int)));
    active_ranks.assign(max_num_ranks, false);
    active_ranks[rank] = true;
    set_active_rank_bound(rank + 1);

    int num_sync_buffer_bytes = max_num_ranks * sizeof(int);
    m_sync_alloc = std::make_unique<vmm_region>(static_cast<size_t>(num_sync_buffer_bytes));
    sync_buffer_ptr = static_cast<int *>(m_sync_alloc->ptr());
    m_sync_count_alloc = std::make_unique<vmm_region>(static_cast<size_t>(num_sync_buffer_bytes));
    sync_count_ptr = static_cast<int *>(m_sync_count_alloc->ptr());
    CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
    CUDA_CHECK(cudaMemset(sync_count_ptr, 0, num_sync_buffer_bytes));

    CUDA_CHECK(cudaDeviceSynchronize());

    my_peer_info.rdma_buffer_ptr = rdma_buffer_ptr;
    my_peer_info.device_id = get_local_device_id();
    my_peer_info.sync_buffer_ptr = sync_buffer_ptr;
    my_peer_info.rank = rank;

    nixl_peer_info.resize(max_num_ranks);
    nixl_peer_info[rank] = my_peer_info;

    _nixl_agent_init();

    _nixl_ep_init();
}

Buffer::~Buffer() noexcept {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before NIXL_EP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(rdma_buffer_ptr) + offset;
    return torch::from_blob(base_ptr, num_rdma_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

int64_t Buffer::get_comm_stream() const {
    EP_HOST_ASSERT(is_available() && "get_comm_stream() called before update_memory_buffers()");
    return reinterpret_cast<int64_t>(comm_stream);
}

void Buffer::destroy() {
    auto warn_cuda = [](cudaError_t status, const char *operation) noexcept {
        cuda::warn(status, "Buffer::destroy()", operation);
    };

    auto warn_nixl = [](nixl_status_t status, const char* operation) noexcept {
        if (status != NIXL_SUCCESS) {
            std::cerr << "WARNING: destroy() failed to " << operation
                      << ": " << nixlEnumStrings::statusStr(status) << '\n';
        }
    };

    if (destroyed) {
        return;
    }

    // Synchronize
    warn_cuda(cudaDeviceSynchronize(), "synchronize device");

    _nixl_ep_destroy();

    if (nixl_agent_info and nixl_agent_info->agent != nullptr) {
        if (getenv("NIXL_ETCD_ENDPOINTS")) {
            warn_nixl(nixl_agent_info->agent->invalidateLocalMD(),
                      "invalidate local metadata");
        }
        warn_nixl(nixl_agent_info->agent->deregisterMem(
                      nixl_agent_info->rdma_reg_descs,
                      &nixl_agent_info->extra_params),
                  "deregister RDMA memory");
        warn_nixl(nixl_agent_info->agent->deregisterMem(
                      nixl_agent_info->sync_reg_descs,
                      &nixl_agent_info->extra_params),
                  "deregister sync memory");
        warn_nixl(nixl_agent_info->agent->deregisterMem(
                      nixl_agent_info->sync_count_reg_descs,
                      &nixl_agent_info->extra_params),
                  "deregister sync-count memory");

        nixl_agent_info.reset();
    }

    m_rdma_alloc.reset();
    rdma_buffer_ptr = nullptr;
    m_mask_alloc.reset();
    mask_buffer_ptr = nullptr;
    m_sync_alloc.reset();
    sync_buffer_ptr = nullptr;
    m_sync_count_alloc.reset();
    sync_count_ptr = nullptr;

    m_workspace_alloc.reset();
    workspace = nullptr;

    destroyed = true;
    available = false;
}

void Buffer::barrier() {
    ep_kernels::barrier(gpu_ctx_ptr, mask_buffer_ptr, timeout_cycles, cuda_stream::get_current());
}

void Buffer::_nixl_agents_connect(const std::vector<int>& ranks, const std::vector<nixl_blob_t>& remote_mds) {
    EP_HOST_ASSERT(!ranks.empty());
    EP_HOST_ASSERT(remote_mds.empty() || remote_mds.size() == ranks.size());

    // Assuming ranks vector does not include current rank and has only new ranks
    remote_ranks.insert(remote_ranks.end(), ranks.begin(), ranks.end());
    for (int remote_rank : ranks) {
        nixl_agent_info->remote_agent_names[remote_rank] = std::to_string(remote_rank);
    }

    // Fire all get metadata requests in parallel
    for (size_t i = 0; i < ranks.size(); i++) {
        int remote_rank = ranks[i];
        std::string agent_name;

        nixl_status_t status = remote_mds.empty()
            ? nixl_agent_info->agent->fetchRemoteMD(nixl_agent_info->remote_agent_names[remote_rank])
            : nixl_agent_info->agent->loadRemoteMD(remote_mds[i], agent_name);

        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to get metadata for remote agent " +
                                    std::to_string(remote_rank) + ", status: " + std::to_string(status));
        }
    }

    // Wait for all remote metadata to be available
    std::vector<bool> peer_ready(max_num_ranks, false);
    int peers_remaining = static_cast<int>(ranks.size());

    while (peers_remaining > 0) {
        for (int remote_rank : ranks) {
            if (peer_ready[remote_rank]) continue;

            nixl_xfer_dlist_t empty_descs(VRAM_SEG);
            if (nixl_agent_info->agent->checkRemoteMD(std::to_string(remote_rank), empty_descs) == NIXL_SUCCESS) {
                peer_ready[remote_rank] = true;
                peers_remaining--;
            }
        }
        if (peers_remaining > 0) {
            sleep_ms(10);
        }
    }
}

void Buffer::_nixl_agents_peer_info_gather(std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        std::string my_peer_info_str(reinterpret_cast<const char*>(&my_peer_info), sizeof(NixlPeerInfo));
        nixl_agent_info->agent->genNotif(std::to_string(remote_rank), my_peer_info_str);
    }

    for (int remote_rank : ranks) {
        do {
            nixl_notifs_t notif_map;
            nixl_agent_info->agent->getNotifs(notif_map);
            for (auto &notif : notif_map) {
                std::string my_peer_info_str = notif.second[0];
                NixlPeerInfo remote_peer_info;
                memcpy(&remote_peer_info, my_peer_info_str.c_str(), sizeof(NixlPeerInfo));
                nixl_peer_info[remote_peer_info.rank] = remote_peer_info;
                nixl_agent_info->wire_up_done[remote_peer_info.rank] = true;
            }
        } while (!nixl_agent_info->wire_up_done[remote_rank]);
    }
}

void Buffer::connect_ranks(const std::vector<int>& remote_ranks_list, const std::optional<std::vector<nixl_blob_t>>& remote_mds, bool activate) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(!remote_mds.has_value() || remote_mds->size() == remote_ranks_list.size());

    std::vector<int> new_ranks;
    std::vector<nixl_blob_t> new_ranks_mds;

    for (size_t i = 0; i < remote_ranks_list.size(); i++) {
        int remote_rank = remote_ranks_list[i];
        EP_HOST_ASSERT(remote_rank >= 0 and remote_rank < max_num_ranks);
        // Skip self and ranks we are already connected to
        if (remote_rank == rank or _is_rank_connected(remote_rank))
            continue;

        new_ranks.push_back(remote_rank);
        CUDA_CHECK(cudaMemset(sync_count_ptr + remote_rank, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(sync_buffer_ptr + remote_rank, 0, sizeof(int)));

        if (remote_mds.has_value())
            new_ranks_mds.push_back((*remote_mds)[i]);
    }

    if (!new_ranks.empty()) {
        pybind11::gil_scoped_release release;
        _nixl_agents_connect(new_ranks, new_ranks_mds);

        _nixl_agents_peer_info_gather(new_ranks);

        _nixl_ep_memory_views_stage();
    }

    if (activate) {
        for (int remote_rank : remote_ranks_list) {
            if (remote_rank != rank)
                update_mask_buffer(remote_rank, false);
        }
    }

    // Ready to use
    available = true;
}

void Buffer::disconnect_ranks(const std::vector<int>& remote_ranks_list) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(remote_ranks_list.size() <= remote_ranks.size());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update mask buffer to mark ranks as inactive
    for (int removed_rank : remote_ranks_list) {
        EP_HOST_ASSERT(removed_rank != rank);
        EP_HOST_ASSERT(_is_rank_connected(removed_rank));
        update_mask_buffer(removed_rank, true);  // mask=true
        CUDA_CHECK(cudaMemset(gpu_ctx.p2p_ptrs + removed_rank, 0, sizeof(void*)));
    }

    _nixl_agents_peer_info_cleanup(remote_ranks_list);

    _nixl_agents_disconnect(remote_ranks_list);

    // Remove ranks from remote_ranks vector (arbitrary order)
    for (int removed_rank : remote_ranks_list) {
        remote_ranks.erase(
            std::remove(remote_ranks.begin(), remote_ranks.end(), removed_rank),
            remote_ranks.end()
        );
    }

    _nixl_ep_memory_views_stage();

    _nixl_ep_memory_views_commit();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank, std::optional<int> num_experts,
                             bool use_fp8, bool round_scale, bool use_ue8m0,
                             bool async, bool return_recv_hook) {
    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    const int rank_bound = get_rank_bound(num_experts);
    const int num_max_recv_tokens = rank_bound * num_max_dispatch_tokens_per_rank;
    EP_HOST_ASSERT(num_max_recv_tokens % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts_per_rank);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == rank_bound);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));

    // Buffer control
    int max_num_experts = max_num_ranks * num_experts_per_rank;
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[buffer_idx];
    auto next_buffer = layout.buffers[buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    cudaStream_t compute_stream = cuda_stream::get_current();
    cudaStream_t launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_experts_per_rank, num_max_recv_tokens, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn: torch::kBFloat16));
    auto packed_recv_src_info = torch::empty({num_experts_per_rank, num_max_recv_tokens}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_experts_per_rank, rank_bound}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_experts_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_experts_per_rank, hidden / 128, num_max_recv_tokens},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_experts_per_rank, hidden / 512, num_max_recv_tokens},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=, this](int phases) {
        ep_kernels::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales_ptr,
                               packed_recv_src_info.data_ptr<int>(), packed_recv_layout_range.data_ptr<int64_t>(),
                               packed_recv_count.data_ptr<int>(),
                               mask_buffer_ptr,
                               cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                               dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                               buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                               buffer.dispatch_rdma_send_buffer,
                              x.data_ptr(), topk_idx.data_ptr<topk_idx_t>(),
                               next_clean_meta.first, next_clean_meta.second,
                               num_tokens, hidden, num_max_dispatch_tokens_per_rank,
                               num_topk, rank_bound, num_experts_per_rank, rank,
                               use_fp8, round_scale, use_ue8m0,
                               timeout_cycles,
                               workspace, num_device_sms,
                               launch_stream, phases, gpu_ctx_ptr);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(EP_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                            const torch::Tensor& src_info, const torch::Tensor& layout_range,
                            const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                            int num_max_dispatch_tokens_per_rank,
                            bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                            const std::optional<torch::Tensor>& out) {
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts_per_rank);
    const int rank_bound = layout_range.size(1);
    EP_HOST_ASSERT(rank_bound >= active_rank_bound and rank_bound <= max_num_ranks);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts_per_rank);
    EP_HOST_ASSERT(x.size(1) == rank_bound * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(src_info.size(1) == rank_bound * num_max_dispatch_tokens_per_rank);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == rank_bound);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    int max_num_experts = max_num_ranks * num_experts_per_rank;
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[buffer_idx];
    auto next_buffer = layout.buffers[buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    cudaStream_t compute_stream = cuda_stream::get_current();
    cudaStream_t launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=, this](int phases) {
        ep_kernels::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer, buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(), topk_idx.data_ptr<topk_idx_t>(), topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(), layout_range.data_ptr<int64_t>(),
                              mask_buffer_ptr,
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first, next_clean_meta.second,
                              num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
                              num_topk, rank_bound, num_experts_per_rank, rank,
                             use_logfmt, timeout_cycles,
                              workspace, num_device_sms,
                              launch_stream, phases, zero_copy, gpu_ctx_ptr);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(EP_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
}

torch::Tensor
Buffer::get_next_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden,
                                int rank_bound) const {
    EP_HOST_ASSERT(rank_bound >= active_rank_bound and rank_bound <= max_num_ranks);
    int max_num_experts = max_num_ranks * num_experts_per_rank;
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts);

    auto buffer = layout.buffers[buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts_per_rank, rank_bound * num_max_dispatch_tokens_per_rank, hidden},
                            {rank_bound * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < max_num_ranks);
    EP_HOST_ASSERT((rank_to_mask != rank or !mask) && "cannot mask the local rank");
    if (!mask)
        EP_HOST_ASSERT(_is_rank_connected(rank_to_mask) && "cannot unmask an unconnected rank");
    _nixl_ep_memory_views_commit();
    active_ranks[rank_to_mask] = !mask;
    _refresh_active_rank_bound();
    ep_kernels::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, cuda_stream::get_current());
}

void
Buffer::query_mask_buffer(const torch::Tensor &mask_status) const {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == max_num_ranks &&
                   mask_status.scalar_type() == torch::kInt32);

    ep_kernels::query_mask_buffer(mask_buffer_ptr,
                                  max_num_ranks,
                                  mask_status.data_ptr<int>(),
                                  cuda_stream::get_current());
}

void Buffer::clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    _nixl_ep_memory_views_commit();
    std::vector<int> mask(max_num_ranks, 1);
    for (int rank_id = 0; rank_id < max_num_ranks; ++rank_id) {
        const bool active = _is_rank_connected(rank_id);
        active_ranks[rank_id] = active;
        mask[rank_id] = !active;
    }
    _refresh_active_rank_bound();
    CUDA_CHECK(cudaMemcpyAsync(mask_buffer_ptr, mask.data(),
                               max_num_ranks * sizeof(int), cudaMemcpyHostToDevice,
                               cuda_stream::get_current()));
}

std::string Buffer::get_local_metadata() const {
    EP_HOST_ASSERT(nixl_agent_info != nullptr && nixl_agent_info->agent != nullptr);
    nixl_blob_t metadata_blob;
    nixl_status_t status = nixl_agent_info->agent->getLocalMD(metadata_blob);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get local metadata, status: " + std::to_string(status));
    }
    return metadata_blob;
}

void Buffer::_nixl_ep_memory_views_stage(void) {
    // TODO: Separate the local memory view so peer updates only rebuild remote views.
    NixlMemoryViews& memory_views = staged_memory_views;
    _nixl_ep_memory_views_destroy(memory_views);
    nixl_remote_dlist_t remote_descs(VRAM_SEG);
    nixl_remote_dlist_t barrier_descs(VRAM_SEG);
    nixl_local_dlist_t local_descs(VRAM_SEG);

    local_descs.addDesc(nixlBlobDesc((uintptr_t)(rdma_buffer_ptr), num_rdma_bytes, get_local_device_id(), ""));
    local_descs.addDesc(nixlBlobDesc((uintptr_t)(sync_count_ptr), max_num_ranks * sizeof(int), get_local_device_id(), ""));

    std::unordered_set<int> remote_set(remote_ranks.begin(), remote_ranks.end());
    for (int r = 0; r < max_num_ranks; r++) {
        std::string remote_agent_name = remote_set.count(r) ? nixl_agent_info->remote_agent_names[r] : nixl_null_agent;
        remote_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].rdma_buffer_ptr, num_rdma_bytes, nixl_peer_info[r].device_id, remote_agent_name));
        barrier_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].sync_buffer_ptr, max_num_ranks * sizeof(int), nixl_peer_info[r].device_id, remote_agent_name));
    }

    EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(local_descs, memory_views.local, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    if (!remote_ranks.empty()) {
        EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(remote_descs, memory_views.remote, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
        EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(barrier_descs, memory_views.barrier, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    }
}

void Buffer::_nixl_ep_memory_views_destroy(NixlMemoryViews& memory_views) {
    if (memory_views.local) nixl_agent_info->agent->releaseMemView(memory_views.local);
    if (memory_views.remote) nixl_agent_info->agent->releaseMemView(memory_views.remote);
    if (memory_views.barrier) nixl_agent_info->agent->releaseMemView(memory_views.barrier);
    memory_views = {};
}

void Buffer::_nixl_ep_memory_views_commit(void) {
    if (!staged_memory_views.local)
        return;
    // Wait for all kernels using the active memory views to finish.
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(active_memory_views, staged_memory_views);
    gpu_ctx.local_mvh = active_memory_views.local;
    gpu_ctx.remote_mvh = active_memory_views.remote;
    gpu_ctx.barrier_mvh = active_memory_views.barrier;
    CUDA_CHECK(cudaMemcpyAsync(gpu_ctx_ptr, &gpu_ctx, sizeof(gpu_ctx),
                               cudaMemcpyHostToDevice, comm_stream));
    for (int remote_rank : remote_ranks)
        ep_kernels::cache_p2p_ptr(gpu_ctx_ptr, remote_rank, comm_stream);
    // Wait for the new GPU context and cached P2P pointers to be ready.
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    _nixl_ep_memory_views_destroy(staged_memory_views);
}

void Buffer::_nixl_ep_init(void) {
    gpu_ctx = {
        .sync_buffer_ptr = sync_buffer_ptr,
        .sync_count_ptr = sync_count_ptr,
        .rdma_buffer_ptr = rdma_buffer_ptr,
        .p2p_ptrs = nullptr,
        .max_num_ranks = max_num_ranks,
        .rank = rank,
    };
    CUDA_CHECK(cudaMalloc(&gpu_ctx.p2p_ptrs, max_num_ranks * sizeof(void*)));
    CUDA_CHECK(cudaMemset(gpu_ctx.p2p_ptrs, 0, max_num_ranks * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx_ptr, sizeof(gpu_nixl_ctx)));
    CUDA_CHECK(cudaMemcpy(gpu_ctx_ptr, &gpu_ctx, sizeof(gpu_ctx), cudaMemcpyHostToDevice));
}

void Buffer::_nixl_ep_destroy(void) {
    _nixl_ep_memory_views_destroy(staged_memory_views);
    _nixl_ep_memory_views_destroy(active_memory_views);
    if (gpu_ctx.p2p_ptrs != nullptr) {
        cudaFree(gpu_ctx.p2p_ptrs);
        gpu_ctx.p2p_ptrs = nullptr;
    }
    if (gpu_ctx_ptr != nullptr) {
        cudaFree(gpu_ctx_ptr);
        gpu_ctx_ptr = nullptr;
    }
}

void Buffer::_nixl_agent_init() {
    std::string agent_name = std::to_string(rank);

    // Device API mode. NIXL itself has no such switch - the proxy is enabled
    // by UCX backend parameters below - so this is purely an EP-level knob for
    // running the same binary either way.
    bool enable_device_proxy = false;
    if (const char *mode_override = std::getenv("NIXL_EP_DEVICE_MODE")) {
        const std::string mode(mode_override);
        if (mode == "proxy") {
            enable_device_proxy = true;
        } else if (mode != "direct") {
            throw std::runtime_error(
                "NIXL_EP_DEVICE_MODE must be exactly 'direct' or 'proxy'; got '" + mode + "'");
        }
    }

    const char *proxy_channels_env = std::getenv("NIXL_EP_PROXY_CHANNELS");
    if (proxy_channels_env == nullptr) {
        proxy_channels_env = std::getenv("NIXL_EP_NUM_CHANNELS");
    }
    const uint32_t proxy_channels = proxy_channels_env ? std::stoul(proxy_channels_env) : 4;

    const char *proxy_workers_env = std::getenv("NIXL_EP_PROXY_WORKER_COUNT");
    const uint32_t proxy_workers = proxy_workers_env ? std::stoul(proxy_workers_env) :
                                                       proxy_channels;

    nixlAgentConfig cfg;
    // The proxy's own threads drive UCX progress, one per ring, so a progress
    // thread would only contend with them.
    cfg.useProgThread = !enable_device_proxy;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    cfg.etcdWatchTimeout = NIXL_ETCD_WATCH_TIMEOUT;
    auto agent = std::make_shared<nixlAgent>(agent_name, cfg);

    // Create UCX backend
    nixl_mem_list_t mems;
    nixl_b_params_t init_params;

    nixl_status_t status = agent->getPluginParams("UCX", mems, init_params);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get UCX plugin parameters for agent " + agent_name +
                                ", status: " + std::to_string(status));
    }

    init_params["ucx_error_handling_mode"] = "none";
    if (enable_device_proxy) {
        init_params["device_proxy"] = "true";
        init_params["proxy_channel_count"] = std::to_string(proxy_channels);
        init_params["proxy_thread_count"] = std::to_string(proxy_workers);
        init_params["proxy_max_peers"] = std::to_string(max_num_ranks);
        // The engine derives num_workers from channels x peers and rejects an
        // explicit value that disagrees. getPluginParams() seeded the plugin
        // default of 1, so it has to be erased rather than merely left unset.
        init_params.erase("num_workers");
        init_params["ucx_num_device_channels"] = "1";
        init_params["engine_config"] = "FENCE_MODE=ep_based";
    } else {
        const char *num_channels_env = std::getenv("NIXL_EP_NUM_CHANNELS");
        init_params["ucx_num_device_channels"] = num_channels_env ? num_channels_env : "4";
        init_params["num_workers"] = std::to_string(1);
    }

    nixlBackendH* ucx_backend = nullptr;
    status = agent->createBackend("UCX", init_params, ucx_backend);
    if (status != NIXL_SUCCESS || !ucx_backend) {
        throw std::runtime_error("Failed to create UCX backend for agent " + agent_name +
                                ", status: " + std::to_string(status));
    }

    nixl_agent_info = std::make_unique<NixlAgentInfo>(agent, ucx_backend, max_num_ranks);
    nixl_agent_info->extra_params.backends.push_back(ucx_backend);
    nixl_agent_info->agent_name = agent_name;

    nixl_agent_info->rdma_reg_descs.clear();
    nixl_agent_info->rdma_reg_descs.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(rdma_buffer_ptr), num_rdma_bytes, device_id, ""));

    nixl_agent_info->sync_reg_descs.clear();
    nixl_agent_info->sync_reg_descs.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(sync_buffer_ptr), max_num_ranks * sizeof(int), device_id, ""));

    nixl_agent_info->sync_count_reg_descs.clear();
    nixl_agent_info->sync_count_reg_descs.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(sync_count_ptr), max_num_ranks * sizeof(int), device_id, ""));

    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->rdma_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->sync_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->sync_count_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);

    if (getenv("NIXL_ETCD_ENDPOINTS")) {
        status = nixl_agent_info->agent->sendLocalMD();
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to send local metadata for agent " +
                                    nixl_agent_info->agent_name + ", status: " + std::to_string(status));
        }
    }
}

void Buffer::_nixl_agents_disconnect(const std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        EP_HOST_ASSERT(remote_rank != rank);
        EP_HOST_ASSERT(remote_rank >= 0 and remote_rank < max_num_ranks);
        nixl_xfer_dlist_t empty_descs(VRAM_SEG);
        if(nixl_agent_info->agent->checkRemoteMD(nixl_agent_info->remote_agent_names[remote_rank], empty_descs) == NIXL_SUCCESS) {
            nixl_status_t status = nixl_agent_info->agent->invalidateRemoteMD(nixl_agent_info->remote_agent_names[remote_rank]);
            // NIXL watchers might invalidate peer metadata, so we ignore NIXL_ERR_NOT_FOUND errors
            if (status != NIXL_SUCCESS && status != NIXL_ERR_NOT_FOUND) {
                printf("WARNING: rank %d Failed to invalidate remote rank %d metadata for agent %s, status: %d\n",
                    rank, remote_rank, std::to_string(remote_rank).c_str(), status); fflush(stdout);
            }
        }
    }
}

void Buffer::_nixl_agents_peer_info_cleanup(const std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        nixl_agent_info->wire_up_done[remote_rank] = false;
        // Clear nixl_peer_info for removed ranks (only do this once per rank, on first channel)
        nixl_peer_info[remote_rank] = NixlPeerInfo{};
    }
}

static std::optional<std::vector<nixl_blob_t>> convert_mds(const std::optional<std::vector<pybind11::bytes>>& remote_mds) {
    if (!remote_mds.has_value()) {
        return std::nullopt;
    }
    std::vector<nixl_blob_t> md_blobs;
    md_blobs.reserve(remote_mds->size());
    for (const auto& md_bytes : *remote_mds) {
        md_blobs.push_back(nixl_blob_t(md_bytes));
    }
    return md_blobs;
}

} // namespace nixl_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NIXL_EP: an efficient expert-parallel communication library";
    m.def("get_low_latency_buffer_size_hint", &nixl_ep::get_low_latency_buffer_size_hint);

    pybind11::class_<nixl_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &nixl_ep::EventHandle::current_stream_wait);

    pybind11::class_<nixl_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, bool, int>())
        .def("update_memory_buffers", &nixl_ep::Buffer::update_memory_buffers)
        .def("barrier", &nixl_ep::Buffer::barrier)
        .def("connect_ranks", [](nixl_ep::Buffer &buffer, const std::vector<int>& remote_ranks, const std::optional<std::vector<pybind11::bytes>>& remote_mds, bool activate) {
            buffer.connect_ranks(remote_ranks, nixl_ep::convert_mds(remote_mds), activate);
        }, py::arg("remote_ranks"), py::arg("remote_mds") = std::nullopt, py::arg("activate") = true)
        .def("disconnect_ranks", &nixl_ep::Buffer::disconnect_ranks)
        .def("is_available", &nixl_ep::Buffer::is_available)
        .def("get_local_device_id", &nixl_ep::Buffer::get_local_device_id)
        .def("get_local_buffer_tensor", &nixl_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &nixl_ep::Buffer::get_comm_stream)
        .def("destroy", &nixl_ep::Buffer::destroy)
        .def("dispatch", &nixl_ep::Buffer::dispatch)
        .def("combine", &nixl_ep::Buffer::combine)
        .def("update_mask_buffer", &nixl_ep::Buffer::update_mask_buffer)
        .def("query_mask_buffer", &nixl_ep::Buffer::query_mask_buffer)
        .def("clean_mask_buffer", &nixl_ep::Buffer::clean_mask_buffer)
        .def("get_next_combine_buffer", &nixl_ep::Buffer::get_next_combine_buffer)
        .def("get_local_metadata", [](const nixl_ep::Buffer &buffer) -> pybind11::bytes {
            return pybind11::bytes(buffer.get_local_metadata());
        });
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<nixl_ep::topk_idx_t>::value);
    m.def("is_sm90_compiled", nixl_ep::is_sm90_compiled);
}
