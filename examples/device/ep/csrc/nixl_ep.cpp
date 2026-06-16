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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <pybind11/functional.h>

#include "nixl_ep.hpp"
#include "kernels/api.cuh"
#include "kernels/configs.cuh"
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include "cuda_warn.hpp"
#include "kernels/exception.cuh"
#include "nixl.h"
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <sstream>
#include <unordered_set>

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#define NIXL_ETCD_WATCH_TIMEOUT std::chrono::microseconds(1000000000) // 1000 seconds

namespace {
void
sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

uint64_t
milliseconds_to_cycles(uint64_t milliseconds, int device_clock_rate_khz) {
    EP_HOST_ASSERT(device_clock_rate_khz > 0);
    return milliseconds * static_cast<uint64_t>(device_clock_rate_khz);
}

int64_t
scalar_type_element_size(torch::headeronly::ScalarType dtype) {
#define CASE_ELEMENT_SIZE(ctype, name)        \
    case torch::headeronly::ScalarType::name: \
        return static_cast<int64_t>(sizeof(ctype));

    switch (dtype) {
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENT_SIZE)
    default:
        STD_TORCH_CHECK(false, "Unknown ScalarType");
    }
#undef CASE_ELEMENT_SIZE
}

constexpr torch::headeronly::ScalarType
topk_idx_scalar_type() {
    EP_STATIC_ASSERT(TOPK_IDX_BITS == 64, "stable ABI combine supports 64-bit topk indices only");
    return torch::headeronly::ScalarType::Long;
}
} // namespace

namespace nixl_ep {

void Buffer::update_memory_buffers(int num_ranks, int num_experts_per_rank, int64_t num_rdma_bytes, int64_t num_nvl_bytes)
{
    if (!available) {
        init(num_ranks, num_experts_per_rank, num_nvl_bytes, num_rdma_bytes);
        available = true;
    } else {
        throw std::runtime_error("Multiple calls to update_memory_buffers are not supported");
    }
}

Buffer::Buffer(int rank, bool explicitly_destroy, bool low_latency_mode, int timeout_ms):
        low_latency_mode(low_latency_mode),
        timeout_ms([timeout_ms] {
            EP_HOST_ASSERT(timeout_ms >= 0);
            return static_cast<uint64_t>(timeout_ms);
        }()),
        rank(rank),
        explicitly_destroy(explicitly_destroy),
        comm_stream(cuda_stream::getFromPool()) {}

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

void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes)
{
    EP_HOST_ASSERT(num_ranks > 0);
    EP_HOST_ASSERT(num_experts_per_rank > 0);

    // Update buffer attributes
    this->max_num_ranks = num_ranks;
    this->num_experts_per_rank = num_experts_per_rank;
    this->num_nvl_bytes = num_nvl_bytes;
    this->num_rdma_bytes = num_rdma_bytes;

    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    // Get device info
    int device_clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_device_sms, cudaDevAttrMultiProcessorCount, device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&device_clock_rate_khz, cudaDevAttrClockRate, device_id));
    timeout_cycles = milliseconds_to_cycles(timeout_ms, device_clock_rate_khz);
    int denom_sms = std::max(1, num_device_sms / 2);
    auto per_channel_bytes = ceil_div<int64_t>(num_rdma_bytes, denom_sms);
    EP_HOST_ASSERT(per_channel_bytes < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));
        CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    m_workspace_alloc = std::make_unique<vmm_region>(NUM_WORKSPACE_BYTES);
    workspace = m_workspace_alloc->ptr();
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    if (!low_latency_mode) {
        // MoE counter
        CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
        *moe_recv_counter = -1;

        // MoE expert-level counter
        CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
        for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)
            moe_recv_expert_counter[i] = -1;

        // MoE RDMA-level counter
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }

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

    if (!low_latency_mode) {
        CUDA_CHECK(cudaMalloc(&local_ht_barrier_counter, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(local_ht_barrier_counter, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&last_ht_barrier_counter, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(last_ht_barrier_counter, 0, sizeof(uint64_t)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    my_peer_info.rdma_buffer_ptr = rdma_buffer_ptr;
    my_peer_info.device_id = get_local_device_id();
    my_peer_info.sync_buffer_ptr = sync_buffer_ptr;
    my_peer_info.ht_barrier_ptr = local_ht_barrier_counter;
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

bool Buffer::is_ht_available() const {
    return is_available() and max_num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

torch::stable::Tensor
Buffer::get_local_buffer_tensor(torch::headeronly::ScalarType dtype,
                                int64_t offset,
                                bool use_rdma_buffer) const {
    const int64_t element_bytes = scalar_type_element_size(dtype);
    auto base_ptr =
        static_cast<uint8_t *>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    const int64_t num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    const int64_t sizes[] = {num_bytes / element_bytes};
    const int64_t strides[] = {1};
    return torch::stable::from_blob(
        base_ptr,
        torch::headeronly::IntHeaderOnlyArrayRef(sizes, 1),
        torch::headeronly::IntHeaderOnlyArrayRef(strides, 1),
        torch::stable::Device(torch::headeronly::DeviceType::CUDA, device_id),
        dtype);
}

void Buffer::destroy() {
    auto warn_cuda = [](cudaError_t status, const char *operation) noexcept {
        warnCuda(status, "destroy()", operation);
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

    if (num_nvl_bytes > 0) {
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, timeout_cycles, comm_stream);
        warn_cuda(cudaDeviceSynchronize(), "synchronize device after intranode barrier");

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++ i) if (i != nvl_rank)
                warn_cuda(cudaIpcCloseMemHandle(buffer_ptrs[i]), "close remote IPC handle");
        }

        // Free local buffer
        warn_cuda(cudaFree(buffer_ptrs[nvl_rank]), "free local NVL buffer");
    }

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
        if (local_ht_barrier_counter != nullptr) {
            warn_nixl(nixl_agent_info->agent->deregisterMem(
                          nixl_agent_info->ht_barrier_reg_descs),
                      "deregister ht barrier memory");
        }

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

    if (!low_latency_mode) {
        warn_cuda(cudaFree(local_ht_barrier_counter), "free local ht barrier counter");
        local_ht_barrier_counter = nullptr;
        warn_cuda(cudaFree(last_ht_barrier_counter), "free last ht barrier counter");
        last_ht_barrier_counter = nullptr;
        warn_cuda(cudaFreeHost(const_cast<int*>(moe_recv_counter)), "free moe receive counter");
        moe_recv_counter = nullptr;
        warn_cuda(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)), "free moe receive expert counter");
        moe_recv_expert_counter = nullptr;
        warn_cuda(cudaFreeHost(const_cast<int*>(moe_recv_rdma_counter)), "free moe receive rdma counter");
        moe_recv_rdma_counter = nullptr;
    }

    m_workspace_alloc.reset();
    workspace = nullptr;

    destroyed = true;
    available = false;
}

void Buffer::barrier() {
    ep_kernels::barrier(gpu_ctx_ptr, mask_buffer_ptr, timeout_cycles, cuda_stream::getCurrent());
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

void Buffer::_ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles = {}) {
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(all_gathered_handles.size() == max_num_ranks);
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++ i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
                CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and barrier signal pointers to GPU
        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void Buffer::connect_ranks(const std::vector<int>& remote_ranks_list, const std::optional<std::vector<nixl_blob_t>>& remote_mds,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles, bool activate) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(!remote_mds.has_value() || remote_mds->size() == remote_ranks_list.size());

    if (!low_latency_mode && num_nvl_bytes > 0) {
        EP_HOST_ASSERT(remote_ranks.empty() && "connect_ranks called more than once in high-throughput mode; elasticity is not yet supported");
    }
    EP_HOST_ASSERT(low_latency_mode || activate);

    std::vector<int> new_ranks;
    std::vector<nixl_blob_t> new_ranks_mds;

    if (all_gathered_handles.size() > 0)
        _ipc_handles_sync(all_gathered_handles);

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
        _nixl_agents_connect(new_ranks, new_ranks_mds);

        _nixl_agents_peer_info_gather(new_ranks);

        _nixl_ep_memory_views_destroy();

        _nixl_ep_memory_views_create();

        for (int remote_rank : new_ranks)
            ep_kernels::cache_p2p_ptr(gpu_ctx_ptr, remote_rank, comm_stream);

        CUDA_CHECK(cudaDeviceSynchronize());
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

    _nixl_ep_memory_views_destroy();

    _nixl_agents_peer_info_cleanup(remote_ranks_list);

    _nixl_agents_disconnect(remote_ranks_list);

    // Remove ranks from remote_ranks vector (arbitrary order)
    for (int removed_rank : remote_ranks_list) {
        remote_ranks.erase(
            std::remove(remote_ranks.begin(), remote_ranks.end(), removed_rank),
            remote_ranks.end()
        );
    }

    _nixl_ep_memory_views_create();
}

std::tuple<torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           torch::stable::Tensor,
           torch::stable::Tensor,
           int64_t>
Buffer::get_dispatch_layout(const torch::stable::Tensor &topk_idx,
                            int num_experts,
                            const EventHandle *previous_event,
                            bool async,
                            bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    cudaStream_t compute_stream = cuda_stream::getCurrent();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event);
        EP_HOST_ASSERT(async);
        cuda_stream::setCurrent(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event) {
        previous_event->streamWait(comm_stream);
    } else {
        streamWait(comm_stream, compute_stream);
    }

    const auto num_tokens = static_cast<int>(topk_idx.size(0));
    const auto num_topk = static_cast<int>(topk_idx.size(1));
    const int num_ranks = max_num_ranks;
    const torch::stable::Device cuda_device{torch::headeronly::DeviceType::CUDA, device_id};

    const int64_t rank_sizes[] = {num_ranks};
    torch::stable::Tensor num_tokens_per_rank =
        torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(rank_sizes, 1),
                             torch::headeronly::ScalarType::Int,
                             std::nullopt,
                             cuda_device,
                             std::nullopt,
                             std::nullopt);
    const int64_t expert_sizes[] = {num_experts};
    torch::stable::Tensor num_tokens_per_expert =
        torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(expert_sizes, 1),
                             torch::headeronly::ScalarType::Int,
                             std::nullopt,
                             cuda_device,
                             std::nullopt,
                             std::nullopt);
    const int64_t token_rank_sizes[] = {num_tokens, num_ranks};
    torch::stable::Tensor is_token_in_rank =
        torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(token_rank_sizes, 2),
                             torch::headeronly::ScalarType::Bool,
                             std::nullopt,
                             cuda_device,
                             std::nullopt,
                             std::nullopt);
    std::optional<torch::stable::Tensor> num_tokens_per_rdma_rank;
    int *num_tokens_per_rdma_rank_ptr = nullptr;
    if (is_ht_available()) {
        const int64_t rdma_rank_sizes[] = {num_rdma_ranks};
        num_tokens_per_rdma_rank =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(rdma_rank_sizes, 1),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);
        num_tokens_per_rdma_rank_ptr = num_tokens_per_rdma_rank->mutable_data_ptr<int>();
    }

    layout::get_dispatch_layout(topk_idx.const_data_ptr<topk_idx_t>(),
                                num_tokens_per_rank.mutable_data_ptr<int>(),
                                num_tokens_per_rdma_rank_ptr,
                                num_tokens_per_expert.mutable_data_ptr<int>(),
                                is_token_in_rank.mutable_data_ptr<bool>(),
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                comm_stream);

    // Wait streams
    const EventHandle *event = nullptr;
    if (async) {
        event = new EventHandle(comm_stream);
    } else {
        streamWait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) {
        cuda_stream::setCurrent(compute_stream);
    }

    return {num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            reinterpret_cast<int64_t>(event)};
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
Buffer::ht_dispatch(const torch::stable::Tensor &x,
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
                    bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(!low_latency_mode &&
                   "ht_dispatch() requires high-throughput mode (low_latency_mode=false)");
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks,
    // which can be quite long. If users of DeepEP need to execute other Python code on other
    // threads, such as KV transfer, their code will get stuck due to GIL unless we release GIL
    // here.
    pybind11::gil_scoped_release release;
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    const torch::stable::Device cuda_device{torch::headeronly::DeviceType::CUDA, device_id};
    const auto x_element_size = scalar_type_element_size(x.scalar_type());
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() ==
                       torch::headeronly::ScalarType::Int);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() ==
                       torch::headeronly::ScalarType::Int);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() ==
                       torch::headeronly::ScalarType::Int);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() ==
                       torch::headeronly::ScalarType::Int);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::headeronly::ScalarType::Int);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::headeronly::ScalarType::Int);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::headeronly::ScalarType::Int);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x_element_size) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and
                       cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                       cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and
                       cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and
                       cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == max_num_ranks and
                       cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and
                       cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == max_num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and
                       num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and
                       num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == max_num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % max_num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / max_num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    const int num_tokens = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int hidden_int4 = static_cast<int>(x.size(1) * x_element_size / sizeof(int4));
    const int num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0));
    const int num_local_experts = num_experts / max_num_ranks;

    // Top-k checks
    int num_topk = 0;
    const topk_idx_t *topk_idx_ptr = nullptr;
    const float *topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::headeronly::ScalarType::Float);
        topk_idx_ptr = topk_idx->const_data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->const_data_ptr<float>();
    }

    // FP8 scales checks
    float *x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x_element_size == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::headeronly::ScalarType::Float or
                       x_scales->scalar_type() == torch::headeronly::ScalarType::Int);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float *>(x_scales->mutable_data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    cudaStream_t compute_stream = cuda_stream::getCurrent();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event);
        EP_HOST_ASSERT(async);
        cuda_stream::setCurrent(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event) {
        previous_event->streamWait(comm_stream);
    } else {
        streamWait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    int num_rdma_recv_tokens = -1;
    torch::stable::Tensor rdma_channel_prefix_matrix;
    torch::stable::Tensor recv_rdma_rank_prefix_sum;
    torch::stable::Tensor gbl_channel_prefix_matrix;
    torch::stable::Tensor recv_gbl_rank_prefix_sum;
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        ht::cached_notify(
            hidden_int4,
            num_scales,
            num_topk,
            num_topk,
            max_num_ranks,
            num_channels,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            rdma_buffer_ptr,
            config.num_max_rdma_chunked_recv_tokens,
            buffer_ptrs_gpu,
            config.num_max_nvl_chunked_recv_tokens,
            barrier_signal_ptrs_gpu,
            rank,
            comm_stream,
            config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), max_num_ranks),
            num_nvl_bytes,
            timeout_cycles,
            true,
            low_latency_mode,
            gpu_ctx);
    } else {
        const int64_t rdma_channel_sizes[] = {num_rdma_ranks, num_channels};
        rdma_channel_prefix_matrix =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(rdma_channel_sizes, 2),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);
        const int64_t rdma_rank_prefix_sizes[] = {num_rdma_ranks};
        recv_rdma_rank_prefix_sum =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(rdma_rank_prefix_sizes, 1),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);
        const int64_t gbl_channel_sizes[] = {max_num_ranks, num_channels};
        gbl_channel_prefix_matrix =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(gbl_channel_sizes, 2),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);
        const int64_t gbl_rank_prefix_sizes[] = {max_num_ranks};
        recv_gbl_rank_prefix_sum =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(gbl_rank_prefix_sizes, 1),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);

        // Send sizes
        *moe_recv_counter = -1;
        *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i) {
            moe_recv_expert_counter[i] = -1;
        }
        ht::notify_dispatch(
            num_tokens_per_rank->const_data_ptr<int>(),
            moe_recv_counter_mapped,
            max_num_ranks,
            num_tokens_per_rdma_rank->const_data_ptr<int>(),
            moe_recv_rdma_counter_mapped,
            num_tokens_per_expert->const_data_ptr<int>(),
            moe_recv_expert_counter_mapped,
            num_experts,
            is_token_in_rank.const_data_ptr<bool>(),
            num_tokens,
            num_channels,
            hidden_int4,
            num_scales,
            num_topk,
            expert_alignment,
            rdma_channel_prefix_matrix.mutable_data_ptr<int>(),
            recv_rdma_rank_prefix_sum.mutable_data_ptr<int>(),
            gbl_channel_prefix_matrix.mutable_data_ptr<int>(),
            recv_gbl_rank_prefix_sum.mutable_data_ptr<int>(),
            rdma_buffer_ptr,
            config.num_max_rdma_chunked_recv_tokens,
            buffer_ptrs_gpu,
            config.num_max_nvl_chunked_recv_tokens,
            barrier_signal_ptrs_gpu,
            rank,
            comm_stream,
            config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), max_num_ranks),
            num_nvl_bytes,
            timeout_cycles,
            low_latency_mode,
            gpu_ctx);

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);
            num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++i) {
                ready &= moe_recv_expert_counter[i] >= 0;
            }

            if (ready) {
                break;
            }

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start_time)
                    .count() > NUM_CPU_TIMEOUT_SECS) {
                for (int i = 0; i < num_local_experts; ++i) {
                    printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                }
                throw std::runtime_error("NixlEP error: timeout (dispatch CPU)");
            }
        }
        num_recv_tokens_per_expert_list =
            std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    const int64_t recv_x_sizes[] = {num_recv_tokens, hidden};
    auto recv_x = torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(recv_x_sizes, 2),
                                       x.scalar_type(),
                                       std::nullopt,
                                       cuda_device,
                                       std::nullopt,
                                       std::nullopt);
    std::optional<torch::stable::Tensor> recv_src_meta;
    std::optional<torch::stable::Tensor> recv_rdma_channel_prefix_matrix;
    std::optional<torch::stable::Tensor> recv_gbl_channel_prefix_matrix;
    std::optional<torch::stable::Tensor> send_rdma_head;
    std::optional<torch::stable::Tensor> send_nvl_head;
    void *recv_src_meta_ptr = nullptr;
    int *recv_rdma_channel_prefix_matrix_ptr = nullptr;
    int *recv_gbl_channel_prefix_matrix_ptr = nullptr;
    int *send_rdma_head_ptr = nullptr;
    int *send_nvl_head_ptr = nullptr;
    if (not cached_mode) {
        const int64_t recv_src_meta_sizes[] = {num_recv_tokens, ht::get_source_meta_bytes()};
        recv_src_meta = torch::stable::empty(
            torch::headeronly::IntHeaderOnlyArrayRef(recv_src_meta_sizes, 2),
            torch::headeronly::ScalarType::Byte,
            std::nullopt,
            cuda_device,
            std::nullopt,
            std::nullopt);
        const int64_t recv_rdma_channel_sizes[] = {num_rdma_ranks, num_channels};
        recv_rdma_channel_prefix_matrix =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(recv_rdma_channel_sizes, 2),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);
        const int64_t recv_gbl_channel_sizes[] = {max_num_ranks, num_channels};
        recv_gbl_channel_prefix_matrix =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(recv_gbl_channel_sizes, 2),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);
        const int64_t send_rdma_head_sizes[] = {num_tokens, num_rdma_ranks};
        send_rdma_head =
            torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(send_rdma_head_sizes, 2),
                                 torch::headeronly::ScalarType::Int,
                                 std::nullopt,
                                 cuda_device,
                                 std::nullopt,
                                 std::nullopt);
        const int64_t send_nvl_head_sizes[] = {num_rdma_recv_tokens, NUM_MAX_NVL_PEERS};
        send_nvl_head = torch::stable::empty(
            torch::headeronly::IntHeaderOnlyArrayRef(send_nvl_head_sizes, 2),
            torch::headeronly::ScalarType::Int,
            std::nullopt,
            cuda_device,
            std::nullopt,
            std::nullopt);
        recv_src_meta_ptr = recv_src_meta->mutable_data_ptr();
        recv_rdma_channel_prefix_matrix_ptr =
            recv_rdma_channel_prefix_matrix->mutable_data_ptr<int>();
        recv_gbl_channel_prefix_matrix_ptr = recv_gbl_channel_prefix_matrix->mutable_data_ptr<int>();
        send_rdma_head_ptr = send_rdma_head->mutable_data_ptr<int>();
        send_nvl_head_ptr = send_nvl_head->mutable_data_ptr<int>();
    }

    // Assign pointers
    std::optional<torch::stable::Tensor> recv_topk_idx;
    std::optional<torch::stable::Tensor> recv_topk_weights;
    topk_idx_t *recv_topk_idx_ptr = nullptr;
    float *recv_topk_weights_ptr = nullptr;
    if (topk_idx) {
        const int64_t recv_topk_sizes[] = {num_recv_tokens, num_topk};
        recv_topk_idx = torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(recv_topk_sizes, 2),
                                             topk_idx_scalar_type(),
                                             std::nullopt,
                                             cuda_device,
                                             std::nullopt,
                                             std::nullopt);
        recv_topk_weights = torch::stable::empty(
            torch::headeronly::IntHeaderOnlyArrayRef(recv_topk_sizes, 2),
            torch::headeronly::ScalarType::Float,
            std::nullopt,
            cuda_device,
            std::nullopt,
            std::nullopt);
        recv_topk_idx_ptr = recv_topk_idx->mutable_data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->mutable_data_ptr<float>();
    }

    std::optional<torch::stable::Tensor> recv_x_scales;
    float *recv_x_scales_ptr = nullptr;
    if (x_scales) {
        const int64_t recv_x_scales_sizes[] = {num_recv_tokens, num_scales};
        recv_x_scales = torch::stable::empty(
            torch::headeronly::IntHeaderOnlyArrayRef(recv_x_scales_sizes, 2),
            x_scales->scalar_type(),
            std::nullopt,
            cuda_device,
            std::nullopt,
            std::nullopt);
        recv_x_scales_ptr = static_cast<float *>(recv_x_scales->mutable_data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    ht::dispatch(recv_x.mutable_data_ptr(),
                 recv_x_scales_ptr,
                 recv_topk_idx_ptr,
                 recv_topk_weights_ptr,
                 recv_src_meta_ptr,
                 x.const_data_ptr(),
                 x_scales_ptr,
                 topk_idx_ptr,
                 topk_weights_ptr,
                 send_rdma_head_ptr,
                 send_nvl_head_ptr,
                 recv_rdma_channel_prefix_matrix_ptr,
                 recv_gbl_channel_prefix_matrix_ptr,
                 rdma_channel_prefix_matrix.mutable_data_ptr<int>(),
                 recv_rdma_rank_prefix_sum.mutable_data_ptr<int>(),
                 gbl_channel_prefix_matrix.mutable_data_ptr<int>(),
                 recv_gbl_rank_prefix_sum.mutable_data_ptr<int>(),
                 is_token_in_rank.const_data_ptr<bool>(),
                 num_tokens,
                 hidden_int4,
                 num_scales,
                 num_topk,
                 num_experts,
                 scale_token_stride,
                 scale_hidden_stride,
                 rdma_buffer_ptr,
                 config.num_max_rdma_chunked_send_tokens,
                 config.num_max_rdma_chunked_recv_tokens,
                 buffer_ptrs_gpu,
                 config.num_max_nvl_chunked_send_tokens,
                 config.num_max_nvl_chunked_recv_tokens,
                 rank,
                 max_num_ranks,
                 cached_mode,
                 comm_stream,
                 num_channels,
                 timeout_cycles,
                 low_latency_mode,
                 gpu_ctx);

    // Wait streams
    const EventHandle *event = nullptr;
    if (async) {
        event = new EventHandle(comm_stream);
    } else {
        streamWait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) {
        cuda_stream::setCurrent(compute_stream);
    }

    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            reinterpret_cast<int64_t>(event)};
}

std::tuple<torch::stable::Tensor, std::optional<torch::stable::Tensor>, int64_t>
Buffer::ht_combine(const torch::stable::Tensor &x,
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
                   bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(!low_latency_mode &&
                   "ht_combine() requires high-throughput mode (low_latency_mode=false)");
    pybind11::gil_scoped_release release;
    const int num_ranks = max_num_ranks;
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    const torch::stable::Device cuda_device{torch::headeronly::DeviceType::CUDA, device_id};
    const auto x_element_size = scalar_type_element_size(x.scalar_type());

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(x.scalar_type() == torch::headeronly::ScalarType::BFloat16);
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and
                   src_meta.scalar_type() == torch::headeronly::ScalarType::Byte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                   is_combined_token_in_rank.scalar_type() == torch::headeronly::ScalarType::Bool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                   rdma_channel_prefix_matrix.scalar_type() == torch::headeronly::ScalarType::Int);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                   rdma_rank_prefix_sum.scalar_type() == torch::headeronly::ScalarType::Int);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                   gbl_channel_prefix_matrix.scalar_type() == torch::headeronly::ScalarType::Int);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                   combined_rdma_head.scalar_type() == torch::headeronly::ScalarType::Int);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and
                   combined_nvl_head.scalar_type() == torch::headeronly::ScalarType::Int);

    const int num_tokens = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int hidden_int4 = static_cast<int>(x.size(1) * x_element_size / sizeof(int4));
    const int num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x_element_size) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == ht::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and
                   rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and
                   gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    cudaStream_t compute_stream = cuda_stream::getCurrent();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event);
        EP_HOST_ASSERT(async);
        cuda_stream::setCurrent(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event) {
        previous_event->streamWait(comm_stream);
    } else {
        streamWait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    std::optional<torch::stable::Tensor> combined_topk_weights;
    const float *topk_weights_ptr = nullptr;
    float *combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::headeronly::ScalarType::Float);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->const_data_ptr<float>();
        const int64_t combined_topk_sizes[] = {num_combined_tokens, num_topk};
        combined_topk_weights = torch::stable::empty(
            torch::headeronly::IntHeaderOnlyArrayRef(combined_topk_sizes, 2),
            torch::headeronly::ScalarType::Float,
            std::nullopt,
            cuda_device,
            std::nullopt,
            std::nullopt);
        combined_topk_weights_ptr = combined_topk_weights->mutable_data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <=
                   config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    ht::cached_notify(hidden_int4,
                      0,
                      0,
                      num_topk,
                      num_ranks,
                      num_channels,
                      num_combined_tokens,
                      combined_rdma_head.mutable_data_ptr<int>(),
                      rdma_channel_prefix_matrix.mutable_data_ptr<int>(),
                      rdma_rank_prefix_sum.mutable_data_ptr<int>(),
                      combined_nvl_head.mutable_data_ptr<int>(),
                      rdma_buffer_ptr,
                      config.num_max_rdma_chunked_recv_tokens,
                      buffer_ptrs_gpu,
                      config.num_max_nvl_chunked_recv_tokens,
                      barrier_signal_ptrs_gpu,
                      rank,
                      comm_stream,
                      config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                      num_nvl_bytes,
                      timeout_cycles,
                      false,
                      low_latency_mode,
                      gpu_ctx);

    // Assign bias pointers
    const std::optional<torch::stable::Tensor> bias_opts[] = {bias_0, bias_1};
    void *bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        if (bias_opts[i].has_value()) {
            const auto &bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.mutable_data_ptr();
        }
    }

    // Launch data combine
    const int64_t combined_x_sizes[] = {num_combined_tokens, hidden};
    auto combined_x = torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(combined_x_sizes, 2),
                                           x.scalar_type(),
                                           std::nullopt,
                                           cuda_device,
                                           std::nullopt,
                                           std::nullopt);
    ht::combine(combined_x.mutable_data_ptr(),
                combined_topk_weights_ptr,
                is_combined_token_in_rank.const_data_ptr<bool>(),
                x.const_data_ptr(),
                topk_weights_ptr,
                bias_ptrs[0],
                bias_ptrs[1],
                combined_rdma_head.mutable_data_ptr<int>(),
                combined_nvl_head.mutable_data_ptr<int>(),
                src_meta.const_data_ptr(),
                rdma_channel_prefix_matrix.mutable_data_ptr<int>(),
                rdma_rank_prefix_sum.mutable_data_ptr<int>(),
                gbl_channel_prefix_matrix.mutable_data_ptr<int>(),
                num_tokens,
                num_combined_tokens,
                hidden,
                num_topk,
                rdma_buffer_ptr,
                config.num_max_rdma_chunked_send_tokens,
                config.num_max_rdma_chunked_recv_tokens,
                buffer_ptrs_gpu,
                config.num_max_nvl_chunked_send_tokens,
                config.num_max_nvl_chunked_recv_tokens,
                rank,
                num_ranks,
                comm_stream,
                num_channels,
                timeout_cycles,
                low_latency_mode,
                gpu_ctx);

    // Wait streams
    const EventHandle *event = nullptr;
    if (async) {
        event = new EventHandle(comm_stream);
    } else {
        streamWait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream) {
        cuda_stream::setCurrent(compute_stream);
    }

    return {combined_x, combined_topk_weights, reinterpret_cast<int64_t>(event)};
}

std::tuple<torch::stable::Tensor,
           std::optional<torch::stable::Tensor>,
           torch::stable::Tensor,
           torch::stable::Tensor,
           torch::stable::Tensor,
           int64_t>
Buffer::dispatch(const torch::stable::Tensor &x,
                 const torch::stable::Tensor &topk_idx,
                 const std::optional<torch::stable::Tensor> &cumulative_local_expert_recv_stats,
                 const std::optional<torch::stable::Tensor> &dispatch_wait_recv_cost_stats,
                 int num_max_dispatch_tokens_per_rank,
                 bool use_fp8,
                 bool round_scale,
                 bool use_ue8m0,
                 bool async,
                 bool return_recv_hook) {
    EP_HOST_ASSERT(low_latency_mode &&
                   "dispatch() requires low-latency mode (low_latency_mode=true)");
    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and
                   x.scalar_type() == torch::headeronly::ScalarType::BFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == topk_idx_scalar_type());

    // Diagnosis tensors
    int *cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() ==
                       torch::headeronly::ScalarType::Int);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and
                       cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts_per_rank);
        cumulative_local_expert_recv_stats_ptr =
            cumulative_local_expert_recv_stats->mutable_data_ptr<int>();
    }

    int64_t *dispatch_wait_recv_cost_stats_ptr = nullptr;
    if (dispatch_wait_recv_cost_stats) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() ==
                       torch::headeronly::ScalarType::Long);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and
                       dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == active_rank_bound);
        dispatch_wait_recv_cost_stats_ptr =
            dispatch_wait_recv_cost_stats->mutable_data_ptr<int64_t>();
    }

    const auto num_tokens = static_cast<int>(x.size(0));
    const auto hidden = static_cast<int>(x.size(1));
    const auto num_topk = static_cast<int>(topk_idx.size(1));
    const torch::stable::Device cuda_device{torch::headeronly::DeviceType::CUDA, device_id};

    // Buffer control
    const int max_num_experts = max_num_ranks * num_experts_per_rank;
    EPLayout layout(
        rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[buffer_idx];
    auto next_buffer = layout.buffers[buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    cudaStream_t compute_stream = cuda_stream::getCurrent();
    cudaStream_t launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook) {
        streamWait(launch_stream, compute_stream);
    }

    // Allocate packed tensors
    const int64_t packed_recv_x_sizes[] = {
        num_experts_per_rank, active_rank_bound * num_max_dispatch_tokens_per_rank, hidden};
    auto packed_recv_x = torch::stable::empty(
        torch::headeronly::IntHeaderOnlyArrayRef(packed_recv_x_sizes, 3),
        use_fp8 ? torch::headeronly::ScalarType::Float8_e4m3fn
                : torch::headeronly::ScalarType::BFloat16,
        std::nullopt,
        cuda_device,
        std::nullopt,
        std::nullopt);
    const int64_t packed_recv_src_info_sizes[] = {
        num_experts_per_rank, active_rank_bound * num_max_dispatch_tokens_per_rank};
    auto packed_recv_src_info = torch::stable::empty(
        torch::headeronly::IntHeaderOnlyArrayRef(packed_recv_src_info_sizes, 2),
        torch::headeronly::ScalarType::Int,
        std::nullopt,
        cuda_device,
        std::nullopt,
        std::nullopt);
    const int64_t packed_recv_layout_range_sizes[] = {num_experts_per_rank, active_rank_bound};
    auto packed_recv_layout_range = torch::stable::empty(
        torch::headeronly::IntHeaderOnlyArrayRef(packed_recv_layout_range_sizes, 2),
        torch::headeronly::ScalarType::Long,
        std::nullopt,
        cuda_device,
        std::nullopt,
        std::nullopt);
    const int64_t packed_recv_count_sizes[] = {num_experts_per_rank};
    auto packed_recv_count = torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(
                                                      packed_recv_count_sizes, 1),
                                                  torch::headeronly::ScalarType::Int,
                                                  std::nullopt,
                                                  cuda_device,
                                                  std::nullopt,
                                                  std::nullopt);

    // Allocate column-majored scales
    std::optional<torch::stable::Tensor> packed_recv_x_scales;
    void *packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((active_rank_bound * num_max_dispatch_tokens_per_rank) % 4 == 0 and
                   "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            const int64_t scale_sizes[] = {num_experts_per_rank,
                                           hidden / 128,
                                           active_rank_bound * num_max_dispatch_tokens_per_rank};
            packed_recv_x_scales = torch::stable::empty(
                torch::headeronly::IntHeaderOnlyArrayRef(scale_sizes, 3),
                torch::headeronly::ScalarType::Float,
                std::nullopt,
                cuda_device,
                std::nullopt,
                std::nullopt);
        } else {
            EP_HOST_ASSERT(round_scale);
            const int64_t scale_sizes[] = {num_experts_per_rank,
                                           hidden / 512,
                                           active_rank_bound * num_max_dispatch_tokens_per_rank};
            packed_recv_x_scales = torch::stable::empty(
                torch::headeronly::IntHeaderOnlyArrayRef(scale_sizes, 3),
                torch::headeronly::ScalarType::Int,
                std::nullopt,
                cuda_device,
                std::nullopt,
                std::nullopt);
        }
        packed_recv_x_scales = torch::stable::transpose(*packed_recv_x_scales, 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->mutable_data_ptr();
    }

    // Kernel launch
    std::pair<uint64_t *, int> next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=, this](int phases) {
        ep_kernels::dispatch(packed_recv_x.mutable_data_ptr(),
                             packed_recv_x_scales_ptr,
                             packed_recv_src_info.mutable_data_ptr<int>(),
                             packed_recv_layout_range.mutable_data_ptr<int64_t>(),
                             packed_recv_count.mutable_data_ptr<int>(),
                             mask_buffer_ptr,
                             cumulative_local_expert_recv_stats_ptr,
                             dispatch_wait_recv_cost_stats_ptr,
                             buffer.dispatch_rdma_recv_data_buffer,
                             buffer.dispatch_rdma_recv_count_buffer,
                             buffer.dispatch_rdma_send_buffer,
                             x.const_data_ptr(),
                             topk_idx.const_data_ptr<topk_idx_t>(),
                             next_clean_meta.first,
                             next_clean_meta.second,
                             num_tokens,
                             hidden,
                             num_max_dispatch_tokens_per_rank,
                             num_topk,
                             active_rank_bound,
                             num_experts_per_rank,
                             rank,
                             use_fp8,
                             round_scale,
                             use_ue8m0,
                             timeout_cycles,
                             workspace,
                             num_device_sms,
                             launch_stream,
                             phases,
                             gpu_ctx_ptr);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    // Wait streams
    const EventHandle *event = nullptr;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait
        // happens, so in Python API, we must wrap all tensors into the event handle.
        event = new EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        streamWait(compute_stream, launch_stream);
    }

    if (return_recv_hook) {
        pending_dispatch_recv_hook = [=, this]() { launcher(EP_RECV_PHASE); };
    } else {
        pending_dispatch_recv_hook = nullptr;
    }

    return {packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            reinterpret_cast<int64_t>(event)};
}

void
Buffer::run_dispatch_recv_hook() {
    EP_HOST_ASSERT(pending_dispatch_recv_hook != nullptr);
    pending_dispatch_recv_hook();
    pending_dispatch_recv_hook = nullptr;
}

std::tuple<torch::stable::Tensor, int64_t>
Buffer::combine(const torch::stable::Tensor &x,
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
    EP_HOST_ASSERT(low_latency_mode &&
                   "combine() requires low-latency mode (low_latency_mode=true)");

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous());
    EP_HOST_ASSERT(x.scalar_type() == torch::headeronly::ScalarType::BFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts_per_rank);
    EP_HOST_ASSERT(x.size(1) == active_rank_bound * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and
                   topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == topk_idx_scalar_type());
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::headeronly::ScalarType::Float);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::headeronly::ScalarType::Int and
                   x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(src_info.size(1) == active_rank_bound * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::headeronly::ScalarType::Long);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts_per_rank and
                   layout_range.size(1) == active_rank_bound);

    int64_t* combine_wait_recv_cost_stats_ptr;
    if (combine_wait_recv_cost_stats) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() ==
                       torch::headeronly::ScalarType::Long);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and
                       combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == active_rank_bound);
        combine_wait_recv_cost_stats_ptr = combine_wait_recv_cost_stats->mutable_data_ptr<int64_t>();
    } else {
        combine_wait_recv_cost_stats_ptr = nullptr;
    }

    const auto hidden = static_cast<int>(x.size(2));
    const auto num_topk = static_cast<int>(topk_weights.size(1));
    const auto num_combined_tokens = static_cast<int>(topk_weights.size(0));
    const torch::stable::Device cuda_device{torch::headeronly::DeviceType::CUDA, device_id};

    // Buffer control
    const int max_num_experts = max_num_ranks * num_experts_per_rank;
    const EPLayout layout{
        rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts};
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    EPBuffer buffer = layout.buffers[buffer_idx];
    EPBuffer next_buffer = layout.buffers[buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    cudaStream_t compute_stream = cuda_stream::getCurrent();
    cudaStream_t launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook) {
        streamWait(launch_stream, compute_stream);
    }

    // Allocate output tensor
    torch::stable::Tensor combined_x;
    if (out) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = *out;
    } else {
        const int64_t sizes[] = {num_combined_tokens, hidden};
        combined_x = torch::stable::empty(torch::headeronly::IntHeaderOnlyArrayRef(sizes, 2),
                                          torch::headeronly::ScalarType::BFloat16,
                                          std::nullopt,
                                          cuda_device,
                                          std::nullopt,
                                          std::nullopt);
    }

    // Kernel launch
    std::pair<uint64_t*, int> next_clean_meta = next_buffer.clean_meta();

    auto launcher = [=, this](int phases) {
        ep_kernels::combine(combined_x.mutable_data_ptr(),
                            buffer.combine_rdma_recv_data_buffer,
                            buffer.combine_rdma_recv_flag_buffer,
                            buffer.combine_rdma_send_buffer,
                            x.const_data_ptr(),
                            topk_idx.const_data_ptr<topk_idx_t>(),
                            topk_weights.const_data_ptr<float>(),
                            src_info.const_data_ptr<int>(),
                            layout_range.const_data_ptr<int64_t>(),
                            mask_buffer_ptr,
                            combine_wait_recv_cost_stats_ptr,
                            next_clean_meta.first,
                            next_clean_meta.second,
                            num_combined_tokens,
                            hidden,
                            num_max_dispatch_tokens_per_rank,
                            num_topk,
                            active_rank_bound,
                            num_experts_per_rank,
                            rank,
                            use_logfmt,
                            timeout_cycles,
                            workspace,
                            num_device_sms,
                            launch_stream,
                            phases,
                            zero_copy,
                            gpu_ctx_ptr);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    // Wait streams
    const EventHandle *event = nullptr;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait
        // happens, so in Python API, we must wrap all tensors into the event handle.
        event = new EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        streamWait(compute_stream, launch_stream);
    }

    if (return_recv_hook) {
        pending_combine_recv_hook = [=, this]() { launcher(EP_RECV_PHASE); };
    } else {
        pending_combine_recv_hook = nullptr;
    }

    return {combined_x, reinterpret_cast<int64_t>(event)};
}

void
Buffer::run_combine_recv_hook() {
    EP_HOST_ASSERT(pending_combine_recv_hook != nullptr);
    pending_combine_recv_hook();
    pending_combine_recv_hook = nullptr;
}

torch::stable::Tensor
Buffer::get_next_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden) const {
    const int max_num_experts = max_num_ranks * num_experts_per_rank;
    const EPLayout layout{
        rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts};
    const EPBuffer &buffer = layout.buffers[buffer_idx];
    constexpr auto dtype_elem_size = sizeof(uint16_t);
    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % dtype_elem_size == 0);

    const auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / dtype_elem_size);
    const int64_t sizes[] = {
        num_experts_per_rank, active_rank_bound * num_max_dispatch_tokens_per_rank, hidden};
    const int64_t strides[] = {
        active_rank_bound * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1};

    return torch::stable::from_blob(
        buffer.combine_rdma_send_buffer_data_start,
        torch::headeronly::IntHeaderOnlyArrayRef(sizes, 3),
        torch::headeronly::IntHeaderOnlyArrayRef(strides, 3),
        torch::stable::Device(torch::headeronly::DeviceType::CUDA, device_id),
        torch::headeronly::ScalarType::BFloat16);
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
    active_ranks[rank_to_mask] = !mask;
    _refresh_active_rank_bound();
    ep_kernels::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, cuda_stream::getCurrent());
}

void
Buffer::query_mask_buffer(const torch::stable::Tensor &mask_status) const {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == max_num_ranks);
    EP_HOST_ASSERT(mask_status.scalar_type() == torch::headeronly::ScalarType::Int);

    ep_kernels::query_mask_buffer(mask_buffer_ptr,
                                  max_num_ranks,
                                  mask_status.mutable_data_ptr<int>(),
                                  cuda_stream::getCurrent());
}

void Buffer::clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    std::vector<int> mask(max_num_ranks, 1);
    for (int rank_id = 0; rank_id < max_num_ranks; ++rank_id) {
        const bool active = _is_rank_connected(rank_id);
        active_ranks[rank_id] = active;
        mask[rank_id] = !active;
    }
    _refresh_active_rank_bound();
    CUDA_CHECK(cudaMemcpyAsync(mask_buffer_ptr, mask.data(),
                               max_num_ranks * sizeof(int), cudaMemcpyHostToDevice,
                               cuda_stream::getCurrent()));
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

void Buffer::_nixl_ep_memory_views_create(void) {
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

    EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(local_descs, gpu_ctx.local_mvh, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    if (!remote_ranks.empty()) {
        EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(remote_descs, gpu_ctx.remote_mvh, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
        EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(barrier_descs, gpu_ctx.barrier_mvh, &nixl_agent_info->extra_params) == NIXL_SUCCESS);

        if (!low_latency_mode && max_num_ranks > NUM_MAX_NVL_PEERS) {
            nixl_remote_dlist_t ht_barrier_descs(VRAM_SEG);
            for (int r = 0; r < max_num_ranks; r++) {
                std::string remote_agent_name = remote_set.count(r) ? nixl_agent_info->remote_agent_names[r] : nixl_null_agent;
                ht_barrier_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].ht_barrier_ptr, sizeof(uint64_t), nixl_peer_info[r].device_id, remote_agent_name));
            }
            EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(ht_barrier_descs, gpu_ctx.ht_barrier_mvh, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
        }
    }
    CUDA_CHECK(cudaMemcpy(gpu_ctx_ptr, &gpu_ctx, sizeof(gpu_ctx), cudaMemcpyHostToDevice));
}

void Buffer::_nixl_ep_memory_views_destroy(void) {
    if (gpu_ctx.local_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.local_mvh);
    if (gpu_ctx.remote_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.remote_mvh);
    if (gpu_ctx.barrier_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.barrier_mvh);
    if (gpu_ctx.ht_barrier_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.ht_barrier_mvh);
    gpu_ctx.local_mvh = nullptr;
    gpu_ctx.remote_mvh = nullptr;
    gpu_ctx.barrier_mvh = nullptr;
    gpu_ctx.ht_barrier_mvh = nullptr;
}

void Buffer::_nixl_ep_init(void) {
    gpu_ctx = {
        .sync_buffer_ptr = sync_buffer_ptr,
        .sync_count_ptr = sync_count_ptr,
        .last_ht_barrier_counter = last_ht_barrier_counter,
        .local_ht_barrier_counter_ptr = local_ht_barrier_counter,
        .rdma_buffer_ptr = rdma_buffer_ptr,
        .p2p_ptrs = nullptr,
        .max_num_ranks = max_num_ranks,
        .num_rdma_ranks = num_rdma_ranks,
        .rank = rank,
    };
    CUDA_CHECK(cudaMalloc(&gpu_ctx.p2p_ptrs, max_num_ranks * sizeof(void*)));
    CUDA_CHECK(cudaMemset(gpu_ctx.p2p_ptrs, 0, max_num_ranks * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx_ptr, sizeof(gpu_nixl_ctx)));
    CUDA_CHECK(cudaMemcpy(gpu_ctx_ptr, &gpu_ctx, sizeof(gpu_ctx), cudaMemcpyHostToDevice));
}

void Buffer::_nixl_ep_destroy(void) {
    _nixl_ep_memory_views_destroy();
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
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
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

    const char* num_channels_env = std::getenv("NIXL_EP_NUM_CHANNELS");
    init_params["ucx_num_device_channels"] = num_channels_env ? num_channels_env : "4";
    init_params["ucx_error_handling_mode"] = "none";
    init_params["num_workers"] = std::to_string(1);

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
    nixl_agent_info->ht_barrier_reg_descs.clear();

    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->rdma_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->sync_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->sync_count_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);

    if (local_ht_barrier_counter) {
        nixl_agent_info->ht_barrier_reg_descs.addDesc(
            nixlBlobDesc((uintptr_t)(local_ht_barrier_counter), sizeof(uint64_t), get_local_device_id(), ""));
        EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->ht_barrier_reg_descs) == NIXL_SUCCESS);
    }

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

    pybind11::class_<nixl_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             pybind11::arg("num_sms") = 20,
             pybind11::arg("num_max_nvl_chunked_send_tokens") = 6,
             pybind11::arg("num_max_nvl_chunked_recv_tokens") = 256,
             pybind11::arg("num_max_rdma_chunked_send_tokens") = 6,
             pybind11::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("_ptr", [](const nixl_ep::Config &self) { return reinterpret_cast<int64_t>(&self); });

    pybind11::class_<nixl_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &nixl_ep::EventHandle::currentStreamWait)
        .def("_ptr", [](nixl_ep::EventHandle &self) { return reinterpret_cast<int64_t>(&self); })
        .def_static("_adopt_ptr", [](int64_t ptr) {
            EP_HOST_ASSERT(ptr != 0);
            auto *raw = reinterpret_cast<nixl_ep::EventHandle *>(ptr);
            nixl_ep::EventHandle handle = std::move(*raw);
            delete raw;
            return handle;
        });

    pybind11::class_<nixl_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, bool, bool, int>())
        .def("update_memory_buffers", &nixl_ep::Buffer::update_memory_buffers)
        .def("barrier", &nixl_ep::Buffer::barrier)
        .def(
            "connect_ranks",
            [](nixl_ep::Buffer &buffer,
               const std::vector<int> &remote_ranks,
               const std::optional<std::vector<pybind11::bytes>> &remote_mds,
               const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
               bool activate) {
                buffer.connect_ranks(
                    remote_ranks, nixl_ep::convert_mds(remote_mds), all_gathered_handles, activate);
            },
            pybind11::arg("remote_ranks"),
            pybind11::arg("remote_mds") = std::nullopt,
            pybind11::arg("ipc_handles") = std::vector<std::optional<pybind11::bytearray>>{},
            pybind11::arg("activate") = true)
        .def("disconnect_ranks", &nixl_ep::Buffer::disconnect_ranks)
        .def("is_available", &nixl_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &nixl_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &nixl_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &nixl_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &nixl_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &nixl_ep::Buffer::get_local_ipc_handle)
        .def("destroy", &nixl_ep::Buffer::destroy)
        .def("update_mask_buffer", &nixl_ep::Buffer::update_mask_buffer)
        .def("query_mask_buffer", &nixl_ep::Buffer::query_mask_buffer)
        .def("clean_mask_buffer", &nixl_ep::Buffer::clean_mask_buffer)
        .def("get_local_metadata",
             [](const nixl_ep::Buffer &buffer) -> pybind11::bytes {
                 return pybind11::bytes(buffer.get_local_metadata());
             })
        .def("_ptr", [](nixl_ep::Buffer &self) { return reinterpret_cast<int64_t>(&self); });
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<nixl_ep::topk_idx_t>::value);
    m.def("is_sm90_compiled", nixl_ep::is_sm90_compiled);
}
