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

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

#include "nixl.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME nixl_ep_cpp
#endif

/* CUDA memory allocator using VMM. */
class cuda_allocator {
public:
    cuda_allocator(size_t size) : m_size(0), m_ptr(0), m_alloc_handle(0)
    {
        if (size == 0) {
            throw std::invalid_argument("cuda_allocator: size must be non-zero");
        }

        CUdevice device;
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA device handle");
        }

        int rdma_vmm_supported = 0;
        cuDeviceGetAttribute(&rdma_vmm_supported,
                             CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                             device);
        if (!rdma_vmm_supported) {
            throw std::runtime_error("GPUDirect RDMA with CUDA VMM is not supported on this device");
        }

        int fabric_supported = 0;
        cuDeviceGetAttribute(&fabric_supported,
                             CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                             device);

        CUmemAllocationProp prop = {};
        prop.type                          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type                 = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id                   = device;
        prop.allocFlags.gpuDirectRDMACapable = 1;
        prop.requestedHandleTypes          = fabric_supported ?
                                             CU_MEM_HANDLE_TYPE_FABRIC :
                                             CU_MEM_HANDLE_TYPE_NONE;

        size_t granularity = 0;
        if (cuMemGetAllocationGranularity(&granularity, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA allocation granularity");
        }

        init_vmm(size, device, prop, granularity);
    }

    ~cuda_allocator()
    {
        if (m_ptr) {
            cuMemUnmap(m_ptr, m_size);
            cuMemAddressFree(m_ptr, m_size);
        }

        if (m_alloc_handle) {
            cuMemRelease(m_alloc_handle);
        }
    }

    void*  ptr()  const { return reinterpret_cast<void*>(m_ptr); }
    size_t size() const { return m_size; }

    cuda_allocator(const cuda_allocator&)            = delete;
    cuda_allocator& operator=(const cuda_allocator&) = delete;

private:
    void init_vmm(size_t size, CUdevice device, const CUmemAllocationProp &prop,
                  size_t granularity)
    {
        CUmemAccessDesc access_desc = {};
        const char *err_msg;

        m_size = nixl_ep::align_up<size_t>(size, granularity);

        if (cuMemCreate(&m_alloc_handle, m_size, &prop, 0) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to create CUDA VMM allocation");
        }

        if (cuMemAddressReserve(&m_ptr, m_size, 0, 0, 0) != CUDA_SUCCESS) {
            err_msg = "Failed to reserve CUDA virtual address";
            goto err_release;
        }

        if (cuMemMap(m_ptr, m_size, 0, m_alloc_handle, 0) != CUDA_SUCCESS) {
            err_msg = "Failed to map CUDA VMM memory";
            goto err_free;
        }

        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id   = device;
        access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cuMemSetAccess(m_ptr, m_size, &access_desc, 1) != CUDA_SUCCESS) {
            err_msg = "Failed to set CUDA memory access";
            goto err_unmap;
        }

        return;

err_unmap:
        cuMemUnmap(m_ptr, m_size);
err_free:
        cuMemAddressFree(m_ptr, m_size);
        m_ptr = 0;
err_release:
        cuMemRelease(m_alloc_handle);
        m_alloc_handle = 0;
        throw std::runtime_error(err_msg);
    }

    size_t                        m_size;
    CUdeviceptr                   m_ptr;
    CUmemGenericAllocationHandle  m_alloc_handle;
};

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

    // Owning allocators (keep raw ptrs above as aliases for use throughout)
    std::unique_ptr<cuda_allocator> m_rdma_alloc;
    std::unique_ptr<cuda_allocator> m_mask_alloc;
    std::unique_ptr<cuda_allocator> m_sync_alloc;
    std::unique_ptr<cuda_allocator> m_sync_count_alloc;
    std::unique_ptr<cuda_allocator> m_workspace_alloc;

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
