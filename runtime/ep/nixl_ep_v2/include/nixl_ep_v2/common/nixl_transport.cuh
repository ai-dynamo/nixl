// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <cstdint>

namespace nixl_ep_v2::elastic::nixl {

// Host-written at every commit, device-read; lives at a fixed device address for the life of the buffer (graphs replay)
struct alignas(16) NIXLDeviceContext {
    void* local_mvh;    // Local memory view: one entry, this rank's region
    void* remote_mvh;   // Remote memory view: `num_world_ranks` entries indexed by world rank (null agent without an endpoint)
    void** p2p_ptrs;    // `[num_world_ranks]`: a peer's region base when reachable with loads/stores, else nullptr
    void* local_base;   // This rank's region base (== the workspace pointer)
    int world_rank;     // This rank's world rank
    int num_channels;   // UCX device channels the kernels may issue on (`ucx_num_device_channels`)
    int num_world_ranks; // Length of the tables (read by the pointer-cache kernel only)
};
static_assert(sizeof(NIXLDeviceContext) == 48, "Unexpected transport context size");

}  // namespace nixl_ep_v2::elastic::nixl

// NIXL's GPU-initiated API (`nixlPut`, `nixlAtomicAdd` over memory views) and UCX's device API under it, for the
// JIT-compiled kernels and the extension's pointer-cache unit (`nixl_device.cu`), which define `EP_NIXL_DEVICE_API`;
// the extension's host units see the context struct alone
#if defined(__CUDACC__) && defined(EP_NIXL_DEVICE_API)
#include <gpu/nixl_device.cuh>
// Without UCX's device header NIXL compiles stubs that fail at run time (`nixlGetPtr` returns nullptr for every peer):
// fail the build instead; the header test tells
#if !__has_include(<ucp/api/device/ucp_device_impl.h>)
#error "UCX device header <ucp/api/device/ucp_device_impl.h> not on the include path: set NIXL_EP_UCX_ROOT_DIR (-I <NIXL_EP_UCX_ROOT_DIR>/include)"
#endif
// Without UCX's CUDA IPC lane `nixlGetPtr` returns nullptr for every peer, the same symptom
#if !defined(UCT_CUDA_IPC_SUPPORTED) || !UCT_CUDA_IPC_SUPPORTED
#error "UCX's device API was built without CUDA IPC (UCT_CUDA_IPC_SUPPORTED is unset): rebuild UCX with CUDA"
#endif

#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/ptx.cuh>

// The transport of the adaptive EP kernels. Every rank registers ONE symmetric region (workspace, then the token
// buffer) with NIXL, all with the identical layout, so one byte offset names the same word on every rank; a peer is
// its world rank (`d * max_num_scaleup_ranks + k`). A peer mapped into our address space (UCX's CUDA IPC lane,
// `p2p_ptrs[world_rank] != nullptr`) gets loads, stores and system-scope atomics on `base + offset`; same-position peers in
// other domains get RDMA. Kernels route through these endpoints only. The fixed context address lets graphs replay across commits.
namespace nixl_ep_v2::elastic::nixl {

// The kernels copy the transport context once per block, field by field through the read-only path into registers
// (a whole-struct copy is materialised in local memory, as for the topology descriptor (`EPTopology`))
__device__ __forceinline__ NIXLDeviceContext load_context(const NIXLDeviceContext* ptr) {
    NIXLDeviceContext ctx;
    ctx.local_mvh = ptx::ldg_ptr(&ptr->local_mvh);
    ctx.remote_mvh = ptx::ldg_ptr(&ptr->remote_mvh);
    ctx.p2p_ptrs = ptx::ldg_ptr(&ptr->p2p_ptrs);
    ctx.local_base = ptx::ldg_ptr(&ptr->local_base);
    ctx.world_rank = __ldg(&ptr->world_rank);
    ctx.num_channels = __ldg(&ptr->num_channels);
    return ctx;
}

// One per warp: the context (copied by the kernel at entry) and the UCX device channel this warp issues on.
// Operations of one warp on one channel to one peer are ordered; the kernels' data-then-announce protocols rely on it
struct NIXLTransport {
    NIXLDeviceContext ctx;
    unsigned channel;

    __device__ __forceinline__ NIXLTransport(const NIXLDeviceContext& ctx, const int& channel):
        ctx(ctx), channel(static_cast<unsigned>(channel)) {
        EP_DEVICE_ASSERT(0 <= channel and channel < ctx.num_channels and "Channel outside the context's channel count");
    }

    // Byte offset of a local symmetric pointer inside the region
    __device__ __forceinline__ size_t get_offset(const void* local_ptr) const {
        return static_cast<size_t>(static_cast<const uint8_t*>(local_ptr) - static_cast<const uint8_t*>(ctx.local_base));
    }

    // The peer's address of the same word, or nullptr when the peer is not reachable with loads/stores (our own table
    // entry holds `local_base`, so for ourselves this is `local_ptr`)
    template <typename dtype_t = void>
    __device__ __forceinline__ dtype_t* get_p2p_ptr(dtype_t* local_ptr, const int& world_rank) const {
        const auto base = static_cast<uint8_t*>(ptx::ldg_ptr(ctx.p2p_ptrs + world_rank));
        return base == nullptr ? nullptr : reinterpret_cast<dtype_t*>(base + get_offset(local_ptr));
    }

    // RDMA write of `num_bytes` from our region to the same offset in the peer's region. `ring_doorbell == false`
    // defers the doorbell: the request is posted with the next operation on this channel that rings it
    __device__ __forceinline__ void rdma_put(void* dst_sym_ptr, const void* src_sym_ptr, const int& num_bytes,
                                             const int& world_rank, const bool& ring_doorbell = true) const {
        const nixlMemViewElem src{ctx.local_mvh, 0, get_offset(src_sym_ptr)};
        const nixlMemViewElem dst{ctx.remote_mvh, static_cast<size_t>(world_rank), get_offset(dst_sym_ptr)};
        const auto status = nixlPut<nixl_gpu_level_t::THREAD>(src, dst, static_cast<size_t>(num_bytes), channel,
                                                              ring_doorbell ? 0 : nixl_gpu_flags::defer);
        if (status != NIXL_IN_PROG) {
            printf("NIXL EP v2 nixlPut from world rank %d to world rank %d failed, status %d (offset %llu, %d bytes)\n",
                   ctx.world_rank, world_rank, static_cast<int>(status), static_cast<unsigned long long>(get_offset(dst_sym_ptr)), num_bytes);
            ptx::trap();
        }
    }

    // Use the transport atomic even when the peer is mapped, to order preceding deferred RDMA puts.
    __device__ __forceinline__ void rdma_add(void* sym_ptr, const uint64_t& value, const int& world_rank,
                                            nixlGpuXferStatusH* completion = nullptr) const {
        const nixlMemViewElem dst{ctx.remote_mvh, static_cast<size_t>(world_rank), get_offset(sym_ptr)};
        const auto status = nixlAtomicAdd<nixl_gpu_level_t::THREAD>(value, dst, channel, 0, completion);
        if (status != NIXL_IN_PROG) {
            printf("NIXL EP v2 nixlAtomicAdd from world rank %d to world rank %d failed, status %d (offset %llu)\n",
                   ctx.world_rank, world_rank, static_cast<int>(status), static_cast<unsigned long long>(get_offset(sym_ptr)));
            ptx::trap();
        }
    }

    // Publish a batch's completion after its payload on the same peer and channel; the atomic rings the deferred puts.
    __device__ __forceinline__ void rdma_put_and_signal(void* dst_sym_ptr, const void* src_sym_ptr, const int& num_bytes,
                                                       int64_t* signal_sym_ptr, const int& world_rank,
                                                       const uint64_t& signal_value) const {
        rdma_put(dst_sym_ptr, src_sym_ptr, num_bytes, world_rank, false);
        rdma_add(signal_sym_ptr, signal_value, world_rank);
    }

    // Release-add on a peer's word: a system-scope atomic over the mapping when the peer is reachable with stores,
    // an RDMA atomic otherwise (64-bit words only; visible only after this channel's earlier writes to the peer have
    // completed). Never mix this with plain stores into the same word. An optional status tracks local RDMA completion;
    // mapped atomics leave it unused
    template <typename dtype_t>
    __device__ __forceinline__ void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& world_rank,
                                                nixlGpuXferStatusH* completion = nullptr) const {
        if (const auto dst_ptr = get_p2p_ptr(sym_ptr, world_rank); dst_ptr != nullptr) {
            ptx::red_add_rel_sys(dst_ptr, value);
        } else {
            EP_STATIC_ASSERT(sizeof(dtype_t) == 8, "RDMA atomics are 64-bit");
            rdma_add(sym_ptr, static_cast<uint64_t>(value), world_rank, completion);
        }
    }

    // Copy `num_bytes` (a multiple of 4) of control words to the same offset on the peer: word stores over the mapping,
    // an RDMA put otherwise; the receiver polls each word on its own, so the words need no order
    __device__ __forceinline__ void put_words(void* dst_sym_ptr, const void* src_sym_ptr, const int& num_bytes,
                                              const int& world_rank, const bool& ring_doorbell = true) const {
        if (const auto dst_ptr = get_p2p_ptr(dst_sym_ptr, world_rank); dst_ptr != nullptr) {
            const auto src = static_cast<const int*>(src_sym_ptr);
            const auto dst = static_cast<int*>(dst_ptr);
            for (int i = 0; i < num_bytes / static_cast<int>(sizeof(int)); ++ i)
                ptx::st_relaxed_sys(dst + i, src[i]);
        } else {
            rdma_put(dst_sym_ptr, src_sym_ptr, num_bytes, world_rank, ring_doorbell);
        }
    }

    // Store one word on a peer reachable with stores (the scale-up peers of the hybrid path always are)
    template <typename dtype_t>
    __device__ __forceinline__ void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& world_rank) const {
        ptx::st_relaxed_sys(get_p2p_ptr(sym_ptr, world_rank), value);
    }
};

}  // namespace nixl_ep_v2::elastic::nixl
#endif  // __CUDACC__ && EP_NIXL_DEVICE_API
