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

#include <cstdint>

namespace nixl_ep_v2::elastic::nixl {

// Host-written at every commit, device-read; lives at a fixed device address for the life of the buffer (graphs replay)
struct alignas(16) NIXLDeviceContext {
    void* local_mvh;    // Local memory view: one entry, this rank's region
    void* remote_mvh;   // Remote memory view: `num_rank_ids` entries indexed by dense rank id (null agent when absent)
    void** p2p_ptrs;    // `[num_rank_ids]`: a peer's region base when reachable with loads/stores, else nullptr
    void* local_base;   // This rank's region base (== the workspace pointer)
    int rank_id;        // This rank's dense id
    int num_channels;   // UCX device channels the kernels may issue on (`ucx_num_device_channels`)
    int num_rank_ids;   // Length of the tables
};
static_assert(sizeof(NIXLDeviceContext) == 48, "Unexpected transport context size");

}  // namespace nixl_ep_v2::elastic::nixl

// NIXL's GPU-initiated API (puts, write-with-immediate and receive CQ polling over memory views) and UCX's device API under it, for the
// JIT-compiled kernels only (`EP_NIXL_DEVICE_API`, a JIT flag); the extension sees the context struct alone
#if defined(__CUDACC__) && defined(EP_NIXL_DEVICE_API)
#include <gpu/nixl_device.cuh>
// As in `nixl_device.cu`: without UCX's device header NIXL compiles stubs that fail at run time; the header test tells
#if !__has_include(<ucp/api/device/ucp_device_impl.h>)
#error "UCX device header <ucp/api/device/ucp_device_impl.h> not on the include path: set NIXL_EP_UCX_ROOT_DIR (-I <NIXL_EP_UCX_ROOT_DIR>/include)"
#endif

#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/ptx.cuh>

// The transport of the adaptive EP kernels. Every rank registers ONE symmetric region (workspace, then the token
// buffer) with NIXL, all with the identical layout, so one byte offset names the same word on every rank; a peer is
// its dense rank id (`d * max_num_scaleup_ranks + k`). A peer mapped into our address space (UCX's CUDA IPC lane,
// `p2p_ptrs[rid] != nullptr`) gets loads, stores and system-scope atomics on `base + offset`; any other peer gets
// RDMA through the memory views. The context lives at a fixed device address, so a captured graph replays across commits.
namespace nixl_ep_v2::elastic::nixl {

// The kernels copy the transport context once per block, field by field through the read-only path into registers
// (a whole-struct copy is materialised in local memory, as for the topology descriptor (`EPTopology`))
__device__ __forceinline__ NIXLDeviceContext load_context(const NIXLDeviceContext* ptr) {
    NIXLDeviceContext ctx;
    ctx.local_mvh = reinterpret_cast<void*>(__ldg(reinterpret_cast<const unsigned long long*>(&ptr->local_mvh)));
    ctx.remote_mvh = reinterpret_cast<void*>(__ldg(reinterpret_cast<const unsigned long long*>(&ptr->remote_mvh)));
    ctx.p2p_ptrs = reinterpret_cast<void**>(__ldg(reinterpret_cast<const unsigned long long*>(&ptr->p2p_ptrs)));
    ctx.local_base = reinterpret_cast<void*>(__ldg(reinterpret_cast<const unsigned long long*>(&ptr->local_base)));
    ctx.rank_id = __ldg(&ptr->rank_id);
    ctx.num_channels = __ldg(&ptr->num_channels);
    ctx.num_rank_ids = __ldg(&ptr->num_rank_ids);
    return ctx;
}

// One per warp: the context (copied by the kernel at entry) and the UCX device channel this warp issues on.
// Operations of one warp on one channel to one peer are ordered; the kernels' data-then-announce protocols rely on it
struct NIXLTransport {
    static constexpr int kDispatchSlotBits = 22;
    static constexpr uint32_t kDispatchSlotMask = (1u << kDispatchSlotBits) - 1;

    // Dispatch packs a 10-bit logical channel and 22-bit receive slot; combine identifies the original source token.
    __device__ static constexpr uint32_t dispatch_payload(const int logical_channel, const int slot) {
        return (static_cast<uint32_t>(logical_channel) << kDispatchSlotBits) | static_cast<uint32_t>(slot);
    }

    __device__ static constexpr int dispatch_channel(const uint32_t payload) {
        return static_cast<int>(payload >> kDispatchSlotBits);
    }

    __device__ static constexpr int dispatch_slot(const uint32_t payload) {
        return static_cast<int>(payload & kDispatchSlotMask);
    }

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

    // The peer's address of the same word, or nullptr when the peer is not reachable with loads/stores
    template <typename dtype_t = void>
    __device__ __forceinline__ dtype_t* get_p2p_ptr(dtype_t* local_ptr, const int& rid) const {
        if (rid == ctx.rank_id)
            return local_ptr;
        const auto base = reinterpret_cast<uint8_t*>(__ldg(reinterpret_cast<const unsigned long long*>(ctx.p2p_ptrs + rid)));
        return base == nullptr ? nullptr : reinterpret_cast<dtype_t*>(base + get_offset(local_ptr));
    }

    // The peer's address of a local pointer given the peer's base (see get_p2p_ptr), for callers that resolve the base once per peer
    template <typename dtype_t = void>
    __device__ __forceinline__ dtype_t* get_p2p_ptr_from_base(uint8_t* base, dtype_t* local_ptr) const {
        return reinterpret_cast<dtype_t*>(base + get_offset(local_ptr));
    }

    // The peer's region base for `get_p2p_ptr_from_base`, or nullptr when the peer is not reachable with loads/stores
    __device__ __forceinline__ uint8_t* get_p2p_base(const int& rid) const {
        if (rid == ctx.rank_id)
            return static_cast<uint8_t*>(ctx.local_base);
        return reinterpret_cast<uint8_t*>(__ldg(reinterpret_cast<const unsigned long long*>(ctx.p2p_ptrs + rid)));
    }

    // RDMA write of `num_bytes` from our region to the same offset in the peer's region. `ring_doorbell == false`
    // defers the doorbell: the request is posted with the next operation on this channel that rings it
    __device__ __forceinline__ void rdma_put(void* dst_sym_ptr, const void* src_sym_ptr, const int& num_bytes,
                                             const int& rid, const bool& ring_doorbell = true) const {
        const nixlMemViewElem src{ctx.local_mvh, 0, get_offset(src_sym_ptr)};
        const nixlMemViewElem dst{ctx.remote_mvh, static_cast<size_t>(rid), get_offset(dst_sym_ptr)};
        const auto status = nixlPut<nixl_gpu_level_t::THREAD>(src, dst, static_cast<size_t>(num_bytes), channel,
                                                              ring_doorbell ? 0 : nixl_gpu_flags::defer);
        EP_DEVICE_ASSERT(status == NIXL_IN_PROG and "nixlPut failed");
    }

    // Fuse a token's notification into its last data write. The receiving CQE covers this write and earlier writes
    // on this same peer/channel QP. No memory word written by the NIC is used as evidence of payload arrival.
    __device__ __forceinline__ void rdma_put_with_immediate(void* dst_sym_ptr, const void* src_sym_ptr, const int& num_bytes,
                                                           const int& rid, const uint32_t immediate,
                                                           const bool& ring_doorbell = true) const {
        const nixlMemViewElem src{ctx.local_mvh, 0, get_offset(src_sym_ptr)};
        const nixlMemViewElem dst{ctx.remote_mvh, static_cast<size_t>(rid), get_offset(dst_sym_ptr)};
        const auto status = nixlPutWithIMM<nixl_gpu_level_t::THREAD>(
            src, dst, static_cast<size_t>(num_bytes), immediate, channel, ring_doorbell ? 0 : nixl_gpu_flags::defer);
        EP_DEVICE_ASSERT(status == NIXL_IN_PROG and "nixlPutWithIMM failed");
    }

    // Polling acquires the payload and advances only this reader's cursor, without publishing CQ consumption.
    __device__ __forceinline__ bool poll_receive_token_at(const int& rid, uint64_t& ci, uint32_t& token) const {
        uint32_t immediate = 0;
        const nixlMemViewElem peer{ctx.remote_mvh, static_cast<size_t>(rid), 0};
        const auto status = nixlPollIMM<nixl_gpu_level_t::THREAD>(peer, channel, &ci, &immediate);
        EP_DEVICE_ASSERT((status == NIXL_SUCCESS or status == NIXL_IN_PROG) and "nixlPollIMM failed");
        if (status == NIXL_SUCCESS) {
            token = immediate;
            return true;
        }
        return false;
    }

    // After all readers finish, return the consumed CQ/RQ credits and the updated CI.
    __device__ __forceinline__ uint64_t advance_rx(const int& rid, const uint64_t consumed) const {
        uint64_t next_ci = 0;
        const nixlMemViewElem peer{ctx.remote_mvh, static_cast<size_t>(rid), 0};
        const auto status = nixlAdvanceRx<nixl_gpu_level_t::THREAD>(peer, channel, consumed, &next_ci);
        EP_DEVICE_ASSERT(status == NIXL_SUCCESS and "nixlAdvanceRx failed");
        return next_ci;
    }

    // Release-add expert token counts and contributor counts to a mapped NVLink peer's counter.
    template <typename dtype_t>
    __device__ __forceinline__ void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& rid) const {
        const auto dst_ptr = get_p2p_ptr(sym_ptr, rid);
        EP_DEVICE_ASSERT(dst_ptr != nullptr and "Expert-count reduction requires an NVLink peer");
        ptx::red_add_rel_sys(dst_ptr, value);
    }

    // Publish a barrier or READY/ARRIVED signal: release-store to a mapped peer, otherwise stage the
    // 64-bit value in this writer's staging slot and send it with an RDMA write.
    __device__ __forceinline__ void put_word_rel(int64_t* dst_sym_ptr, int64_t* staging_sym_ptr, const int64_t& value,
                                                 const int& rid, const bool& ring_doorbell = true) const {
        if (const auto dst_ptr = get_p2p_ptr(dst_sym_ptr, rid); dst_ptr != nullptr) {
            ptx::st_release_sys(dst_ptr, value);
        } else {
            ptx::st_relaxed_sys(staging_sym_ptr, value);
            rdma_put(dst_sym_ptr, staging_sym_ptr, sizeof(int64_t), rid, ring_doorbell);
        }
    }

    // Send dispatch's count metadata with mapped word stores or a plain RDMA write; `num_bytes` is a
    // multiple of 4. The receiver polls each encoded word independently, so inter-word order is irrelevant.
    __device__ __forceinline__ void put_words(void* dst_sym_ptr, const void* src_sym_ptr, const int& num_bytes,
                                              const int& rid, const bool& ring_doorbell = true) const {
        if (const auto dst_ptr = get_p2p_ptr(dst_sym_ptr, rid); dst_ptr != nullptr) {
            const auto src = static_cast<const int*>(src_sym_ptr);
            const auto dst = static_cast<int*>(dst_ptr);
            for (int i = 0; i < num_bytes / static_cast<int>(sizeof(int)); ++ i)
                ptx::st_relaxed_sys(dst + i, src[i]);
        } else {
            rdma_put(dst_sym_ptr, src_sym_ptr, num_bytes, rid, ring_doorbell);
        }
    }

    // Relaxed-store rank/expert counts to a mapped NVLink peer, as a scalar or a 16-byte count pair.
    template <typename dtype_t>
    __device__ __forceinline__ void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& rid) const {
        const auto dst_ptr = get_p2p_ptr(sym_ptr, rid);
        EP_DEVICE_ASSERT(dst_ptr != nullptr and "A scale-up peer is not reachable with loads/stores");
        ptx::st_relaxed_sys(dst_ptr, value);
    }
};

}  // namespace nixl_ep_v2::elastic::nixl
#endif  // __CUDACC__ && EP_NIXL_DEVICE_API
