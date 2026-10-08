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

// NIXL's GPU-initiated API (`nixlPut`, `nixlAtomicAdd` over memory views) and UCX's device API under it, for the
// JIT-compiled kernels only (`EP_NIXL_DEVICE_API`, a JIT flag); the extension sees the context struct alone
#if defined(__CUDACC__) && defined(EP_NIXL_DEVICE_API)
#include <gpu/nixl_device.cuh>
// As in `nixl_device.cu`: without UCX's device header NIXL compiles stubs that fail at run time; the header test tells
#if !__has_include(<ucp/api/device/ucp_device_impl.h>)
#error "UCX device header <ucp/api/device/ucp_device_impl.h> not on the include path: set EP_UCX_ROOT_DIR (-I <EP_UCX_ROOT_DIR>/include)"
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

    // Release-add on a peer's word: a system-scope atomic over the mapping when the peer is reachable with stores,
    // an RDMA atomic otherwise (64-bit words only; visible only after this channel's earlier writes to the peer have
    // completed). Never mix this with plain stores into the same word
    template <typename dtype_t>
    __device__ __forceinline__ void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& rid,
                                                const bool& ring_doorbell = true) const {
        if (const auto dst_ptr = get_p2p_ptr(sym_ptr, rid); dst_ptr != nullptr) {
            ptx::red_add_rel_sys(dst_ptr, value);
        } else {
            EP_STATIC_ASSERT(sizeof(dtype_t) == 8, "RDMA atomics are 64-bit");
            const nixlMemViewElem dst{ctx.remote_mvh, static_cast<size_t>(rid), get_offset(sym_ptr)};
            const auto status = nixlAtomicAdd<nixl_gpu_level_t::THREAD>(static_cast<uint64_t>(value), dst, channel,
                                                                        ring_doorbell ? 0 : nixl_gpu_flags::defer);
            EP_DEVICE_ASSERT(status == NIXL_IN_PROG and "nixlAtomicAdd failed");
        }
    }

    // Release-store of a 64-bit word the writer knows in full: a system-scope store over the mapping, else the value is
    // staged in `staging_sym_ptr` (a word of our region only this writer uses) and put by RDMA. Preferred to a
    // release-add where the result is known: the NIC serialises RDMA atomics, a write travels like the data it follows
    __device__ __forceinline__ void put_word_rel(int64_t* dst_sym_ptr, int64_t* staging_sym_ptr, const int64_t& value,
                                                 const int& rid, const bool& ring_doorbell = true) const {
        if (const auto dst_ptr = get_p2p_ptr(dst_sym_ptr, rid); dst_ptr != nullptr) {
            ptx::st_release_sys(dst_ptr, value);
        } else {
            ptx::st_relaxed_sys(staging_sym_ptr, value);
            rdma_put(dst_sym_ptr, staging_sym_ptr, sizeof(int64_t), rid, ring_doorbell);
        }
    }

    // Copy `num_bytes` (a multiple of 4) of control words to the same offset on the peer: word stores over the mapping,
    // an RDMA put otherwise; the receiver polls each word on its own, so the words need no order
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

    // Store one word on a peer reachable with stores (the scale-up peers of the hybrid path always are)
    template <typename dtype_t>
    __device__ __forceinline__ void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& rid) const {
        const auto dst_ptr = get_p2p_ptr(sym_ptr, rid);
        EP_DEVICE_ASSERT(dst_ptr != nullptr and "A scale-up peer is not reachable with loads/stores");
        ptx::st_relaxed_sys(dst_ptr, value);
    }
};

}  // namespace nixl_ep_v2::elastic::nixl
#endif  // __CUDACC__ && EP_NIXL_DEVICE_API
