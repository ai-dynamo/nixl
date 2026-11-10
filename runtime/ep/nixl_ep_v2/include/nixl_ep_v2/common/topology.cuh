// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <cstdlib>

#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/ptx.cuh>

namespace nixl_ep_v2::elastic::topology {

// The adaptive EP kernels are launched with a FIXED grid so that a captured CUDA graph survives a change of
// topology. Blocks with `blockIdx.x >= num_active_sms` exit at once.
// Clusters have two blocks. Use 24 as the baseline launch capacity; the active block count is selected at runtime.
constexpr int kNumLaunchSMs = 24;
// Hard cap used for buffer sizing (the launch grid may be raised up to this with NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS)
constexpr int kNumMaxLaunchSMs = 64;

// Hard caps of the rank decomposition: the workspace control words are sized by them (the transport tables by the
// maximum world, which they bound), so no offset depends on the committed world; a world-rank-indexed word depends on
// the maximum world only through `world_rank` (128 positions hold a GB200 NVL72 domain)
constexpr int kNumMaxScaleoutRanksCap = 32;
static_assert(kNumMaxScaleoutRanksCap <= 32, "The scale-out mask is one 32-bit word");
constexpr int kNumMaxScaleupRanksCap = 128;
constexpr int kNumScaleupMaskWords = kNumMaxScaleupRanksCap / 64;
constexpr int kNumMaxRanksCap = kNumMaxScaleoutRanksCap * kNumMaxScaleupRanksCap;   // The caps' product: the world-rank-indexed workspace rows
static_assert(kNumMaxScaleupRanksCap % 64 == 0, "The position mask is whole 64-bit words");
// Communication worlds, including graph replays; bootstrap and teardown cohorts may be smaller.
constexpr int kMinNumActiveRanks = 2;

// The active positions of a domain (bit k across the words): a plain array, so the topology descriptor (`EPTopology`) stays a POD
struct ScaleupMask {
    uint64_t words[kNumScaleupMaskWords];

    __device__ __host__ __forceinline__ bool test(const int& idx) const {
        // A predicate chain, not a dynamic index (a dynamically indexed register array goes to local memory; per token and lane)
        uint64_t word = 0ull;
        #pragma unroll
        for (int i = 0; i < kNumScaleupMaskWords; ++ i)
            word = (idx >> 6) == i ? words[i] : word;
        return idx >= 0 and idx < kNumMaxScaleupRanksCap and ((word >> (idx & 63)) & 1ull);
    }

    __device__ __host__ __forceinline__ int popc() const {
        int count = 0;
        #pragma unroll
        for (int i = 0; i < kNumScaleupMaskWords; ++ i) {
#ifdef __CUDA_ARCH__
            count += __popcll(words[i]);
#else
            count += __builtin_popcountll(words[i]);
#endif
        }
        return count;
    }

    // Highest active position + 1 (0 when empty): the extent of the active positions (host, when a topology is staged)
    __host__ int extent() const {
        for (int i = kNumScaleupMaskWords - 1; i >= 0; -- i)
            if (words[i] != 0)
                return i * 64 + 64 - __builtin_clzll(words[i]);
        return 0;
    }
};

// World rank inside a declared maximum world of `max_num_scaleup_ranks` positions per domain (the transport tables' index)
__device__ __host__ __forceinline__ constexpr int world_rank(const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                                             const int& max_num_scaleup_ranks) {
    return scaleout_rank_idx * max_num_scaleup_ranks + scaleup_rank_idx;
}

#ifndef __CUDA_ARCH__
// Host: the launch grid actually used, `kNumLaunchSMs` unless `NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS` overrides it (even,
// <= kNumMaxLaunchSMs). An A/B knob: the grid is a template argument of the kernels, so changing it recompiles.
inline int get_num_launch_sms() {
    static int value = [] {
        const char* env = std::getenv("NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS");
        int v = env != nullptr and env[0] != '\0' ? std::atoi(env) : kNumLaunchSMs;
        if (not (2 <= v and v <= kNumMaxLaunchSMs and v % 2 == 0))
            throw EPExceptionWithLineInfo("Assertion", std::string("NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS=") + env + ": an even number between 2 and " + std::to_string(kNumMaxLaunchSMs) + " is required");
        return v;
    }();
    return value;
}
#endif

// Kernel mode, decided by the host from the world's NVLink domains (never a user input)
enum Mode : int {
    kModeDirect = 0,  // a single NVLink domain
    kModeHybrid = 1,  // RDMA between NVLink domains, NVLink inside them
};

// Device-resident topology descriptor (`EPTopology`): written by the host between launches, read by every adaptive kernel block at entry
// The active ranks are the product of the active domains (`scaleout_mask`, bit d) and the active positions of a domain
// (`scaleup_mask`, bit k): rank (d, k) takes part iff both bits are set; `num_scaleout_ranks` / `num_scaleup_ranks`
// are the extents of the masks (highest active index + 1), so physical coordinates do not change with holes.
// With S fixed physical expert slots per rank, routing slot e has extent rank e / S, whose coordinates
// are (rank / num_scaleup_ranks, rank % num_scaleup_ranks). Holes retain their slots; they are not packed
// away. The slot range is num_scaleout_ranks * num_scaleup_ranks * S, not the active rank count times S.
struct alignas(128) EPTopology {
    Mode mode;
    int num_scaleout_ranks, num_scaleup_ranks;
    int scaleout_rank_idx, scaleup_rank_idx;
    // Number of blocks that do work in this launch (<= the launched grid, even)
    int num_active_sms;
    // Active domains (bit d) and active positions (bit k, `kNumScaleupMaskWords` words)
    uint32_t scaleout_mask;
    ScaleupMask scaleup_mask;
    // The transport context (memory views, p2p pointers, channels): loaded from here, so a change of peers needs no re-capture
    const nixl::NIXLDeviceContext* transport;

    __device__ __host__ __forceinline__ bool is_scaleout_active(const int& idx) const {
        return idx >= 0 and idx < num_scaleout_ranks and ((scaleout_mask >> idx) & 1u);
    }

    // The range check is `ScaleupMask::test`'s: bits at or beyond the extent `num_scaleup_ranks` are zero by construction
    // (`stage_topology` takes the extent from the mask)
    __device__ __host__ __forceinline__ bool is_scaleup_active(const int& idx) const {
        return scaleup_mask.test(idx);
    }

    __device__ __host__ __forceinline__ bool is_active(const int& scaleout_idx, const int& scaleup_idx) const {
        return is_scaleout_active(scaleout_idx) and is_scaleup_active(scaleup_idx);
    }

    __device__ __host__ __forceinline__ int num_active_scaleout_ranks() const {
#ifdef __CUDA_ARCH__
        return __popc(scaleout_mask);
#else
        return __builtin_popcount(scaleout_mask);
#endif
    }

    __device__ __host__ __forceinline__ int num_active_scaleup_ranks() const {
        return scaleup_mask.popc();
    }

    // Whether this rank is active (in the committed world): the host refuses `dispatch` / `combine` for a rank outside
    // it; `barrier_impl` is launched by every rank of the collective call and returns at once for an outsider
    __device__ __host__ __forceinline__ bool is_active() const {
        return is_active(scaleout_rank_idx, scaleup_rank_idx);
    }

    __device__ __host__ __forceinline__ void check_launch() const {
        EP_UNIFIED_ASSERT(is_active() and "This rank is not active in the committed world (see `stage_topology`)");
        EP_UNIFIED_ASSERT(num_active_scaleout_ranks() * num_active_scaleup_ranks() >= kMinNumActiveRanks and
                          "Dispatch/combine require at least two active GPUs: commit a larger world before launching or replaying");
    }
};
static_assert(sizeof(EPTopology) == 128, "The topology descriptor is one 128-byte line");

// Membership specialised on the declared maximum: with at most 64 positions only word 0 can hold an active bit (per token and lane)
template <int kNumMaxScaleupRanks>
__device__ __forceinline__ bool is_scaleup_active(const EPTopology& topo, const int& idx) {
    if constexpr (kNumMaxScaleupRanks <= 64)
        return idx >= 0 and idx < topo.num_scaleup_ranks and ((topo.scaleup_mask.words[0] >> idx) & 1ull);
    return topo.is_scaleup_active(idx);
}

template <int kNumMaxScaleupRanks>
__device__ __forceinline__ bool is_rank_active(const EPTopology& topo, const int& rank_idx) {
    return topo.is_scaleout_active(rank_idx / topo.num_scaleup_ranks) and is_scaleup_active<kNumMaxScaleupRanks>(topo, rank_idx % topo.num_scaleup_ranks);
}

// World ranks of our peers (`world_rank`): the scale-up peer at position `k` of our domain, the rail peer in domain `d`
template <int kNumMaxScaleupRanks>
__device__ __host__ __forceinline__ int scaleup_peer(const EPTopology& topo, const int& k) {
    return world_rank(topo.scaleout_rank_idx, k, kNumMaxScaleupRanks);
}

template <int kNumMaxScaleupRanks>
__device__ __host__ __forceinline__ int scaleout_peer(const EPTopology& topo, const int& d) {
    return world_rank(d, topo.scaleup_rank_idx, kNumMaxScaleupRanks);
}

// The kernels read the topology descriptor (`EPTopology`) once per block, field by field through the read-only
// path into registers; a whole-struct copy is materialised in local memory and read back from the stack
__device__ __forceinline__ EPTopology load(const EPTopology* ptr) {
#if defined(__CUDA_ARCH__)
    EPTopology topo;
    topo.mode = static_cast<Mode>(__ldg(reinterpret_cast<const int*>(&ptr->mode)));
    topo.num_scaleout_ranks = __ldg(&ptr->num_scaleout_ranks);
    topo.num_scaleup_ranks = __ldg(&ptr->num_scaleup_ranks);
    topo.scaleout_rank_idx = __ldg(&ptr->scaleout_rank_idx);
    topo.scaleup_rank_idx = __ldg(&ptr->scaleup_rank_idx);
    topo.num_active_sms = __ldg(&ptr->num_active_sms);
    topo.scaleout_mask = __ldg(&ptr->scaleout_mask);
    #pragma unroll
    for (int i = 0; i < kNumScaleupMaskWords; ++ i)
        topo.scaleup_mask.words[i] = __ldg(&ptr->scaleup_mask.words[i]);
    topo.transport = ptx::ldg_ptr(&ptr->transport);
    return topo;
#else
    return *ptr;
#endif
}

}  // namespace nixl_ep_v2::elastic::topology

namespace nixl_ep_v2::elastic::adaptive {

// Compile-time channel maxima: the block shape and the channel slack of the buffers follow the fixed grid.
constexpr int kNumMaxChannelsPerSM = 8;
constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * topology::kNumMaxLaunchSMs;
EP_STATIC_ASSERT(kNumMaxChannels <= nixl_ep_v2::kNumMaxChannels, "The workspace channel arrays are sized by `nixl_ep_v2::kNumMaxChannels`");

}  // namespace nixl_ep_v2::elastic::adaptive
