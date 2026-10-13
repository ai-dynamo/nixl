#pragma once

#include <cstdlib>

#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>

namespace nixl_ep_v2::elastic::topology {

// The adaptive EP kernels are launched with a FIXED grid so that a captured CUDA graph survives a change of
// topology. Blocks with `blockIdx.x >= num_active_sms` exit at once.
// Even, because the kernels are launched in clusters of two. the upstream kernels run on 24 SMs; 26 measured
// slightly faster for the adaptive kernels and is a tentative value.
constexpr int kNumLaunchSMs = 26;
// Hard cap used for buffer sizing (the launch grid may be raised up to this with EP_ADAPTIVE_NUM_LAUNCH_SMS)
constexpr int kNumMaxLaunchSMs = 64;

// Hard caps of the rank decomposition: the workspace control words and the transport tables are sized by them, so
// no position depends on the declared or live world (128 positions hold a GB200 NVL72 domain)
constexpr int kNumMaxScaleoutRanksCap = 32;
static_assert(kNumMaxScaleoutRanksCap <= 32, "The scale-out mask is one 32-bit word");
constexpr int kNumMaxScaleupRanksCap = 128;
constexpr int kNumScaleupMaskWords = kNumMaxScaleupRanksCap / 64;
constexpr int kNumMaxRankIds = kNumMaxScaleoutRanksCap * kNumMaxScaleupRanksCap;
static_assert(kNumMaxScaleupRanksCap % 64 == 0, "The position mask is whole 64-bit words");

// The live positions of a domain (bit k across the words): a plain array, so the topology descriptor (`EPTopology`) stays a POD
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

    // Highest live position + 1 (0 when empty): the bound of the live world (host, when a topology is staged)
    __host__ int bound() const {
        for (int i = kNumScaleupMaskWords - 1; i >= 0; -- i)
            if (words[i] != 0)
                return i * 64 + 64 - __builtin_clzll(words[i]);
        return 0;
    }
};

// Dense rank id inside a declared maximum world of `max_num_scaleup_ranks` positions per domain (the transport tables' index)
__device__ __host__ __forceinline__ constexpr int rank_id(const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                                          const int& max_num_scaleup_ranks) {
    return scaleout_rank_idx * max_num_scaleup_ranks + scaleup_rank_idx;
}

#ifndef __CUDA_ARCH__
// Host: the launch grid actually used, `kNumLaunchSMs` unless `EP_ADAPTIVE_NUM_LAUNCH_SMS` overrides it (even,
// <= kNumMaxLaunchSMs). An A/B knob: the grid is a template argument of the kernels, so changing it recompiles.
inline int get_num_launch_sms() {
    static int value = [] {
        const char* env = std::getenv("EP_ADAPTIVE_NUM_LAUNCH_SMS");
        int v = env != nullptr and env[0] != '\0' ? std::atoi(env) : kNumLaunchSMs;
        EP_HOST_ASSERT(2 <= v and v <= kNumMaxLaunchSMs and v % 2 == 0 and "EP_ADAPTIVE_NUM_LAUNCH_SMS must be even and <= 64");
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
// The live world is the product of the live domains (`scaleout_mask`, bit d) and the live positions of a domain
// (`scaleup_mask`, bit k): rank (d, k) takes part iff both bits are set; `num_scaleout_ranks` / `num_scaleup_ranks`
// are the bounds of the masks (highest live index + 1). Slot ownership uses `domain * num_scaleup_ranks + position`,
// with a fixed slot count per rank; changing the width changes this numbering. Transport peer IDs use the maximum
// width and remain fixed. Selections of masked owners are dropped; serving callers use prefix masks and route only
// to active slots.
struct alignas(128) EPTopology {
    Mode mode;
    int num_scaleout_ranks, num_scaleup_ranks;
    int scaleout_rank_idx, scaleup_rank_idx;
    // Number of blocks that do work in this launch (<= the launched grid, even)
    int num_active_sms;
    // Live domains (bit d) and live positions (bit k, `kNumScaleupMaskWords` words)
    uint32_t scaleout_mask;
    ScaleupMask scaleup_mask;
    // The transport context (memory views, p2p pointers, channels): loaded from here, so a change of peers needs no re-capture
    const nixl::NIXLDeviceContext* transport;

    __device__ __host__ __forceinline__ bool is_scaleout_live(const int& idx) const {
        return idx >= 0 and idx < num_scaleout_ranks and ((scaleout_mask >> idx) & 1u);
    }

    __device__ __host__ __forceinline__ bool is_scaleup_live(const int& idx) const {
        return idx >= 0 and idx < num_scaleup_ranks and scaleup_mask.test(idx);
    }

    __device__ __host__ __forceinline__ bool is_live(const int& scaleout_idx, const int& scaleup_idx) const {
        return is_scaleout_live(scaleout_idx) and is_scaleup_live(scaleup_idx);
    }

    __device__ __host__ __forceinline__ int num_live_scaleout_ranks() const {
#ifdef __CUDA_ARCH__
        return __popc(scaleout_mask);
#else
        return __builtin_popcount(scaleout_mask);
#endif
    }

    // Whether this rank is part of the live world (a rank outside it must not launch the EP kernels)
    __device__ __host__ __forceinline__ bool is_participant() const {
        return is_live(scaleout_rank_idx, scaleup_rank_idx);
    }
};
static_assert(sizeof(EPTopology) == 128, "The topology descriptor is one 128-byte line");

// Liveness specialised on the declared maximum: with at most 64 positions only word 0 can hold a live bit (per token and lane)
template <int kNumMaxScaleupRanks>
__device__ __forceinline__ bool is_scaleup_live(const EPTopology& topo, const int& idx) {
    if constexpr (kNumMaxScaleupRanks <= 64)
        return idx >= 0 and idx < topo.num_scaleup_ranks and ((topo.scaleup_mask.words[0] >> idx) & 1ull);
    return topo.is_scaleup_live(idx);
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
        topo.scaleup_mask.words[i] = __ldg(reinterpret_cast<const unsigned long long*>(&ptr->scaleup_mask.words[i]));
    topo.transport = reinterpret_cast<const nixl::NIXLDeviceContext*>(__ldg(reinterpret_cast<const unsigned long long*>(&ptr->transport)));
    return topo;
#else
    return *ptr;
#endif
}

}  // namespace nixl_ep_v2::elastic::topology
