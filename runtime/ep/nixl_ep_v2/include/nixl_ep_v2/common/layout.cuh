// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/math.cuh>
#include <nixl_ep_v2/common/profile.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>

namespace nixl_ep_v2::elastic::layout {

// The control region of the symmetric buffer: every word a peer may write is zero at rest (each kernel zeroes what it
// used), except the combine readiness counters and monotonic per-peer barrier words
struct WorkspaceLayout {
    void* workspace;

    // We want to fix the layout position for all settings,
    // so that one buffer can be reused for all cases
    static constexpr int kNumMaxExperts = 2048;

    __forceinline__ __device__ __host__
    explicit WorkspaceLayout(void* workspace): workspace(workspace) {}

    // Arrival counter and round number of the active-block grid barrier (two words, padded to a line)
    static constexpr int64_t kNumActiveGridBarrierBytes = 128;
    // Notify reduction workspace: rank and expert counts
    static constexpr int64_t kNumNotifyReductionBytes = (topology::kNumMaxRanksCap + kNumMaxExperts) * sizeof(int64_t);
    // Scale-up rank and expert send/recv counts: `kNumMaxRanksCap` rank words then `kNumMaxExperts` expert words, each way
    static constexpr int kNumScaleupCountWords = topology::kNumMaxRanksCap + kNumMaxExperts;
    static constexpr int64_t kNumScaleupCountBytes = 2 * static_cast<int64_t>(kNumScaleupCountWords) * sizeof(int64_t);
    // Scale-up atomic sender counters: one word per rank
    static constexpr int64_t kNumScaleupAtomicCounterBytes = topology::kNumMaxRanksCap * sizeof(int);
    // Scale-out rank send/recv counts: a row of maximum positions per domain, each way
    static constexpr int64_t kNumScaleoutRankCountBytes = 2 * topology::kNumMaxRanksCap * sizeof(int);
    // Scale-out expert send/recv counts: a row of `kNumMaxExperts` per domain, each way
    static constexpr int64_t kNumScaleoutExpertCountBytes = 2 * static_cast<int64_t>(topology::kNumMaxScaleoutRanksCap) * kNumMaxExperts * sizeof(int);
    // Channel tails of the scale-up domain: a row of `kNumMaxScaleupRanksCap` per channel
    static constexpr int64_t kNumChannelScaleupTailBytes = static_cast<int64_t>(kNumMaxChannels) * topology::kNumMaxScaleupRanksCap * sizeof(int);
    // Per-peer barrier: one arrival counter and one expectation per world rank
    static constexpr int64_t kNumPeerBarrierRowBytes = topology::kNumMaxRanksCap * sizeof(int64_t);
    static constexpr int64_t kNumPeerBarrierBytes = 2 * kNumPeerBarrierRowBytes;
    // Separate completion words: dispatch clears its counters; combine keeps watermarks for the reduction epilogue
    static constexpr int64_t kNumReadyAtomicBytes = static_cast<int64_t>(adaptive::kNumMaxChannels) * topology::kNumMaxScaleoutRanksCap * sizeof(int64_t);

    static int64_t get_num_bytes() {
        return kNumNotifyReductionBytes + kNumScaleupCountBytes + kNumScaleupAtomicCounterBytes +
               kNumScaleoutRankCountBytes + kNumScaleoutExpertCountBytes + kNumChannelScaleupTailBytes +
               kNumActiveGridBarrierBytes + kNumPeerBarrierBytes + 2 * kNumReadyAtomicBytes + sizeof(profile::Header);
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        return static_cast<int64_t*>(workspace);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int64_t>(get_notify_reduction_workspace_ptr(), kNumNotifyReductionBytes);
        return base_ptr + (kIsSendBuffer ? 0 : kNumScaleupCountWords);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + topology::kNumMaxRanksCap;
    }

    // Word `i` of the contiguous "rank counts then expert counts" view with `num_rank_slots` rank slots
    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_count_ptr(const int& i, const int& num_rank_slots) const {
        return i < num_rank_slots ?
            get_scaleup_rank_count_ptr<kIsSendBuffer>() + i :
            get_scaleup_expert_count_ptr<kIsSendBuffer>() + (i - num_rank_slots);
    }

    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {
        return math::advance_ptr<int>(
            get_scaleup_rank_expert_count_ptr<true>(), kNumScaleupCountBytes);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {
        const auto base_ptr = math::advance_ptr<int>(get_scaleup_atomic_sender_counter(), kNumScaleupAtomicCounterBytes) +
                              (kIsSendBuffer ? 0 : topology::kNumMaxRanksCap);
        return base_ptr + scaleout_rank_idx * topology::kNumMaxScaleupRanksCap + scaleup_rank_idx;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        const auto base_ptr = math::advance_ptr<int>(get_scaleout_rank_count_ptr<true>(), kNumScaleoutRankCountBytes) +
                              (kIsSendBuffer ? 0 : topology::kNumMaxScaleoutRanksCap * kNumMaxExperts);
        return base_ptr + scaleout_rank_idx * kNumMaxExperts + expert_idx;
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_scaleout_expert_count_ptr<true>(), kNumScaleoutExpertCountBytes);
        return base_ptr + (channel_idx * topology::kNumMaxScaleupRanksCap + scaleup_rank_idx);
    }

    // `[0]` arrivals, `[1]` round number; both start at zero (the workspace is zero-initialised)
    __forceinline__ __device__ __host__ unsigned* get_active_grid_barrier_ptr() const {
        return math::advance_ptr<unsigned>(
            get_channel_scaleup_tail_ptr(0, 0), kNumChannelScaleupTailBytes);
    }

    // Each connected pair counts arrivals monotonically; reconnect restarts both words.
    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_signal_ptr(const int& world_rank) const {
        return math::advance_ptr<int64_t>(get_active_grid_barrier_ptr(), kNumActiveGridBarrierBytes) + world_rank;
    }

    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_expected_ptr(const int& world_rank) const {
        return math::advance_ptr<int64_t>(get_peer_barrier_signal_ptr(0), kNumPeerBarrierRowBytes) + world_rank;
    }

    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr() const {
        return math::advance_ptr<int64_t>(get_peer_barrier_expected_ptr(0), kNumPeerBarrierRowBytes);
    }

    __forceinline__ __device__ __host__ int64_t* get_combine_ready_atomic_buff() const {
        return math::advance_ptr<int64_t>(get_scaleout_channel_signaled_tail_ptr(), kNumReadyAtomicBytes);
    }

    __forceinline__ __device__ __host__ profile::Header* get_profile_header() const {
        return math::advance_ptr<profile::Header>(get_combine_ready_atomic_buff(), kNumReadyAtomicBytes);
    }

};

// Notify shared memory: rank and expert counts, in whole strides of the notify threads.
__forceinline__ __device__ __host__ constexpr int get_num_notify_smem_bytes(const int& num_max_ranks, const int& num_experts,
                                                                           const int& num_notify_threads) {
    return math::constexpr_align(num_max_ranks + num_experts, num_notify_threads) * sizeof(int);
}

struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    // NOTES: the top-k index is always 32-bit
    int num_topk, num_metadata_bytes;
    void* base;

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        // Metadata includes: top-k indices, weight and source rank/token index
        num_topk(num_topk),
        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +
                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),
        base(base) {
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
    }

    template <bool kWithMBarrier, typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +
                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);
        return static_cast<dtype_t>(num_bytes);
    }

    __forceinline__ __device__ __host__ void* get_base_ptr() const {
        return base;
    }

    __forceinline__ __device__ __host__ void set_base_ptr(void* ptr) {
        base = ptr;
    }

    __forceinline__ __device__ __host__ void* get_hidden_ptr() const {
        return get_base_ptr();
    }

    __forceinline__ __device__ __host__ sf_pack_t* get_sf_ptr() const {
        return math::advance_ptr<sf_pack_t>(base, math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_metadata_ptr() const {
        return math::advance_ptr<int>(get_sf_ptr(), math::align(num_sf_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_topk_idx_ptr() const {
        return get_metadata_ptr();
    }

    __forceinline__ __device__ __host__ float* get_topk_weights_ptr() const {
        return math::advance_ptr<float>(get_metadata_ptr(), num_topk * sizeof(int));
    }

    __forceinline__ __device__ __host__ int* get_src_token_global_idx_ptr() const {
        return math::advance_ptr<int>(get_topk_weights_ptr(), num_topk * sizeof(float));
    }

    __forceinline__ __device__ __host__ int* get_linked_list_idx_ptr() const {
        return get_src_token_global_idx_ptr() + 1;
    }

    __forceinline__ __device__ ptx::mbarrier* get_mbarrier_ptr() const {
        return math::advance_ptr<ptx::mbarrier>(get_metadata_ptr(), math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes));
    }
};

template <bool kWithMBarrier>
struct BufferLayout {
    TokenLayout token_layout;
    int num_ranks;
    int num_max_tokens_per_rank;

    void* base;

    __forceinline__ __device__ __host__
    BufferLayout(const TokenLayout& token_layout,
                 const int& num_ranks,
                 const int& max_num_tokens_per_rank,
                 void* base = nullptr) :
        token_layout(token_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_token() const {
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    __forceinline__ __device__ __host__
    BufferLayout get_rank_buffer(const int& rank_idx) const {
        return BufferLayout(token_layout,
                            1, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx, const int& num_tokens_per_channel) const {
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % num_tokens_per_channel == 0);
        return BufferLayout(token_layout,
                            // Do not use `num_max_tokens_per_rank / num_tokens_per_channel` as the false stride
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * num_tokens_per_channel * channel_idx);
    }

    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        auto token_buffer = token_layout;
        token_buffer.set_base_ptr(static_cast<int8_t*>(base) + get_num_bytes_per_token() * token_idx);
        return token_buffer;
    }
};

}  // namespace nixl_ep_v2::elastic::layout
