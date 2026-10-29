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

#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/math.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>

// Debug-only phase timestamps: `%globaltimer` (ns) written into workspace slots. `EP_PROFILE_STAMP`: SM 0, thread 0;
// `EP_PROFILE_STAMP_WARP`: SM 0, lane 0 of the given warp; `EP_PROFILE_STAMP_LAST`: every warp, folded with a max (below).
// Compiled in only with `EP_ADAPTIVE_PROFILE=1` (JIT flag), otherwise a no-op.
// The macros use `sm_idx` (block index) and `thread_idx` (thread index), which must be in scope at the call site.
// `EP_PROFILE_STAMP_THREAD`: SM 0, the given thread. Accumulators (below): `EP_PROFILE_ACCUM_BEGIN`/`EP_PROFILE_ACCUM_END`
// add the time between the two to a slot, `EP_PROFILE_COUNT` adds one; both take their writer as the `cond` argument.
#ifdef EP_ADAPTIVE_PROFILE
#define EP_PROFILE_STAMP(workspace_layout, slot)                                              \
    do {                                                                                      \
        if (sm_idx == 0 and thread_idx == 0)                                                  \
            (workspace_layout).get_profile_ptr()[(slot)] = static_cast<int64_t>(ptx::globaltimer()); \
    } while (0)
#define EP_PROFILE_STAMP_THREAD(workspace_layout, slot, thread)                               \
    do {                                                                                      \
        if (sm_idx == 0 and thread_idx == (thread))                                           \
            (workspace_layout).get_profile_ptr()[(slot)] = static_cast<int64_t>(ptx::globaltimer()); \
    } while (0)
#define EP_PROFILE_STAMP_WARP(workspace_layout, slot, warp)                                   \
    do {                                                                                      \
        if (sm_idx == 0 and (thread_idx / 32) == (warp) and (thread_idx % 32) == 0)          \
            (workspace_layout).get_profile_ptr()[(slot)] = static_cast<int64_t>(ptx::globaltimer()); \
    } while (0)
// Every warp's lane 0 folds its timestamp in with a max: the slot ends up holding the finish time of the
// last warp of the last block, i.e. the true end of the kernel (for the PDL-launched epilogues, whose Kineto
// duration also counts the time their blocks sit in `cudaGridDependencySynchronize` waiting for the main
// kernel). Per warp, not per block: the hybrid roles' warps finish at very different times.
#define EP_PROFILE_STAMP_LAST(workspace_layout, slot)                                         \
    do {                                                                                      \
        if ((thread_idx % 32) == 0)                                                           \
            atomicMax(reinterpret_cast<unsigned long long*>((workspace_layout).get_profile_ptr() + (slot)), \
                      static_cast<unsigned long long>(ptx::globaltimer()));                   \
    } while (0)
// `EP_PROFILE_ACCUM_BEGIN` reads the clock into a local, `EP_PROFILE_ACCUM_END` adds the nanoseconds since it to the
// slot when `cond` holds, `EP_PROFILE_COUNT` adds one; the local may stay unused in some instantiations ([[maybe_unused]])
#define EP_PROFILE_ACCUM_BEGIN(name) [[maybe_unused]] const uint64_t name = ptx::globaltimer()
#define EP_PROFILE_ACCUM_END(workspace_layout, slot, name, cond)                              \
    do {                                                                                      \
        if (cond)                                                                             \
            atomicAdd(reinterpret_cast<unsigned long long*>((workspace_layout).get_profile_ptr() + (slot)), \
                      static_cast<unsigned long long>(ptx::globaltimer() - (name)));          \
    } while (0)
#define EP_PROFILE_COUNT(workspace_layout, slot, cond)                                        \
    do {                                                                                      \
        if (cond)                                                                             \
            atomicAdd(reinterpret_cast<unsigned long long*>((workspace_layout).get_profile_ptr() + (slot)), 1ull); \
    } while (0)
#else
#define EP_PROFILE_STAMP(workspace_layout, slot) do {} while (0)
#define EP_PROFILE_STAMP_THREAD(workspace_layout, slot, thread) do {} while (0)
#define EP_PROFILE_STAMP_WARP(workspace_layout, slot, warp) do {} while (0)
#define EP_PROFILE_STAMP_LAST(workspace_layout, slot) do {} while (0)
#define EP_PROFILE_ACCUM_BEGIN(name) do {} while (0)
#define EP_PROFILE_ACCUM_END(workspace_layout, slot, name, cond) do {} while (0)
#define EP_PROFILE_COUNT(workspace_layout, slot, cond) do {} while (0)
#endif

namespace nixl_ep_v2::elastic::layout {

// The forwarding metadata of a hybrid dispatch (`token_metadata_at_forward`), replayed by the combine's forward warps. Per
// channel: one row per forwarded token, `num_scaleout_ranks * num_max_tokens_per_channel` at most, then a footer. A row holds
// the source token's global index, the top-k scale-up peers, their destination slots, and the scale-up tail each selection
// needs in the combine (the list capacity minus the token's position in that peer's channel list; zero without a peer). The
// footer holds -(number of token rows + 1), then one forwarded-token count per source domain; it takes further rows when
// `1 + num_scaleout_ranks` exceeds the row width. The host allocates with it, both kernels index with it: the kernels give
// their top-k count as the template argument, so that the row stride is a compile-time constant; the host leaves it at 0 and
// passes the runtime count as the third member
template <int kNumTopk = 0>
struct ForwardMetadataLayout {
    int num_scaleout_ranks, num_max_tokens_per_channel, num_topk = kNumTopk;

    __forceinline__ __device__ __host__ constexpr int topk() const { return kNumTopk > 0 ? kNumTopk : num_topk; }
    __forceinline__ __device__ __host__ constexpr int num_dims() const { return 1 + topk() * 3; }
    __forceinline__ __device__ __host__ int num_token_rows() const { return num_scaleout_ranks * num_max_tokens_per_channel; }
    __forceinline__ __device__ __host__ int num_footer_rows() const { return math::ceil_div(1 + num_scaleout_ranks, num_dims()); }
    __forceinline__ __device__ __host__ int num_rows() const { return num_token_rows() + num_footer_rows(); }
    __forceinline__ __device__ __host__ int num_words_per_channel() const { return num_rows() * num_dims(); }   // fits an int, as the kernels' indices do

    // Words of token row `row`
    __forceinline__ __device__ __host__ int src_token_word(const int& row) const { return row * num_dims(); }
    __forceinline__ __device__ __host__ int peer_word(const int& row, const int& k) const { return row * num_dims() + 1 + k; }
    __forceinline__ __device__ __host__ int slot_word(const int& row, const int& k) const { return row * num_dims() + 1 + topk() + k; }
    __forceinline__ __device__ __host__ int required_tail_word(const int& row, const int& k) const { return row * num_dims() + 1 + topk() * 2 + k; }

    // Footer words
    __forceinline__ __device__ __host__ int length_word() const { return num_token_rows() * num_dims(); }
    __forceinline__ __device__ __host__ int domain_count_word(const int& scaleout_rank_idx) const { return length_word() + 1 + scaleout_rank_idx; }
};

// The control region of the symmetric buffer: every word a peer may write sits at an offset fixed by compile-time
// capacities, and is zero at rest (each kernel zeroes what it used), except the combine readiness counters,
// monotonic per-peer barrier words and the zero-copy words (call-tagged, never cleaned in-kernel,
// reset by the host on a world change)
struct WorkspaceLayout {
    void* workspace;

    int num_ranks;
    int num_scaleout_ranks, num_scaleup_ranks;
    int num_experts, num_experts_per_rank;

    static constexpr int kNumMaxRanks = topology::kNumMaxRankIds;   // dense rank ids of the largest declared world
    static constexpr int kNumMaxExperts = 2048;
    static constexpr int kNumMaxExpertsPerRank = 256;

    __forceinline__ __device__ __host__
    WorkspaceLayout(void* workspace,
                    const int& num_scaleout_ranks,
                    const int& num_scaleup_ranks,
                    const int& num_experts):
        workspace(workspace),
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),
        num_scaleout_ranks(num_scaleout_ranks),
        num_scaleup_ranks(num_scaleup_ranks),
        num_experts(num_experts) {
        num_experts_per_rank = num_experts / num_ranks;
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        EP_UNIFIED_ASSERT(num_experts <= kNumMaxExperts);
        EP_UNIFIED_ASSERT(num_experts_per_rank <= kNumMaxExpertsPerRank);
    }

    // Arrival counter and round number of the active-block grid barrier (two words, padded to a line)
    static constexpr int64_t kNumActiveGridBarrierBytes = 128;
    // Token take counters of the direct paths (`[0]` dispatch tokens, `[1]` combine row chunks), a line of their own:
    // zero at rest (the workspace is zero-initialised and each direct launch zeroes its counter at its end, once every take
    // of the launch is behind the grid arrival of the peer-signalled end sequence or the end barrier's grid sync with zero
    // copy), so a launch needs no ordering at its start
    static constexpr int64_t kNumTakeCounterBytes = 128;
    // Profile slots: stamps below 32 (ns, `%globaltimer`), accumulators and counts above; `read_profile()` zeroes all.
    // Dispatch stamps: 0 entry, 1 entry sync done (hybrid: start barrier; peer-signalled direct: READY published; zero copy: none),
    // 8 token counting done, 2 grid reduction done, 3/4 peer count writes issued / counts received (direct; peer-signalled: 3 -> 4
    // includes SM 0's end-sequence hold for its data warps), 18/19 around one thread's count put, 20 scale-out count put issued,
    // 21 rank counts received and forwarded, 22 expert counts received and forwarded (18 to 22 hybrid), 5 prefix sums done, 6 end
    // reached by SM 0 thread 0 (peer-signalled: its end sequence already ran), 7 end done (peer-signalled direct: last warp done,
    // max over warps; otherwise SM 0 past the end barrier), 9 a barrier's start grid sync done (not stamped by the peer-signalled
    // direct kernels), 10/11 first/last data warp of SM 0 done (direct) or last scale-out sender / last forward warp done (hybrid,
    // max over warps).
    // Copy epilogue 12 work start / 13 last block done; combine 14 entry / 15 last block done, 27 last scale-up warp /
    // 31 last forward warp done (hybrid, max over warps); reduce epilogue 16 work start / 17 last block done.
    // Dispatch accumulators (SM 0's first scale-out sender warp, per token): 32 loop body, 33 load wait
    // + send-buffer store, 34 own-domain store + wait, 35 completion release, 36 RDMA puts, 37 ringing puts; counts 38 RDMA
    // tokens, 39 of them ringing. Combine accumulators (SM 0's first forward warp, per row): 40 RDMA puts, 41 ringing
    // puts, 42 count of rings, 43 store wait, 44 scale-up tail waits, 45 reduce, 46 payload store, 47/49
    // own-domain/RDMA row, 48/50 their counts. Zero copy: 3/4 count row put / rows swept (SM 0), 23 bases held,
    // 24 count pass done, 25 first token stored. Unused: 26, 28 to 30, 51 to 63.
    static constexpr int kNumProfileSlots = 64;

    static constexpr int64_t kNumZeroCopyControlBytes = 128;

    // Each array holds one atomic token ready counter per channel and source domain.
    static constexpr int64_t kNumReadyAtomicBytes =
        static_cast<int64_t>(adaptive::kNumMaxChannels) * topology::kNumMaxScaleoutRanksCap * sizeof(int64_t);

    static constexpr int64_t get_num_bytes() {
        int64_t num_bytes = 0;

        // Notify reduction workspace (rank counts, expert counts, then the per-channel record counts per domain)
        num_bytes += (kNumMaxRanks + kNumMaxExperts + static_cast<int64_t>(kNumMaxChannels) * topology::kNumMaxScaleoutRanksCap) * sizeof(int64_t);

        // Scaleup notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 2;
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int64_t) * 2;

        // Scaleup atomic sender count
        num_bytes += kNumMaxRanks * sizeof(int);

        // Scaleout notify threads
        // Expert send/recv count
        num_bytes += static_cast<int64_t>(topology::kNumMaxScaleoutRanksCap) * kNumMaxExperts * sizeof(int) * 2;

        // Channel aggregated into the scaleup domains
        // Also reused for channel scaleup tail
        num_bytes += static_cast<int64_t>(kNumMaxChannels) * topology::kNumMaxScaleupRanksCap * sizeof(int);

        // Adaptive kernels: arrival counter and round number of the active-block grid barrier, then the take counters
        num_bytes += kNumActiveGridBarrierBytes;
        num_bytes += kNumTakeCounterBytes;

        // Debug: phase timestamps (see EP_PROFILE_STAMP)
        num_bytes += kNumProfileSlots * sizeof(int64_t);

        // Per-peer barrier: signals (two parities), expectations, staged arrivals (two slots)
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 5;

        // Separate dispatch and combine atomic token ready counters
        num_bytes += 2 * kNumReadyAtomicBytes;

        // Zero copy: the row-base words (one per position) and the control line (the fresh-call counter)
        num_bytes += topology::kNumMaxScaleupRanksCap * sizeof(int64_t);
        num_bytes += kNumZeroCopyControlBytes;

        return num_bytes;
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        return static_cast<int64_t*>(workspace);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int64_t>(get_notify_reduction_workspace_ptr(), (kNumMaxRanks + kNumMaxExperts + static_cast<int64_t>(kNumMaxChannels) * topology::kNumMaxScaleoutRanksCap) * sizeof(int64_t));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + kNumMaxRanks;
    }
    // The direct dispatch writes and polls the receive-side rank and expert count slots in 16-byte pairs (`ptx::st_relaxed_sys` of an
    // int4, `ptx::ld_volatile_int4`): both slot bases must stay 16-byte aligned, i.e. an even number of 8-byte words before each. The
    // rank slots start 2 (kNumMaxRanks + kNumMaxExperts) + kNumMaxChannels * kNumMaxScaleoutRanksCap words into the workspace, the
    // expert slots kNumMaxRanks words later
    static_assert((static_cast<int64_t>(kNumMaxChannels) * topology::kNumMaxScaleoutRanksCap) % 2 == 0 and kNumMaxRanks % 2 == 0,
                  "The receive-side count slots must be 16-byte aligned");

    // Word `i` of the contiguous "rank counts then expert counts" view with `num_rank_slots` rank slots
    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_count_ptr(const int& i, const int& num_rank_slots) const {
        return i < num_rank_slots ?
            get_scaleup_rank_count_ptr<kIsSendBuffer>() + i :
            get_scaleup_expert_count_ptr<kIsSendBuffer>() + (i - num_rank_slots);
    }

    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {
        return math::advance_ptr<int>(
            get_scaleup_rank_expert_count_ptr<true>(), 2 * (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int>(get_scaleup_atomic_sender_counter(), kNumMaxRanks * sizeof(int));
        return base_ptr + (kIsSendBuffer ? 0 : topology::kNumMaxScaleoutRanksCap * kNumMaxExperts);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>();
        return base_ptr + scaleout_rank_idx * kNumMaxExperts + expert_idx;
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_scaleout_rank_expert_count_ptr<true>(),
            2 * static_cast<int64_t>(topology::kNumMaxScaleoutRanksCap) * kNumMaxExperts * sizeof(int));
        return base_ptr + (channel_idx * topology::kNumMaxScaleupRanksCap + scaleup_rank_idx);
    }

    // `[0]` arrivals, `[1]` round number; both start at zero (the workspace is zero-initialised)
    __forceinline__ __device__ __host__ unsigned* get_active_grid_barrier_ptr() const {
        return math::advance_ptr<unsigned>(
            get_channel_scaleup_tail_ptr(0, 0), static_cast<int64_t>(kNumMaxChannels) * topology::kNumMaxScaleupRanksCap * sizeof(int));
    }

    // The direct paths' token take counters (see `kNumTakeCounterBytes`)
    __forceinline__ __device__ __host__ unsigned* get_take_counter_ptr(const int& idx) const {
        return math::advance_ptr<unsigned>(get_active_grid_barrier_ptr(), kNumActiveGridBarrierBytes) + idx;
    }

    __forceinline__ __device__ __host__ int64_t* get_profile_ptr() const {
        return math::advance_ptr<int64_t>(get_active_grid_barrier_ptr(), kNumActiveGridBarrierBytes + kNumTakeCounterBytes);
    }

    // Per-peer barrier words: the count peer `rank_id` wrote (the hybrid barrier legs index the word by arrival parity, the direct
    // kernels give the two words fixed roles, `comm::kPeerReadyWord` / `kPeerArrivedWord`), our expectation, our staged arrival
    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_signal_ptr(const int& rank_id, const int& word) const {
        return math::advance_ptr<int64_t>(get_profile_ptr(), kNumProfileSlots * sizeof(int64_t)) + word * kNumMaxRanks + rank_id;
    }

    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_expected_ptr(const int& rank_id) const {
        return math::advance_ptr<int64_t>(get_peer_barrier_signal_ptr(0, 0), 2 * kNumMaxRanks * sizeof(int64_t)) + rank_id;
    }

    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_staging_ptr(const int& rank_id, const int& word) const {
        return math::advance_ptr<int64_t>(get_peer_barrier_expected_ptr(0), kNumMaxRanks * sizeof(int64_t)) + word * kNumMaxRanks + rank_id;
    }

    __forceinline__ __device__ __host__ int64_t* get_dispatch_ready_atomic_buff() const {
        return math::advance_ptr<int64_t>(get_peer_barrier_staging_ptr(0, 0), 2 * kNumMaxRanks * sizeof(int64_t));
    }

    __forceinline__ __device__ __host__ int64_t* get_combine_ready_atomic_buff() const {
        return math::advance_ptr<int64_t>(get_dispatch_ready_atomic_buff(), kNumReadyAtomicBytes);
    }

    // Zero copy: `[dst position]` -> this rank's first compact row at that destination, epoch-tagged (`encode_zc_base`)
    __forceinline__ __device__ __host__ int64_t* get_zc_rank_base_ptr() const {
        return math::advance_ptr<int64_t>(get_combine_ready_atomic_buff(), kNumReadyAtomicBytes);
    }

    __forceinline__ __device__ __host__ unsigned long long* get_zc_fresh_call_counter_ptr() const {
        return math::advance_ptr<unsigned long long>(get_zc_rank_base_ptr(), topology::kNumMaxScaleupRanksCap * sizeof(int64_t));
    }

    // Count word (int): bit 31 = 0 | 30..28 slot | 27 expand | 26..20 tag (1..127, the call index halved) | 19..0 count + 1
    // Base word (int64): [epoch = call index + 1 | base]; a zeroed word of either kind never matches
    static constexpr int kNumZcCountBits = 20, kNumZcTagBits = 7, kNumZcTags = 127;
    static constexpr int kZcTagShift = kNumZcCountBits, kZcExpandBit = 27, kZcSlotShift = 28;
    static constexpr int kNumMaxZeroCopySlots = 8;
    static constexpr uint32_t kZcCountMask = (1u << kNumZcCountBits) - 1u;
    static constexpr uint32_t kZcTagMask = (1u << kNumZcTagBits) - 1u;
    static constexpr uint32_t kZcSlotMask = static_cast<uint32_t>(kNumMaxZeroCopySlots) - 1u;
    static constexpr uint32_t kZcFieldsMask = ~kZcCountMask & 0x7fffffffu;   // slot, expand and tag
    __host__ __device__ static constexpr uint32_t zc_call_parity(uint32_t idx) { return idx & 1u; }
    __host__ __device__ static constexpr uint32_t zc_call_tag(uint32_t idx) { return ((idx >> 1) % kNumZcTags) + 1u; }   // [1, 127], period 254 fresh calls
    __host__ __device__ static constexpr uint32_t zc_fields(uint32_t tag, int slot, bool expand) {
        return ((static_cast<uint32_t>(slot) & kZcSlotMask) << kZcSlotShift) | (static_cast<uint32_t>(expand) << kZcExpandBit) | ((tag & kZcTagMask) << kZcTagShift);
    }
    __host__ __device__ static constexpr int zc_make_count_word(uint32_t tag, int slot, bool expand, int count) {
        return static_cast<int>(zc_fields(tag, slot, expand) | ((static_cast<uint32_t>(count) + 1u) & kZcCountMask));
    }
    __host__ __device__ static constexpr bool zc_word_matches(int word, uint32_t tag, int slot, bool expand) {
        return (static_cast<uint32_t>(word) & kZcFieldsMask) == zc_fields(tag, slot, expand);
    }
    __host__ __device__ static constexpr bool zc_word_is_foreign(int word, uint32_t tag, int slot, bool expand) {   // tag equal, slot or expand not: a mismatched collective call
        return ((static_cast<uint32_t>(word) >> kZcTagShift) & kZcTagMask) == (tag & kZcTagMask) and not zc_word_matches(word, tag, slot, expand);
    }
    __host__ __device__ static constexpr int decode_zc_tagged_count(int word) { return static_cast<int>(static_cast<uint32_t>(word) & kZcCountMask) - 1; }
    __host__ __device__ static constexpr uint32_t zc_base_epoch(uint32_t idx) { return (idx & 0x7fffffffu) + 1u; }
    __host__ __device__ static constexpr int64_t encode_zc_base(uint32_t idx, int base) {
        return static_cast<int64_t>((static_cast<uint64_t>(zc_base_epoch(idx)) << 32) | static_cast<uint32_t>(base));
    }
    __host__ __device__ static constexpr bool is_zc_base_ready(int64_t slot, uint32_t idx) {
        return static_cast<uint32_t>(static_cast<uint64_t>(slot) >> 32) == zc_base_epoch(idx);
    }
    __host__ __device__ static constexpr int decode_zc_base(int64_t slot) { return static_cast<int>(slot & 0xffffffffll); }
};

static_assert((WorkspaceLayout::get_num_bytes() - WorkspaceLayout::kNumZeroCopyControlBytes -
               topology::kNumMaxScaleupRanksCap * sizeof(int64_t)) % 128 == 0 and
              (WorkspaceLayout::get_num_bytes() - WorkspaceLayout::kNumZeroCopyControlBytes) % 128 == 0 and
              WorkspaceLayout::kNumZeroCopyControlBytes % 16 == 0, "Unaligned zero-copy words");
static_assert(WorkspaceLayout::get_num_bytes() <= 2 * 1024 * 1024, "The workspace outgrew its 2 MB alignment unit");
static_assert(WorkspaceLayout::kNumZcCountBits + WorkspaceLayout::kNumZcTagBits + 1 + 3 == 31 and
              WorkspaceLayout::kNumZcTags == (1 << WorkspaceLayout::kNumZcTagBits) - 1 and
              WorkspaceLayout::kNumMaxZeroCopySlots == (1 << (31 - WorkspaceLayout::kZcSlotShift)), "Count word fields do not tile bits 0..30");
static_assert(WorkspaceLayout::zc_call_tag(0) == 1 and WorkspaceLayout::zc_call_tag(0xffffffffu) >= 1 and
              WorkspaceLayout::zc_call_tag(0xffffffffu) <= WorkspaceLayout::kNumZcTags and
              WorkspaceLayout::zc_call_tag(2 * WorkspaceLayout::kNumZcTags) == 1 and
              WorkspaceLayout::zc_call_tag(2 * WorkspaceLayout::kNumZcTags - 1) == WorkspaceLayout::kNumZcTags, "Bad call tag");
static_assert(WorkspaceLayout::zc_make_count_word(WorkspaceLayout::kNumZcTags, WorkspaceLayout::kNumMaxZeroCopySlots - 1, true, (1 << 20) - 2) > 0 and
              WorkspaceLayout::zc_make_count_word(WorkspaceLayout::kNumZcTags, WorkspaceLayout::kNumMaxZeroCopySlots, true, (1 << 20) - 2) > 0 and
              WorkspaceLayout::zc_make_count_word(1, 0, false, 0) == ((1 << 20) | 1) and
              WorkspaceLayout::decode_zc_tagged_count(WorkspaceLayout::zc_make_count_word(5, 3, false, 4095)) == 4095 and
              WorkspaceLayout::zc_word_matches(WorkspaceLayout::zc_make_count_word(5, 3, false, 4095), 5, 3, false) and
              not WorkspaceLayout::zc_word_matches(0, 1, 0, false) and not WorkspaceLayout::zc_word_is_foreign(0, 1, 0, false) and
              not WorkspaceLayout::zc_word_matches(WorkspaceLayout::zc_make_count_word(5, 3, false, 7), 6, 3, false) and
              not WorkspaceLayout::zc_word_is_foreign(WorkspaceLayout::zc_make_count_word(5, 3, false, 7), 6, 3, false) and
              WorkspaceLayout::zc_word_is_foreign(WorkspaceLayout::zc_make_count_word(5, 1, false, 7), 5, 0, false) and
              WorkspaceLayout::zc_word_is_foreign(WorkspaceLayout::zc_make_count_word(5, 0, true, 7), 5, 0, false), "Bad count word");
static_assert(not WorkspaceLayout::is_zc_base_ready(0, 0) and not WorkspaceLayout::is_zc_base_ready(0, 0xffffffffu) and
              WorkspaceLayout::is_zc_base_ready(WorkspaceLayout::encode_zc_base(0x7fffffffu, 5), 0x7fffffffu) and
              WorkspaceLayout::is_zc_base_ready(WorkspaceLayout::encode_zc_base(0xffffffffu, 7), 0xffffffffu) and
              not WorkspaceLayout::is_zc_base_ready(WorkspaceLayout::encode_zc_base(3, 7), 4) and
              WorkspaceLayout::decode_zc_base(WorkspaceLayout::encode_zc_base(3, 123456)) == 123456 and
              WorkspaceLayout::decode_zc_base(WorkspaceLayout::encode_zc_base(0xffffffffu, 0x7fffffff)) == 0x7fffffff, "Bad base slot");

// A token record: `[hidden][SF][metadata]`, each part aligned to 32 bytes.
// Completion counters live in dedicated arrays and are never copied with the payload.
struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    // NOTES: the top-k index is always 32-bit
    bool with_metadata;
    int num_topk, num_metadata_bytes;
    void* base;

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        // Metadata includes: top-k indices, weight and source rank/token index
        with_metadata(with_metadata),
        num_topk(num_topk),
        num_metadata_bytes(math::align<int>(num_topk * (sizeof(int) + sizeof(float)) +
                                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0), ptx::kNumTMAAlignBytes)),
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

    template <int kNumTokensPerChannel>
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx) const {
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % kNumTokensPerChannel == 0);
        return BufferLayout(token_layout,
                            // Do not use `num_max_tokens_per_rank / kNumTokensPerChannel` as the false stride
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * kNumTokensPerChannel * channel_idx);
    }

    // Runtime channel size (adaptive kernels: the channel count follows the active SM count)
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx, const int& num_tokens_per_channel) const {
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % num_tokens_per_channel == 0);
        return BufferLayout(token_layout,
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * num_tokens_per_channel * channel_idx);
    }

    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        return TokenLayout(token_layout.num_hidden_bytes, token_layout.num_sf_bytes, token_layout.num_topk, token_layout.with_metadata,
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx);
    }
};

}  // namespace nixl_ep_v2::elastic
