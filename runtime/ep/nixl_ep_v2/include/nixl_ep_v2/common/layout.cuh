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

// The control region of the symmetric buffer: every word a peer may write sits at an offset fixed by compile-time
// capacities, and is zero at rest (each kernel zeroes what it used), except the monotonic per-peer barrier words
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
    // Profile slots: stamps below 32 (ns, `%globaltimer`), accumulators and counts above; `read_profile()` zeroes all.
    // Dispatch stamps: 0 entry, 1 start barrier done, 8 token counting done, 2 grid reduction done, 3/4 peer count
    // writes issued/arrived (direct), 18/19 around one thread's count put, 20 scale-out count put issued, 21 rank
    // counts received and forwarded, 22 expert counts received and forwarded (18 to 22 hybrid), 5 prefix sums done,
    // 6/7 end barrier reached/done, 9 any barrier's start grid sync done, 10/11 first/last data warp of SM 0 done
    // (direct) or last scale-out sender / last forward warp done (hybrid, max over warps).
    // Copy epilogue 12 work start / 13 last block done; combine 14 entry / 15 last block done, 27 last scale-up warp /
    // 31 last forward warp done (hybrid, max over warps); reduce epilogue 16 work start / 17 last block done.
    // Dispatch accumulators (SM 0's first scale-out sender warp, per token): 32 loop body, 33 load wait + flag staging
    // + send-buffer store, 34 own-domain store + wait, 35 flag release, 36 RDMA puts, 37 ringing puts; counts 38 RDMA
    // tokens, 39 of them ringing. Combine accumulators (SM 0's first forward warp, per row): 40 RDMA puts, 41 ringing
    // puts, 42 count of rings, 43 store wait, 44 scale-up tail waits, 45 reduce, 46 flag staging + store, 47/49
    // own-domain/RDMA row, 48/50 their counts. Unused: 23 to 26, 28 to 30, 51 to 63.
    static constexpr int kNumProfileSlots = 64;

    static int64_t get_num_bytes() {
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

        // Adaptive kernels: arrival counter and round number of the active-block grid barrier
        num_bytes += kNumActiveGridBarrierBytes;

        // Debug: phase timestamps (see EP_PROFILE_STAMP)
        num_bytes += kNumProfileSlots * sizeof(int64_t);

        // Per-peer barrier: signals (two parities), expectations, staged arrivals (two slots)
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 5;

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

    __forceinline__ __device__ __host__ int64_t* get_profile_ptr() const {
        return math::advance_ptr<int64_t>(get_active_grid_barrier_ptr(), kNumActiveGridBarrierBytes);
    }

    // Per-peer barrier words: the count peer `rank_id` wrote (word by arrival parity), our expectation, our staged arrival
    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_signal_ptr(const int& rank_id, const int& parity) const {
        return math::advance_ptr<int64_t>(get_profile_ptr(), kNumProfileSlots * sizeof(int64_t)) + parity * kNumMaxRanks + rank_id;
    }

    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_expected_ptr(const int& rank_id) const {
        return math::advance_ptr<int64_t>(get_peer_barrier_signal_ptr(0, 0), 2 * kNumMaxRanks * sizeof(int64_t)) + rank_id;
    }

    __forceinline__ __device__ __host__ int64_t* get_peer_barrier_staging_ptr(const int& rank_id, const int& slot) const {
        return math::advance_ptr<int64_t>(get_peer_barrier_expected_ptr(0), kNumMaxRanks * sizeof(int64_t)) + slot * kNumMaxRanks + rank_id;
    }
};

// A token record: `[hidden][SF][metadata]`, each part aligned to `ptx::kNumTMAAlignBytes` (32 bytes), plus (`with_flags`, the communication records
// of the adaptive kernels) the arrival flags of the scale-out path: the packets of one RDMA write land in any order,
// so the last 16 bytes of every `kNumChunkBytes` packet hold a flag; the payload an interior flag displaces goes to a
// spill slot behind the metadata, the last packet's flag is a dedicated slot at the end of the record (strict PCIe
// ordering on the receiving region: `UCX_IB_PCI_RELAXED_ORDERING=n`). The receiver tests every flag word for EQUALITY
// with `kFlagValue`: at rest a flag slot holds the other kernel's payload bytes, so a nonzero test is not enough.
struct TokenLayout {
    static constexpr int kNumChunkBytes = 4096;   // the RDMA packet size: the path MTU, pinned by `ElasticBuffer` (`UCX_IB_PATH_MTU=4096`)
    static constexpr int kNumFlagBytes = 16;      // one flag slot: a 64-bit word and 8 unused bytes
    static constexpr int64_t kFlagValue = 0x5A3C96E17B4DA5C3ll;   // at rest a flag slot holds payload, equal to this with probability 2^-64 per word; zeroed by `clear_flags` once consumed

    int num_hidden_bytes, num_sf_bytes;
    // NOTES: the top-k index is always 32-bit
    bool with_metadata, with_flags;
    int num_topk, num_metadata_bytes;
    int num_metadata_field_bytes;   // the metadata fields, aligned; the spill and flag slots follow
    int num_spill_slots;            // interior packet boundaries of the record (0 without flags)
    void* base;

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr, const bool& with_flags = true) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        // Metadata includes: top-k indices, weight and source rank/token index
        with_metadata(with_metadata), with_flags(with_flags),
        num_topk(num_topk),
        num_metadata_field_bytes(math::align<int>(num_topk * (sizeof(int) + sizeof(float)) +
                                                  (with_metadata ? (1 + num_topk) * sizeof(int) : 0), ptx::kNumTMAAlignBytes)),
        num_spill_slots(0),
        base(base) {
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
        // The slot count depends on the record size, which depends on the slot count: iterate to the fixed point
        while (with_flags) {
            const int num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +
                                  math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                                  num_metadata_field_bytes + kNumFlagBytes * (num_spill_slots + 1);
            const int num_interior = (num_bytes - 1) / kNumChunkBytes;
            if (num_interior == num_spill_slots)
                break;
            num_spill_slots = num_interior;
        }
        EP_UNIFIED_ASSERT(kNumChunkBytes * num_spill_slots <= num_hidden_bytes + math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) + num_metadata_field_bytes and "An interior flag of this record size would land on a spill slot");
        num_metadata_bytes = num_metadata_field_bytes + (with_flags ? kNumFlagBytes * (num_spill_slots + 1) : 0);
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

    // Arrival flags (`with_flags`): flag `k < num_spill_slots` sits at the end of interior packet `k` (over displaced
    // payload bytes), flag `num_spill_slots` is the dedicated slot at the end of the record
    __forceinline__ __device__ __host__ int get_flag_offset(const int& k) const {
        return k < num_spill_slots ? kNumChunkBytes * (k + 1) - kNumFlagBytes : get_num_bytes<false>() - kNumFlagBytes;
    }

    __forceinline__ __device__ __host__ int64_t* get_flag_ptr(const int& k) const {
        return math::advance_ptr<int64_t>(base, get_flag_offset(k));
    }

    // Spill slot `k`: the 16 payload bytes displaced by interior flag `k`
    __forceinline__ __device__ __host__ int4* get_spill_ptr(const int& k) const {
        return math::advance_ptr<int4>(get_metadata_ptr(), num_metadata_field_bytes + kNumFlagBytes * k);
    }

#ifdef __CUDACC__
    // Device-side helpers (the header is also compiled by the host units, which have no PTX wrappers)

    // Sender, on the staged record (one thread): payload under the interior flags to the spill slots, then every flag
    __forceinline__ __device__ void spill_and_set_flags() const {
        for (int k = 0; k < num_spill_slots; ++ k)
            *get_spill_ptr(k) = *reinterpret_cast<const int4*>(get_flag_ptr(k));
        for (int k = 0; k <= num_spill_slots; ++ k)
            *reinterpret_cast<int4*>(get_flag_ptr(k)) = make_int4(static_cast<int>(kFlagValue), static_cast<int>(kFlagValue >> 32), 0, 0);
    }

    // Receiver: every flag word holds the pattern. Relaxed loads, so that all the words are in flight together; the
    // caller issues ONE `fence_acq_rel_sys` after a positive answer and before it reads the record
    __forceinline__ __device__ bool are_flags_set() const {
        bool set = true;
        for (int k = 0; k <= num_spill_slots; ++ k)
            set &= ptx::ld_volatile<int64_t>(get_flag_ptr(k)) == kFlagValue;
        return set;
    }

    // Receiver, after the flags are set (one thread): the displaced payload bytes back over the interior flags
    __forceinline__ __device__ void restore_spills() const {
        for (int k = 0; k < num_spill_slots; ++ k)
            *reinterpret_cast<int4*>(get_flag_ptr(k)) = *get_spill_ptr(k);
    }

    // Receiver, after the record was consumed (one thread): every flag word back to zero
    __forceinline__ __device__ void clear_flags() const {
        for (int k = 0; k <= num_spill_slots; ++ k)
            *get_flag_ptr(k) = 0;
    }

    // Forwarder, on the consumed record in shared memory (one thread): zero the dedicated last flag (the interior ones
    // hold payload again after `restore_spills`), so that no copy outside a live receive slot carries the pattern
    __forceinline__ __device__ void clear_final_flag() const {
        *reinterpret_cast<int4*>(get_flag_ptr(num_spill_slots)) = make_int4(0, 0, 0, 0);
    }

    // Sender, into our own receive slot (one thread): the staged record without its flag slots, as `num_spill_slots + 1`
    // TMA stores (a flag must not travel with unordered bulk data); the caller then fences once and calls `store_flags`
    __forceinline__ __device__ void tma_store_skipping_flags(void* dst_ptr) const {
        int begin = 0;
        for (int k = 0; k <= num_spill_slots; ++ k) {
            const int end = get_flag_offset(k);
            if (end > begin)
                ptx::tma_store_1d(math::advance_ptr(dst_ptr, begin), math::advance_ptr(base, begin), end - begin);
            begin = end + kNumFlagBytes;
        }
    }

    // The flag stores of a record whose bulk stores completed, into our own receive slot (GPU scope; one fence per batch)
    __forceinline__ __device__ void store_flags(void* dst_ptr) const {
        for (int k = 0; k <= num_spill_slots; ++ k)
            ptx::st_relaxed_gpu(math::advance_ptr(dst_ptr, get_flag_offset(k)), kFlagValue);
    }
#endif  // __CUDACC__
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
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx, token_layout.with_flags);
    }
};

}  // namespace nixl_ep_v2::elastic
