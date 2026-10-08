#pragma once

#include <nixl_ep_v2/common/comm.cuh>
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/math.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>


namespace nixl_ep_v2::elastic::adaptive {

// DeepSeek's hybrid dispatch on a fixed grid; rank decomposition and active block count come from the topology descriptor (`EPTopology`)
// Elastic on the NIXL transport: the live world is the mask product of `topology.cuh`; masked peers are not counted, written, signalled or waited for
template <bool kDoCPUSync,
          int kNumLaunchSMs,
          int kNumNotifyWarps, int kNumChannelsPerSM,
          int kNumMaxScaleoutRanks, int kNumMaxScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExpertsPerRank, int kNumTopk, int kExpertAlignment,
          int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumMaxScaleupRanks, 32),
          int kNumScaleoutWarps = kNumChannelsPerSM,
          int kNumForwardWarps = kNumChannelsPerSM,
          int kScaleoutUpdateInterval = 6,
          int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,
          int kNumMaxRanks = kNumMaxScaleoutRanks * kNumMaxScaleupRanks,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,
          int kNumForwardThreads = kNumForwardWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
adaptive_dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* token_metadata_at_forward,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    void* buffer, void* send_buffer,
    void* workspace, void* mapped_host_workspace,
    const topology::EPTopology* topology_ptr) {
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    EP_STATIC_ASSERT(kNumMaxScaleoutRanks <= topology::kNumMaxScaleoutRanksCap, "Too many scale-out ranks");
    EP_STATIC_ASSERT(kNumMaxScaleupRanks <= topology::kNumMaxScaleupRanksCap, "Too many scale-up ranks");
    EP_STATIC_ASSERT(kNumMaxScaleoutRanks <= kNumNotifyThreads, "kNumMaxScaleoutRanks must be less than kNumNotifyThreads");
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumChannelsPerSM and kNumForwardWarps == kNumChannelsPerSM, "The channel indices assume one scale-out and one forward warp per channel");
    EP_STATIC_ASSERT(kNumLaunchSMs % 2 == 0, "The adaptive grid is launched in clusters of two");

    // Topology: uniform per block, read once
    const auto topo = topology::load(topology_ptr);
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);

    // Surplus blocks of the fixed grid release their SM at once (before any barrier or mbarrier init)
    if (sm_idx >= topo.num_active_sms)
        return;

    // The transport context comes from the topology descriptor (`EPTopology`), not from the launch
    const auto ctx = nixl::load_context(topo.transport);

    // Runtime topology-derived values (compile-time constants in the hybrid kernel)
    const int num_scaleout_ranks = topo.num_scaleout_ranks, num_scaleup_ranks = topo.num_scaleup_ranks;
    const int scaleout_rank_idx = topo.scaleout_rank_idx, scaleup_rank_idx = topo.scaleup_rank_idx;
    const int num_active_sms = topo.num_active_sms;
    const int num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    const int num_live_scaleup_ranks = topo.scaleup_mask.popc();
    // Fixed expert slots per rank: the live expert count follows the live rank count (slot `e` is owned by rank `e / slots`)
    const int num_experts_per_rank = kNumExpertsPerRank;
    const int num_experts = num_ranks * kNumExpertsPerRank;
    const int num_experts_per_scaleout = num_scaleup_ranks * kNumExpertsPerRank;
    const int num_channels = kNumChannelsPerSM * num_active_sms;
    const int num_max_tokens_per_channel = math::ceil_div(kNumMaxTokensPerRank, num_channels);
    EP_DEVICE_ASSERT(num_scaleout_ranks <= kNumMaxScaleoutRanks and num_scaleup_ranks <= kNumMaxScaleupRanks);
    EP_DEVICE_ASSERT(num_active_sms <= kNumLaunchSMs and num_active_sms % 2 == 0);
    EP_DEVICE_ASSERT(topo.is_participant() and "A rank outside the live world must not launch");

    // Direct mode (MNNVL): one flat NVLink team, which the topology descriptor (`EPTopology`) expresses as `num_scaleout_ranks` equal to 1;
    // the data warps write straight into the peers' scale-up buffers instead of staging through the
    // scale-out send buffer and the forward warps
    const bool is_direct = topo.mode == topology::kModeDirect;
    EP_DEVICE_ASSERT(not is_direct or num_scaleout_ranks == 1);

    // Utils
    // NOTES: a warp is a channel (several channels may share one transport channel, see `comm::get_channel_idx`)
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto rank_idx = scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;

    // Dense rank ids of our peers: the scale-up peer at position `k` of our domain, the rail peer in domain `d`
    const auto scaleup_peer_id = [=](const int& k) { return topology::rank_id(scaleout_rank_idx, k, kNumMaxScaleupRanks); };
    const auto scaleout_peer_id = [=](const int& d) { return topology::rank_id(d, scaleup_rank_idx, kNumMaxScaleupRanks); };

    // Whether the rank owning a (global) expert is live; a selection of a dead expert is dropped
    const auto is_expert_owner_live = [=](const int& expert_idx) {
        const int owner = expert_idx / num_experts_per_rank;
        return topo.is_scaleout_live(owner / num_scaleup_ranks) and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, owner % num_scaleup_ranks);
    };

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, num_scaleout_ranks, num_scaleup_ranks, num_experts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, num_scaleout_ranks, num_scaleup_ranks, num_experts);
    EP_PROFILE_STAMP(workspace_layout, 0);

    // A domain's count row: its expert counts, then the rank counts (word `num_experts_per_scaleout + k`), then the
    // per-channel record counts (word `num_experts_per_scaleout + num_scaleup_ranks + c`); ONE put per rail peer
    // Our own domain's words go straight into our receive copy, a rail peer's into the send copy
    const auto scaleout_count_ptr = [&](const int& d, const int& word) {
        return d == scaleout_rank_idx ? workspace_layout.get_scaleout_expert_count_ptr<false>(d, word)
                                      : workspace_layout.get_scaleout_expert_count_ptr<true>(d, word);
    };

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    // The notify region is sized for the compile-time maxima (the expert slots of the maximum world: its ranks times the slots per rank)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumMaxChannelsInGrid = kNumLaunchSMs * kNumChannelsPerSM;
    constexpr int kNumMaxExpertSlots = kNumMaxRanks * kNumExpertsPerRank;
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumMaxRanks + kNumMaxExpertSlots + kNumMaxChannelsInGrid * kNumMaxScaleoutRanks, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // Transport handle. Each warp is a channel, spread over the launch grid and the context's channel count (`comm::get_channel_idx`)
    const auto channel_idx = comm::get_channel_idx<kNumLaunchSMs, kNumChannelsPerSM, (kNumNotifyWarps > 0)>(
        sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, ctx.num_channels, warp_idx < kNumNotifyWarps);
    const auto transport = nixl::NIXLTransport(ctx, channel_idx);

    // Global parallel barriers for scale-out subteam and scale-up subteam
    comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kHybridDispatchTag0, kNumMaxScaleupRanks, false, false, true>(
        transport, workspace_layout, topo, sm_idx, thread_idx);
    EP_PROFILE_STAMP(workspace_layout, 1);

    // The golden layout during the whole process for both scale-out and forward warps
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumScaleoutWarps + kNumForwardWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);

    // All the buffers
    // The scale-out send buffer (the RDMA sources) is a region of its own: the NIC reads it after this kernel has ended; the next
    // dispatch rewrites it only past its start barrier, which every receiver reaches after consuming this launch's records (see `comm::gpu_barrier_rt`)
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, num_scaleup_ranks, num_scaleout_ranks * kNumMaxTokensPerRank, buffer);
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, 1, kNumMaxTokensPerRank, send_buffer);
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, num_scaleout_ranks, num_channels * num_max_tokens_per_channel, scaleup_buffer.get_buffer_end_ptr());

    // Init TMA for scale-out and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // Assign shared memory
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + num_ranks;
        int* channel_count = rank_expert_count + num_ranks + num_experts;   // `[channel][domain]`, hybrid only
        const int num_reduction_entries = num_ranks + num_experts + (is_direct ? 0 : num_channels * num_scaleout_ranks);
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * num_active_sms) {
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0 and not is_expert_owner_live(dst_expert_idx))
                dst_expert_idx = -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / num_experts_per_rank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);

            // Hybrid: the token's channel sends one record to every domain with a live selection
            if (not is_direct) {
                const auto domain_mask = ptx::reduce_or(dst_expert_idx >= 0 ? (1u << (dst_expert_idx / num_experts_per_scaleout)) : 0u);
                if (lane_idx < num_scaleout_ranks and ((domain_mask >> lane_idx) & 1))
                    atomicAdd_block(channel_count + (i % num_channels) * num_scaleout_ranks + lane_idx, 1);
            }
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
        EP_PROFILE_STAMP(workspace_layout, 8);

        // Do full-grid reduction
        for (int i = thread_idx; i < num_reduction_entries; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0
        if (sm_idx == 0) {
            // Reduce all SM's count
            // Wait all ACTIVE SMs' arrival
            for (int i = thread_idx; i < num_reduction_entries; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == num_active_sms) {
                        // Encode and write into the send buffer (hybrid) or shared memory (direct)
                        const auto encoded = math::encode_decode_positive<int>(status & 0xffffffffll);
                        if (is_direct) {
                            rank_expert_count[i] = encoded;
                        } else {
                            // Hybrid: rank count of rank `i`, expert count of expert `i - num_ranks`, or record count of channel `c`
                            // towards domain `d`, each into its domain's count row
                            int d, word;
                            if (i < num_ranks) {
                                d = i / num_scaleup_ranks, word = num_experts_per_scaleout + i % num_scaleup_ranks;
                            } else if (i < num_ranks + num_experts) {
                                d = (i - num_ranks) / num_experts_per_scaleout, word = (i - num_ranks) % num_experts_per_scaleout;
                            } else {
                                const int j = i - num_ranks - num_experts;
                                d = j % num_scaleout_ranks, word = num_experts_per_scaleout + num_scaleup_ranks + j / num_scaleout_ranks;
                            }
                            *scaleout_count_ptr(d, word) = encoded;
                        }

                        // Clean for the next usage
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("NIXL EP V2 adaptive notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, num_scaleout_ranks, scaleup_rank_idx, num_scaleup_ranks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), num_active_sms);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
            EP_PROFILE_STAMP(workspace_layout, 2);

            if (is_direct) {
                // Direct mode: DeepSeek's direct kernel's single-stage exchange over the flat NVLink team. Every
                // rank writes its (encoded) per-rank and per-expert counts into its own slots of every peer,
                // waits for all peers' slots, and reduces the expert counts locally. No release atomics.
                // Issue scaleup rank count writes to peers
                for (int i = thread_idx; i < num_ranks; i += kNumNotifyThreads) {
                    if (not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, i))
                        continue;
                    const auto dst_rank_counter = workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;
                    transport.put_value<int64_t>(dst_rank_counter, static_cast<int64_t>(rank_count[i]), scaleup_peer_id(i));
                }
                __syncwarp();

                // Issue scaleup expert count writes to peers (per-element NVLink stores)
                for (int i = thread_idx; i < num_experts; i += kNumNotifyThreads) {
                    const auto dst_rank = i / num_experts_per_rank;
                    if (not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, dst_rank))
                        continue;
                    const auto idx = num_experts_per_rank * rank_idx + (i % num_experts_per_rank);
                    transport.put_value<int64_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), scaleup_peer_id(dst_rank));
                }

                // This is necessary, as the waited results will rewrite the shared memory
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                EP_PROFILE_STAMP(workspace_layout, 3);

                // Wait for rank and expert count
                const auto start_clock = clock64();
                for (int i = thread_idx; i < num_ranks + num_experts; i += kNumNotifyThreads) {
                    const int writer = i < num_ranks ? i : (i - num_ranks) / num_experts_per_rank;
                    if (not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, writer)) {
                        rank_expert_count[i] = 0;
                        continue;
                    }
                    const auto slot_ptr = workspace_layout.get_scaleup_count_ptr<false>(i, num_ranks);
                    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                        // NOTES: the global memory type has 64 bits
                        const auto count = static_cast<int>(ptx::ld_volatile<int64_t>(slot_ptr));
                        const auto decoded = math::encode_decode_positive(count);
                        if (math::is_decoded_positive_ready(decoded)) {
                            *slot_ptr = 0;
                            rank_expert_count[i] = decoded;
                            return true;
                        }

                        if (is_last_check)
                            printf("NIXL EP V2 adaptive notify (direct) timeout, rank: %d, slot: %d, count: %d\n", rank_idx, i, decoded);
                        return false;
                    }, start_clock);
                }
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                EP_PROFILE_STAMP(workspace_layout, 4);

                // Reduce expert count and add stats
                for (int i = thread_idx; i < num_experts_per_rank; i += kNumNotifyThreads) {
                    int sum = 0;
                    for (int j = 0; j < num_ranks; ++ j)
                        sum += expert_count[j * num_experts_per_rank + i];

                    // Write unaligned count before aligning
                    if (num_unaligned_recv_tokens_per_expert != nullptr)
                        num_unaligned_recv_tokens_per_expert[i] = sum;

                    expert_count[i] = math::align(sum, kExpertAlignment);

                    // Update statistics counters
                    if (cumulative_local_expert_recv_stats != nullptr)
                        atomicAdd(cumulative_local_expert_recv_stats + i, sum);
                }
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

                // Write host workspace
                if constexpr (kDoCPUSync) {
                    for (int i = thread_idx; i < num_ranks + num_experts_per_rank; i += kNumNotifyThreads) {
                        *host_workspace_layout.get_scaleup_count_ptr<false>(i, num_ranks) =
                            math::encode_decode_positive(rank_expert_count[i]);
                    }
                    __syncwarp();
                }
            } else {
                // Hybrid mode: two-stage exchange, scale-out over the rail team then scale-up over NVLink
                // ONE put per live rail peer carries the domain's count row, issued by lane 0 of notify warp `d` (lane `d` of warp 0 when the warps do not suffice)
                const bool warp_per_domain = num_scaleout_ranks <= kNumNotifyWarps;
                const int issue_domain = warp_per_domain ? ((thread_idx % 32 == 0) ? thread_idx / 32 : num_scaleout_ranks) : thread_idx;
                // Profile: the thread issuing to domain 1 (on domain 0) or to domain 0 (elsewhere) stamps its put (slots 18, 19)
                [[maybe_unused]] const int profile_put_thread = (warp_per_domain ? 32 : 1) * (scaleout_rank_idx == 0 ? 1 : 0);
                if (issue_domain < num_scaleout_ranks and topo.is_scaleout_live(issue_domain) and issue_domain != scaleout_rank_idx) {
                    const auto dst_id = scaleout_peer_id(issue_domain);
                    EP_PROFILE_STAMP_THREAD(workspace_layout, 18, profile_put_thread);
                    transport.put_words(
                        workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx, 0),
                        workspace_layout.get_scaleout_expert_count_ptr<true>(issue_domain, 0),
                        (num_experts_per_scaleout + num_scaleup_ranks + num_channels) * sizeof(int), dst_id, true);
                    EP_PROFILE_STAMP_THREAD(workspace_layout, 19, profile_put_thread);
                }
                __syncwarp();
                EP_PROFILE_STAMP(workspace_layout, 20);  // notify: warp 0's scale-out count put issued

                // Util functions to get metadata from scale-out peers
                // NOTES: each count word is polled on its own until it reads as ready: a 4-byte word lands whole (see `put_words`)
                const auto recv_and_reduce = [=](const auto& get_ptr_func, const bool& is_expert_reduction = false) -> int {
                    int count = 0;
                    for (int j = 0; j < num_scaleout_ranks; ++ j) {
                        // Masked domains send nothing
                        if (not topo.is_scaleout_live(j))
                            continue;
                        const auto ptr = get_ptr_func(j);
                        int decoded;
                        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){
                            decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));
                            if (math::is_decoded_positive_ready(decoded))
                                return true;

                            if (is_last_check) {
                                printf("NIXL EP V2 adaptive notify (scale-out %s reduction) timeout, "
                                       "scale-out: %d, scale-up: %d, "
                                       "thread: %d, wait scale-out: %d, decoded: %d\n",
                                       is_expert_reduction ? "expert" : "rank",
                                       scaleout_rank_idx, scaleup_rank_idx, thread_idx, j,
                                       decoded);
                            }
                            return false;
                        });

                        // Add and clean for next usages
                        count += decoded, *ptr = 0;
                    }
                    return count;
                };

                // Write into all scale-up peers' rank-level counters
                for (int i = thread_idx; i < num_scaleup_ranks; i += kNumNotifyThreads) {
                    // Wait scale-out arrival and reduce (a masked position's words too: the live domains send an encoded zero for it)
                    const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                        return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, num_experts_per_scaleout + i);
                    });
                    if (not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, i))
                        continue;

                    // Write into the remote scale-up peer
                    const int64_t counter = (static_cast<int64_t>(num_live_scaleup_ranks) << 32ll) | count;
                    transport.put_value<int64_t>(
                        workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
                        counter, scaleup_peer_id(i));
                }
                __syncwarp();
                EP_PROFILE_STAMP(workspace_layout, 21);  // notify: rank counts received from every domain, forwarded

                // Atomic add into all scale-up peers' expert-level counters
                for (int i = thread_idx; i < num_experts_per_scaleout; i += kNumNotifyThreads) {
                    // Wait scale-out arrival and reduce
                    const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                        return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                    }, true);

                    // Write into the remote scale-up peer
                    const auto dst_scaleup_rank_idx = i / num_experts_per_rank;
                    if (not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, dst_scaleup_rank_idx))
                        continue;
                    const int64_t counter = (1ll << 32ll) | count;
                    const auto expert_idx_in_dst_rank = i % num_experts_per_rank;
                    transport.red_add_rel<int64_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
                        counter, scaleup_peer_id(dst_scaleup_rank_idx));
                }
                // There are shared memory reads above, a barrier is necessary
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                EP_PROFILE_STAMP(workspace_layout, 22);  // notify: expert counts received from every domain, forwarded

                // NOTES: from now on, the `rank` and `expert`s size change into the local size
                expert_count = rank_expert_count + num_scaleup_ranks;

                // Wait local counters to be ready
                // NOTES: here we only care the prefix sum by scale-up peers (used for later epilogue), not all ranks
                // Strided over the notify threads: a small world can have more counters than the notify warps have threads
                comm::timeout_while_strided<kNumTimeoutCycles>(thread_idx, num_scaleup_ranks + num_experts_per_rank, kNumNotifyThreads,
                    [&](const int& i, const bool& is_last_check) {
                    if (i < num_scaleup_ranks and not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, i)) {
                        rank_expert_count[i] = 0;
                        if constexpr (kDoCPUSync)
                            *host_workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks) = math::encode_decode_positive(0);
                        return true;
                    }
                    const auto slot_ptr = workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks);
                    const auto status = ptx::ld_volatile<int64_t>(slot_ptr);
                    if ((status >> 32ull) == num_live_scaleup_ranks) {
                        // Clean GPU workspace and write into host workspace
                        const auto count = static_cast<int>(status & 0xffffffffll);
                        const auto aligned_count = math::align<int>(
                            count, i < num_scaleup_ranks ? 1 : kExpertAlignment);

                        *slot_ptr = 0;
                        if constexpr (kDoCPUSync) {
                            *host_workspace_layout.get_scaleup_count_ptr<false>(i, num_scaleup_ranks) =
                                math::encode_decode_positive(aligned_count);
                        }

                        // Update statistics counters
                        if (cumulative_local_expert_recv_stats != nullptr and i >= num_scaleup_ranks)
                            atomicAdd(cumulative_local_expert_recv_stats + (i - num_scaleup_ranks), count);

                        // Write unaligned count before aligning
                        if (num_unaligned_recv_tokens_per_expert != nullptr and i >= num_scaleup_ranks)
                            num_unaligned_recv_tokens_per_expert[i - num_scaleup_ranks] = count;

                        // Save for later prefix sum calculation
                        rank_expert_count[i] = aligned_count;
                        return true;
                    }

                    if (is_last_check) {
                        printf("NIXL EP V2 adaptive notify (scale-up reduction) timeout, "
                               "scale-out: %d/%d, scale-up: %d/%d, "
                               "slot: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, num_scaleout_ranks, scaleup_rank_idx, num_scaleup_ranks, i,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), num_live_scaleup_ranks);
                    }
                    return false;
                });
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
            }  // hybrid mode

            // Do prefix sum by the warps of the first SM
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            // NOTES: in direct mode `expert_count` still points at `rank_expert_count + num_ranks` (== num_scaleup_ranks)
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // Store into global memory
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    psum = ptx::exchange(sum, 31);
                }
                return psum;
            };
            if (warp_idx == 0) {
                // Inclusive prefix sum
                // The total goes into the fixed slot `[kNumMaxScaleupRanks]`, which the epilogue and the combine read under any live world
                const auto total = do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, num_scaleup_ranks, 0);
                if (lane_idx == 0)
                    psum_num_recv_tokens_per_scaleup_rank[kNumMaxScaleupRanks] = total;
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, num_experts_per_rank, 1);
            }
            EP_PROFILE_STAMP(workspace_layout, 5);
        }
    } else if (is_direct) {
        // Direct mode: every data warp (scale-out and forward warps alike) is a channel that TMA-stores its
        // tokens straight into the destination peers' scale-up buffers over NVLink (DeepSeek's direct kernel's
        // data path over the shared token layout; the linked-list slots of the layout stay unused)
        constexpr int kNumDataWarps = kNumScaleoutWarps + kNumForwardWarps;
        const int data_warp_idx = warp_idx - kNumNotifyWarps;
        // Warp-major channel numbering (as DeepSeek's direct kernel): consecutive tokens go to different SMs, so a
        // small batch uses every SM's write bandwidth instead of piling onto SM 0 (the hybrid roles number
        // channels SM-major because their per-channel buffers and tails are laid out that way)
        const int num_data_channels = kNumDataWarps * num_active_sms;
        const int channel_idx = data_warp_idx * num_active_sms + sm_idx;

        // My region in every peer's scale-up buffer (`[num_scaleup_ranks][num_max_tokens_per_rank]` with one scale-out rank)
        const auto recv_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Iterate all tokens
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += num_data_channels) {
            const auto token_i64_idx = static_cast<int64_t>(token_idx);

            // Wait TMA store arrivals
            ptx::tma_store_wait();
            __syncwarp();

            // Issue data TMA
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue SF `cp.async`
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }

            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / num_experts_per_rank : -1;
                if (stored_dst_rank_idx >= 0 and not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, stored_dst_rank_idx))
                    stored_dst_rank_idx = -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            // Please ensure no TMA buffer shared memory writes after this part
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
            if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
                stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
            __syncwarp();

            // Wait TMA load arrival
            // NOTES: this arrive must be after the `ptx::cp_async_mbarrier_arrive`
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();

            // Issue TMA NVLink stores
            const auto dst_ptr = stored_dst_slot_idx >= 0 ?
                transport.get_p2p_ptr(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), scaleup_peer_id(stored_dst_rank_idx)) :
                nullptr;
            EP_DEVICE_ASSERT((stored_dst_slot_idx < 0 or dst_ptr != nullptr) and "Direct mode needs every live peer reachable with stores");
            if (dst_ptr != nullptr)
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            ptx::tma_store_commit();
            __syncwarp();
        }
        EP_PROFILE_STAMP_WARP(workspace_layout, 10, kNumNotifyWarps);
        EP_PROFILE_STAMP_WARP(workspace_layout, 11, kNumThreads / 32 - 1);
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        const int scaleout_warp_idx = warp_idx - kNumNotifyWarps;
        const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer(channel_idx, num_max_tokens_per_channel);

        // How many records this channel sends to each domain (lane `d`): the channel's last put to a domain rings the
        // doorbell of its deferred puts (no tail travels; the receiver learns the count from the notify exchange)
        int stored_num_records_to_domain = 0;
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += num_channels) {
            const auto dst_expert_idx = lane_idx < kNumTopk ? static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx)) : -1;
            const bool live = dst_expert_idx >= 0 and is_expert_owner_live(dst_expert_idx);
            const auto domain_mask = ptx::reduce_or(live ? (1u << (dst_expert_idx / num_experts_per_scaleout)) : 0u);
            stored_num_records_to_domain += (domain_mask >> lane_idx) & 1;
        }
        int stored_scaleout_tail = 0;   // lane `d`: records sent to domain `d` so far, i.e. the next slot there

        // Profile accumulators of SM 0's first scale-out sender warp, slots 32 to 39 (the slot map is in `layout.cuh`)
        [[maybe_unused]] const bool profile_sender = sm_idx == 0 and warp_idx == kNumNotifyWarps and lane_idx == 0;

        // Preload next token
        const auto preload_next_token = [&](const int& token_idx) {
            if (token_idx >= num_tokens)
                return;

            // Issue TMA load
            const auto token_i64_idx = static_cast<int64_t>(token_idx);
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue SF `cp.async`
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }
        };

        // Iterate all tokens
        preload_next_token(channel_idx);
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += num_channels) {
            EP_PROFILE_ACCUM_BEGIN(profile_t_tok);
            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_scaleout_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 and is_expert_owner_live(dst_expert_idx) ?
                    dst_expert_idx / num_experts_per_scaleout : -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
            const auto stored_old_slot_idx = ptx::exchange(
                stored_scaleout_tail, stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            if (ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and stored_dst_scaleout_rank_idx >= 0)
                stored_dst_slot_idx = stored_old_slot_idx;

            // Update scale-out tail
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);
            stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // Our own domain's record is stored into our receive slot; another domain's goes through the send buffer and an RDMA put
            const bool needs_rdma = stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx;
            const bool any_rdma = ptx::any(needs_rdma);
            // How many records this channel sends to the lane's destination domain in total (warp-uniform shuffle)
            const auto num_records_to_dst = ptx::exchange(stored_num_records_to_domain,
                                                          stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            // The doorbell rings every `kScaleoutUpdateInterval` records to a domain and on the channel's last record to it
            const bool ring_now = stored_dst_slot_idx + 1 == num_records_to_dst or (stored_dst_slot_idx + 1) % kScaleoutUpdateInterval == 0;

            // Wait TMA arrival, stage the flags (spills, flag words, TMA proxy fence), store into the send buffer when any lane goes by RDMA
            EP_PROFILE_ACCUM_BEGIN(profile_t_ldst);
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                tma_buffer.spill_and_set_flags();
                ptx::tma_store_fence();
            }
            __syncwarp();
            // `any_rdma` is a full-warp vote: consume it in warp-uniform control flow, never inside the elected lane's block alone
            if (any_rdma) {
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                                      tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                }
                __syncwarp();
            }
            EP_PROFILE_ACCUM_END(workspace_layout, 33, profile_t_ldst, profile_sender);

            EP_PROFILE_ACCUM_BEGIN(profile_t_own);
            // Own-domain rows: the record without its flag slots (a bulk store is not ordered internally), the flags after the stores completed
            // Local rank can be bypassed
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx) {
                tma_buffer.tma_store_skipping_flags(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr());
            }
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            __syncwarp();
            EP_PROFILE_ACCUM_END(workspace_layout, 34, profile_t_own, profile_sender);
            EP_PROFILE_ACCUM_BEGIN(profile_t_rel);
            // Flag release once per doorbell interval (the RDMA doorbell's rule): the rows stored since the previous release
            // have consecutive slots and completed stores, so ONE GPU-scope fence orders them all before the flag stores
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx and ring_now) {
                ptx::fence_acq_rel_gpu();
                const int first_slot_idx = stored_dst_slot_idx - stored_dst_slot_idx % kScaleoutUpdateInterval;
                for (int slot_idx = first_slot_idx; slot_idx <= stored_dst_slot_idx; ++ slot_idx)
                    tma_buffer.store_flags(scaleout_recv_buffer.get_token_buffer(slot_idx).get_base_ptr());
            }
            __syncwarp();
            EP_PROFILE_ACCUM_END(workspace_layout, 35, profile_t_rel, profile_sender);

            // Preload the next token (overlapping with the RDMA puts)
            preload_next_token(token_idx + num_channels);

            // Issue RDMA requests under the doorbell rule above (the NIC drains while the channel keeps sending)
            EP_PROFILE_ACCUM_BEGIN(profile_t_put);
            if (needs_rdma) {
                transport.rdma_put(
                        scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                        tma_buffer.get_num_bytes<false>(),
                        scaleout_peer_id(stored_dst_scaleout_rank_idx),
                        ring_now);
            }
            __syncwarp();
#ifdef EP_ADAPTIVE_PROFILE
            // Profiling votes, consumed in warp-uniform control flow (slots 36/38: any lane by RDMA; 37/39: any lane rang)
            if (any_rdma) {
                EP_PROFILE_ACCUM_END(workspace_layout, 36, profile_t_put, profile_sender);
                EP_PROFILE_COUNT(workspace_layout, 38, profile_sender);
                if (ptx::any(needs_rdma and ring_now)) {
                    EP_PROFILE_ACCUM_END(workspace_layout, 37, profile_t_put, profile_sender);
                    EP_PROFILE_COUNT(workspace_layout, 39, profile_sender);
                }
            }
#endif
            EP_PROFILE_ACCUM_END(workspace_layout, 32, profile_t_tok, profile_sender);
        }
        EP_PROFILE_STAMP_LAST(workspace_layout, 10);   // hybrid: the last scale-out sender warp is done
    } else {
        const int forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);
        const int channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer(channel_idx, num_max_tokens_per_channel);
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Shape of `token_metadata_at_forward`: `[num_channels, num_scaleout_ranks * num_max_tokens_per_channel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        const int num_tokens_in_linked_list = num_max_tokens_per_channel * num_scaleout_ranks + 1;
        token_metadata_at_forward += channel_idx * (num_tokens_in_linked_list * kNumForwardMetadataDims);

        // Transform linked list index
        const auto transform_linked_list_idx = [=](const int& idx) {
            return channel_idx * (num_tokens_in_linked_list * num_scaleup_ranks) +
                idx * num_scaleup_ranks + scaleup_rank_idx;
        };

        // Forward tokens from scale-out ranks. Lane `d` tracks domain `d`: the record count from the notify exchange (-1 until
        // it arrived) and how many it has consumed; a record is complete when every flag inside it is set (see `TokenLayout`)
        int num_tokens_processed = 0;
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};
        const bool lane_tracks_live_domain = lane_idx < num_scaleout_ranks and topo.is_scaleout_live(lane_idx);
        const auto count_word_ptr = workspace_layout.get_scaleout_expert_count_ptr<false>(
            lane_tracks_live_domain ? lane_idx : 0, num_experts_per_scaleout + num_scaleup_ranks + channel_idx);
        int stored_num_records = lane_tracks_live_domain ? -1 : 0, stored_num_consumed = 0;
        const auto lane_recv_buffer = scaleout_recv_buffer.get_rank_buffer(lane_tracks_live_domain ? lane_idx : 0);
        int recv_scaleout_rank_idx = channel_idx % num_scaleout_ranks;
        uint32_t wip_mask;
        while ((wip_mask = ptx::gather(stored_num_records < 0 or stored_num_consumed < stored_num_records))) {
            // Wait until some pending domain has its next record complete (lane `d` polls domain `d` and its count word)
            uint32_t ready_mask = 0;
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                bool ready = false;
                if ((wip_mask >> lane_idx) & 1) {
                    if (stored_num_records < 0) {
                        const auto decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(count_word_ptr));
                        if (math::is_decoded_positive_ready(decoded))
                            stored_num_records = decoded, *count_word_ptr = 0;
                    }
                    ready = (stored_num_records < 0 or stored_num_consumed < stored_num_records) and
                            stored_num_consumed < num_max_tokens_per_channel and
                            lane_recv_buffer.get_token_buffer(stored_num_consumed).are_flags_set();
                }
                ready_mask = ptx::gather(ready);
                wip_mask = ptx::gather(stored_num_records < 0 or stored_num_consumed < stored_num_records);
                if (ready_mask != 0 or wip_mask == 0)
                    return true;

                if (is_last_check and lane_tracks_live_domain) {
                    printf("NIXL EP V2 hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                           "channel: %d, lane: %d, records: %d, consumed: %d\n",
                           scaleout_rank_idx, scaleup_rank_idx, channel_idx, lane_idx,
                           stored_num_records, stored_num_consumed);
                }
                return false;
            });
            if (ready_mask == 0)
                continue;   // Every remaining domain turned out complete (its count arrived and was consumed)

            // Pick next rank in round-robin
            const auto offset = (recv_scaleout_rank_idx + 1) % num_scaleout_ranks;
            const auto hi_mask = (ready_mask >> offset) << offset;
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(ready_mask);

            // Process the run of complete records at the head of that domain's queue, up to one chunk (lane `j` probes slot `start + j`)
            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);
            const auto start_slot_idx = ptx::exchange(stored_num_consumed, recv_scaleout_rank_idx);
            const auto num_records = ptx::exchange(stored_num_records, recv_scaleout_rank_idx);
            const auto probe_slot_idx = start_slot_idx + lane_idx;
            const bool slot_complete = lane_idx < kNumSlotsPerForwardChunk and probe_slot_idx < num_max_tokens_per_channel and
                                       (num_records < 0 or probe_slot_idx < num_records) and
                                       (lane_idx == 0 or recv_buffer.get_token_buffer(probe_slot_idx).are_flags_set());
            const auto end_slot_idx = start_slot_idx + ptx::ffs(~ptx::gather(slot_complete));
            if (lane_idx == recv_scaleout_rank_idx)
                stored_num_consumed = end_slot_idx;

            // The flags were observed with relaxed loads by the probing lanes: the acquire fence runs in every lane, GPU scope for our
            // own domain's records, system scope for a NIC's or another GPU's. The memory model requires it, the hardware does not (the
            // TMA load follows the vote on the flags and both bypass L1): removing it saves 8-18 us per dispatch, 5% at small batches
            if (recv_scaleout_rank_idx == scaleout_rank_idx) {
                ptx::fence_acq_rel_gpu();
            } else {
                ptx::fence_acq_rel_sys();
            }
            __syncwarp();

            for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {
                const auto token_buffer = recv_buffer.get_token_buffer(slot_idx);

                // Wait TMA arrival
                ptx::tma_store_wait();
                __syncwarp();

                // The record into shared memory as it lies, the slot's flag words back to zero, then the copy made whole (spills restored,
                // the last flag slot zeroed: no copy outside a live receive slot may carry the flag pattern)
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), token_buffer.get_base_ptr(),
                                     mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    token_buffer.clear_flags();
                    tma_buffer.restore_spills();
                    tma_buffer.clear_final_flag();
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Read top-k indices
                EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
                int stored_dst_scaleup_rank_idx = -1;
                auto dst_expert_idx = lane_idx < kNumTopk ? tma_buffer.get_topk_idx_ptr()[lane_idx] : -1;
                dst_expert_idx -= scaleout_rank_idx * num_experts_per_scaleout;
                stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < num_experts_per_scaleout ?
                    dst_expert_idx / num_experts_per_rank : -1;
                if (stored_dst_scaleup_rank_idx >= 0 and not topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, stored_dst_scaleup_rank_idx))
                    stored_dst_scaleup_rank_idx = -1;

                // Write the per-scaleup channel index for this token
                int linked_list_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;
                    const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;
                    const auto exchanged = ptx::exchange(
                        stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                    linked_list_idx = valid ? exchanged : linked_list_idx;
                }
                if (lane_idx < kNumTopk) {
                    tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Deduplicate for scale-up ranks
                int stored_dst_slot_idx = -1;
                if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                __syncwarp();

                // Issue TMAs
                if (stored_dst_slot_idx >= 0) {
                    const auto dst_ptr = transport.get_p2p_ptr(
                        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        scaleup_peer_id(stored_dst_scaleup_rank_idx));
                    EP_DEVICE_ASSERT(dst_ptr != nullptr and "A scale-up peer is not reachable with stores");
                    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // Add per-scale-up counter: one 32-bit mask per lane group of positions (lane `l` of group `j` is position `j * 32 + l`)
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto group_mask = ptx::reduce_or(
                        (stored_dst_scaleup_rank_idx >= 0 and (stored_dst_scaleup_rank_idx >> 5) == j) ?
                        (1u << (stored_dst_scaleup_rank_idx & 31)) : 0u);
                    stored_scaleup_send_counters[j] += (group_mask >> lane_idx) & 1;
                }

                // Record metadata at forward
                EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of selections");
                const auto metadata_ptr = token_metadata_at_forward +
                    num_tokens_processed * kNumForwardMetadataDims;

                // Source token index and last token index flag
                if (ptx::elect_one_sync()) {
                    metadata_ptr[0] = tma_buffer.get_src_token_global_idx_ptr()[0];
                    metadata_ptr[1] = slot_idx == (end_slot_idx - 1);
                }

                // Second, original top-k indices and destination slots
                if (lane_idx < kNumTopk) {
                    metadata_ptr[2 + lane_idx] = stored_dst_scaleup_rank_idx;
                    metadata_ptr[2 + kNumTopk + lane_idx] = stored_dst_slot_idx;
                }
                num_tokens_processed += 1;
                __syncwarp();
            }
        }

        EP_PROFILE_STAMP_LAST(workspace_layout, 11);   // hybrid: the last forward warp has consumed its records

        // Assign the source token index part of the metadata into `-1` as an ending mark
        if (ptx::elect_one_sync())
            token_metadata_at_forward[num_tokens_processed * kNumForwardMetadataDims] = -1;
        __syncwarp();

        // Update linked list's ending position
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
            if (const auto j = i * 32 + lane_idx; j < num_scaleup_ranks and topology::is_scaleup_live<kNumMaxScaleupRanks>(topo, j)) {
                const auto dst_ptr = transport.get_p2p_ptr(tail_ptr, scaleup_peer_id(j));
                EP_DEVICE_ASSERT(dst_ptr != nullptr and "A scale-up peer is not reachable with stores");
                ptx::st_relaxed_sys(dst_ptr, transform_linked_list_idx(stored_scaleup_send_counters[i]));
            }
        }
        __syncwarp();
    }

    // Scale-up barrier to ensure data arrival
    // As scale-out tokens have already been consumed by forwarders, no need to do scale-out barrier again
    EP_PROFILE_STAMP(workspace_layout, 6);
    comm::gpu_barrier_rt<kNumThreads, kNumTimeoutCycles, comm::kHybridDispatchTag1, kNumMaxScaleupRanks, true, true, false>(
        transport, workspace_layout, topo, sm_idx, thread_idx, /* do not scale-out */ false, true);
    EP_PROFILE_STAMP(workspace_layout, 7);

    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Clean scale-up counters
    // All scale-out counters should be cleaned before
    EP_DEVICE_ASSERT(num_scaleup_ranks <= kNumThreads and "Insufficient threads");
    if (sm_idx == 0 and thread_idx < num_scaleup_ranks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace nixl_ep_v2::elastic::adaptive
