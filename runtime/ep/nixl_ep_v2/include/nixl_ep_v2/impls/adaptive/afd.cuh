// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nixl_ep_v2/common/comm.cuh>

namespace nixl_ep_v2::elastic::adaptive {

template <int kDirection, int kExpertsPerRank>
__device__ __forceinline__ int physical_expert(const topology::EPTopology& topo, int64_t expert) {
    if constexpr (kDirection == kFused)
        return expert;
    if (expert < 0 || expert / kExpertsPerRank >= topo.afd->num_ffn_ranks)
        return -1;
    const int rank = topo.scaleout_rank_idx * topo.num_scaleup_ranks + topo.afd->ffn_ranks[expert / kExpertsPerRank];
    return rank * kExpertsPerRank + expert % kExpertsPerRank;
}

// Translate a record offset within the destination's token inbox to its mapped NVLink address.
__device__ __forceinline__ void* afd_inbox_record(
    const nixl::NIXLTransport& transport, const topology::EPTopology& topo, int peer, int pos,
    uint64_t record_offset, uint64_t record_bytes) {
    const uint64_t offset = topo.afd->inbox_offset + record_offset;
    EP_DEVICE_ASSERT(offset + record_bytes <= topo.afd->registered_end[pos]);
    auto* base = transport.get_p2p_base(peer);
    EP_DEVICE_ASSERT(base != nullptr);
    return base + offset;
}

// Both endpoints advance the same per-peer epoch. The consumer publishes READY at entry,
// and the producer publishes ARRIVED after every data and count store has completed.
template <int kDirection, int64_t kTimeout, int kTag, int kMaxPositions>
__device__ __forceinline__ void finish_direct_exchange(const nixl::NIXLTransport& transport,
        const layout::WorkspaceLayout& workspace, const topology::EPTopology& topo, int lane) {
    if constexpr (kDirection == kFused) {
        comm::peer_arrived_exchange_rt<kTimeout, kTag, kMaxPositions>(transport, workspace, topo, lane);
    } else {
        const int self = topology::rank_id(topo.scaleout_rank_idx, topo.scaleup_rank_idx, kMaxPositions);
        for (int pos = lane; pos < topo.num_scaleup_ranks; pos += 32) {
            if (!topology::is_scaleup_live<kMaxPositions>(topo, pos))
                continue;
            const int peer = topology::rank_id(topo.scaleout_rank_idx, pos, kMaxPositions);
            auto* expected = workspace.get_peer_barrier_expected_ptr(peer);
            const int64_t next = *expected + 2;
            if constexpr (kDirection == kSend) {
                comm::peer_word_wait_rt<kTimeout, kTag>(workspace, topo, pos, peer, comm::kPeerReadyWord, next - 1);
                transport.put_word_rel(workspace.get_peer_barrier_signal_ptr(self, comm::kPeerArrivedWord),
                                       workspace.get_peer_barrier_staging_ptr(peer, comm::kPeerArrivedWord), next, peer);
            } else {
                comm::peer_word_wait_rt<kTimeout, kTag>(workspace, topo, pos, peer, comm::kPeerArrivedWord, next);
            }
            *expected = next;
        }
        __syncwarp();
    }
}

}  // namespace nixl_ep_v2::elastic::adaptive
