// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <nixl_ep_v2/common/comm.cuh>
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>

namespace nixl_ep_v2::elastic {

// Host-level barrier of the active ranks (`ElasticBuffer::barrier`): drain RDMA, then run the scale-out and scale-up
// peer barriers. Membership, peers and transport come from the topology descriptor (`EPTopology`).
template <int kNumThreads, int kNumMaxScaleupRanks, int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
barrier_impl(const topology::EPTopology* topology_ptr) {
    const auto topo = topology::load(topology_ptr);
    if (not topo.is_active())
        return;

    const auto ctx = nixl::load_context(topo.transport);
    const auto workspace_layout = layout::WorkspaceLayout(ctx.local_base);
    const auto transport = nixl::NIXLTransport(ctx, 0);
    const auto thread_idx = static_cast<int>(threadIdx.x);

    // A zero-add leaves the atomic-only arrival word unchanged. Drain every channel before announcing arrival, so a
    // peer cannot retire its registration while a drain is still addressing it. Earlier kernels are stream-ordered.
    if (topo.num_active_scaleout_ranks() > 1) {
        const auto start_clock = clock64();
        for (int i = thread_idx; i < topo.num_scaleout_ranks * ctx.num_channels; i += kNumThreads) {
            const int domain = i / ctx.num_channels;
            if (not topo.is_scaleout_active(domain) or domain == topo.scaleout_rank_idx)
                continue;
            const auto channel_transport = nixl::NIXLTransport(ctx, i % ctx.num_channels);
            comm::red_add_rel_and_wait<kNumTimeoutCycles>(
                channel_transport, workspace_layout.get_peer_barrier_signal_ptr(ctx.world_rank), 0,
                topology::scaleout_peer<kNumMaxScaleupRanks>(topo, domain), start_clock);
        }
    }
    __syncthreads();

    // Incoming arrivals certify the peers' drains; await our own arrival completions before returning too.
    if (topo.num_active_scaleout_ranks() > 1)
        comm::peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, comm::kKernelBarrierTag, kNumMaxScaleupRanks, true>(
            transport, workspace_layout, topo, comm::Leg::kScaleout, thread_idx);
    if (topo.num_active_scaleup_ranks() > 1)
        comm::peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, comm::kKernelBarrierTag, kNumMaxScaleupRanks, true>(
            transport, workspace_layout, topo, comm::Leg::kScaleup, thread_idx);
}

} // namespace nixl_ep_v2::elastic
