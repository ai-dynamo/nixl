#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/nixl_transport.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/topology.cuh>

namespace deep_ep::elastic {

// Host-level barrier of the live world (`ElasticBuffer::barrier`): the two per-peer barrier legs of the adaptive kernels
// (scale-out, then scale-up) run by one block; membership, peers and transport come from the topology descriptor (`EPTopology`)
template <int kNumThreads, int kNumMaxScaleupRanks, int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
barrier_impl(void* workspace, const topology::EPTopology* topology_ptr) {
    const auto topo = topology::load(topology_ptr);
    if (not topo.is_participant())
        return;

    // The barrier touches only the per-peer barrier words, whose offsets are fixed by
    // the `kNumMax*` constants, so `num_experts` may be 0 here
    const auto workspace_layout = layout::WorkspaceLayout(workspace, topo.num_scaleout_ranks, topo.num_scaleup_ranks, 0);
    const auto ctx = nixl::load_context(topo.transport);
    const auto transport = nixl::NIXLTransport(ctx, 0);
    const auto thread_idx = static_cast<int>(threadIdx.x);

    // Pending TMA and generic stores of earlier kernels are complete (stream order); the arrivals publish them, and a peer's
    // arrival shows it consumed every put of the earlier launches (see `comm::gpu_barrier_rt`): once it has passed this barrier too, it may be disconnected
    if (topo.num_live_scaleout_ranks() > 1)
        comm::peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, comm::kKernelBarrierTag, kNumMaxScaleupRanks>(
            transport, workspace_layout, topo, true, 0, thread_idx);
    if (topo.scaleup_mask.popc() > 1)
        comm::peer_barrier_leg_rt<kNumThreads, kNumTimeoutCycles, comm::kKernelBarrierTag, kNumMaxScaleupRanks>(
            transport, workspace_layout, topo, false, 0, thread_idx);
}

} // namespace deep_ep::elastic
