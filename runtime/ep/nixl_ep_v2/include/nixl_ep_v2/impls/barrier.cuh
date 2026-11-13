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

#include <nixl_ep_v2/common/comm.cuh>
#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/layout.cuh>
#include <nixl_ep_v2/common/nixl_transport.cuh>
#include <nixl_ep_v2/common/ptx.cuh>
#include <nixl_ep_v2/common/topology.cuh>

namespace nixl_ep_v2::elastic {

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

} // namespace nixl_ep_v2::elastic
