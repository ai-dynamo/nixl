// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <nixl_ep_v2/common/topology.cuh>

namespace nixl_ep_v2::elastic::adaptive::combine_policy {

// Measured small-batch wins justify incremental completion only in hybrid mode.
constexpr int kMaxIncrementalTokens = 16;
constexpr int kDispatchFinished = 1;
constexpr int kDispatchIncrementalCombine = 2;

__forceinline__ __device__ __host__ bool use_incremental(const topology::EPTopology& topo, const int num_tokens) {
    return topo.mode == topology::kModeHybrid and topo.num_active_scaleout_ranks() > 1 and num_tokens <= kMaxIncrementalTokens;
}

// The low word of dispatch's packed tail carries completion and the source's combine policy.
__forceinline__ __device__ __host__ int dispatch_finish_flag(const bool finished, const bool incremental) {
    return finished ? kDispatchFinished | (incremental ? kDispatchIncrementalCombine : 0) : 0;
}

__forceinline__ __device__ __host__ bool finish_uses_incremental(const int flag) {
    return (flag & kDispatchIncrementalCombine) != 0;
}

}  // namespace nixl_ep_v2::elastic::adaptive::combine_policy
