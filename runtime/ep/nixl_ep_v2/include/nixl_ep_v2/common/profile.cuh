// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <cstdint>

#include <nixl_ep_v2/common/exception.cuh>

namespace nixl_ep_v2::elastic::profile {

// One schema for device indices and the names returned to Python. A zero slot means that writer did not reach the phase.
#define EP_PROFILE_PHASES(X) \
    X(DispatchEntry, "dispatch.entry") \
    X(DispatchEntryBarrier, "dispatch.entry_barrier") \
    X(DispatchCountLocal, "dispatch.count_local") \
    X(DispatchCountReduction, "dispatch.count_reduction") \
    X(DispatchCountExchange, "dispatch.count_exchange") \
    X(DispatchCountPrefix, "dispatch.count_prefix") \
    X(DispatchSendSubmitted, "dispatch.send_submitted") \
    X(DispatchForwardSubmitted, "dispatch.forward_submitted") \
    X(DispatchFinalBarrier, "dispatch.final_barrier") \
    X(DispatchCopyEntry, "dispatch.copy_entry") \
    X(DispatchCopyReady, "dispatch.copy_ready") \
    X(DispatchCopySubmitted, "dispatch.copy_submitted") \
    X(CombineEntry, "combine.entry") \
    X(CombineEntryBarrier, "combine.entry_barrier") \
    X(CombineScaleupSubmitted, "combine.scaleup_submitted") \
    X(CombineForwardSubmitted, "combine.forward_submitted") \
    X(CombineMainEnd, "combine.main_end") \
    X(CombineReduceEntry, "combine.reduce_entry") \
    X(CombineReduceReady, "combine.reduce_ready") \
    X(CombineReduceSubmitted, "combine.reduce_submitted")

enum class Phase {
#define EP_PROFILE_ENUM(name, label) name,
    EP_PROFILE_PHASES(EP_PROFILE_ENUM)
#undef EP_PROFILE_ENUM
    Count
};
inline constexpr int kNumPhases = static_cast<int>(Phase::Count);
inline constexpr const char* kPhaseNames[] = {
#define EP_PROFILE_NAME(name, label) label,
    EP_PROFILE_PHASES(EP_PROFILE_NAME)
#undef EP_PROFILE_NAME
};
#undef EP_PROFILE_PHASES

// The optional tensor is owned by the buffer and outlives every kernel and captured graph using it.
struct Header {
    int64_t* values;
    int num_blocks, num_warps;
};

#if defined(__CUDACC__) && defined(EP_ADAPTIVE_PROFILE) && EP_ADAPTIVE_PROFILE
__device__ __forceinline__ void stamp(const Header* header, const Phase phase) {
    if (threadIdx.x % warpSize == 0) {
        const auto view = *header;
        EP_DEVICE_ASSERT(view.values != nullptr and blockIdx.x < view.num_blocks and
                         blockDim.x <= view.num_warps * warpSize);
        uint64_t timestamp;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) :: "memory");
        const auto index = (static_cast<int64_t>(phase) * view.num_blocks + blockIdx.x) * view.num_warps + threadIdx.x / warpSize;
        view.values[index] = static_cast<int64_t>(timestamp);
    }
}
#define EP_PROFILE_STAMP(workspace_base, phase) \
    ::nixl_ep_v2::elastic::profile::stamp(::nixl_ep_v2::elastic::layout::WorkspaceLayout(workspace_base).get_profile_header(), \
                                      ::nixl_ep_v2::elastic::profile::Phase::phase)
#else
#define EP_PROFILE_STAMP(workspace_base, phase) do {} while (0)
#endif

}  // namespace nixl_ep_v2::elastic::profile
