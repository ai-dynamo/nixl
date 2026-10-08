#pragma once

#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/topology.cuh>

#include "../../../jit/compiler.hpp"
#include "../../../jit/launch_runtime.hpp"

namespace nixl_ep_v2::elastic::adaptive {

// Hard caps of the rank decomposition; they size the per-lane register arrays and the notify smem region
using topology::kNumMaxScaleoutRanksCap;
using topology::kNumMaxScaleupRanksCap;
// Compile-time channel maxima: the block shape and the channel slack of the buffers follow the fixed grid.
constexpr int kNumMaxChannelsPerSM = 8;
constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * topology::kNumMaxLaunchSMs;
EP_STATIC_ASSERT(kNumMaxChannels <= nixl_ep_v2::kNumMaxChannels, "The workspace channel arrays are sized by `nixl_ep_v2::kNumMaxChannels`");

class DispatchRuntime final : public jit::LaunchRuntime<DispatchRuntime> {
public:
    struct Args {
        // Templated arguments
        bool do_cpu_sync;
        int num_notify_warps, num_channels_per_sm;
        int num_max_scaleout_ranks, num_max_scaleup_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts_per_rank, num_topk, expert_alignment;
        int64_t num_timeout_cycles;

        // Parameters
        void* x; sf_pack_t* sf; topk_idx_t* topk_idx; float* topk_weights;
        topk_idx_t* copied_topk_idx;
        int* cumulative_local_expert_recv_stats;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        int* num_unaligned_recv_tokens_per_expert;
        int* token_metadata_at_forward;
        int num_tokens;
        int sf_token_stride, sf_hidden_stride;
        void* buffer; void* send_buffer;
        void* workspace; void* mapped_host_workspace;
        const topology::EPTopology* topology;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <nixl_ep_v2/impls/adaptive/dispatch.cuh>

using namespace nixl_ep_v2::elastic::adaptive;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&adaptive_dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
            args.do_cpu_sync,
            args.launch_args.grid_dim.first,
            args.num_notify_warps, args.num_channels_per_sm,
            args.num_max_scaleout_ranks, args.num_max_scaleup_ranks,
            args.num_hidden_bytes, args.num_sf_packs,
            args.num_max_tokens_per_rank,
            args.num_experts_per_rank, args.num_topk, args.expert_alignment,
            args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.x, args.sf, args.topk_idx, args.topk_weights,
            args.copied_topk_idx,
            args.cumulative_local_expert_recv_stats,
            args.psum_num_recv_tokens_per_scaleup_rank,
            args.psum_num_recv_tokens_per_expert,
            args.num_unaligned_recv_tokens_per_expert,
            args.token_metadata_at_forward,
            args.num_tokens,
            args.sf_token_stride, args.sf_hidden_stride,
            args.buffer, args.send_buffer,
            args.workspace, args.mapped_host_workspace,
            args.topology));
    }
};

constexpr int kNumNotifyWarps = 4;

// Notify shared memory: rank counts, expert slot counts and the per-channel record counts per domain, at the kernel's
// template maxima (the expert slots of the declared maximum world: ranks times the fixed slots per rank)
static int get_num_notify_smem_bytes(const int& num_max_ranks, const int& num_max_expert_slots, const int& num_max_scaleout_ranks) {
    return math::align(num_max_ranks + num_max_expert_slots + topology::get_num_launch_sms() * kNumMaxChannelsPerSM * num_max_scaleout_ranks, kNumNotifyWarps * 32) * sizeof(int);
}

// DeepSeek's hybrid token layout plus the arrival flag and spill slots of the scale-out path (`with_flags`; direct mode ignores them)
static layout::TokenLayout get_dispatch_token_layout(
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
}

static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

// Decide the number of channels per SM by shared memory consumption (both dispatch and combine must fit); fixed at the first dispatch
static int get_num_channels_per_sm(const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk,
                                   const int& num_max_expert_slots, const int& num_max_ranks, const int& num_max_scaleout_ranks,
                                   const int& num_smem_bytes,
                                   const bool& prefer_overlap_with_compute) {
    const auto dispatch_token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);
    const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    int num_channels_per_sm = std::min<int>(
        (num_smem_bytes - get_num_notify_smem_bytes(num_max_ranks, num_max_expert_slots, num_max_scaleout_ranks)) / dispatch_token_layout.get_num_bytes<true>(),
        32 - kNumNotifyWarps);
    num_channels_per_sm = std::min<int>(
        num_smem_bytes / combine_token_layout.get_num_bytes<true>(),
        num_channels_per_sm);
    num_channels_per_sm = std::min<int>(
        /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
    if (not prefer_overlap_with_compute)
        num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
    EP_HOST_ASSERT(num_channels_per_sm >= 1);
    return num_channels_per_sm;
}

static void launch_dispatch(void* x, void* sf,
                            topk_idx_t* topk_idx, float* topk_weights,
                            topk_idx_t* copied_topk_idx,
                            int* cumulative_local_expert_recv_stats,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* psum_num_recv_tokens_per_expert,
                            int* num_unaligned_recv_tokens_per_expert,
                            int* token_metadata_at_forward,
                            const int& num_tokens, const int& num_max_tokens_per_rank,
                            const int& hidden, const int& elem_size,
                            const int& num_sf_packs, const int& sf_token_stride, const int& sf_hidden_stride,
                            const int& num_experts_per_rank, const int& num_topk, const int& expert_alignment,
                            void* buffer, void* send_buffer,
                            void* workspace, void* mapped_host_workspace,
                            const topology::EPTopology* topology,
                            const int& num_max_scaleout_ranks, const int& num_max_scaleup_ranks,
                            const int& num_channels_per_sm,
                            const int& num_smem_bytes,
                            const int64_t& num_timeout_cycles,
                            const bool& do_cpu_sync,
                            const at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(num_max_scaleout_ranks <= kNumMaxScaleoutRanksCap and num_max_scaleup_ranks <= kNumMaxScaleupRanksCap);
    EP_HOST_ASSERT(1 <= num_channels_per_sm and num_channels_per_sm <= kNumMaxChannelsPerSM);

    // Notify warps, then the scale-out and forward warps: each channel owns one warp of each kind
    const int num_threads = (kNumNotifyWarps + 2 * num_channels_per_sm) * 32;

    // Generate, build and launch
    const DispatchRuntime::Args args = {
        .do_cpu_sync = do_cpu_sync,
        .num_notify_warps = kNumNotifyWarps, .num_channels_per_sm = num_channels_per_sm,
        .num_max_scaleout_ranks = num_max_scaleout_ranks, .num_max_scaleup_ranks = num_max_scaleup_ranks,
        .num_hidden_bytes = hidden * elem_size, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts_per_rank = num_experts_per_rank, .num_topk = num_topk, .expert_alignment = expert_alignment,
        .num_timeout_cycles = num_timeout_cycles,
        .x = x, .sf = static_cast<sf_pack_t*>(sf), .topk_idx = topk_idx, .topk_weights = topk_weights,
        .copied_topk_idx = copied_topk_idx,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert,
        .token_metadata_at_forward = token_metadata_at_forward,
        .num_tokens = num_tokens,
        .sf_token_stride = sf_token_stride, .sf_hidden_stride = sf_hidden_stride,
        .buffer = buffer, .send_buffer = send_buffer,
        .workspace = workspace, .mapped_host_workspace = mapped_host_workspace,
        .topology = topology,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(topology::get_num_launch_sms(), num_threads, num_smem_bytes, 2, true)};
    const auto code = DispatchRuntime::generate(args);
    const auto runtime = jit::compiler->build("adaptive_dispatch", code);
    DispatchRuntime::launch(runtime, args, stream);
}

class DispatchCopyEpilogueRuntime final : public jit::LaunchRuntime<DispatchCopyEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool do_expand, do_zero_padding;
        int num_channels_per_sm;
        int num_warps;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts_per_rank, num_topk, expert_alignment;

        // Parameters
        void *buffer, *workspace;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        void* recv_x; void* recv_sf;
        topk_idx_t* recv_topk_idx; float* recv_topk_weights;
        int* recv_src_metadata;
        int* channel_linked_list;
        int* num_unaligned_recv_tokens_per_expert;
        int num_recv_tokens, num_max_scaleup_ranks;
        int recv_sf_token_stride, recv_sf_hidden_stride;
        const topology::EPTopology* topology;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <nixl_ep_v2/impls/adaptive/dispatch_copy_epilogue.cuh>

using namespace nixl_ep_v2::elastic::adaptive;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.do_expand, args.do_zero_padding,
                           args.launch_args.grid_dim.first, args.num_channels_per_sm, args.num_warps,
                           args.num_hidden_bytes, args.num_sf_packs,
                           args.num_max_tokens_per_rank,
                           args.num_experts_per_rank, args.num_topk, args.expert_alignment);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.buffer, args.workspace,
                                                 args.psum_num_recv_tokens_per_scaleup_rank,
                                                 args.psum_num_recv_tokens_per_expert,
                                                 args.recv_x, args.recv_sf, args.recv_topk_idx, args.recv_topk_weights,
                                                 args.recv_src_metadata,
                                                 args.channel_linked_list,
                                                 args.num_unaligned_recv_tokens_per_expert,
                                                 args.num_recv_tokens, args.num_max_scaleup_ranks,
                                                 args.recv_sf_token_stride, args.recv_sf_hidden_stride,
                                                 args.topology));
    }
};

static void launch_dispatch_copy_epilogue(void* buffer, void* workspace,
                                          int* psum_num_recv_tokens_per_scaleup_rank,
                                          int* psum_num_recv_tokens_per_expert,
                                          void* recv_x, void* recv_sf,
                                          topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                                          int* recv_src_metadata,
                                          int* channel_linked_list,
                                          int* num_unaligned_recv_tokens_per_expert,
                                          const int& num_recv_tokens, const int& num_max_scaleup_ranks, const int& num_max_tokens_per_rank,
                                          const int& num_hidden_bytes,
                                          const int& num_sf_packs, const int& recv_sf_token_stride, const int& recv_sf_hidden_stride,
                                          const int& num_experts_per_rank, const int& num_topk, const int& expert_alignment,
                                          const topology::EPTopology* topology,
                                          const int& num_sms, const int& num_smem_bytes,
                                          const int& num_channels_per_sm,
                                          const bool& do_expand, const bool& do_zero_padding,
                                          const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    // A margin below the opt-in maximum: a layout that filled it to the byte faulted on the last warp's mbarrier
    constexpr int kNumSmemReserveBytes = 1024;
    const auto token_layout = layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    const auto num_warps = std::min((num_smem_bytes - kNumSmemReserveBytes) / token_layout.get_num_bytes<true>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const DispatchCopyEpilogueRuntime::Args args = {
        .do_expand = do_expand, .do_zero_padding = do_zero_padding,
        .num_channels_per_sm = num_channels_per_sm, .num_warps = num_warps,
        .num_hidden_bytes = num_hidden_bytes, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts_per_rank = num_experts_per_rank, .num_topk = num_topk, .expert_alignment = expert_alignment,
        .buffer = buffer, .workspace = workspace,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .recv_x = recv_x, .recv_sf = recv_sf,
        .recv_topk_idx = recv_topk_idx, .recv_topk_weights = recv_topk_weights,
        .recv_src_metadata = recv_src_metadata,
        .channel_linked_list = channel_linked_list,
        .num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert,
        .num_recv_tokens = num_recv_tokens, .num_max_scaleup_ranks = num_max_scaleup_ranks,
        .recv_sf_token_stride = recv_sf_token_stride, .recv_sf_hidden_stride = recv_sf_hidden_stride,
        .topology = topology,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)};
    const auto code = DispatchCopyEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("adaptive_dispatch_copy_epilogue", code);
    DispatchCopyEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace nixl_ep_v2::elastic::adaptive
