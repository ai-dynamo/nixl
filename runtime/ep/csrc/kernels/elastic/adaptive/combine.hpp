#pragma once

#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/topology.cuh>

#include "../../../jit/compiler.hpp"
#include "../../../jit/launch_runtime.hpp"
#include "dispatch.hpp"

namespace nixl_ep_v2::elastic::adaptive {

class CombineRuntime final : public jit::LaunchRuntime<CombineRuntime> {
public:
    struct Args {
        // Templated arguments
        bool use_expanded_layout, allow_multiple_reduction;
        int num_channels_per_sm;
        int num_max_scaleout_ranks, num_max_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts_per_rank;
        int num_topk;
        int64_t num_timeout_cycles;

        // Parameters
        nv_bfloat16* x;
        float* topk_weights;
        int* src_metadata;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* token_metadata_at_forward;
        int* channel_linked_list;
        void* buffer; void* send_buffer;
        void* workspace;
        const topology::EPTopology* topology;
        int num_reduced_tokens;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <nixl_ep_v2/impls/adaptive/combine.cuh>

using namespace nixl_ep_v2::elastic::adaptive;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&adaptive_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.use_expanded_layout, args.allow_multiple_reduction,
                           args.launch_args.grid_dim.first,
                           args.num_channels_per_sm,
                           args.num_max_scaleout_ranks, args.num_max_scaleup_ranks,
                           args.hidden,
                           args.num_max_tokens_per_rank,
                           args.num_experts_per_rank,
                           args.num_topk,
                           args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.x, args.topk_weights,
                                                 args.src_metadata,
                                                 args.psum_num_recv_tokens_per_scaleup_rank,
                                                 args.token_metadata_at_forward,
                                                 args.channel_linked_list,
                                                 args.buffer, args.send_buffer, args.workspace,
                                                 args.topology,
                                                 args.num_reduced_tokens));
    }
};

static void launch_combine(void* x,
                           void* topk_weights,
                           int* src_metadata,
                           int* psum_num_recv_tokens_per_scaleup_rank,
                           int* token_metadata_at_forward,
                           int* channel_linked_list,
                           void* buffer, void* send_buffer, void* workspace,
                           const topology::EPTopology* topology,
                           const int& num_reduced_tokens, const int& num_max_tokens_per_rank,
                           const int& hidden,
                           const int& num_experts_per_rank, const int& num_topk,
                           const int64_t& num_timeout_cycles,
                           const int& num_max_scaleout_ranks, const int& num_max_scaleup_ranks,
                           const int& num_channels_per_sm,
                           const int& num_smem_bytes,
                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                           const at::cuda::CUDAStream& stream) {
    // Each channel owns one scale-up warp and one forward warp; the block shape is fixed with the buffer
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    EP_HOST_ASSERT(1 <= num_channels_per_sm and num_channels_per_sm <= kNumMaxChannelsPerSM);
    const int num_warps = 2 * num_channels_per_sm;
    EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                   "The combine block needs token_layout bytes for each of its 2 * num_channels_per_sm warps, "
                   "which exceeds the dynamic shared memory num_smem_bytes");
    EP_HOST_ASSERT(num_max_scaleout_ranks <= kNumMaxScaleoutRanksCap and num_max_scaleup_ranks <= kNumMaxScaleupRanksCap);

    // Generate, build and launch
    const auto num_threads = num_warps * 32;
    const CombineRuntime::Args args = {
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .num_channels_per_sm = num_channels_per_sm,
        .num_max_scaleout_ranks = num_max_scaleout_ranks, .num_max_scaleup_ranks = num_max_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts_per_rank = num_experts_per_rank,
        .num_topk = num_topk,
        .num_timeout_cycles = num_timeout_cycles,
        .x = static_cast<nv_bfloat16*>(x),
        .topk_weights = static_cast<float*>(topk_weights),
        .src_metadata = src_metadata,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .token_metadata_at_forward = token_metadata_at_forward,
        .channel_linked_list = channel_linked_list,
        .buffer = buffer, .send_buffer = send_buffer, .workspace = workspace,
        .topology = topology,
        .num_reduced_tokens = num_reduced_tokens,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(topology::get_num_launch_sms(), num_threads, num_smem_bytes, 2, true)
    };
    const auto code = CombineRuntime::generate(args);
    const auto runtime = jit::compiler->build("adaptive_combine", code);
    CombineRuntime::launch(runtime, args, stream);
}

class CombineReduceEpilogueRuntime final : public jit::LaunchRuntime<CombineReduceEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool use_expanded_layout, allow_multiple_reduction;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts_per_rank, num_topk;
        int64_t num_timeout_cycles;

        // Parameters
        nv_bfloat16* combined_x;
        float* combined_topk_weights;
        topk_idx_t* combined_topk_idx;
        void* buffer;
        void* workspace;
        void* bias_0;
        void* bias_1;
        int num_combined_tokens;
        const topology::EPTopology* topology;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <nixl_ep_v2/impls/adaptive/combine_reduce_epilogue.cuh>

using namespace nixl_ep_v2::elastic::adaptive;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                        args.use_expanded_layout, args.allow_multiple_reduction,
                           args.launch_args.grid_dim.first,
                           args.launch_args.num_threads / 32,
                           args.hidden,
                           args.num_max_tokens_per_rank,
                           args.num_experts_per_rank, args.num_topk,
                           args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.combined_x,
                                                 args.combined_topk_weights,
                                                 args.combined_topk_idx,
                                                 args.buffer, args.workspace,
                                                 args.bias_0, args.bias_1,
                                                 args.num_combined_tokens,
                                                 args.topology));
    }
};

static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights,
                                           topk_idx_t* combined_topk_idx,
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,
                                           const int& hidden,
                                           const int& num_experts_per_rank, const int& num_topk,
                                           const int64_t& num_timeout_cycles,
                                           void* buffer, void* workspace,
                                           void* bias_0, void* bias_1,
                                           const topology::EPTopology* topology,
                                           const int& num_sms, const int& num_smem_bytes,
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                                           const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false, nullptr, /* with_flags */ false);
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const CombineReduceEpilogueRuntime::Args args = {
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts_per_rank = num_experts_per_rank, .num_topk = num_topk,
        .num_timeout_cycles = num_timeout_cycles,
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        .combined_topk_weights = combined_topk_weights,
        .combined_topk_idx = combined_topk_idx,
        .buffer = buffer, .workspace = workspace,
        .bias_0 = bias_0, .bias_1 = bias_1,
        .num_combined_tokens = num_combined_tokens,
        .topology = topology,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)
    };
    const auto code = CombineReduceEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("adaptive_combine_reduce_epilogue", code);
    CombineReduceEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace nixl_ep_v2::elastic::adaptive
