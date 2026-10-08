#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/topology.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class BarrierRuntime final : public jit::LaunchRuntime<BarrierRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_scaleup_ranks;
        int64_t num_timeout_cycles;

        // Parameters
        void* workspace;
        const topology::EPTopology* topology;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/barrier.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&barrier_impl<{}, {}, {}>);
}}
)",                        args.launch_args.num_threads, args.num_max_scaleup_ranks, args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config, args.workspace, args.topology));
    }
};

static void launch_barrier(void* workspace, const topology::EPTopology* topology,
                           const int& num_max_scaleup_ranks, const int64_t& num_timeout_cycles,
                           const at::cuda::CUDAStream& stream) {
    constexpr auto kNumThreads = 512;
    const BarrierRuntime::Args args = {
        .num_max_scaleup_ranks = num_max_scaleup_ranks, .num_timeout_cycles = num_timeout_cycles,
        .workspace = workspace, .topology = topology,
        .launch_args = jit::LaunchArgs(1, kNumThreads)};
    const auto code = BarrierRuntime::generate(args);
    const auto runtime = jit::compiler->build("barrier", code);
    BarrierRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
