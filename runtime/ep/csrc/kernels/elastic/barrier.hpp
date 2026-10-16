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

#include <nixl_ep_v2/common/compiled.cuh>
#include <nixl_ep_v2/common/exception.cuh>
#include <nixl_ep_v2/common/topology.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace nixl_ep_v2::elastic {

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
#include <nixl_ep_v2/impls/barrier.cuh>

using namespace nixl_ep_v2::elastic;

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

}  // namespace nixl_ep_v2::elastic
