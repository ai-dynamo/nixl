// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

// Compile-only probe for the adaptive kernels (they are normally JIT-compiled on first use, which hides
// template errors until a multi-GPU run). Build inside the release container with the JIT's flags, e.g.
//   nvcc -std=c++20 -O3 --expt-relaxed-constexpr --expt-extended-lambda -arch=sm_100a \
//        -I nixl_ep_v2/include -I $NIXL_EP_NIXL_ROOT_DIR/include -I $NIXL_EP_UCX_ROOT_DIR/include -DEP_NIXL_DEVICE_API=1 \
//        -c tests/elastic/compile_probe_adaptive.cu -o /dev/null
#include <nixl_ep_v2/impls/adaptive/dispatch.cuh>
#include <nixl_ep_v2/impls/adaptive/combine.cuh>
#include <nixl_ep_v2/impls/adaptive/dispatch_copy_epilogue.cuh>
#include <nixl_ep_v2/impls/adaptive/combine_reduce_epilogue.cuh>
#include <nixl_ep_v2/impls/barrier.cuh>

using namespace nixl_ep_v2::elastic::adaptive;
namespace topology = nixl_ep_v2::elastic::topology;

constexpr int64_t kNumTimeoutCycles = nixl_ep_v2::elastic::comm::kNumOneSecCycles;

static void __instantiate_kernels() {
    auto pb = reinterpret_cast<void*>(&nixl_ep_v2::elastic::barrier_impl<128, 8, kNumTimeoutCycles>);
    // hidden 7168 BF16, top-8, 8 slots per rank, 7 channels per SM, the launch grid;
    // rank maxima 4 x 8 (p0 to p3; p4 to p7 are the two epilogues; the main combine takes the scale-up maximum only and no expert count) and the caps 32 x 128 (p8, p9: four lane groups of positions)
    auto p0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, topology::kNumLaunchSMs, 4, 7, 4, 8, 14336, 0, 8, 8, 1, kNumTimeoutCycles>);
    auto p1 = reinterpret_cast<void*>(&adaptive_dispatch_impl<true, topology::kNumLaunchSMs, 4, 7, 4, 8, 14336, 0, 8, 8, 1, kNumTimeoutCycles>);
    // FP8 wire: hidden 7168 and 56 FP32 scale factors
    auto pf = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, topology::kNumLaunchSMs, 4, 7, 4, 8, 7168, 56, 8, 8, 1, kNumTimeoutCycles>);
    // NVFP4: logical hidden 7168, 3584 code bytes and 112 packed block scales.
    auto pn = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, topology::kNumLaunchSMs, 4, 8, 4, 8, 3584, 112, 8, 8, 1, kNumTimeoutCycles>);
    auto pn6 = reinterpret_cast<void*>(&adaptive_dispatch_impl<true, topology::kNumLaunchSMs, 4, 8, 4, 8, 3584, 112, 8, 6, 1, kNumTimeoutCycles>);
    auto p2 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, topology::kNumLaunchSMs, 7, 8, 7168, 8, kNumTimeoutCycles>);
    auto p3 = reinterpret_cast<void*>(&adaptive_combine_impl<true, false, topology::kNumLaunchSMs, 7, 8, 7168, 8, kNumTimeoutCycles>);
    auto p4 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, 148, 7, 16, 8, 14336, 0, 8, 8, 1>);
    auto p5 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<true, true, 148, 7, 16, 8, 14336, 0, 8, 8, 32>);
    auto pfp = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, 148, 7, 16, 8, 7168, 56, 8, 8, 1>);
    auto pnp = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, 148, 8, 32, 8, 3584, 112, 8, 8, 1>);
    auto pne = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<true, true, 148, 8, 32, 8, 3584, 112, 8, 6, 16>);
    auto pnc = reinterpret_cast<void*>(&adaptive_combine_impl<true, false, topology::kNumLaunchSMs, 8, 8, 7168, 6, kNumTimeoutCycles>);
    auto pnr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<true, false, 148, 16, 8, 7168, 8, 6, kNumTimeoutCycles>);
    auto p6 = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<false, true, 148, 16, 7, 7168, 8, 8, kNumTimeoutCycles>);
    auto p7 = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<true, false, 148, 16, 7, 7168, 8, 8, kNumTimeoutCycles>);
    auto p8 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, topology::kNumLaunchSMs, 4, 7, topology::kNumMaxScaleoutRanksCap, topology::kNumMaxScaleupRanksCap, 14336, 0, 1, 8, 1, kNumTimeoutCycles>);
    auto p9 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, topology::kNumLaunchSMs, 7, topology::kNumMaxScaleupRanksCap, 7168, 8, kNumTimeoutCycles>);
    // Wide domains: a 1 x 72 world (an NVL72 domain, 288 experts: three lane groups of positions) and 2 x 128
    // (512 experts); exercises the multi-word position masks of the kernels
    auto pw = reinterpret_cast<void*>(&nixl_ep_v2::elastic::barrier_impl<128, 72, kNumTimeoutCycles>);
    auto q0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, topology::kNumLaunchSMs, 4, 7, 1, 72, 14336, 0, 4, 8, 1, kNumTimeoutCycles>);
    auto q1 = reinterpret_cast<void*>(&adaptive_dispatch_impl<true, topology::kNumLaunchSMs, 4, 7, 1, 72, 14336, 0, 4, 8, 1, kNumTimeoutCycles>);
    auto q2 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, topology::kNumLaunchSMs, 7, 72, 7168, 8, kNumTimeoutCycles>);
    auto q3 = reinterpret_cast<void*>(&adaptive_combine_impl<true, false, topology::kNumLaunchSMs, 7, 72, 7168, 8, kNumTimeoutCycles>);
    auto q4 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, topology::kNumLaunchSMs, 4, 7, 2, 128, 14336, 0, 2, 8, 1, kNumTimeoutCycles>);
    auto q5 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, topology::kNumLaunchSMs, 7, 128, 7168, 8, kNumTimeoutCycles>);
    (void) pb; (void) p0; (void) p1; (void) p2; (void) p3; (void) p4; (void) p5; (void) p6; (void) p7; (void) p8; (void) p9; (void) pw; (void) q0; (void) q1; (void) q2; (void) q3; (void) q4; (void) q5;
    (void) pn; (void) pn6; (void) pnp; (void) pne; (void) pnc; (void) pnr;
}
