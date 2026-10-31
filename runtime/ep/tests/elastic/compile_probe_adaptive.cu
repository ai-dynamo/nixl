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

static void __instantiate_kernels() {
    auto pb = reinterpret_cast<void*>(&nixl_ep_v2::elastic::barrier_impl<128, 8, 2000000000ll>);
    // hidden 7168 BF16, top-8, 8 expert slots per rank (256 at the 4 x 8 maximum), max 4096 tokens per rank, 7 channels
    // per SM, 26-block grid; rank maxima 4 x 8 (p0 to p7) and the caps 32 x 128 with one slot per rank (p8, p9: four
    // lane groups of positions)
    auto p0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 4, 8, 14336, 0, 4096, 8, 8, 1, 2000000000ll>);
    auto p1 = reinterpret_cast<void*>(&adaptive_dispatch_impl<true, false, 26, 4, 7, 4, 8, 14336, 0, 4096, 8, 8, 1, 2000000000ll>);
    auto pf = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 4, 8, 7168, 56, 4096, 8, 8, 1, 2000000000ll>);
    // The acquire-flag protocol (`acquire_flags=True`, kAcquireFlags): the hybrid dispatch at the 4 x 8 maximum on the three wires and
    // the reduce epilogue; the forms above are the default (relaxed polls + acq_rel fences)
    auto pa0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 4, 8, 14336, 0, 4096, 8, 8, 1, 2000000000ll, true>);
    auto pa1 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 4, 8, 7168, 56, 4096, 8, 8, 1, 2000000000ll, true>);
    auto pa2 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 4, 4, 8, 3584, 112, 4096, 8, 8, 1, 2000000000ll, true>);
    auto pa3 = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<false, true, 148, 16, 7168, 4096, 8, 8, 2000000000ll, true>);
    auto p2 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, 26, 7, 4, 8, 7168, 4096, 8, 8, 2000000000ll>);
    auto p3 = reinterpret_cast<void*>(&adaptive_combine_impl<true, false, 26, 7, 4, 8, 7168, 4096, 8, 8, 2000000000ll>);
    auto p4 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, false, 148, 7, 16, 14336, 0, 4096, 8, 8, 1>);
    auto p5 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<true, true, false, 148, 7, 16, 14336, 0, 4096, 8, 8, 32>);
    auto p6 = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<false, true, 148, 16, 7168, 4096, 8, 8, 2000000000ll>);
    auto p7 = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<true, false, 148, 16, 7168, 4096, 8, 8, 2000000000ll>);
    auto p8 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 32, 128, 14336, 0, 4096, 1, 8, 1, 2000000000ll>);
    auto p9 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, 26, 7, 32, 128, 7168, 4096, 1, 8, 2000000000ll>);
    // FP8 wire at hidden 7168 (56 SF packs): top-6 (top-8 is `pf` above)
    auto pg = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 4, 8, 7168, 56, 4096, 8, 6, 1, 2000000000ll>);
    auto ph = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, false, 148, 7, 16, 7168, 56, 4096, 8, 8, 1>);
    // NVFP4 wire at hidden 7168 (3584 code bytes, 112 SF packs, the 4192-byte record at top-8 (4224 with the sector pad), 4160 at top-6): the
    // dispatch at 4 channels per SM (top-8, top-6) and at 8 (this wire's count under `prefer_overlap_with_compute`); the copy
    // epilogue at its 32 warps (a 64-register budget), compact and expanded with zero padding at alignment 128
    auto pn0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 4, 4, 8, 3584, 112, 4096, 8, 8, 1, 2000000000ll>);
    auto pn1 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 4, 4, 8, 3584, 112, 4096, 8, 6, 1, 2000000000ll>);
    auto pn2 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 8, 4, 8, 3584, 112, 4096, 8, 8, 1, 2000000000ll>);
    auto pn3 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, false, 148, 4, 32, 3584, 112, 4096, 8, 8, 1>);
    auto pn4 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<true, true, false, 148, 4, 32, 3584, 112, 4096, 8, 8, 128>);
    // The perf worlds of this wire (fp4d_perf_2x4 / fp4d_perf_1x16: the compact dispatch at 4 channels per SM, 26 SMs, top-8): a
    // 2 x 4 world with 32 expert slots per rank and a 1 x 16 world with 16, each with its 32-warp compact copy epilogue (pn0-pn4
    // are the 4 x 8 maximum world, 8 slots per rank). The spill gate then covers the perf worlds' 4096-max-token forms at the
    // probe's constants (kDoCPUSync false, 148 SMs); the 1024-max-token arms and the CPU-sync dispatch the bench times are not in it
    auto pn5 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 4, 2, 4, 3584, 112, 4096, 32, 8, 1, 2000000000ll>);
    auto pn6 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, false, 148, 4, 32, 3584, 112, 4096, 32, 8, 1>);
    auto pn7 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 4, 1, 16, 3584, 112, 4096, 16, 8, 1, 2000000000ll>);
    auto pn8 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, false, 148, 4, 32, 3584, 112, 4096, 16, 8, 1>);
    // Wide domains: a 1 x 72 world (an NVL72 domain, 4 slots per rank = 288: three lane groups of positions) and 2 x 128
    // (2 slots per rank = 512); exercises the multi-word position masks of the kernels
    auto pw = reinterpret_cast<void*>(&nixl_ep_v2::elastic::barrier_impl<128, 72, 2000000000ll>);
    auto q0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 1, 72, 14336, 0, 4096, 4, 8, 1, 2000000000ll>);
    auto q1 = reinterpret_cast<void*>(&adaptive_dispatch_impl<true, false, 26, 4, 7, 1, 72, 14336, 0, 4096, 4, 8, 1, 2000000000ll>);
    auto q2 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, 26, 7, 1, 72, 7168, 4096, 4, 8, 2000000000ll>);
    auto q3 = reinterpret_cast<void*>(&adaptive_combine_impl<true, false, 26, 7, 1, 72, 7168, 4096, 4, 8, 2000000000ll>);
    auto q4 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, false, 26, 4, 7, 2, 128, 14336, 0, 4096, 2, 8, 1, 2000000000ll>);
    auto q5 = reinterpret_cast<void*>(&adaptive_combine_impl<false, true, 26, 7, 2, 128, 7168, 4096, 2, 8, 2000000000ll>);
    // Zero copy: the 1 x 16 perf shape, FP8 at 8 channels per SM, a 1 x 72 domain, the 32 x 128 and 1 x 128 caps, the 2 x 4 cohort
    // (hybrid under zero copy) and the zero-copy copy epilogue
    auto z0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 7, 1, 16, 14336, 0, 4096, 16, 8, 1, 2000000000ll>);
    auto z1 = reinterpret_cast<void*>(&adaptive_dispatch_impl<true, true, 26, 4, 7, 1, 16, 14336, 0, 4096, 16, 8, 1, 2000000000ll>);
    auto z2 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 8, 1, 16, 7168, 56, 4096, 16, 8, 1, 2000000000ll>);
    auto z3 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 7, 1, 72, 14336, 0, 4096, 4, 8, 1, 2000000000ll>);
    auto z4 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 7, 32, 128, 14336, 0, 4096, 1, 8, 1, 2000000000ll>);
    auto z5 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 7, 1, 128, 14336, 0, 4096, 8, 8, 1, 2000000000ll>);
    auto z6 = reinterpret_cast<void*>(&adaptive_dispatch_impl<true, true, 26, 4, 8, 1, 16, 7168, 56, 4096, 16, 8, 1, 2000000000ll>);
    auto z7 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 7, 2, 4, 14336, 0, 4096, 32, 8, 1, 2000000000ll>);
    auto z8 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, true, 148, 7, 16, 14336, 0, 4096, 8, 8, 1>);
    (void) pb; (void) p0; (void) p1; (void) pf; (void) p2; (void) p3; (void) p4; (void) p5; (void) p6; (void) p7; (void) p8; (void) p9; (void) pw; (void) q0; (void) q1; (void) q2; (void) q3; (void) q4; (void) q5;
    (void) pg; (void) ph; (void) pn0; (void) pn1; (void) pn2; (void) pn3; (void) pn4; (void) pn5; (void) pn6; (void) pn7; (void) pn8;
    // Zero copy on the NVFP4 wire: the 1 x 16 perf shape at 4 channels per SM (the tests' 4-channel block) and at 8 (this wire's count
    // under `prefer_overlap_with_compute`), each with its 32-warp compact copy epilogue
    auto zn0 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 4, 1, 16, 3584, 112, 4096, 16, 8, 1, 2000000000ll>);
    auto zn1 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, true, 148, 4, 32, 3584, 112, 4096, 16, 8, 1>);
    auto zn2 = reinterpret_cast<void*>(&adaptive_dispatch_impl<false, true, 26, 4, 8, 1, 16, 3584, 112, 4096, 16, 8, 1, 2000000000ll>);
    auto zn3 = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<false, false, true, 148, 8, 32, 3584, 112, 4096, 16, 8, 1>);
    (void) z0; (void) z1; (void) z2; (void) z3; (void) z4; (void) z5; (void) z6; (void) z7; (void) z8; (void) zn0; (void) zn1; (void) zn2; (void) zn3;
}
