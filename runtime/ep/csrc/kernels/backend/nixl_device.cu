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

// The pointer-cache kernel of the NIXL transport (see `nixl_device.cuh`)
#include "nixl_device.cuh"

#include <gpu/nixl_device.cuh>
// Without UCX's device header NIXL compiles a `getPtr` stub that returns nullptr for every peer: fail the build instead (see `nixl_transport.cuh`)
#if !__has_include(<ucp/api/device/ucp_device_impl.h>)
#error "UCX device header <ucp/api/device/ucp_device_impl.h> not on the include path: set NIXL_EP_UCX_ROOT_DIR (-I <NIXL_EP_UCX_ROOT_DIR>/include)"
#endif
// Without UCX's CUDA IPC lane `nixlGetPtr` returns nullptr for every peer, the same symptom
#if !defined(UCT_CUDA_IPC_SUPPORTED) || !UCT_CUDA_IPC_SUPPORTED
#error "UCX's device API was built without CUDA IPC (UCT_CUDA_IPC_SUPPORTED is unset): rebuild UCX with CUDA"
#endif

namespace nixl_ep_v2::elastic::nixl {
namespace {

__global__ void cache_peer_pointers_kernel(const NIXLDeviceContext context, const int* connected) {
    for (int peer = threadIdx.x; peer < context.num_rank_ids; peer += blockDim.x) {
        void* ptr = nullptr;
        if (peer == context.rank_id) {
            ptr = context.local_base;
        } else if (connected[peer]) {
            // Null-agent holes are not valid `nixlGetPtr` inputs; the view owns the pointer's lifetime
            ptr = nixlGetPtr(context.remote_mvh, peer);
        }
        context.p2p_ptrs[peer] = ptr;
    }
}

}  // namespace

cudaError_t cache_peer_pointers(const NIXLDeviceContext& context, const int* connected, cudaStream_t stream) {
    cache_peer_pointers_kernel<<<1, 128, 0, stream>>>(context, connected);
    return cudaGetLastError();
}

}  // namespace nixl_ep_v2::elastic::nixl
