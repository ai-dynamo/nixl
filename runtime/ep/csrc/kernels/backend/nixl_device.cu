// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

// The pointer-cache kernel of the NIXL transport (see `nixl_device.cuh`)
#define EP_NIXL_DEVICE_API 1   // `nixl_transport.cuh` then includes NIXL's GPU API and checks the UCX build
#include "nixl_device.cuh"

namespace nixl_ep_v2::elastic::nixl {
namespace {

__global__ void cache_peer_pointers_kernel(const NIXLDeviceContext context, const int* connected) {
    for (int peer = threadIdx.x; peer < context.num_world_ranks; peer += blockDim.x) {
        void* ptr = nullptr;
        if (peer == context.world_rank) {
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
