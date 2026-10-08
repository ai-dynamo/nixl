#pragma once

#include <cuda_runtime.h>
#include <nixl_ep_v2/common/nixl_transport.cuh>

namespace nixl_ep_v2::elastic::nixl {

// Fill `context.p2p_ptrs` from `context.remote_mvh`: `nixlGetPtr` (a device function) for every connected peer (a pointer
// iff UCX gave the peer its CUDA IPC lane), the local base for self, nullptr otherwise; the context travels by value (a staged view, a scratch table)
cudaError_t cache_peer_pointers(const NIXLDeviceContext& context, const int* connected, cudaStream_t stream);

}  // namespace nixl_ep_v2::elastic::nixl
