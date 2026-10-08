#pragma once

#include <cuda_runtime.h>
#include <deep_ep/common/nixl_transport.cuh>

namespace deep_ep::elastic::nixl {

// Fill `context.p2p_ptrs` from `context.remote_mvh`: `nixlGetPtr` (a device function) for every connected peer (a pointer
// iff UCX gave the peer its CUDA IPC lane), the local base for self, nullptr otherwise; the context travels by value (a staged view, a scratch table)
cudaError_t cache_peer_pointers(const NIXLDeviceContext& context, const int* connected, cudaStream_t stream);

}  // namespace deep_ep::elastic::nixl
