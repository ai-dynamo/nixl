// The pointer-cache kernel of the NIXL transport (see `nixl_device.cuh`)
#include "nixl_device.cuh"

#include <gpu/nixl_device.cuh>
// Without UCX's device header NIXL compiles a `getPtr` stub that returns nullptr for every peer: fail the build instead (see `nixl_transport.cuh`)
#if !__has_include(<ucp/api/device/ucp_device_impl.h>)
#error "UCX device header <ucp/api/device/ucp_device_impl.h> not on the include path: set EP_UCX_ROOT_DIR (-I <EP_UCX_ROOT_DIR>/include)"
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
