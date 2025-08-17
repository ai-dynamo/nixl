#ifndef __NIXL_UCX_DEVICE_CUH
#define __NIXL_UCX_DEVICE_CUH

#include <cuda_runtime.h>
#include <ucp/api/cuda/ucp_dev.cuh>
#include <nixl_types.h>

struct nixlGpuXferStatusH {
    ucp_dev_request_t ucp_request;
};

enum nixl_gpu_xfer_coordination_level_t {
    NIXL_GPU_XFER_COORDINATION_THREAD = UCP_DEV_SCALE_THREAD,
    NIXL_GPU_XFER_COORDINATION_WARP = UCP_DEV_SCALE_WARP,
    NIXL_GPU_XFER_COORDINATION_BLOCK = UCP_DEV_SCALE_BLOCK,
    // NIXL_GPU_XFER_COORDINATION_GRID TODO
};

template<nixl_gpu_xfer_coordination_level_t level = NIXL_GPU_XFER_COORDINATION_BLOCK>
__device__ static inline nixl_status_t nixlPostGpuXferReq(nixlGpuXferReqH* req_hndl,
                                                          uint64_t signal_inc,
                                                          nixlGpuXferStatusH* xfer_status) {
    ucp_dev_batch_flags flags = UCP_DEV_BATCH_FLAG_RMA_IOV;
    ucs_status_t status;

    if (!req_hndl) {
        return NIXL_ERR_INVALID_PARAM;
    }

    /*
     * UCP_DEV_BATCH_FLAG_RMA_IOV : always set
     * UCP_DEV_BATCH_FLAG_ATOMIC  : set if signal_inc != 0
     * UCP_DEV_BATCH_FLAG_COMP    : set if xfer_status != nullptr
     * UCP_DEV_BATCH_FLAG_NODELAY : always set
     */
    if (signal_inc != 0) {
        flags = static_cast<ucp_dev_batch_flags>(flags | UCP_DEV_BATCH_FLAG_ATOMIC);
    }

    flags = static_cast<ucp_dev_batch_flags>(flags | UCP_DEV_BATCH_FLAG_NODELAY);

    if (xfer_status != nullptr) {
        flags = static_cast<ucp_dev_batch_flags>(flags | UCP_DEV_BATCH_FLAG_COMP);
    }

    // Cast void* to ucp_batch_h in the kernel
    ucp_batch_h ucp_batch = static_cast<ucp_batch_h>(req_hndl);

    status = ucp_dev_batch_execute<static_cast<ucp_dev_scale_t>(level)>(
            ucp_batch, flags, signal_inc, &xfer_status->ucp_request);

    switch (status) {
        case UCS_OK:
            return NIXL_SUCCESS;
        case UCS_INPROGRESS:
            return NIXL_IN_PROG;
        default:
            return NIXL_ERR_BACKEND;
    }
}

template<nixl_gpu_xfer_coordination_level_t level = NIXL_GPU_XFER_COORDINATION_BLOCK>
__device__ static inline nixl_status_t
nixlPostPartialGpuXferReq(nixlGpuXferReqH* req_hndl,
                          uint64_t signal_inc,
                          size_t count,
                          int* indices,
                          size_t* sizes,
                          const size_t* src_offsets,
                          const size_t* dst_offsets,
                          nixlGpuXferStatusH* xfer_status,
                          bool is_no_delay = true)
{
    uint64_t flags = is_no_delay ? UCP_DEV_BATCH_FLAG_NODELAY : 0;
    ucs_status_t status;
    ucp_dev_request_t* ucp_request = nullptr;

    if (!req_hndl) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (xfer_status != nullptr) {
        ucp_request = &xfer_status->ucp_request;
    }

    /*
     * UCP_DEV_BATCH_FLAG_RMA_IOV : set if count > 0
     * UCP_DEV_BATCH_FLAG_ATOMIC  : set if signal_inc != 0
     * UCP_DEV_BATCH_FLAG_COMP    : set if xfer_status != nullptr
     * UCP_DEV_BATCH_FLAG_NODELAY : set if is_no_delay is true
     */
    if (signal_inc != 0) {
        // TODO pass signal_inc as optional / add template parameter
        flags |= UCP_DEV_BATCH_FLAG_ATOMIC;
    }

    if (xfer_status != nullptr) {
        flags |= UCP_DEV_BATCH_FLAG_COMP;
    }

    if (count > 0) {
        flags |= UCP_DEV_BATCH_FLAG_RMA_IOV;
    }

    // Cast void* to ucp_batch_h in the kernel
    ucp_batch_h ucp_batch = static_cast<ucp_batch_h>(req_hndl);

    status = ucp_dev_batch_execute_part<static_cast<ucp_dev_scale_t>(level)>(
            ucp_batch, flags, signal_inc, count, indices, src_offsets,
            dst_offsets, sizes, ucp_request);

    switch (status) {
        case UCS_OK:
            return NIXL_SUCCESS;
        case UCS_INPROGRESS:
            return NIXL_IN_PROG;
        default:
            return NIXL_ERR_BACKEND;
    }
}

template<nixl_gpu_xfer_coordination_level_t level = NIXL_GPU_XFER_COORDINATION_BLOCK>
__device__ static inline nixl_status_t nixlGetGpuXferReqStatus(nixlGpuXferStatusH* xfer_status) {
    // TODO Artemy needs to add nonbliocking get status and here need to use it
    ucs_status_t ucp_status = ucp_dev_request_progress<static_cast<ucp_dev_scale_t>(level)>(
            &xfer_status->ucp_request);

    switch (ucp_status) {
        case UCS_OK:
            return NIXL_SUCCESS;
        case UCS_INPROGRESS:
            return NIXL_IN_PROG;
        default:
            return NIXL_ERR_BACKEND;
    }
}

#endif // __NIXL_UCX_DEVICE_CUH
