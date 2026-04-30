/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _NIXL_DEVICE_CUH
#define _NIXL_DEVICE_CUH

#include <nixl_types.h>
#include <ucp/api/device/ucp_device_impl.h>

#include <cassert>

struct nixlGpuXferStatusH {
    ucp_device_request_t device_request;
};

/**
 * @enum  nixl_gpu_level_t
 * @brief An enumeration of different levels for GPU transfer requests.
 */
enum class nixl_gpu_level_t : uint64_t {
    THREAD = UCS_DEVICE_LEVEL_THREAD,
    WARP = UCS_DEVICE_LEVEL_WARP,
    BLOCK = UCS_DEVICE_LEVEL_BLOCK,
    GRID = UCS_DEVICE_LEVEL_GRID
};

namespace nixl_gpu_flags {
constexpr uint64_t defer = 1;

__device__ inline uint64_t
to_ucp_flags(uint64_t nixl_flags) noexcept {
    constexpr uint64_t all_known_nixl_flags{defer};
    assert((nixl_flags & ~all_known_nixl_flags) == 0);

    uint64_t ucp_flags{UCP_DEVICE_FLAG_NODELAY};
    if (nixl_flags & defer) {
        ucp_flags &= ~UCP_DEVICE_FLAG_NODELAY;
    }
    return ucp_flags;
}
} // namespace nixl_gpu_flags

struct nixlMemViewElem {
    nixlMemViewH mvh;
    size_t index; /**< Index in the memory view */
    size_t offset; /**< Offset within the buffer */
};

/**
 * @brief Convert UCS status to NIXL status.
 *
 * @param status [in] UCS status code.
 *
 * @return NIXL_IN_PROG     If the UCS status is not an error.
 * @return NIXL_ERR_BACKEND If the UCS status is an error.
 */
__device__ inline nixl_status_t
nixlGpuConvertUcsStatus(ucs_status_t status) {
    if (!UCS_STATUS_IS_ERR(status)) {
        return NIXL_IN_PROG;
    }
    printf("UCX returned error: %d\n", status);
    return NIXL_ERR_BACKEND;
}

/**
 * @brief Get the status of the transfer request.
 *
 * @param xfer_status [in]  Status of the transfer.
 *
 * @return NIXL_SUCCESS     The request has completed, no more operations are in progress.
 * @return NIXL_IN_PROG     One or more operations in the request have not completed.
 * @return NIXL_ERR_BACKEND An error occurred in UCX backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status) {
    const auto status = ucp_device_progress_req<static_cast<ucs_device_level_t>(level)>(
        &xfer_status.device_request);

    switch (status) {
    case UCS_OK:
        return NIXL_SUCCESS;
    case UCS_INPROGRESS:
        return NIXL_IN_PROG;
    default:
        return NIXL_ERR_BACKEND;
    }
}

/**
 * @brief Post a single-region memory transfer from local to remote GPU.
 *
 * This function creates and posts a transfer request using memory view elements @a src and @a dst.
 *
 * @param src         [in]  Source memory view element
 * @param dst         [in]  Destination memory view element
 * @param size        [in]  Size in bytes to transfer
 * @param channel_id  [in]  Channel ID to use for the transfer
 * @param flags       [in]  Transfer flags
 * @param xfer_status [in,out] Optional status handle (use @ref nixlGpuGetXferStatus)
 *
 * @return NIXL_IN_PROG     Transfer posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in UCX backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr) {
    auto src_mem_list = static_cast<ucp_device_local_mem_list_h>(src.mvh);
    auto dst_mem_list = static_cast<ucp_device_remote_mem_list_h>(dst.mvh);
    ucp_device_request_t *ucp_request{xfer_status ? &xfer_status->device_request : nullptr};
    const auto status =
        ucp_device_put<static_cast<ucs_device_level_t>(level)>(src_mem_list,
                                                               src.index,
                                                               src.offset,
                                                               dst_mem_list,
                                                               dst.index,
                                                               dst.offset,
                                                               size,
                                                               channel_id,
                                                               nixl_gpu_flags::to_ucp_flags(flags),
                                                               ucp_request);
    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Atomic add to remote GPU memory.
 *
 * This function performs an atomic increment on a remote counter.
 * The increment is visible only after previous writes complete.
 *
 * @param value       [in]  Value to add to the counter
 * @param counter     [in]  Counter memory view element
 * @param channel_id  [in]  Channel ID to use for the transfer
 * @param flags       [in]  Transfer flags
 * @param xfer_status [in,out] Optional status handle (use @ref nixlGpuGetXferStatus)
 *
 * @return NIXL_IN_PROG     Atomic add posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in UCX backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr) {
    auto mem_list = static_cast<ucp_device_remote_mem_list_h>(counter.mvh);
    ucp_device_request_t *ucp_request{xfer_status ? &xfer_status->device_request : nullptr};
    const auto status = ucp_device_counter_inc<static_cast<ucs_device_level_t>(level)>(
        value,
        mem_list,
        counter.index,
        counter.offset,
        channel_id,
        nixl_gpu_flags::to_ucp_flags(flags),
        ucp_request);
    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Get a local pointer to remote memory.
 *
 * This function returns a local pointer to the mapped memory of the
 * remote memory view handle at the given index.
 * The memory view must be prepared on the host using @ref nixlAgent::prepMemView.
 *
 * @param mem_view  [in]  Memory view handle (remote buffers)
 * @param index     [in]  Index in the memory view

 * @return Pointer to the mapped memory, or nullptr if not available.
 */
__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index) {
    auto mem_list = static_cast<ucp_device_remote_mem_list_h>(mvh);
    void *ptr = nullptr;
    ucp_device_get_ptr(mem_list, index, &ptr);
    return ptr;
}

#endif // _NIXL_DEVICE_CUH
