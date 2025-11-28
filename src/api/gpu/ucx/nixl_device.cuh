/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @brief Parameters for GPU transfer requests with safe type conversion.
 */
struct nixlGpuXferReqParams {
    nixlGpuXferReqParams() = delete;

    __device__
    nixlGpuXferReqParams(nixlGpuXferReqH req_hndl,
                         bool is_no_delay,
                         nixlGpuXferStatusH *xfer_status)
        : mem_list{static_cast<ucp_device_mem_list_handle_h>(req_hndl)},
          flags{is_no_delay ? static_cast<uint64_t>(UCP_DEVICE_FLAG_NODELAY) : 0},
          ucp_request{xfer_status ? &xfer_status->device_request : nullptr} {}

    ucp_device_mem_list_handle_h mem_list;
    uint64_t flags;
    ucp_device_request_t *ucp_request;
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
 * @brief Post a memory transfer request to the GPU.
 *
 * @param req_hndl      [in]  Request handle.
 * @param desc_index    [in]  Index of the memory descriptor in the transfer request.
 * @param local_offset  [in]  Local offset of the memory to be transferred.
 * @param remote_offset [in]  Remote offset of the memory to be transferred to.
 * @param size          [in]  Size in bytes of the memory to be transferred.
 * @param channel_id    [in]  Channel ID to use for the transfer.
 * @param is_no_delay   [in]  Whether to use no-delay mode.
 * @param xfer_status   [out] Status of the transfer. If not null, use @ref
 *                            nixlGpuGetXferStatus to check for completion.
 *
 * @return NIXL_IN_PROG       Transfer posted successfully.
 * @return NIXL_ERR_BACKEND   An error occurred in UCX backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostSingleWriteXferReq(nixlGpuXferReqH req_hndl,
                              unsigned desc_index,
                              size_t local_offset,
                              size_t remote_offset,
                              size_t size,
                              unsigned channel_id = 0,
                              bool is_no_delay = true,
                              nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status = ucp_device_put_single<static_cast<ucs_device_level_t>(level)>(
        params.mem_list, desc_index, local_offset, remote_offset, size, channel_id, params.flags, params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Post a signal transfer request to the GPU.
 *
 * @param req_hndl           [in]  Request handle.
 * @param signal_desc_index  [in]  Index of the signal descriptor to be sent.
 * @param signal_inc         [in]  Increment value for the signal.
 * @param signal_offset      [in]  Offset of the signal to be sent.
 * @param channel_id         [in]  Channel ID to use for the transfer.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 * @param xfer_status        [out] Status of the transfer. If not null, use @ref
 *                                 nixlGpuGetXferStatus to check for completion.
 *
 * @return NIXL_IN_PROG            Transfer posted successfully.
 * @return NIXL_ERR_BACKEND        An error occurred in UCX backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostSignalXferReq(nixlGpuXferReqH req_hndl,
                         unsigned signal_desc_index,
                         uint64_t signal_inc,
                         size_t signal_offset,
                         unsigned channel_id = 0,
                         bool is_no_delay = true,
                         nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status = ucp_device_counter_inc<static_cast<ucs_device_level_t>(level)>(
        params.mem_list, signal_desc_index, signal_inc, signal_offset, channel_id, params.flags, params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Post a partial memory transfer request to the GPU.
 *
 * @param req_hndl           [in]  Request handle.
 * @param count              [in]  Number of blocks to send. This is also the length of the arrays
 *                                 @a desc_indices, @a sizes, @a local_offsets, and @a remote_offsets.
 * @param desc_indices       [in]  Indices of the memory descriptors to send.
 * @param sizes              [in]  Sizes of the blocks to send.
 * @param local_offsets      [in]  Local offsets of the blocks to send.
 * @param remote_offsets     [in]  Remote offsets of the blocks to send to.
 * @param signal_desc_index  [in]  Index of the signal descriptor to be sent.
 * @param signal_inc         [in]  Increment value for the signal. The signal will only be posted if signal_inc != 0.
 * @param signal_offset      [in]  Offset of the signal to be sent.
 * @param channel_id         [in]  Channel ID to use for the transfer.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 * @param xfer_status        [out] Status of the transfer. If not null, use @ref
 *                                 nixlGpuGetXferStatus to check for completion.
 *
 * @return NIXL_IN_PROG            Transfer posted successfully.
 * @return NIXL_ERR_BACKEND        An error occurred in UCX backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostPartialWriteXferReq(nixlGpuXferReqH req_hndl,
                               size_t count,
                               const unsigned *desc_indices,
                               const size_t *sizes,
                               const size_t *local_offsets,
                               const size_t *remote_offsets,
                               unsigned signal_desc_index,
                               uint64_t signal_inc,
                               size_t signal_offset,
                               unsigned channel_id = 0,
                               bool is_no_delay = true,
                               nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status =
        ucp_device_put_multi_partial<static_cast<ucs_device_level_t>(level)>(params.mem_list,
                                                                             desc_indices,
                                                                             count,
                                                                             local_offsets,
                                                                             remote_offsets,
                                                                             sizes,
                                                                             signal_desc_index,
                                                                             signal_inc,
                                                                             signal_offset,
                                                                             channel_id,
                                                                             params.flags,
                                                                             params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Post a memory transfer request to the GPU.
 *
 * @param req_hndl           [in]  Request handle.
 * @param signal_inc         [in]  Increment value for the signal. The signal will only be posted if signal_inc != 0.
 * @param channel_id         [in]  Channel ID to use for the transfer.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 * @param xfer_status        [out] Status of the transfer. If not null, use @ref
 *                                 nixlGpuGetXferStatus to check for completion.
 *
 * @return NIXL_IN_PROG            Transfer posted successfully.
 * @return NIXL_ERR_BACKEND        An error occurred in UCX backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostWriteXferReq(nixlGpuXferReqH req_hndl,
                        uint64_t signal_inc,
                        unsigned channel_id = 0,
                        bool is_no_delay = true,
                        nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status =
        ucp_device_put_multi<static_cast<ucs_device_level_t>(level)>(params.mem_list,
                                                                     signal_inc,
                                                                     channel_id,
                                                                     params.flags,
                                                                     params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
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
 * @brief Read the signal.
 *
 * The signal must be initialized with the host function @ref prepGpuSignal.
 *
 * @param signal [in]  Address of the signal.
 *
 * @return The signal.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ uint64_t
nixlGpuReadSignal(const void *signal) {
    return ucp_device_counter_read<static_cast<ucs_device_level_t>(level)>(signal);
}

/**
 * @brief Write value to the local signal.
 *
 * This function can be used to set a signal to a specific value.
 *
 * The signal must be initialized with the host function @ref prepGpuSignal.
 *
 * @param signal [in,out]  Address of the signal.
 * @param value  [in]      Value to write to the signal.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ void
nixlGpuWriteSignal(void *signal, uint64_t value) {
    ucp_device_counter_write<static_cast<ucs_device_level_t>(level)>(signal, value);
}

#endif // _NIXL_DEVICE_CUH
