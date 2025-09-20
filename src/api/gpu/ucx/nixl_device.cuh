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

struct nixlGpuSignal {
    uint64_t inc = 0;
    uint64_t remote_addr = 0;
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
 * @return nixl_status_t Corresponding NIXL status code.
 */
__device__ inline nixl_status_t
nixlGpuConvertUcsStatus(ucs_status_t status) {
    return status == UCS_OK ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

/**
 * @brief Post a memory transfer request to the GPU.
 *
 * @param req_hndl    [in]  Request handle.
 * @param index       [in]  Index of the memory descriptor in the transfer request.
 * @param addr        [in]  Local address of the memory to be transferred.
 * @param remote_addr [in]  Remote address of the memory to be transferred to.
 * @param size        [in]  Size of the memory to be transferred.
 * @param is_no_delay [in]  Whether to use no-delay mode.
 * @param xfer_status [out] Status of the transfer. If null, the status is not reported.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostSingleWriteXferReq(nixlGpuXferReqH req_hndl,
                              unsigned index,
                              const void *addr,
                              uint64_t remote_addr,
                              size_t size,
                              bool is_no_delay = true,
                              nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status = ucp_device_put_single<static_cast<ucs_device_level_t>(level)>(
        params.mem_list, index, addr, remote_addr, size, params.flags, params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Post a signal transfer request to the GPU.
 *
 * @param req_hndl           [in]  Request handle.
 * @param index              [in]  Index of the signal to be transferred.
 * @param signal             [in]  Signal to be sent.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 * @param xfer_status        [out] Status of the transfer. If null, the status is not reported.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostSignalXferReq(nixlGpuXferReqH req_hndl,
                         unsigned index,
                         const nixlGpuSignal &signal,
                         bool is_no_delay = true,
                         nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status = ucp_device_counter_inc<static_cast<ucs_device_level_t>(level)>(
        params.mem_list, index, signal.inc, signal.remote_addr, params.flags, params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Post a partial memory transfer request to the GPU.
 *
 * @param req_hndl           [in]  Request handle.
 * @param count              [in]  Number of blocks to send. This is also the length of the arrays
 *                                 @a indices, @a sizes, @a addrs, and @a remote_addrs.
 * @param indices            [in]  Indices of the blocks to send.
 * @param sizes              [in]  Sizes of the blocks to send.
 * @param addrs              [in]  Addresses of the blocks to send.
 * @param remote_addrs       [in]  Remote addresses of the blocks to send to.
 * @param signal             [in]  Signal to be sent.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 * @param xfer_status        [out] Status of the transfer. If null, the status is not reported.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostPartialWriteXferReq(nixlGpuXferReqH req_hndl,
                               size_t count,
                               const unsigned *indices,
                               const size_t *sizes,
                               void *const *addrs,
                               const uint64_t *remote_addrs,
                               const nixlGpuSignal &signal,
                               unsigned signal_index,
                               bool is_no_delay = true,
                               nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status =
        ucp_device_put_multi_partial<static_cast<ucs_device_level_t>(level)>(params.mem_list,
                                                                             indices,
                                                                             count,
                                                                             addrs,
                                                                             remote_addrs,
                                                                             sizes,
                                                                             signal_index,
                                                                             signal.inc,
                                                                             signal.remote_addr,
                                                                             params.flags,
                                                                             params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Post a memory transfer request to the GPU.
 *
 * @param req_hndl           [in]  Request handle.
 * @param sizes              [in]  Sizes of the blocks to send.
 * @param addrs              [in]  Addresses of the blocks to send.
 * @param remote_addrs       [in]  Remote addresses of the blocks to send to.
 * @param signal             [in]  Signal to be sent.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 * @param xfer_status        [out] Status of the transfer. If null, the status is not reported.
 *
 * @note The arrays @a sizes, @a addrs, and @a remote_addrs must have the same length, which
 *       corresponds to the number of blocks to transfer as specified in @a req_hndl.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuPostWriteXferReq(nixlGpuXferReqH req_hndl,
                        const size_t *sizes,
                        void *const *addrs,
                        const uint64_t *remote_addrs,
                        const nixlGpuSignal &signal,
                        bool is_no_delay = true,
                        nixlGpuXferStatusH *xfer_status = nullptr) {
    const nixlGpuXferReqParams params{req_hndl, is_no_delay, xfer_status};

    ucs_status_t status =
        ucp_device_put_multi<static_cast<ucs_device_level_t>(level)>(params.mem_list,
                                                                     addrs,
                                                                     remote_addrs,
                                                                     sizes,
                                                                     signal.inc,
                                                                     signal.remote_addr,
                                                                     params.flags,
                                                                     params.ucp_request);

    return nixlGpuConvertUcsStatus(status);
}

/**
 * @brief Get the status of the transfer request.
 *
 * @param xfer_status [in]  Status of the transfer.
 *
 * @return NIXL_SUCCESS  The request has completed, no more operations are in progress.
 * @return NIXL_IN_PROG  One or more operations in the request have not completed.
 * @return Error code if call was not successful
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
