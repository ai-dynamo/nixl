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

struct nixlGpuXferStatusH;

struct nixlGpuSignal {
    uint64_t inc = 0;
    uint64_t remote_addr = 0;
};

/**
 * @enum  nixl_gpu_level_t
 * @brief An enumeration of different levels for GPU transfer requests.
 */
enum class nixl_gpu_level_t : uint64_t {
    THREAD,
    WARP,
    BLOCK,
    GRID
};

/**
 * @brief Post a memory transfer request to the GPU.
 *
 * @param req_hndl    [in]  Request handle.
 * @param addr        [in]  Local address of the memory to be transferred.
 * @param remote_addr [in]  Remote address of the memory to be transferred to.
 * @param is_no_delay [in]  Whether to use no-delay mode.
 * @param xfer_status [out] Status of the transfer. If null, the status is not reported.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ nixl_status_t
nixlGpuPostSingleWriteXferReq(nixlGpuXferReqH *req_hndl,
                              const void *addr,
                              uint64_t remote_addr,
                              bool is_no_delay = true,
                              nixlGpuXferStatusH *xfer_status = nullptr)
{
    return NIXL_ERR_NOT_SUPPORTED;
}

/**
 * @brief Post a signal transfer request to the GPU.
 *
 * @param req_hndl           [in]  Request handle.
 * @param signal             [in]  Signal to be sent.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 * @param xfer_status        [out] Status of the transfer. If null, the status is not reported.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ nixl_status_t
nixlGpuPostSignalXferReq(nixlGpuXferReqH *req_hndl,
                         const nixlGpuSignal &signal,
                         bool is_no_delay = true,
                         nixlGpuXferStatusH *xfer_status = nullptr)
{
    return NIXL_ERR_NOT_SUPPORTED;
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
template<nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ nixl_status_t
nixlGpuPostPartialWriteXferReq(nixlGpuXferReqH *req_hndl,
                               size_t count,
                               const int *indices,
                               const size_t *sizes,
                               void *const *addrs,
                               const uint64_t *remote_addrs,
                               const nixlGpuSignal &signal,
                               bool is_no_delay = true,
                               nixlGpuXferStatusH *xfer_status = nullptr)
{
    return NIXL_ERR_NOT_SUPPORTED;
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
template<nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ nixl_status_t
nixlGpuPostWriteXferReq(nixlGpuXferReqH *req_hndl,
                        const size_t *sizes,
                        void *const *addrs,
                        const uint64_t *remote_addrs,
                        const nixlGpuSignal &signal,
                        bool is_no_delay = true,
                        nixlGpuXferStatusH *xfer_status = nullptr)
{
    return NIXL_ERR_NOT_SUPPORTED;
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
template<nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ nixl_status_t
nixlGpuGetXferStatus(const nixlGpuXferStatusH &xfer_status)
{
    return NIXL_ERR_NOT_SUPPORTED;
}

/**
 * @brief Read the value of a signal.
 *
 * @param signal_addr [in]  Address of the signal.
 *
 * @return The value of the signal.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ uint64_t
nixlGpuReadSignalValue(const void *signal_addr) {
    return 0;
}

#endif
