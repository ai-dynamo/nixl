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

/**
 * @brief A struct for a nixlGpuXferStatusH
 */
struct nixlGpuXferStatusH;

/**
 * @enum  nixl_gpu_coordination_level_t
 * @brief An enumeration of different coordination levels for GPU transfer requests.
 */
enum nixl_gpu_coordination_level_t {
    NIXL_GPU_COORDINATION_THREAD,
    NIXL_GPU_COORDINATION_WARP,
    NIXL_GPU_COORDINATION_BLOCK,
    NIXL_GPU_COORDINATION_GRID
};

/**
 * @brief Post a memory transfer request to the GPU.
 * 
 * @param req_hndl    [in]  The request handle.
 * @param address     [in]  The local address of the memory to be transferred.
 * @param remote_addr [in]  The remote address of the memory to be transferred.
 * @param xfer_status [out] The status of the transfer. If null the status is not reported.
 * @param is_no_delay [in]  Whether to use no-delay mode. True by default.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_coordination_level_t level = NIXL_GPU_COORDINATION_BLOCK> 
__device__ static inline nixl_status_t 
nixlGpuPostMemXferReq(nixlGpuXferReqH* req_hndl, 
                      void *address, 
                      uint64_t remote_addr,
                      nixlGpuXferStatusH* xfer_status = nullptr,
                      bool is_no_delay = true)
{
    return NIXL_ERR_NOT_SUPPORTED;
}

/**
 * @brief Post a signal transfer request to the GPU.
 * 
 * @param req_hndl           [in]  The request handle.
 * @param signal_inc         [in]  The increment of the signal.
 * @param signal_remote_addr [in]  The remote address of the signal.
 * @param xfer_status        [out] The status of the transfer. If null the status is not reported.
 * @param is_no_delay        [in]  Whether to use no-delay mode. True by default.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_coordination_level_t level = NIXL_GPU_COORDINATION_BLOCK> 
__device__ static inline nixl_status_t 
nixlGpuPostSignalXferReq(nixlGpuXferReqH* req_hndl, 
                         uint64_t signal_inc, 
                         uint64_t signal_remote_addr, 
                         nixlGpuXferStatusH* xfer_status = nullptr, 
                         bool is_no_delay = true)
{
    return NIXL_ERR_NOT_SUPPORTED;
}

/**
 * @brief Post a partial memory transfer request to the GPU.
 * 
 * @param req_hndl           [in]  The request handle.
 * @param count              [in]  The number of blocks to send.
 * @param indices            [in]  The indices of the blocks to send.
 * @param sizes              [in]  The sizes of the blocks to send.
 * @param addreses           [in]  The addresses of the blocks to send.
 * @param remote_addrs       [in]  The remote addresses of the blocks to send.
 * @param signal_remote_addr [in]  The remote address of the signal. If address is 0, no signal.
 * @param signal_inc         [in]  The increment of the signal.
 * @param xfer_status        [out] The status of the transfer. If null the status is not reported.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_coordination_level_t level = NIXL_GPU_COORDINATION_BLOCK> 
__device__ static inline nixl_status_t 
nixlGpuPostPartialMemXferReq(nixlGpuXferReqH* req_hndl, 
                             size_t count, 
                             const int* indices,
                             const size_t* sizes, 
                             const void** addreses, 
                             const uint64_t* remote_addrs, 
                             uint64_t signal_remote_addr,
                             uint64_t signal_inc, 
                             nixlGpuXferStatusH* xfer_status = nullptr, 
                             bool is_no_delay = true)
{
    return NIXL_ERR_NOT_SUPPORTED;
}

/**
 * @brief Post a memory transfer request to the GPU.
 * 
 * @param req_hndl           [in]  The request handle.
 * @param sizes              [in]  The sizes of the blocks to send.
 * @param addreses           [in]  The addresses of the blocks to send.
 * @param remote_addrs       [in]  The remote addresses of the blocks to send.
 * @param signal_remote_addr [in]  The remote address of the signal. If address is 0, no signal.
 * @param signal_inc         [in]  The increment of the signal.
 * @param xfer_status        [out] The status of the transfer. If null the status is not reported.
 * @param is_no_delay        [in]  Whether to use no-delay mode.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_coordination_level_t level = NIXL_GPU_COORDINATION_BLOCK> 
__device__ static inline nixl_status_t 
nixlPostGpuXferReq(nixlGpuXferReqH* req_hndl, 
                   const size_t* sizes, 
                   const void** addreses, 
                   const uint64_t* remote_addrs, 
                   uint64_t signal_remote_addr,
                   uint64_t signal_inc, 
                   nixlGpuXferStatusH* xfer_status = nullptr, 
                   bool is_no_delay = true) 
{
    return NIXL_ERR_NOT_SUPPORTED;
}

/**
 * @brief Get the status of a transfer request.
 * 
 * @param xfer_status [in]  The status of the transfer.
 *
 * @return nixl_status_t    Error code if call was not successful
 */
template<nixl_gpu_coordination_level_t level = NIXL_GPU_COORDINATION_BLOCK> 
__device__ static inline nixl_status_t 
nixlGpuGetXferStatus(nixlGpuXferStatusH* xfer_status)
{
    return NIXL_ERR_NOT_SUPPORTED;
}
