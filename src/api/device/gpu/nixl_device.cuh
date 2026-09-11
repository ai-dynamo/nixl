/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/**
 * @file nixl_device.cuh
 * @brief GPU Device API for in-kernel memory transfers
 *
 * Include this header from CUDA device code. The namespaced nixl::gpu API is
 * the canonical interface; the nixlPut / nixlGpuGetXferStatus wrappers keep
 * the original C-style names.
 */

#ifndef NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
#define NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH

#include <gpu/impl/device_dispatch.cuh>

namespace nixl::gpu {

/**
 * @brief Get the status of the transfer request.
 *
 * @param xfer_status [in]  Status of the transfer.
 *
 * @return NIXL_SUCCESS     The request has completed, no more operations are
 *                          in progress.
 * @return NIXL_IN_PROG     One or more operations in the request have not
 *                          completed.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<level_t level = level_t::THREAD>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    return impl::getXferStatus<level>(xfer_status);
}

/**
 * @brief Post a single-region memory transfer from local to remote GPU.
 *
 * This function creates and posts a transfer request using memory view
 * elements @a src and @a dst.
 *
 * @param src         [in]  Source memory view element
 * @param dst         [in]  Destination memory view element
 * @param size        [in]  Size in bytes to transfer
 * @param channel_id  [in]  Channel ID to use for the transfer
 * @param flags       [in]  Transfer flags
 * @param xfer_status [in,out] Optional status handle
 *                            (use @ref nixl::gpu::getXferStatus)
 *
 * @return NIXL_IN_PROG     Transfer posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<level_t level = level_t::THREAD>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    xferStatusH *xfer_status = nullptr) {
    return impl::put<level>(src, dst, size, channel_id, flags, xfer_status);
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
 * @param xfer_status [in,out] Optional status handle
 *                            (use @ref nixl::gpu::getXferStatus)
 *
 * @return NIXL_IN_PROG     Atomic add posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<level_t level = level_t::THREAD>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id = 0,
          uint64_t flags = 0,
          xferStatusH *xfer_status = nullptr) {
    return impl::atomicAdd<level>(value, counter, channel_id, flags, xfer_status);
}

/**
 * @brief Get a local pointer to remote memory.
 *
 * This function returns a local pointer to the mapped memory of the
 * remote memory view handle at the given index.
 * The memory view must be prepared on the host using
 * @ref nixlAgent::prepMemView.
 *
 * @param mvh    [in]  Memory view handle (remote buffers)
 * @param index  [in]  Index in the memory view
 *
 * @return Pointer to the mapped memory, or nullptr if not available.
 */
__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    return impl::getPtr(mvh, index);
}

} // namespace nixl::gpu

/**
 * @brief Get the status of the transfer request.
 *
 * @param xfer_status [in]  Status of the transfer.
 *
 * @return NIXL_SUCCESS     The request has completed, no more operations are
 *                          in progress.
 * @return NIXL_IN_PROG     One or more operations in the request have not
 *                          completed.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status) {
    return nixl::gpu::getXferStatus<level>(xfer_status);
}

/**
 * @brief Post a single-region memory transfer from local to remote GPU.
 *
 * This function creates and posts a transfer request using memory view
 * elements @a src and @a dst.
 *
 * @param src         [in]  Source memory view element
 * @param dst         [in]  Destination memory view element
 * @param size        [in]  Size in bytes to transfer
 * @param channel_id  [in]  Channel ID to use for the transfer
 * @param flags       [in]  Transfer flags
 * @param xfer_status [in,out] Optional status handle
 *                            (use @ref nixlGpuGetXferStatus)
 *
 * @return NIXL_IN_PROG     Transfer posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::put<level>(src, dst, size, channel_id, flags, xfer_status);
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
 * @param xfer_status [in,out] Optional status handle
 *                            (use @ref nixlGpuGetXferStatus)
 *
 * @return NIXL_IN_PROG     Atomic add posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::atomicAdd<level>(value, counter, channel_id, flags, xfer_status);
}

/**
 * @brief Get a local pointer to remote memory.
 *
 * This function returns a local pointer to the mapped memory of the
 * remote memory view handle at the given index.
 * The memory view must be prepared on the host using
 * @ref nixlAgent::prepMemView.
 *
 * @param mvh    [in]  Memory view handle (remote buffers)
 * @param index  [in]  Index in the memory view
 *
 * @return Pointer to the mapped memory, or nullptr if not available.
 */
__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index) {
    return nixl::gpu::getPtr(mvh, index);
}

#endif // NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
