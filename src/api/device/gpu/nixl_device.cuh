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
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param xfer_status [in]  Status of the transfer.
 *
 * @return NIXL_SUCCESS     The request has completed, no more operations are
 *                          in progress.
 * @return NIXL_IN_PROG     One or more operations in the request have not
 *                          completed.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<level_t level = level_t::THREAD, exec_mode_t mode = exec_mode_t::DYNAMIC>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    return impl::getXferStatus<level, mode>(xfer_status);
}

/**
 * @brief Post a single-region memory transfer from local to remote GPU.
 *
 * This function creates and posts a transfer request using memory view
 * elements @a src and @a dst.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
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
template<level_t level = level_t::THREAD, exec_mode_t mode = exec_mode_t::DYNAMIC>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    xferStatusH *xfer_status = nullptr) {
    return impl::put<level, mode>(src, dst, size, channel_id, flags, xfer_status);
}

/**
 * @brief Post a single-region memory transfer with a 32-bit immediate value.
 *
 * This function creates and posts a transfer request using memory view
 * elements @a src and @a dst, both using UCX_DIRECT.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param src         [in]  Source memory view element
 * @param dst         [in]  Destination memory view element
 * @param size        [in]  Size in bytes to transfer
 * @param immediate   [in]  Value delivered in the receiver's CQE
 * @param channel_id  [in]  Channel ID to use for the transfer
 * @param flags       [in]  Transfer flags
 * @param xfer_status [in,out] Optional sender-completion status handle (use @ref nixl::gpu::getXferStatus)
 *
 * @return NIXL_IN_PROG     Transfer posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<level_t level = level_t::THREAD, exec_mode_t mode = exec_mode_t::DYNAMIC>
__device__ nixl_status_t
putWithIMM(const memViewElem &src,
           const memViewElem &dst,
           size_t size,
           uint32_t immediate,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           xferStatusH *xfer_status = nullptr) {
    return impl::putWithIMM<level, mode>(src, dst, size, immediate, channel_id, flags, xfer_status);
}

/**
 * @brief Poll a receive CQ for a completion with an immediate value.
 *
 * This function polls the local RX completion queue using a caller-owned index.
 * THREAD only. completion queue indices and receive credits are unchanged.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param peer           [in]     Remote peer's memory view element
 * @param channel_id     [in]     Channel ID to use for the poll
 * @param consumer_index [in,out] Caller-owned completion cursor. incremented on success
 * @param immediate      [out]    Received 32-bit value, valid on success
 *
 * @return NIXL_SUCCESS     Completion received; consumer index advanced.
 * @return NIXL_IN_PROG     No completion available; consumer index unchanged.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<level_t level = level_t::THREAD, exec_mode_t mode = exec_mode_t::DYNAMIC>
__device__ nixl_status_t
pollIMM(const memViewElem &peer,
        unsigned channel_id,
        uint64_t *consumer_index,
        uint32_t *immediate) {
    return impl::pollIMM<level, mode>(peer, channel_id, consumer_index, immediate);
}

/**
 * @brief Advance the HW completion queue consumer index and return RX credits.
 *
 * THREAD only. One caller advances the queue after all readers finish
 * using the retired entries and their payloads.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param peer           [in]  Remote peer's memory view element; offset is ignored
 * @param channel_id     [in]  Channel ID to advance
 * @param consumed       [in]  Consumed completions since the last advanceRx
 * @param consumer_index [out] Updated HW completion queue consumer index
 *
 * @return NIXL_SUCCESS     HW completion queue consumer index advanced and credits returned.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<level_t level = level_t::THREAD, exec_mode_t mode = exec_mode_t::DYNAMIC>
__device__ nixl_status_t
advanceRx(const memViewElem &peer,
          unsigned channel_id,
          uint64_t consumed,
          uint64_t *consumer_index) {
    return impl::advanceRx<level, mode>(peer, channel_id, consumed, consumer_index);
}

/**
 * @brief Atomic add to remote GPU memory.
 *
 * This function performs an atomic increment on a remote counter.
 * The increment is visible only after previous writes complete.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
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
template<level_t level = level_t::THREAD, exec_mode_t mode = exec_mode_t::DYNAMIC>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id = 0,
          uint64_t flags = 0,
          xferStatusH *xfer_status = nullptr) {
    return impl::atomicAdd<level, mode>(value, counter, channel_id, flags, xfer_status);
}

/**
 * @brief Get a local pointer to remote memory.
 *
 * This function returns a local pointer to the mapped memory of the
 * remote memory view handle at the given index.
 * The memory view must be prepared on the host using
 * @ref nixlAgent::prepMemView.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param mvh    [in]  Memory view handle (remote buffers)
 * @param index  [in]  Index in the memory view
 *
 * @return Pointer to the mapped memory, or nullptr if not available.
 */
template<exec_mode_t mode = exec_mode_t::DYNAMIC>
__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    return impl::getPtr<mode>(mvh, index);
}

} // namespace nixl::gpu

/**
 * @brief Get the status of the transfer request.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param xfer_status [in]  Status of the transfer.
 *
 * @return NIXL_SUCCESS     The request has completed, no more operations are
 *                          in progress.
 * @return NIXL_IN_PROG     One or more operations in the request have not
 *                          completed.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status) {
    return nixl::gpu::getXferStatus<level, mode>(xfer_status);
}

/**
 * @brief Post a single-region memory transfer from local to remote GPU.
 *
 * This function creates and posts a transfer request using memory view
 * elements @a src and @a dst.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
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
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::put<level, mode>(src, dst, size, channel_id, flags, xfer_status);
}

/**
 * @brief Post a single-region memory transfer with a 32-bit immediate value.
 *
 * This function creates and posts a transfer request using memory view
 * elements @a src and @a dst, both using UCX_DIRECT.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param src         [in]  Source memory view element
 * @param dst         [in]  Destination memory view element
 * @param size        [in]  Size in bytes to transfer
 * @param immediate   [in]  Value delivered in the receiver's CQE
 * @param channel_id  [in]  Channel ID to use for the transfer
 * @param flags       [in]  Transfer flags
 * @param xfer_status [in,out] Optional sender-completion status handle (use @ref nixlGpuGetXferStatus)
 *
 * @return NIXL_IN_PROG     Transfer posted successfully.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlPutWithIMM(const nixlMemViewElem &src,
               const nixlMemViewElem &dst,
               size_t size,
               uint32_t immediate,
               unsigned channel_id = 0,
               uint64_t flags = 0,
               nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::putWithIMM<level, mode>(src, dst, size, immediate, channel_id, flags, xfer_status);
}

/**
 * @brief Poll a receive CQ for a completion with an immediate value.
 *
 * This function polls the local RX completion queue using a caller-owned index.
 * THREAD only. completion queue indices and receive credits are unchanged.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param peer           [in]     Remote peer's memory view element
 * @param channel_id     [in]     Channel ID to use for the poll
 * @param consumer_index [in,out] Caller-owned completion cursor. incremented on success
 * @param immediate      [out]    Received 32-bit value, valid on success
 *
 * @return NIXL_SUCCESS     Completion received; consumer index advanced.
 * @return NIXL_IN_PROG     No completion available; consumer index unchanged.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlPollIMM(const nixlMemViewElem &peer,
            unsigned channel_id,
            uint64_t *consumer_index,
            uint32_t *immediate) {
    return nixl::gpu::pollIMM<level, mode>(peer, channel_id, consumer_index, immediate);
}

/**
 * @brief Advance the HW completion queue consumer index and return RX credits.
 *
 * THREAD only. One caller advances the queue after all readers finish
 * using the retired entries and their payloads.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param peer           [in]  Remote peer's memory view element; offset is ignored
 * @param channel_id     [in]  Channel ID to advance
 * @param consumed       [in]  Consumed completions since the last nixlAdvanceRx
 * @param consumer_index [out] Updated HW completion queue consumer index
 *
 * @return NIXL_SUCCESS     HW completion queue consumer index advanced and credits returned.
 * @return NIXL_ERR_BACKEND An error occurred in the backend.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlAdvanceRx(const nixlMemViewElem &peer,
              unsigned channel_id,
              uint64_t consumed,
              uint64_t *consumer_index) {
    return nixl::gpu::advanceRx<level, mode>(peer, channel_id, consumed, consumer_index);
}

/**
 * @brief Atomic add to remote GPU memory.
 *
 * This function performs an atomic increment on a remote counter.
 * The increment is visible only after previous writes complete.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
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
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::atomicAdd<level, mode>(value, counter, channel_id, flags, xfer_status);
}

/**
 * @brief Get a local pointer to remote memory.
 *
 * This function returns a local pointer to the mapped memory of the
 * remote memory view handle at the given index.
 * The memory view must be prepared on the host using
 * @ref nixlAgent::prepMemView.
 *
 * @tparam mode     Execution mode; DYNAMIC (default) reads it from the memory view at run
 *                  time, a fixed mode must match the view.
 *
 * @param mvh    [in]  Memory view handle (remote buffers)
 * @param index  [in]  Index in the memory view
 *
 * @return Pointer to the mapped memory, or nullptr if not available.
 */
template<nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index) {
    return nixl::gpu::getPtr<mode>(mvh, index);
}

#endif // NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
