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
#ifndef DELTA_KERNEL_CUH
#define DELTA_KERNEL_CUH

#include <cuda_runtime.h>

namespace nixlMarshal {

/**
 * @brief  Element-wise XOR of two device buffers into a third, on a CUDA stream.
 *
 * @tparam T        Unsigned integer element type (uint8_t / uint16_t / uint32_t /
 *                  uint64_t). Only these instantiations are provided.
 * @param  dst      Device pointer to the destination buffer.
 * @param  src      Device pointer to the source buffer.
 * @param  ref      Device pointer to the reference buffer.
 * @param  n_bytes  Size of each buffer in bytes. Must be a multiple of sizeof(T).
 * @param  stream   CUDA stream on which to enqueue the kernel.
 *
 * @return cudaSuccess on a successful launch (the kernel is asynchronous).
 *         cudaErrorInvalidValue if n_bytes is not a multiple of sizeof(T).
 *         Any error returned by cudaGetLastError() after the launch otherwise.
 */
template<typename T>
cudaError_t
cudaXorKernel(void *dst, const void *src, const void *ref, size_t n_bytes, cudaStream_t stream);

} // namespace nixlMarshal

#endif // DELTA_KERNEL_CUH
