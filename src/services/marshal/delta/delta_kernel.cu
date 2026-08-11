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

#include "delta_kernel.cuh"

#include <cstdint>

namespace nixlMarshal {

constexpr int threads_per_block = 256;

template<typename T>
__global__ void
xorKernel(T *dst, const T *src, const T *ref, size_t n_bytes) {
    const size_t n_elems = n_bytes / sizeof(T);
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_elems;
         i += stride) {
        dst[i] = src[i] ^ ref[i];
    }
}

template<typename T>
cudaError_t
cudaXorKernel(void *dst, const void *src, const void *ref, size_t n_bytes, cudaStream_t stream) {
    if (n_bytes % sizeof(T) != 0) {
        return cudaErrorInvalidValue;
    }
    const auto n_elems = n_bytes / sizeof(T);
    const auto raw_blocks = (n_elems + threads_per_block - 1) / threads_per_block;
    const auto blocks = static_cast<int>(std::min<size_t>(raw_blocks, INT_MAX));
    // reinterpret_cast is safe because the buffers are guaranteed to be of type T*
    xorKernel<T><<<blocks, threads_per_block, 0, stream>>>(reinterpret_cast<T *>(dst),
                                                           reinterpret_cast<const T *>(src),
                                                           reinterpret_cast<const T *>(ref),
                                                           n_bytes);
    return cudaGetLastError();
}

// Explicit template instantiations for each supported element type
template cudaError_t
cudaXorKernel<uint8_t>(void *, const void *, const void *, size_t, cudaStream_t);
template cudaError_t
cudaXorKernel<uint16_t>(void *, const void *, const void *, size_t, cudaStream_t);
template cudaError_t
cudaXorKernel<uint32_t>(void *, const void *, const void *, size_t, cudaStream_t);
template cudaError_t
cudaXorKernel<uint64_t>(void *, const void *, const void *, size_t, cudaStream_t);

} // namespace nixlMarshal
