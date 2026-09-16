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

#include "coalesce_kernels.cuh"

#include <cstdint>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace nixlMarshal {

constexpr int kThreadsPerBlock = 256;
constexpr size_t kAlignment = 8;

__global__ void
coalesceKernel(void *dst,
               const void *const *ptrs,
               size_t *sizes,
               size_t *offsets,
               size_t n_chunks) {
    for (size_t chunk_idx = blockIdx.x; chunk_idx < n_chunks; chunk_idx += gridDim.x) {
        size_t n_elems = (sizes[chunk_idx] + sizeof(size_t) - 1) / sizeof(size_t);
        size_t *dst_ptr = reinterpret_cast<size_t *>(static_cast<char *>(dst) + offsets[chunk_idx]);
        const size_t *src_ptr = reinterpret_cast<const size_t *>(ptrs[chunk_idx]);
        for (size_t i = threadIdx.x; i < n_elems; i += blockDim.x) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

cudaError_t
cudaCoalesceKernel(void *dst,
                   const void *const *ptrs,
                   size_t *sizes,
                   size_t *offsets,
                   size_t n_chunks,
                   cudaStream_t stream) {
    coalesceKernel<<<n_chunks, kThreadsPerBlock, 0, stream>>>(dst, ptrs, sizes, offsets, n_chunks);
    return cudaGetLastError();
}

cudaError_t
cudaChunkPtrsFromOffsets(void **out_ptrs,
                         const size_t *offsets,
                         void *base,
                         size_t n_chunks,
                         cudaStream_t stream) {
    using thrust::placeholders::_1;
    thrust::device_ptr<const size_t> tOffsets(offsets);
    thrust::device_ptr<uintptr_t> tOut(reinterpret_cast<uintptr_t *>(out_ptrs));
    const auto base_addr = reinterpret_cast<uintptr_t>(base);
    thrust::transform(
        thrust::cuda::par_nosync.on(stream), tOffsets, tOffsets + n_chunks, tOut, _1 + base_addr);
    return cudaGetLastError();
}

template<size_t Alignment> struct alignUp {
    static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0,
                  "Alignment must be a non-zero power of two");

    __host__ __device__ size_t
    operator()(size_t x) const {
        return (x + (Alignment - 1)) & ~(Alignment - 1);
    }
};

cudaError_t
cudaExclusivePrefixSum(const size_t *in, size_t *out, size_t n, cudaStream_t stream) {
    thrust::device_ptr<const size_t> tIn(in);
    thrust::device_ptr<size_t> tOut(out);
    auto t_in_aligned = thrust::make_transform_iterator(tIn, alignUp<kAlignment>{});
    thrust::exclusive_scan(
        thrust::cuda::par_nosync.on(stream), t_in_aligned, t_in_aligned + n, tOut);
    return cudaGetLastError();
}

} // namespace nixlMarshal
