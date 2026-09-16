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
#ifndef COALESCE_KERNELS_CUH
#define COALESCE_KERNELS_CUH

#include <cuda_runtime.h>

namespace nixlMarshal {

cudaError_t
cudaCoalesceKernel(void *dst,
                   const void *const *ptrs,
                   size_t *sizes,
                   size_t *offsets,
                   size_t n_chunks,
                   cudaStream_t stream);

/**
 * @brief  Compute per-chunk device pointers from a device offsets array and a base pointer.
 *
 * outPtrs[i] = base + offsets[i]. Used on the decompress path to reconstruct the
 * compressed-chunk pointer array from the offsets stored at the head of the packed buffer.
 *
 * @param  outPtrs  Device array of nChunks pointers to fill.
 * @param  offsets  Device array of nChunks byte offsets (relative to base).
 * @param  base     Base device pointer that offsets are relative to.
 * @param  nChunks  Number of chunks.
 * @param  stream   CUDA stream on which to enqueue the kernel.
 *
 * @return cudaSuccess on success, otherwise the error from cudaGetLastError().
 */
cudaError_t
cudaChunkPtrsFromOffsets(void **out_ptrs,
                         const size_t *offsets,
                         void *base,
                         size_t n_chunks,
                         cudaStream_t stream);

/**
 * @brief  Stream-ordered exclusive prefix sum (scan) of n size_t values on the device.
 *
 * Thrust device algorithms must be compiled by nvcc, so this wrapper lives in the
 * .cu translation unit and is called from host (.cpp) code.
 *
 * @param  in      Device pointer to n input values.
 * @param  out     Device pointer to n output values (may alias in). out[0] = 0 and
 *                 out[i] = sum(in[0..i-1]).
 * @param  n       Number of elements.
 * @param  stream  CUDA stream on which to enqueue the scan.
 *
 * @return cudaSuccess on success, otherwise the error from cudaGetLastError().
 */
cudaError_t
cudaExclusivePrefixSum(const size_t *in, size_t *out, size_t n, cudaStream_t stream);

} // namespace nixlMarshal

#endif // COALESCE_KERNELS_CUH
