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

#ifndef _DEVICE_UTILS_CUH
#define _DEVICE_UTILS_CUH

#include <cuda_runtime.h>
#include "nixl.h"
#include <nixl_device.cuh>

#define MAX_THREADS 1024
#define NSEC_PER_SEC 1000000000ul
#define NSEC_PER_USEC 1000ul
#define NS_TO_USEC(ns) ((ns) * 1.0 / (NSEC_PER_USEC))

__device__ inline unsigned long long
getTimeNs() {
    unsigned long long globaltimer;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

template<nixl_gpu_level_t level>
__device__ constexpr size_t
getReqIdx() {
    switch (level) {
    case nixl_gpu_level_t::THREAD:
        return threadIdx.x;
    case nixl_gpu_level_t::WARP:
        return threadIdx.x / warpSize;
    case nixl_gpu_level_t::BLOCK:
        return 0;
    default:
        return 0;
    }
}

__device__ inline void
printProgressError(int thread_id, size_t iteration, nixl_status_t status) {
    printf("Thread %d: Failed to progress transfer iteration %d: status=%d\n",
           thread_id,
           (int)iteration, // cast to int because printf issue size_t
           status);
}

__device__ inline void
printCompletionError(int thread_id, size_t iteration, nixl_status_t status) {
    printf("Thread %d: Transfer completion failed iteration %d: status=%d\n",
           thread_id,
           (int)iteration, // cast to int because printf issue size_t
           status);
}

static inline nixl_status_t
handleCudaErrors() {
    nixl_status_t ret = NIXL_SUCCESS;

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    return ret;
}

#endif // _DEVICE_UTILS_CUH
