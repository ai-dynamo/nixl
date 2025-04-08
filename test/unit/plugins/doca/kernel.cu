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

#include "common.h"

#define VOLATILE(x) (*(volatile typeof(x) *)&(x))

__global__ void warmup_kernel(uintptr_t addr, size_t size)
{
    printf(">>>>>>> CUDA warmup! addr %p size %d\n", (void*)addr, (uint32_t)size);
}

__global__ void target_kernel(uintptr_t addr, size_t size)
{
    printf(">>>>>>> CUDA target waiting on addr %p size %d\n", (void*)addr, (uint32_t)size);
    while(VOLATILE(((uint8_t*)addr)[0]) == 0);
    printf(">>>>>>> CUDA target now addr %p is %d\n", (void*)addr, VOLATILE(((uint8_t*)addr)[0]));
}

__global__ void initiator_kernel(uintptr_t addr, size_t size)
{
    printf(">>>>>>> CUDA initiator send on addr %p size %d\n", (void*)addr, (uint32_t)size);
    for (int i = 0; i < (int)size; i++)
        ((uint8_t*)addr)[i] = 0x1;
}

extern "C" {

int launch_warmup_kernel(cudaStream_t stream, uintptr_t addr, size_t size)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    warmup_kernel<<<1, 1, 0, stream>>>(addr, size);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

int launch_target_wait_kernel(cudaStream_t stream, uintptr_t addr, size_t size)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    target_kernel<<<1, 1, 0, stream>>>(addr, size);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

int launch_initiator_send_kernel(cudaStream_t stream, uintptr_t addr, size_t size)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    initiator_kernel<<<1, 1, 0, stream>>>(addr, size);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

} /* extern C */
