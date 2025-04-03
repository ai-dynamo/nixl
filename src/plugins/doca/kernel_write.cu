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

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_rdma.cuh>

#include "doca_backend.h"

__global__ void kernel_write()
{
    printf("Hey! I'm the rdma write kernel\n");
}

extern "C" {

doca_error_t doca_kernel_write(cudaStream_t stream)
{
    cudaError_t result = cudaSuccess;

    // if (rdma_gpu == NULL || server_local_buf_arr_A == NULL || server_remote_buf_arr_F == NULL) {
    //     DOCA_LOG_ERR("kernel_write_server invalid input values");
    //     return DOCA_ERROR_INVALID_VALUE;
    // }

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_write<<<1, 1, 0, stream>>>();
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}


} /* extern C */
    