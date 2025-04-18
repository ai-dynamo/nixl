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

#ifndef __BACKEND_PLUGIN_DOCA_DEVICE_H
#define __BACKEND_PLUGIN_DOCA_DEVICE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_rdma.cuh>
#include "backend_plugin_doca_common.h"

__device__ inline void prepXferReqGpu_DocaBlock(uintptr_t backendHandleGpu)
{
    doca_error_t result;
    struct doca_gpu_buf *lbuf;
    struct doca_gpu_buf *rbuf;
    const int connection_index = 0;
    uint32_t num_ops=0;
    struct docaXferReqGpu *treq = (struct docaXferReqGpu *)backendHandleGpu;

    if (threadIdx.x >= treq->num)
        return;

    doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)treq->larr[threadIdx.x], 0, &lbuf);
    doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)treq->rarr[threadIdx.x], 0, &rbuf);

    // printf(">>>>>>> CUDA rdma write kernel thread %d pos %d size %d\n", threadIdx.x, pos, (int)treq->size[threadIdx.x]);

    //Each thread should send a different buffer
    result = doca_gpu_dev_rdma_write_strong(treq->rdma_gpu, connection_index, rbuf, 0, lbuf, 0, treq->size[threadIdx.x], 0, DOCA_GPU_RDMA_WRITE_FLAG_NONE);
    if (result != DOCA_SUCCESS)
        printf("Error %d doca_gpu_dev_rdma_write_strong\n", result);

    __syncthreads();

    if (threadIdx.x == 0) {
        result = doca_gpu_dev_rdma_commit_strong(treq->rdma_gpu, connection_index);
        if (result != DOCA_SUCCESS)
            printf("Error %d doca_gpu_dev_rdma_push\n", result);

        result = doca_gpu_dev_rdma_wait_all(treq->rdma_gpu, &num_ops);
        if (result != DOCA_SUCCESS)
            printf("Error %d doca_gpu_dev_rdma_wait_all\n", result);

        // printf(">>>>>>> CUDA rdma write kernel pos %d num %d completed %d ops\n", pos, treq->num, num_ops);

        treq->num = 0;
    }
}

#endif