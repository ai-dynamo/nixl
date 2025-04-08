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

__global__ void kernel_read(struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
    //struct doca_gpu_buf_arr *lbarr, struct doca_gpu_buf_arr *rbarr, size_t size
    doca_error_t result;
	struct doca_gpu_buf *lbuf;
	struct doca_gpu_buf *rbuf;
    const int connection_index = 0;
    uint32_t num_ops=0;

    //Warmup
    if (xferReqRing == nullptr)
        return;

    if (threadIdx.x >= xferReqRing[pos].num)
        return;

    doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].larr[threadIdx.x], 0, &lbuf);
	doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].rarr[threadIdx.x], 0, &rbuf);

    // printf(">>>>>>> CUDA rdma read kernel thread %d pos %d size %d\n", threadIdx.x, pos, (int)xferReqRing[pos].size[threadIdx.x]);

    //Each thread should send a different buffer
    result = doca_gpu_dev_rdma_read_strong(rdma_gpu, connection_index, rbuf, 0, lbuf, 0, xferReqRing[pos].size[threadIdx.x], 0);
    if (result != DOCA_SUCCESS)
        printf("Error %d doca_gpu_dev_rdma_read_strong\n", result);

    __syncthreads();

    if (threadIdx.x == 0) {
        result = doca_gpu_dev_rdma_commit_strong(rdma_gpu, connection_index);
        if (result != DOCA_SUCCESS)
            printf("Error %d doca_gpu_dev_rdma_push\n", result);

        result = doca_gpu_dev_rdma_wait_all(rdma_gpu, &num_ops);
        if (result != DOCA_SUCCESS)
            printf("Error %d doca_gpu_dev_rdma_wait_all\n", result);

        // printf(">>>>>>> CUDA rdma read kernel pos %d num %d completed %d ops\n", pos, xferReqRing[pos].num, num_ops);

        xferReqRing[pos].num = 0;
        __threadfence_system();
    }

}

extern "C" {

doca_error_t doca_kernel_read(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
    cudaError_t result = cudaSuccess;

    if (rdma_gpu == NULL) {
        fprintf(stderr, "kernel_read_server invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_read<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(rdma_gpu, xferReqRing, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}


} /* extern C */
    