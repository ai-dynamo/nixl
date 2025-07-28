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

#include <doca_gpunetio_dev_verbs_onesided.cuh>
#include <cuda.h>
#include <cuda/atomic>

#include "gpunetio_backend.h"

#define ENABLE_DEBUG 0

__global__ void
kernel_read (struct doca_gpu_dev_verbs_qp *qp, struct docaXferReqGpu *xferReqRing, uint32_t pos) {
	uint64_t wqe_idx = 0;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
    __shared__ uint32_t base_wqe_idx;
    uint32_t tot_wqe = xferReqRing[pos].num;
    uint32_t idx = 0;

    if (threadIdx.x == 0) {
        if (qp->need_dump == true)
            base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, tot_wqe + 1);
        else
            base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, tot_wqe);
    }
    __syncthreads();

	for (idx = threadIdx.x; idx < tot_wqe; idx += blockDim.x) {
        if (idx == (tot_wqe - 1))
		    cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
        wqe_idx = base_wqe_idx + idx;
		wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        doca_gpu_dev_verbs_wqe_prepare_read(qp,
							  wqe_ptr,
							  wqe_idx,
							  cflag,
							  (uint64_t)(xferReqRing[pos].rbuf[idx]),
                              xferReqRing[pos].rkey[idx],
							  (uint64_t)(xferReqRing[pos].lbuf[idx]),
                              xferReqRing[pos].lkey[idx],
							  xferReqRing[pos].size[idx]);
	}
    __syncthreads();

    if (idx == (tot_wqe - 1)) {
        if (qp->need_dump == true) {
            wqe_idx++;
	    	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    		doca_gpu_dev_verbs_wqe_prepare_dump(qp,
						    wqe_ptr,
						    wqe_idx,
						    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                            (uint64_t)(xferReqRing[pos].lbuf[tot_wqe - 1]),
                            xferReqRing[pos].lkey[tot_wqe - 1],
						    1);
        }
        doca_gpu_dev_verbs_mark_wqes_ready(qp, base_wqe_idx, wqe_idx);
        doca_gpu_dev_verbs_submit(qp, wqe_idx + 1);
        // Wait for final CQE in block of iterations
        if (doca_gpu_dev_verbs_poll_cq_at(doca_gpu_dev_verbs_qp_get_cq_sq(qp), wqe_idx) != 0)
            printf("Error CQE!\n");

        DOCA_GPUNETIO_VOLATILE (xferReqRing[pos].in_use) = 1;
    }
#if ENABLE_DEBUG == 1
    printf (">>>>>>> CUDA rdma read kernel pos %d posted %d buffers\n",
            pos,
            xferReqRing[pos].num);
#endif
}

__global__ void
kernel_write (struct doca_gpu_dev_verbs_qp *qp,
              struct docaXferReqGpu *xferReqRing,
              uint32_t pos) {
	uint64_t wqe_idx = 0;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
    __shared__ uint32_t base_wqe_idx;
    uint32_t tot_wqe = xferReqRing[pos].num;
    uint32_t idx = 0;

    if (threadIdx.x == 0)
        base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, tot_wqe);
    __syncthreads();

	for (idx = threadIdx.x; idx < tot_wqe; idx += blockDim.x) {
        if (idx == (tot_wqe - 1))
		    cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
        wqe_idx = base_wqe_idx + idx;
		wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        doca_gpu_dev_verbs_wqe_prepare_write(qp,
							  wqe_ptr,
							  wqe_idx,
							  MLX5_OPCODE_RDMA_WRITE,
							  cflag,
							  0,
                              (uint64_t)(xferReqRing[pos].rbuf[idx]),
                              xferReqRing[pos].rkey[idx],
							  (uint64_t)(xferReqRing[pos].lbuf[idx]),
                              xferReqRing[pos].lkey[idx],
							  xferReqRing[pos].size[idx]);
	}
    __syncthreads();

    if (idx == (tot_wqe - 1)) {
        doca_gpu_dev_verbs_mark_wqes_ready(qp, base_wqe_idx, wqe_idx);
        doca_gpu_dev_verbs_submit(qp, wqe_idx + 1);

        DOCA_GPUNETIO_VOLATILE (xferReqRing[pos].last_wqe) = wqe_idx;
        DOCA_GPUNETIO_VOLATILE (xferReqRing[pos].in_use) = 1;
    }
#if ENABLE_DEBUG == 1
    printf (">>>>>>> CUDA rdma write kernel pos %d posted %d buffers\n",
            pos,
            xferReqRing[pos].num);
#endif
}

__global__ void
kernel_progress (struct docaXferCompletion *completion_list,
                 struct docaNotif *notif_fill,
                 struct docaNotif *notif_progress,
                 struct docaNotif *notif_send_gpu,
                 uint32_t *exit_flag) {
    uint32_t num_msg = 0, num_msg_notif = 0;
    uint32_t index = 0;
    struct doca_gpu_buf *recv_buf;
    struct doca_gpu_buf *send_buf;
    struct doca_gpu_dev_rdma_r *rdma_gpu_r;
    doca_gpu_dev_verbs_ticket_t out_ticket;

    // Warmup
    if (completion_list == nullptr) return;

    // Wait Xfer & notify
    if (blockIdx.x == 0) {
        while (DOCA_GPUNETIO_VOLATILE (*exit_flag) == 0) {
            // Check xfer completion and send notif
            if (DOCA_GPUNETIO_VOLATILE (completion_list[index].xferReqRingGpu) != nullptr) {
                if (DOCA_GPUNETIO_VOLATILE (completion_list[index].completed) == 0 &&
                    DOCA_GPUNETIO_VOLATILE (completion_list[index].xferReqRingGpu->in_use) == 1) {
                    // Wait for final CQE in block of iterations
                    if (doca_gpu_dev_verbs_poll_cq_at(doca_gpu_dev_verbs_qp_get_cq_sq(completion_list[index].xferReqRingGpu->qp_data_gpu), DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].last_wqe)) != 0) {
                        DOCA_GPUNETIO_VOLATILE (*exit_flag) = 1;
                        printf("Error CQE!\n");
                    }

                    // result = doca_gpu_dev_rdma_wait_all (
                    //         completion_list[index].xferReqRingGpu->qp_data_gpu, &num_msg);
                    // if (result != DOCA_SUCCESS) {
                    //     printf ("Error %d doca_gpu_dev_rdma_wait_all xfer\n", result);
                    //     DOCA_GPUNETIO_VOLATILE (*exit_flag) = 1;
                    // }

                    while (num_msg > 0) {
#if ENABLE_DEBUG == 1
                        printf ("Completion %d num recv %d\n", index, num_msg);
#endif

                        if (DOCA_GPUNETIO_VOLATILE (
                                    completion_list[index].xferReqRingGpu->has_notif_msg_idx) !=
                            DOCA_NOTIF_NULL) {
#if ENABLE_DEBUG == 1
                            printf ("Notif after completion at %d id %d sz %d\n",
                                    index,
                                    DOCA_GPUNETIO_VOLATILE (
                                            completion_list[index]
                                                    .xferReqRingGpu->has_notif_msg_idx),
                                    (int)completion_list[index].xferReqRingGpu->msg_sz);
#endif

                            doca_gpu_dev_verbs_send(
                                completion_list[index].xferReqRingGpu->qp_gpu_notif,
                                doca_gpu_dev_verbs_addr{.addr = (uint64_t)(completion_list[index].xferReqRingGpu->lbuf_notif +
                                                            (completion_list[index].xferReqRingGpu->has_notif_msg_idx * completion_list[index].xferReqRingGpu->lstride_notif)),
                                                        .key = completion_list[index].xferReqRingGpu->lkey_notif},
                                completion_list[index].xferReqRingGpu->msg_sz,
                                &out_ticket);

                            doca_gpu_dev_verbs_wait(completion_list[index].xferReqRingGpu->qp_gpu_notif, out_ticket);
#if ENABLE_DEBUG == 1
                            printf ("Notif correctly sent %d\n", num_msg_notif);
#endif
                        }
                        DOCA_GPUNETIO_VOLATILE (completion_list[index].completed) = 1;
                        num_msg--;
                        index = (index + 1) & DOCA_MAX_COMPLETION_INFLIGHT_MASK;
                    }
                }
            }
        }
    }

    // Receive notif: fill recv in new queue and progress queue
    if (blockIdx.x == 1) {
        while (DOCA_GPUNETIO_VOLATILE (*exit_flag) == 0) {
            // Check received notifications
            if (DOCA_GPUNETIO_VOLATILE (notif_progress->qp) != nullptr) {
                doca_gpu_dev_rdma_get_recv (notif_progress->qp, &rdma_gpu_r);
                result = doca_gpu_dev_rdma_recv_wait_all (
                        rdma_gpu_r, DOCA_GPU_RDMA_RECV_WAIT_FLAG_NB, &num_msg, nullptr, nullptr);
                if (result != DOCA_SUCCESS) {
                    printf ("Error %d doca_gpu_dev_rdma_recv_wait_all\n", result);
                    DOCA_GPUNETIO_VOLATILE (*exit_flag) = 1;
                }

#if ENABLE_DEBUG == 1
                if (num_msg > 0) {
                    printf ("kernel received %d notifications\n", num_msg);
                }
#endif

                doca_gpu_dev_rdma_recv_commit_weak (rdma_gpu_r, num_msg);
                DOCA_GPUNETIO_VOLATILE (notif_progress->num_msg) = num_msg;
                asm volatile("fence.release.sys;");
                DOCA_GPUNETIO_VOLATILE (notif_progress->qp) = nullptr;
            }

            if (DOCA_GPUNETIO_VOLATILE (notif_fill->qp) != nullptr) {
                result = doca_gpu_dev_rdma_get_recv (notif_fill->qp, &rdma_gpu_r);
                if (result != DOCA_SUCCESS)
                    printf ("Error %d doca_gpu_dev_rdma_get_recv\n", result);

                for (int idx = 0; idx < DOCA_MAX_NOTIF_INFLIGHT; idx++) {
                    doca_gpu_dev_buf_get_buf (notif_fill->bbuf_gpu, idx, &recv_buf);
                    result = doca_gpu_dev_rdma_recv_weak (
                            rdma_gpu_r, recv_buf, DOCA_MAX_NOTIF_MESSAGE_SIZE, 0, 0, idx);
                    if (result != DOCA_SUCCESS)
                        printf ("Error %d doca_gpu_dev_rdma_recv_strong\n", result);
                }

                result = doca_gpu_dev_rdma_recv_commit_weak (rdma_gpu_r, DOCA_MAX_NOTIF_INFLIGHT);
                if (result != DOCA_SUCCESS)
                    printf ("Error %d doca_gpu_dev_rdma_recv_commit_strong\n", result);


                DOCA_GPUNETIO_VOLATILE (notif_fill->qp) = nullptr;
            }
        }
    }

    // Send standalone notifications
    if (blockIdx.x == 2) {
        while (DOCA_GPUNETIO_VOLATILE (*exit_flag) == 0) {
            if (DOCA_GPUNETIO_VOLATILE (notif_send_gpu->qp) != nullptr) {
#if ENABLE_DEBUG == 1
                printf ("Notif standalone id %d\n",
                        DOCA_GPUNETIO_VOLATILE (notif_send_gpu->buf_idx));
#endif

                doca_gpu_dev_buf_get_buf (DOCA_GPUNETIO_VOLATILE (notif_send_gpu->bbuf_gpu),
                                          DOCA_GPUNETIO_VOLATILE (notif_send_gpu->buf_idx),
                                          &send_buf);
                result = doca_gpu_dev_rdma_send_strong (notif_send_gpu->qp,
                                                        0,
                                                        send_buf,
                                                        0,
                                                        notif_send_gpu->msg_sz,
                                                        0,
                                                        DOCA_GPU_RDMA_SEND_FLAG_NONE);
                if (result != DOCA_SUCCESS)
                    printf ("Error %d doca_gpu_dev_rdma_send_strong\n", result);

                result = doca_gpu_dev_rdma_commit_strong (notif_send_gpu->qp, 0);
                if (result != DOCA_SUCCESS) printf ("Error %d doca_gpu_dev_rdma_push\n", result);

                result = doca_gpu_dev_rdma_wait_all (notif_send_gpu->qp, &num_msg_notif);
                if (result != DOCA_SUCCESS) {
                    printf ("Error %d doca_gpu_dev_rdma_wait_all standalone\n", result);
                    DOCA_GPUNETIO_VOLATILE (*exit_flag) = 1;
                }
#if ENABLE_DEBUG == 1
                printf ("Notif correctly sent %d\n", num_msg_notif);
#endif

                DOCA_GPUNETIO_VOLATILE (notif_send_gpu->qp) = nullptr;
            }
        }
    }
}

doca_error_t
doca_kernel_write (cudaStream_t stream,
                   struct doca_gpu_dev_verbs_qp *qp,
                   struct docaXferReqGpu *xferReqRing,
                   uint32_t pos) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_write<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>> (qp, xferReqRing, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t
doca_kernel_read (cudaStream_t stream,
                  struct doca_gpu_dev_verbs_qp *qp,
                  struct docaXferReqGpu *xferReqRing,
                  uint32_t pos) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_read<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>> (qp, xferReqRing, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t
doca_kernel_progress (cudaStream_t stream,
                      struct docaXferCompletion *completion_list,
                      struct docaNotif *notif_fill,
                      struct docaNotif *notif_progress,
                      struct docaNotif *notif_send_gpu,
                      uint32_t *exit_flag) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return DOCA_ERROR_BAD_STATE;
    }

    // kernel_progress<<<3, 1, 0, stream>>> (
    //         completion_list, notif_fill, notif_progress, notif_send_gpu, exit_flag);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf (stderr,
                 "[%s:%d] cuda failed with %s",
                 __FILE__,
                 __LINE__,
                 cudaGetErrorString (result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
