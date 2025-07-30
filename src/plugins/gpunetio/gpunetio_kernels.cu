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
#include <doca_gpunetio_dev_verbs_twosided.cuh>
#include <cuda.h>
#include <cuda/atomic>

#include "gpunetio_backend.h"

#define ENABLE_DEBUG 0

__global__ void
kernel_read(struct doca_gpu_dev_verbs_qp *qp, struct docaXferReqGpu *xferReqRing, uint32_t pos) {
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
        if (idx == (tot_wqe - 1)) cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
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

        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].last_wqe) = wqe_idx;
        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].in_use) = 1;
    }
#if ENABLE_DEBUG == 1
    printf(">>>>>>> CUDA rdma read kernel pos %d posted %d buffers\n", pos, xferReqRing[pos].num);
#endif
}

__global__ void
kernel_write(struct doca_gpu_dev_verbs_qp *qp, struct docaXferReqGpu *xferReqRing, uint32_t pos) {
    uint64_t wqe_idx = 0;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
    __shared__ uint32_t base_wqe_idx;
    uint32_t tot_wqe = xferReqRing[pos].num;
    uint32_t idx = 0;

    if (threadIdx.x == 0) base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, tot_wqe);
    __syncthreads();

    for (idx = threadIdx.x; idx < tot_wqe; idx += blockDim.x) {
        if (idx == (tot_wqe - 1)) cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
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

        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].last_wqe) = wqe_idx;
        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].in_use) = 1;
    }
#if ENABLE_DEBUG == 1
    printf(">>>>>>> CUDA rdma write kernel pos %d posted %d buffers\n", pos, xferReqRing[pos].num);
#endif
}

__global__ void
kernel_progress(struct docaXferCompletion *completion_list,
                struct docaNotif *notif_fill,
                struct docaNotif *notif_progress,
                struct docaNotif *notif_send_gpu,
                uint32_t *exit_flag) {
    uint32_t index = 0;
    doca_gpu_dev_verbs_ticket_t out_ticket;

    // Warmup
    if (completion_list == nullptr) return;

    // Wait Xfer & notify
    if (blockIdx.x == 0) {
        while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
            // Check xfer completion and send notif
            if (DOCA_GPUNETIO_VOLATILE(completion_list[index].xferReqRingGpu) != nullptr) {
                if (DOCA_GPUNETIO_VOLATILE(completion_list[index].completed) == 0 &&
                    DOCA_GPUNETIO_VOLATILE(completion_list[index].xferReqRingGpu->in_use) == 1) {
                    // Wait for final CQE in block of iterations
                    if (doca_gpu_dev_verbs_poll_cq_at(
                            doca_gpu_dev_verbs_qp_get_cq_sq(
                                completion_list[index].xferReqRingGpu->qp_data),
                            DOCA_GPUNETIO_VOLATILE(
                                completion_list[index].xferReqRingGpu->last_wqe)) != 0) {
                        DOCA_GPUNETIO_VOLATILE(*exit_flag) = 1;
                        printf("Error CQE!\n");
                        break;
                    }

                    if (DOCA_GPUNETIO_VOLATILE(
                            completion_list[index].xferReqRingGpu->has_notif_msg_idx) !=
                        DOCA_NOTIF_NULL) {
#if ENABLE_DEBUG == 1
                        printf("Notif after completion at %d id %d sz %d\n",
                               index,
                               DOCA_GPUNETIO_VOLATILE(
                                   completion_list[index].xferReqRingGpu->has_notif_msg_idx),
                               (int)completion_list[index].xferReqRingGpu->msg_sz);
#endif
                        doca_gpu_dev_verbs_send(
                            completion_list[index].xferReqRingGpu->qp_notif,
                            doca_gpu_dev_verbs_addr{
                                .addr =
                                    (uint64_t)(completion_list[index].xferReqRingGpu->lbuf_notif),
                                .key = completion_list[index].xferReqRingGpu->lkey_notif},
                            completion_list[index].xferReqRingGpu->msg_sz,
                            &out_ticket);

                        doca_gpu_dev_verbs_wait(completion_list[index].xferReqRingGpu->qp_notif,
                                                &out_ticket);
#if ENABLE_DEBUG == 1
                        printf("Notif correctly sent %ld\n", out_ticket);
#endif
                    }
                    DOCA_GPUNETIO_VOLATILE(completion_list[index].completed) = 1;
                    index = (index + 1) & DOCA_MAX_COMPLETION_INFLIGHT_MASK;
                }
            }
        }
    } // Block 0

    // Receive notif: fill recv in new queue and progress queue
    if (blockIdx.x == 1) {
        while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
            // Check received notifications
            if (DOCA_GPUNETIO_VOLATILE(notif_progress->qp_gpu) != nullptr) {
#if ENABLE_DEBUG == 1
                printf("waiting for notification at %ld\n", notif_progress->msg_last);
#endif
                if (doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                                      DOCA_GPUNETIO_VERBS_QP_RQ>(
                        doca_gpu_dev_verbs_qp_get_cq_rq(notif_progress->qp_gpu),
                        notif_progress->msg_last) != EBUSY) {
#if ENABLE_DEBUG == 1
                    printf("kernel received notification at %ld\n", notif_progress->msg_last);
#endif
                    DOCA_GPUNETIO_VOLATILE(notif_progress->msg_num) = 1;
                    DOCA_GPUNETIO_VOLATILE(notif_progress->msg_last) =
                        (notif_progress->msg_last + 1) %
                        doca_gpu_dev_verbs_qp_get_cq_rq(notif_progress->qp_gpu)->cqe_mask;
                    asm volatile("fence.release.sys;");
                    DOCA_GPUNETIO_VOLATILE(notif_progress->qp_gpu) = nullptr;

                    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                              DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                              DOCA_GPUNETIO_VERBS_QP_RQ>(
                        notif_fill->qp_gpu, notif_fill->qp_gpu->rq_wqe_pi + 1);
                }
            }

            if (DOCA_GPUNETIO_VOLATILE(notif_fill->qp_gpu) != nullptr) {
                for (int idx = 0; idx < DOCA_MAX_NOTIF_INFLIGHT; idx++) {
                    struct mlx5_wqe_data_seg *rwqe_ptr =
                        doca_gpu_dev_verbs_get_rwqe_ptr(notif_fill->qp_gpu, idx);
                    doca_gpu_dev_verbs_wqe_prepare_recv(
                        notif_fill->qp_gpu,
                        rwqe_ptr,
                        (uint64_t)(notif_fill->msg_buf + (notif_fill->msg_size * idx)),
                        notif_fill->msg_lkey,
                        notif_fill->msg_size);
                }

                doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                          DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                          DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                          DOCA_GPUNETIO_VERBS_QP_RQ>(notif_fill->qp_gpu,
                                                                     DOCA_MAX_NOTIF_INFLIGHT);

                DOCA_GPUNETIO_VOLATILE(notif_fill->qp_gpu) = nullptr;
            }
        }
    }

    // Send standalone notifications
    if (blockIdx.x == 2) {
        while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
            if (DOCA_GPUNETIO_VOLATILE(notif_send_gpu->qp_gpu) != nullptr) {
#if ENABLE_DEBUG == 1
                printf("Notif standalone id %d\n", DOCA_GPUNETIO_VOLATILE(notif_send_gpu->buf_idx));
#endif

                doca_gpu_dev_verbs_send(
                    notif_send_gpu->qp_gpu,
                    doca_gpu_dev_verbs_addr{.addr = (uint64_t)notif_send_gpu->msg_buf,
                                            .key = notif_send_gpu->msg_lkey},
                    notif_send_gpu->msg_size,
                    &out_ticket);

                doca_gpu_dev_verbs_wait(notif_send_gpu->qp_gpu, &out_ticket);
#if ENABLE_DEBUG == 1
                printf("Notif correctly sent %d\n", out_ticket);
#endif
                DOCA_GPUNETIO_VOLATILE(notif_send_gpu->qp_gpu) = nullptr;
            }
        }
    }
}

doca_error_t
doca_kernel_write(cudaStream_t stream,
                  struct doca_gpu_dev_verbs_qp *qp,
                  struct docaXferReqGpu *xferReqRing,
                  uint32_t pos) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_write<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(qp, xferReqRing, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t
doca_kernel_read(cudaStream_t stream,
                 struct doca_gpu_dev_verbs_qp *qp,
                 struct docaXferReqGpu *xferReqRing,
                 uint32_t pos) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_read<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(qp, xferReqRing, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t
doca_kernel_progress(cudaStream_t stream,
                     struct docaXferCompletion *completion_list,
                     struct docaNotif *notif_fill,
                     struct docaNotif *notif_progress,
                     struct docaNotif *notif_send_gpu,
                     uint32_t *exit_flag) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    // kernel_progress<<<3, 1, 0, stream>>> (
    //         completion_list, notif_fill, notif_progress, notif_send_gpu, exit_flag);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
