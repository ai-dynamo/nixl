/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gpunetio_completion.cuh"

#define ENABLE_DEBUG 0

__global__ void
kernel_read(doca_gpu_dev_verbs_qp *qp, struct docaXferReqGpu *xferReqRing, uint32_t pos) {
    uint64_t wqe_idx = 0;
    doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
    uint32_t tot_wqe, idx = 0;
    __shared__ uint64_t base_wqe_idx;

    // Warmup
    if (xferReqRing == nullptr) {
        return;
    }

    tot_wqe = xferReqRing[pos].num;

    if (threadIdx.x == 0) {
        if (qp->need_mcst == true) {
            base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, tot_wqe + 1);
        } else {
            base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, tot_wqe);
        }
    }
    __syncthreads();

    for (idx = threadIdx.x; idx < tot_wqe; idx += blockDim.x) {
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

    if ((idx - blockDim.x) == (tot_wqe - 1)) {
        if (qp->need_mcst == true) {
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
        if (doca_gpu_dev_verbs_poll_cq_at(qp, wqe_idx) != 0) {
            printf("kernel_read: Error CQE!\n");
        }

        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].last_wqe) = wqe_idx;
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].in_use) = 1;
    }

#if ENABLE_DEBUG == 1
    if (threadIdx.x == 0) {
        printf(">>>>>>> CUDA rdma read kernel pos %d posted %d buffers from base_wqe_idx %ld\n",
               pos,
               xferReqRing[pos].num,
               base_wqe_idx);
    }
#endif
}

__device__ __forceinline__ bool
nixl_gpunetio_dev_write_chunk(doca_gpu_dev_verbs_qp *qp,
                              struct docaXferReqGpu *xferReqRing,
                              uint32_t pos) {
    uint64_t wqe_idx = 0;
    doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
    uint32_t tot_wqe, idx = 0;
    __shared__ uint64_t base_wqe_idx;
    __shared__ bool abort_write;

    // Warmup
    if (xferReqRing == nullptr) {
        return false;
    }

    tot_wqe = xferReqRing[pos].num;

    if (threadIdx.x == 0) {
        abort_write = false;
        for (uint32_t dependency = 0; dependency < xferReqRing[pos].wait_completion_count;
             ++dependency) {
            const uint32_t position = xferReqRing[pos].wait_positions[dependency];
            while (DOCA_GPUNETIO_VOLATILE(xferReqRing[position].done) == 0) {
                if (DOCA_GPUNETIO_VOLATILE(*xferReqRing[pos].wait_exit) != 0) {
                    abort_write = true;
                    break;
                }
            }
            if (abort_write) {
                break;
            }
        }
        if (xferReqRing[pos].peer_write_completed != nullptr) {
            while (!abort_write &&
                   DOCA_GPUNETIO_VOLATILE(*xferReqRing[pos].peer_write_completed) <
                       xferReqRing[pos].write_ticket - 1) {
                if (DOCA_GPUNETIO_VOLATILE(*xferReqRing[pos].wait_exit) != 0) {
                    abort_write = true;
                }
            }
        }
        if (!abort_write) {
            if (xferReqRing[pos].wait_completion_count != 0 ||
                xferReqRing[pos].peer_write_completed != nullptr) {
                doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
            }
            base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, tot_wqe);
        }
    }
    __syncthreads();
    if (abort_write) {
        return false;
    }

    for (idx = threadIdx.x; idx < tot_wqe; idx += blockDim.x) {
        wqe_idx = base_wqe_idx + idx;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

#if ENABLE_DEBUG == 1
        printf("prepare_write radd %lx rkey %x ladd %lx lkey %x size %ld\n",
               (uint64_t)(xferReqRing[pos].rbuf[idx]),
               xferReqRing[pos].rkey[idx],
               (uint64_t)(xferReqRing[pos].lbuf[idx]),
               xferReqRing[pos].lkey[idx],
               (uint64_t)xferReqRing[pos].size[idx]);
#endif
        if (xferReqRing[pos].has_two_sge && xferReqRing[pos].num_sge[idx] == 2) {
            doca_gpu_dev_verbs_wqe_prepare_write(qp,
                                                 wqe_ptr,
                                                 wqe_idx,
                                                 MLX5_OPCODE_RDMA_WRITE,
                                                 cflag,
                                                 0,
                                                 (uint64_t)xferReqRing[pos].rbuf[idx],
                                                 xferReqRing[pos].rkey[idx],
                                                 (uint64_t)xferReqRing[pos].lbuf[idx],
                                                 xferReqRing[pos].lkey[idx],
                                                 xferReqRing[pos].size[idx],
                                                 (uint64_t)xferReqRing[pos].lbuf2[idx],
                                                 xferReqRing[pos].lkey2[idx],
                                                 xferReqRing[pos].size2[idx]);
        } else {
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
    }
    __syncthreads();

    if ((idx - blockDim.x) == (tot_wqe - 1)) {
        doca_gpu_dev_verbs_mark_wqes_ready(qp, base_wqe_idx, wqe_idx);
        doca_gpu_dev_verbs_submit(qp, wqe_idx + 1);

        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].last_wqe) = wqe_idx;
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        DOCA_GPUNETIO_VOLATILE(xferReqRing[pos].in_use) = 1;
    }

#if ENABLE_DEBUG == 1
    if (threadIdx.x == 0) {
        printf(">>>>>>> CUDA rdma write kernel pos %d posted %d buffers from base_wqe_idx %ld\n",
               pos,
               tot_wqe,
               base_wqe_idx);
    }
#endif
    return true;
}

__global__ void
kernel_write(doca_gpu_dev_verbs_qp *qp, struct docaXferReqGpu *xferReqRing, uint32_t pos) {
    nixl_gpunetio_dev_write_chunk(qp, xferReqRing, pos);
}

__global__ void
kernel_write_striped(struct docaXferReqGpu *xferReqRing, docaStripedWriteLaunch launch) {
    if (xferReqRing == nullptr) {
        return;
    }
    __shared__ bool final_owner;
    const uint32_t final_pos = launch.positions[launch.count - 1];

    for (uint32_t ordinal = blockIdx.x; ordinal + 1 < launch.count; ordinal += gridDim.x) {
        const uint32_t pos = launch.positions[ordinal];
        const bool posted =
            nixl_gpunetio_dev_write_chunk(xferReqRing[pos].qp_data, xferReqRing, pos);
        // Reuse the helper's CTA-local state only after the posting lane finishes.
        __syncthreads();
        if (!posted) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        final_owner = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                          xferReqRing[final_pos].posting_ctas_done)
                          .fetch_add(1U, cuda::std::memory_order_acq_rel) == gridDim.x - 1;
    }
    __syncthreads();
    if (final_owner) {
        // The last arrival acquired every CTA's completed posting publications.
        nixl_gpunetio_dev_write_chunk(xferReqRing[final_pos].qp_data, xferReqRing, final_pos);
    }
}

// Block 0 of kernel_progress is one warp with a lane per transfer ring slot.
constexpr uint32_t NIXL_GPUNETIO_PROGRESS_LANES = 32;
static_assert(DOCA_XFER_REQ_MAX <= NIXL_GPUNETIO_PROGRESS_LANES,
              "kernel_progress polls every ring slot from its own lane of one warp");

enum nixlGpunetioSlotPhase : uint32_t {
    NIXL_GPUNETIO_SLOT_IDLE, // waiting for a posting kernel to set in_use
    NIXL_GPUNETIO_SLOT_DATA, // polling the data CQ at the chunk's last WQE
    NIXL_GPUNETIO_SLOT_SEND, // data complete, notification SEND not posted yet
    NIXL_GPUNETIO_SLOT_NOTIF, // polling the notification CQ at the SEND WQE
};

__global__ void
kernel_progress(struct docaXferReqGpu *xfer_req_ring,
                uint32_t *slot_done,
                struct docaNotif *notif_fill,
                struct docaNotif *notif_progress,
                struct docaNotif *notif_send_gpu,
                uint32_t *exit_flag) {
    doca_gpu_dev_verbs_ticket_t out_ticket;

    // Warmup
    if (xfer_req_ring == nullptr) {
        return;
    }

    // Wait Xfer & notify. Each posted chunk is polled on every pass, so a slow chunk delays only
    // itself. An RC QP completes its WQEs in order and DOCA treats indexes below the CQ consumer
    // index as complete, so chunks of one QP can be polled in any order.
    if (blockIdx.x == 0) {
        const uint32_t lane = threadIdx.x;
        const bool owns_slot = lane < DOCA_XFER_REQ_MAX;
        struct docaXferReqGpu *const req = &xfer_req_ring[owns_slot ? lane : 0];
        uint32_t phase = NIXL_GPUNETIO_SLOT_IDLE;
        doca_gpu_dev_verbs_cq *cq = nullptr;
        uint64_t wait_wqe = 0;

        // A warp-wide exit decision keeps every lane in the ballot below.
        while (__all_sync(0xffffffffU, DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0)) {
            if (owns_slot && phase == NIXL_GPUNETIO_SLOT_IDLE &&
                DOCA_GPUNETIO_VOLATILE(req->in_use) == 1) {
                // Pairs with the posting kernel's release of last_wqe before in_use.
                doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
                cq = doca_gpu_dev_verbs_qp_get_cq_sq(req->qp_data);
                wait_wqe = req->last_wqe;
                phase = NIXL_GPUNETIO_SLOT_DATA;
            }

            if (phase == NIXL_GPUNETIO_SLOT_DATA || phase == NIXL_GPUNETIO_SLOT_NOTIF) {
                const int poll_status =
                    nixl_gpunetio_dev_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                                     DOCA_GPUNETIO_VERBS_QP_SQ>(cq, wait_wqe);
                if (poll_status == 0) {
                    if (phase == NIXL_GPUNETIO_SLOT_DATA &&
                        req->has_notif_msg_idx != DOCA_NOTIF_NULL) {
                        phase = NIXL_GPUNETIO_SLOT_SEND;
                    } else {
                        if (req->peer_write_completed != nullptr) {
                            doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
                            DOCA_GPUNETIO_VOLATILE(*req->peer_write_completed) = req->write_ticket;
                        }
                        DOCA_GPUNETIO_VOLATILE(req->done) = 1;
                        DOCA_GPUNETIO_VOLATILE(req->in_use) = 0;
                        // The host may repost the slot as soon as it sees slot_done.
                        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
                        DOCA_GPUNETIO_VOLATILE(slot_done[lane]) = 1;
                        phase = NIXL_GPUNETIO_SLOT_IDLE;
                    }
                } else if (poll_status != EBUSY) {
                    DOCA_GPUNETIO_VOLATILE(*exit_flag) = 1;
                    printf("kernel_progress: slot %u %s CQ error %d wqe %llu\n",
                           lane,
                           phase == NIXL_GPUNETIO_SLOT_NOTIF ? "notification" : "data",
                           poll_status,
                           (unsigned long long)wait_wqe);
                }
            }

            // Slots of one peer share its notification QP: post their SENDs one lane at a time.
            uint32_t senders = __ballot_sync(0xffffffffU, phase == NIXL_GPUNETIO_SLOT_SEND);
            while (senders != 0U) {
                if (lane == static_cast<uint32_t>(__ffs(senders) - 1)) {
#if ENABLE_DEBUG == 1
                    printf("Notif after completion at slot %u id %u sz %u\n",
                           lane,
                           req->has_notif_msg_idx,
                           req->msg_sz);
#endif
                    doca_gpu_dev_verbs_send(
                        req->qp_notif,
                        doca_gpu_dev_verbs_addr{.addr = (uint64_t)(req->lbuf_notif),
                                                .key = req->lkey_notif},
                        req->msg_sz,
                        &out_ticket);
                    cq = doca_gpu_dev_verbs_qp_get_cq_sq(req->qp_notif);
                    wait_wqe = out_ticket;
                    phase = NIXL_GPUNETIO_SLOT_NOTIF;
                }
                senders &= senders - 1;
                __syncwarp();
            }
        }
        return;
    }

    // The notification blocks run on one thread.
    if (threadIdx.x != 0) {
        return;
    }

    // Receive notif: fill recv in new queue and progress queue
    if (blockIdx.x == 1) {
        while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
            // Check received notifications
            if (DOCA_GPUNETIO_VOLATILE(notif_progress->qp_gpu) != nullptr) {
                uint32_t msg_last = DOCA_GPUNETIO_VOLATILE(notif_progress->msg_last);
                int ret =
                    nixl_gpunetio_dev_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                                     DOCA_GPUNETIO_VERBS_QP_RQ>(
                        doca_gpu_dev_verbs_qp_get_cq_rq(notif_progress->qp_gpu), msg_last);
                if (ret == 0) {
#if ENABLE_DEBUG == 1
                    printf("kernel received notification at %d ret %d\n", msg_last, ret);
#endif

                    DOCA_GPUNETIO_VOLATILE(notif_progress->msg_num) = 1;
                    DOCA_GPUNETIO_VOLATILE(notif_progress->msg_last) = (msg_last + 1);

                    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                              DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                              DOCA_GPUNETIO_VERBS_QP_RQ>(
                        DOCA_GPUNETIO_VOLATILE(notif_progress->qp_gpu),
                        notif_progress->qp_gpu->rq_wqe_pi + 1);

#if ENABLE_DEBUG == 1
                    printf("kernel flush recv notification pi %ld last %d\n",
                           notif_progress->qp_gpu->rq_wqe_pi + 1,
                           notif_progress->msg_last);
#endif

                    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
                    DOCA_GPUNETIO_VOLATILE(notif_progress->qp_gpu) = nullptr;
                } else {
#if ENABLE_DEBUG == 1
                    if (ret == EBUSY) {
                        printf("kernel received notification EBUSY at %d ret %d\n", msg_last, ret);
                    } else {
                        printf("kernel notification CQ poll failed at %d ret %d\n", msg_last, ret);
                    }
#endif
                    DOCA_GPUNETIO_VOLATILE(notif_progress->msg_num) = 0;
                    if (ret < 0) {
                        DOCA_GPUNETIO_VOLATILE(*exit_flag) = 1;
                    }
                    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
                    DOCA_GPUNETIO_VOLATILE(notif_progress->qp_gpu) = nullptr;
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
                doca_gpu_dev_verbs_send(
                    notif_send_gpu->qp_gpu,
                    doca_gpu_dev_verbs_addr{.addr = (uint64_t)notif_send_gpu->msg_buf,
                                            .key = notif_send_gpu->msg_lkey},
                    notif_send_gpu->msg_size,
                    &out_ticket);
                int notif_status;
                do {
                    notif_status = nixl_gpunetio_dev_poll_one_cq_at<
                        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                        DOCA_GPUNETIO_VERBS_QP_SQ>(
                        doca_gpu_dev_verbs_qp_get_cq_sq(notif_send_gpu->qp_gpu), out_ticket);
                } while (notif_status == EBUSY && DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0);
                if (notif_status != 0) {
                    DOCA_GPUNETIO_VOLATILE(*exit_flag) = 1;
                    printf("kernel_progress: standalone notification CQ error %d wqe %ld\n",
                           notif_status,
                           out_ticket);
                }
#if ENABLE_DEBUG == 1
                if (notif_status == 0) {
                    printf("Notif correctly sent %ld addr %lx msg_lkey %x qp %p size %d\n",
                           out_ticket,
                           notif_send_gpu->msg_buf,
                           notif_send_gpu->msg_lkey,
                           (void *)notif_send_gpu->qp_gpu,
                           (int)notif_send_gpu->msg_size);
                }
#endif
                DOCA_GPUNETIO_VOLATILE(notif_send_gpu->qp_gpu) = nullptr;
            }
        }
    }
}

doca_error_t
doca_kernel_write(cudaStream_t stream,
                  doca_gpu_dev_verbs_qp *qp,
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
doca_kernel_write_striped(cudaStream_t stream,
                          struct docaXferReqGpu *xferReqRing,
                          docaStripedWriteLaunch launch,
                          uint32_t qp_count) {
    if (xferReqRing != nullptr &&
        (launch.count == 0 || launch.count > DOCA_XFER_REQ_MAX || qp_count < 2 ||
         qp_count > DOCA_POST_STREAM_NUM)) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        return DOCA_ERROR_BAD_STATE;
    }
    const uint32_t cta_count = xferReqRing == nullptr ? 1U : qp_count;
    kernel_write_striped<<<cta_count, DOCA_XFER_REQ_SIZE, 0, stream>>>(xferReqRing, launch);
    result = cudaGetLastError();
    return result == cudaSuccess ? DOCA_SUCCESS : DOCA_ERROR_BAD_STATE;
}

doca_error_t
doca_kernel_read(cudaStream_t stream,
                 doca_gpu_dev_verbs_qp *qp,
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
                     struct docaXferReqGpu *xfer_req_ring,
                     uint32_t *slot_done,
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

    kernel_progress<<<3, NIXL_GPUNETIO_PROGRESS_LANES, 0, stream>>>(
        xfer_req_ring, slot_done, notif_fill, notif_progress, notif_send_gpu, exit_flag);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
