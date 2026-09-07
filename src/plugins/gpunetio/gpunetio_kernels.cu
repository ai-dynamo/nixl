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

#define ENABLE_DEBUG 0

__device__ uint32_t
nixl_gpunetio_dev_load_host_state(uint32_t &state) {
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(state).load(
        cuda::std::memory_order_acquire);
}

__device__ void
nixl_gpunetio_dev_store_host_state(uint32_t &state, uint32_t value) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(state).store(
        value, cuda::std::memory_order_release);
}

__device__ inline void
nixl_gpunetio_dev_cq_print_cqe_err(struct mlx5_cqe64 *cqe64) {
    struct mlx5_err_cqe_ex *err_cqe = (struct mlx5_err_cqe_ex *)cqe64;

    printf("got completion with err: "
           "syndrome=%#x, vendor_err_synd=%#x, "
           "hw_err_synd=%#x, hw_synd_type=%#x, wqe_counter=%u wqe_qpn=%x\n",
           err_cqe->syndrome,
           err_cqe->vendor_err_synd,
           err_cqe->hw_err_synd,
           err_cqe->hw_synd_type,
           err_cqe->wqe_counter,
           err_cqe->s_wqe_opcode_qpn);
}

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index respecting NIXL
 * requirements. Non-blocking polling, just one-time CQE check.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template<enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
             DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
         enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ int
nixl_gpunetio_dev_priv_poll_one_cq_at(doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    uint8_t *cqe = (uint8_t *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t idx = cons_index & (cqe_num - 1);
    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)(cqe + (idx * DOCA_GPUNETIO_VERBS_CQE_SIZE));

    uint8_t opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
    uint8_t opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

    bool observed_completion = !((opown & MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num));
    observed_completion = observed_completion && (opcode != MLX5_CQE_INVALID);
    if (!observed_completion) {
        return EBUSY;
    }

    if ((opcode == MLX5_CQE_REQ_ERR || opcode == MLX5_CQE_RESP_ERR) * -EIO) {
        nixl_gpunetio_dev_cq_print_cqe_err(cqe64);
    }

    return ((opcode == MLX5_CQE_REQ_ERR || opcode == MLX5_CQE_RESP_ERR) * -EIO);
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index.
 * Non-blocking polling, just one-time CQE check.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, nixl_gpunetio_dev_poll_one_cq_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template<enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
             DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
         enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ int
nixl_gpunetio_dev_poll_one_cq_at(doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status =
        nixl_gpunetio_dev_priv_poll_one_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
    if (status != EBUSY) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
}

__device__ bool
nixl_gpunetio_dev_has_sq_credit(doca_gpu_dev_verbs_qp *qp, uint32_t count) {
    auto *cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
    const uint64_t reserved = atomicAdd((unsigned long long *)&qp->sq_rsvd_index, 0);
    const uint64_t completed = atomicAdd((unsigned long long *)&cq->cqe_ci, 0);
    return reserved + count <= completed + __ldg(&qp->sq_wqe_num);
}

__device__ void
nixl_gpunetio_dev_terminal_error(docaXferReqGpu *request,
                                 docaProgressState *progress_state,
                                 uint32_t pos) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(progress_state->active_bitmap)
        .fetch_and(~(1U << pos), cuda::std::memory_order_release);
    nixl_gpunetio_dev_store_host_state(request->state, DOCA_XFER_STATE_ERROR);
}

__device__ void
nixl_gpunetio_dev_fail_request(docaXferReqGpu *request,
                               docaProgressState *progress_state,
                               uint32_t pos) {
    nixl_gpunetio_dev_terminal_error(request, progress_state, pos);
    nixl_gpunetio_dev_store_host_state(progress_state->failed, 1U);
}

__device__ void
nixl_gpunetio_dev_complete_request(docaXferReqGpu *request,
                                   docaProgressState *progress_state,
                                   uint32_t pos) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(progress_state->active_bitmap)
        .fetch_and(~(1U << pos), cuda::std::memory_order_release);
    nixl_gpunetio_dev_store_host_state(request->state, DOCA_XFER_STATE_COMPLETE);
}

__device__ bool
nixl_gpunetio_dev_reserve_data(docaXferReqGpu *request,
                               uint32_t count,
                               docaProgressState *progress_state,
                               uint32_t *exit_flag,
                               uint64_t *base_wqe_idx) {
    docaQpProgress *progress = request->qp_progress;
    while (nixl_gpunetio_dev_load_host_state(*exit_flag) == 0U &&
           nixl_gpunetio_dev_load_host_state(progress_state->failed) == 0U) {
        if (atomicCAS(&progress->data_producer_lock, 0U, 1U) != 0U) {
            continue;
        }
        if (nixl_gpunetio_dev_load_host_state(progress_state->failed) != 0U) {
            atomicExch(&progress->data_producer_lock, 0U);
            return false;
        }
        if (!nixl_gpunetio_dev_has_sq_credit(request->qp_data, count)) {
            atomicExch(&progress->data_producer_lock, 0U);
            continue;
        }

        *base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                                DOCA_GPUNETIO_VERBS_QP_SQ,
                                                false>(request->qp_data, count);
        request->data_ticket = atomicAdd((unsigned long long *)&progress->next_data_ticket, 1ULL);
        return true;
    }
    return false;
}

__device__ void
nixl_gpunetio_dev_publish_data(docaXferReqGpu *request,
                               docaProgressState *progress_state,
                               uint32_t pos) {
    atomicExch(&request->data_state, DOCA_XFER_DATA_POSTED);
    nixl_gpunetio_dev_store_host_state(request->state, DOCA_XFER_STATE_DATA_POSTED);
    progress_state->active_generation[pos] = request->generation;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(progress_state->active_bitmap)
        .fetch_or(1U << pos, cuda::std::memory_order_release);
}

__global__ void
kernel_read(doca_gpu_dev_verbs_qp *qp,
            struct docaXferReqGpu *xferReqRing,
            docaProgressState *progress_state,
            uint32_t *exit_flag,
            uint32_t pos) {
    uint64_t wqe_idx = 0;
    doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
    uint32_t tot_wqe, idx = 0;
    __shared__ uint64_t base_wqe_idx;
    __shared__ uint64_t last_wqe_idx;
    __shared__ uint32_t reserved;

    // Warmup
    if (xferReqRing == nullptr) {
        return;
    }

    tot_wqe = xferReqRing[pos].num;

    if (threadIdx.x == 0) {
        if (nixl_gpunetio_dev_load_host_state(xferReqRing[pos].state) != DOCA_XFER_STATE_PREPARED) {
            reserved = 0U;
            nixl_gpunetio_dev_fail_request(&xferReqRing[pos], progress_state, pos);
        } else {
            const uint32_t count = tot_wqe + (qp->need_mcst ? 1 : 0);
            reserved = nixl_gpunetio_dev_reserve_data(
                &xferReqRing[pos], count, progress_state, exit_flag, &base_wqe_idx);
            if (reserved == 0U) {
                nixl_gpunetio_dev_fail_request(&xferReqRing[pos], progress_state, pos);
            }
        }
    }
    __syncthreads();
    if (reserved == 0U) {
        return;
    }

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

    if (threadIdx.x == 0) {
        last_wqe_idx = base_wqe_idx + tot_wqe - 1;
        if (qp->need_mcst == true) {
            ++last_wqe_idx;
            wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, last_wqe_idx);

            doca_gpu_dev_verbs_wqe_prepare_dump(qp,
                                                wqe_ptr,
                                                last_wqe_idx,
                                                DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                                                (uint64_t)(xferReqRing[pos].lbuf[tot_wqe - 1]),
                                                xferReqRing[pos].lkey[tot_wqe - 1],
                                                1);
        }
        doca_gpu_dev_verbs_mark_wqes_ready(qp, base_wqe_idx, last_wqe_idx);
        doca_gpu_dev_verbs_submit(qp, last_wqe_idx + 1);
        xferReqRing[pos].last_wqe = last_wqe_idx;
        atomicExch(&xferReqRing[pos].qp_progress->data_producer_lock, 0U);
        nixl_gpunetio_dev_publish_data(&xferReqRing[pos], progress_state, pos);
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

__global__ void
kernel_write(doca_gpu_dev_verbs_qp *qp,
             struct docaXferReqGpu *xferReqRing,
             docaProgressState *progress_state,
             uint32_t *exit_flag,
             uint32_t pos) {
    uint64_t wqe_idx = 0;
    doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
    uint32_t tot_wqe, idx = 0;
    __shared__ uint64_t base_wqe_idx;
    __shared__ uint32_t reserved;

    // Warmup
    if (xferReqRing == nullptr) {
        return;
    }

    tot_wqe = xferReqRing[pos].num;

    if (threadIdx.x == 0) {
        if (nixl_gpunetio_dev_load_host_state(xferReqRing[pos].state) != DOCA_XFER_STATE_PREPARED) {
            reserved = 0U;
            nixl_gpunetio_dev_fail_request(&xferReqRing[pos], progress_state, pos);
        } else {
            reserved = nixl_gpunetio_dev_reserve_data(
                &xferReqRing[pos], tot_wqe, progress_state, exit_flag, &base_wqe_idx);
            if (reserved == 0U) {
                nixl_gpunetio_dev_fail_request(&xferReqRing[pos], progress_state, pos);
            }
        }
    }
    __syncthreads();
    if (reserved == 0U) {
        return;
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

    if (threadIdx.x == 0) {
        const uint64_t last_wqe_idx = base_wqe_idx + tot_wqe - 1;
        doca_gpu_dev_verbs_mark_wqes_ready(qp, base_wqe_idx, last_wqe_idx);
        doca_gpu_dev_verbs_submit(qp, last_wqe_idx + 1);
        xferReqRing[pos].last_wqe = last_wqe_idx;
        atomicExch(&xferReqRing[pos].qp_progress->data_producer_lock, 0U);
        nixl_gpunetio_dev_publish_data(&xferReqRing[pos], progress_state, pos);
    }

#if ENABLE_DEBUG == 1
    if (threadIdx.x == 0) {
        printf(">>>>>>> CUDA rdma write kernel pos %d posted %d buffers from base_wqe_idx %ld\n",
               pos,
               xferReqRing[pos].num,
               base_wqe_idx);
    }
#endif
}

__global__ void
kernel_publish_notif(docaXferReqGpu *xfer_req_ring,
                     docaProgressState *progress_state,
                     uint32_t pos) {
    if (nixl_gpunetio_dev_load_host_state(xfer_req_ring[pos].state) !=
        DOCA_XFER_STATE_NOTIF_PENDING) {
        return;
    }
    progress_state->active_generation[pos] = xfer_req_ring[pos].generation;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(progress_state->active_bitmap)
        .fetch_or(1U << pos, cuda::std::memory_order_release);
}

__device__ bool
nixl_gpunetio_dev_try_post_notif(docaXferReqGpu *request,
                                 docaProgressState *progress_state,
                                 uint32_t *exit_flag) {
    docaQpProgress *progress = request->qp_progress;
    doca_gpu_dev_verbs_qp *qp = request->qp_notif;

    // Data-coupled and standalone notifications share this per-peer SQ owner.
    if (atomicCAS(&progress->notif_producer_lock, 0U, 1U) != 0U) {
        return false;
    }
    if (nixl_gpunetio_dev_load_host_state(*exit_flag) != 0U ||
        nixl_gpunetio_dev_load_host_state(progress_state->failed) != 0U ||
        !nixl_gpunetio_dev_has_sq_credit(qp, 1)) {
        atomicExch(&progress->notif_producer_lock, 0U);
        return false;
    }

    const uint64_t wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                            DOCA_GPUNETIO_VERBS_QP_SQ,
                                            false>(qp, 1);
    doca_gpu_dev_verbs_wqe *wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_send(qp,
                                        wqe_ptr,
                                        wqe_idx,
                                        DOCA_GPUNETIO_MLX5_OPCODE_SEND,
                                        DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                                        0,
                                        request->lbuf_notif,
                                        request->lkey_notif,
                                        request->msg_sz);
    doca_gpu_dev_verbs_mark_wqes_ready(qp, wqe_idx, wqe_idx);
    doca_gpu_dev_verbs_submit(qp, wqe_idx + 1);
    request->notif_wqe = wqe_idx;
    request->notif_ticket = atomicAdd((unsigned long long *)&progress->next_notif_ticket, 1ULL);
    atomicExch(&request->notif_state, DOCA_XFER_NOTIF_POSTED);
    nixl_gpunetio_dev_store_host_state(request->state, DOCA_XFER_STATE_NOTIF_POSTED);
    atomicExch(&progress->notif_producer_lock, 0U);
    return true;
}

__global__ void
kernel_progress(struct docaXferReqGpu *xfer_req_ring,
                struct docaProgressState *progress_state,
                struct docaNotif *notif_fill,
                struct docaNotif *notif_progress,
                uint32_t *exit_flag) {
    if (xfer_req_ring == nullptr) {
        return;
    }

    // One scheduler owns all data-SQ and notification-SQ completion credits.
    if (blockIdx.x == 0) {
        while (nixl_gpunetio_dev_load_host_state(*exit_flag) == 0U) {
            const uint32_t active =
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(progress_state->active_bitmap)
                    .load(cuda::std::memory_order_acquire);
            const uint32_t start = atomicAdd(&progress_state->progress_cursor, 0U);

            for (uint32_t offset = 0; offset < DOCA_XFER_REQ_MAX; ++offset) {
                const uint32_t pos = (start + offset) & DOCA_XFER_REQ_MASK;
                if ((active & (1U << pos)) == 0U) {
                    continue;
                }

                docaXferReqGpu *request = &xfer_req_ring[pos];
                if (progress_state->active_generation[pos] != request->generation) {
                    // Do not release a possibly newer owner on a stale generation.
                    nixl_gpunetio_dev_store_host_state(progress_state->failed, 1U);
                    continue;
                }
                docaQpProgress *progress = request->qp_progress;
                const uint32_t request_state = nixl_gpunetio_dev_load_host_state(request->state);

                if (request_state == DOCA_XFER_STATE_DATA_POSTED &&
                    request->data_ticket ==
                        atomicAdd((unsigned long long *)&progress->head_data_ticket, 0ULL)) {
                    const int poll_status = nixl_gpunetio_dev_poll_one_cq_at<
                        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                        DOCA_GPUNETIO_VERBS_QP_SQ>(
                        doca_gpu_dev_verbs_qp_get_cq_sq(request->qp_data), request->last_wqe);
                    if (poll_status == 0) {
                        atomicAdd((unsigned long long *)&progress->head_data_ticket, 1ULL);
                        atomicExch(&request->data_state, DOCA_XFER_DATA_COMPLETE);
                        if (request->has_notif_msg_idx == DOCA_NOTIF_NULL) {
                            nixl_gpunetio_dev_complete_request(request, progress_state, pos);
                            continue;
                        } else {
                            atomicExch(&request->notif_state, DOCA_XFER_NOTIF_PENDING);
                            nixl_gpunetio_dev_store_host_state(request->state,
                                                               DOCA_XFER_STATE_NOTIF_PENDING);
                        }
                    } else if (poll_status != EBUSY) {
                        atomicAdd((unsigned long long *)&progress->head_data_ticket, 1ULL);
                        nixl_gpunetio_dev_fail_request(request, progress_state, pos);
                        continue;
                    }
                }

                if (nixl_gpunetio_dev_load_host_state(request->state) ==
                    DOCA_XFER_STATE_NOTIF_PENDING) {
                    if (nixl_gpunetio_dev_load_host_state(progress_state->failed) != 0U) {
                        nixl_gpunetio_dev_terminal_error(request, progress_state, pos);
                        continue;
                    }
                    nixl_gpunetio_dev_try_post_notif(request, progress_state, exit_flag);
                }

                if (nixl_gpunetio_dev_load_host_state(request->state) ==
                        DOCA_XFER_STATE_NOTIF_POSTED &&
                    request->notif_ticket ==
                        atomicAdd((unsigned long long *)&progress->head_notif_ticket, 0ULL)) {
                    const int poll_status = nixl_gpunetio_dev_poll_one_cq_at<
                        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                        DOCA_GPUNETIO_VERBS_QP_SQ>(
                        doca_gpu_dev_verbs_qp_get_cq_sq(request->qp_notif), request->notif_wqe);
                    if (poll_status == 0) {
                        atomicAdd((unsigned long long *)&progress->head_notif_ticket, 1ULL);
                        atomicExch(&request->notif_state, DOCA_XFER_NOTIF_COMPLETE);
                        nixl_gpunetio_dev_complete_request(request, progress_state, pos);
                        continue;
                    } else if (poll_status != EBUSY) {
                        atomicAdd((unsigned long long *)&progress->head_notif_ticket, 1ULL);
                        nixl_gpunetio_dev_fail_request(request, progress_state, pos);
                        continue;
                    }
                }
            }
            atomicExch(&progress_state->progress_cursor, (start + 1) & DOCA_XFER_REQ_MASK);
        }
    }

    // Receive notif: fill recv in new queue and progress queue
    if (blockIdx.x == 1) {
        while (nixl_gpunetio_dev_load_host_state(*exit_flag) == 0U) {
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
                } else if (ret == EBUSY) {
#if ENABLE_DEBUG == 1
                    printf("kernel received notification EBUSY at %d ret %d\n", msg_last, ret);
#endif
                    DOCA_GPUNETIO_VOLATILE(notif_progress->msg_num) = 0;
                    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
                    DOCA_GPUNETIO_VOLATILE(notif_progress->qp_gpu) = nullptr;
                } else {
                    nixl_gpunetio_dev_store_host_state(progress_state->failed, 1U);
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
}

doca_error_t
doca_kernel_write(cudaStream_t stream,
                  doca_gpu_dev_verbs_qp *qp,
                  struct docaXferReqGpu *xferReqRing,
                  struct docaProgressState *progress_state,
                  uint32_t *exit_flag,
                  uint32_t pos) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_write<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(
        qp, xferReqRing, progress_state, exit_flag, pos);
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
                 doca_gpu_dev_verbs_qp *qp,
                 struct docaXferReqGpu *xferReqRing,
                 struct docaProgressState *progress_state,
                 uint32_t *exit_flag,
                 uint32_t pos) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_read<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(
        qp, xferReqRing, progress_state, exit_flag, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t
doca_kernel_publish_notif(cudaStream_t stream,
                          struct docaXferReqGpu *xfer_req_ring,
                          struct docaProgressState *progress_state,
                          uint32_t pos) {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_publish_notif<<<1, 1, 0, stream>>>(xfer_req_ring, progress_state, pos);
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
                     struct docaProgressState *progress_state,
                     struct docaNotif *notif_fill,
                     struct docaNotif *notif_progress,
                     uint32_t *exit_flag) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_progress<<<2, 1, 0, stream>>>(
        xfer_req_ring, progress_state, notif_fill, notif_progress, exit_flag);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
