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

__device__ bool
nixl_gpunetio_dev_has_sq_credit(doca_gpu_dev_verbs_qp *qp, uint32_t count) {
    auto *cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
    const uint64_t reserved =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &qp->sq_rsvd_index);
    const uint64_t completed =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &cq->cqe_ci);
    return reserved + count <= completed + __ldg(&qp->sq_wqe_num);
}

__device__ void
nixl_gpunetio_dev_mark_failed(docaProgressState *progress_state) {
    nixl_gpunetio_dev_store_host_state(progress_state->failed, 1U);
    nixl_gpunetio_dev_store_host_state(progress_state->host->failed, 1U);
}

__device__ void
nixl_gpunetio_dev_finish_request(docaProgressState *progress_state,
                                 uint32_t pos,
                                 uint32_t terminal) {
    auto *completion = &progress_state->host->completions[pos];
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(progress_state->active_bitmap)
        .fetch_and(~(1U << pos), cuda::std::memory_order_release);
    // Last publication: CPU may rearm/release the slot immediately afterwards.
    nixl_gpunetio_dev_store_host_state(completion->state, terminal);
}

__device__ void
nixl_gpunetio_dev_terminal_error(docaProgressState *progress_state, uint32_t pos) {
    nixl_gpunetio_dev_finish_request(progress_state, pos, DOCA_XFER_STATE_ERROR);
}

__device__ void
nixl_gpunetio_dev_fail_request(docaProgressState *progress_state, uint32_t pos) {
    nixl_gpunetio_dev_mark_failed(progress_state);
    nixl_gpunetio_dev_terminal_error(progress_state, pos);
}

__device__ void
nixl_gpunetio_dev_complete_request(docaXferReqGpu *request,
                                   docaProgressState *progress_state,
                                   uint32_t pos) {
    if (request->peer_write_completed != nullptr) {
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        DOCA_GPUNETIO_VOLATILE(*request->peer_write_completed) = request->write_ticket;
    }
    nixl_gpunetio_dev_finish_request(progress_state, pos, DOCA_XFER_STATE_COMPLETE);
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
        if (atomicCAS(&progress->data_producer_lock, 0, 1) != 0) {
            continue;
        }
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        if (nixl_gpunetio_dev_load_host_state(progress_state->failed) != 0U) {
            doca_gpu_dev_verbs_unlock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &progress->data_producer_lock);
            return false;
        }
        if (!nixl_gpunetio_dev_has_sq_credit(request->qp_data, count)) {
            doca_gpu_dev_verbs_unlock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &progress->data_producer_lock);
            continue;
        }

        *base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                                DOCA_GPUNETIO_VERBS_QP_SQ>(
                request->qp_data, count, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        return true;
    }
    return false;
}

__device__ void
nixl_gpunetio_dev_publish_data(docaXferReqGpu *request,
                               docaProgressState *progress_state,
                               uint32_t pos) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(request->data_state)
        .store(DOCA_XFER_DATA_POSTED, cuda::std::memory_order_relaxed);
    nixl_gpunetio_dev_store_host_state(request->state, DOCA_XFER_STATE_DATA_POSTED);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(progress_state->active_bitmap)
        .fetch_or(1U << pos, cuda::std::memory_order_release);
}

__global__ void
kernel_read(doca_gpu_dev_verbs_qp *qp,
            struct docaXferReqGpu *xferReqRing,
            docaProgressState *progress_state,
            uint32_t *exit_flag,
            uint32_t pos,
            bool wait_for_completion) {
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
            nixl_gpunetio_dev_fail_request(progress_state, pos);
        } else {
            const uint32_t count = tot_wqe + (qp->need_mcst ? 1 : 0);
            reserved = nixl_gpunetio_dev_reserve_data(
                &xferReqRing[pos], count, progress_state, exit_flag, &base_wqe_idx);
            if (reserved == 0U) {
                nixl_gpunetio_dev_fail_request(progress_state, pos);
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
        doca_gpu_dev_verbs_cq *const cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
        const uint64_t completion_wqe = last_wqe_idx;
        doca_gpu_dev_verbs_mark_wqes_ready(qp, base_wqe_idx, last_wqe_idx);
        doca_gpu_dev_verbs_submit(qp, last_wqe_idx + 1);
        xferReqRing[pos].last_wqe = last_wqe_idx;
        doca_gpu_dev_verbs_unlock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &xferReqRing[pos].qp_progress->data_producer_lock);
        nixl_gpunetio_dev_publish_data(&xferReqRing[pos], progress_state, pos);
        if (wait_for_completion) {
            while (nixl_gpunetio_dev_load_host_state(*exit_flag) == 0U &&
                   nixl_gpunetio_dev_load_host_state(progress_state->failed) == 0U &&
                   doca_gpu_dev_verbs_atomic_read<uint64_t,
                                                  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                       &cq->cqe_ci) <= completion_wqe) {}
            if (doca_gpu_dev_verbs_atomic_read<uint64_t,
                                               DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                    &cq->cqe_ci) > completion_wqe) {
                doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
            }
        }
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
        return false;
    }

    tot_wqe = xferReqRing[pos].num;

    if (threadIdx.x == 0) {
        reserved = 1U;
        for (uint32_t dependency = 0; dependency < xferReqRing[pos].wait_completion_count;
             ++dependency) {
            const uint32_t position = xferReqRing[pos].wait_completion_positions[dependency];
            while (cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                       xferReqRing[position].data_state)
                       .load(cuda::std::memory_order_acquire) != DOCA_XFER_DATA_COMPLETE) {
                if (nixl_gpunetio_dev_load_host_state(*exit_flag) != 0U ||
                    nixl_gpunetio_dev_load_host_state(progress_state->failed) != 0U) {
                    reserved = 0U;
                    break;
                }
            }
            if (reserved == 0U) {
                break;
            }
        }
        if (xferReqRing[pos].peer_write_completed != nullptr) {
            while (reserved != 0U &&
                   DOCA_GPUNETIO_VOLATILE(*xferReqRing[pos].peer_write_completed) <
                       xferReqRing[pos].write_ticket - 1) {
                if (nixl_gpunetio_dev_load_host_state(*exit_flag) != 0U ||
                    nixl_gpunetio_dev_load_host_state(progress_state->failed) != 0U) {
                    reserved = 0U;
                }
            }
        }
        if (reserved != 0U) {
            if (xferReqRing[pos].wait_completion_count != 0 ||
                xferReqRing[pos].peer_write_completed != nullptr) {
                doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
            }
        }
        if (reserved == 0U ||
            nixl_gpunetio_dev_load_host_state(xferReqRing[pos].state) != DOCA_XFER_STATE_PREPARED) {
            reserved = 0U;
            nixl_gpunetio_dev_fail_request(progress_state, pos);
        } else {
            reserved = nixl_gpunetio_dev_reserve_data(
                &xferReqRing[pos], tot_wqe, progress_state, exit_flag, &base_wqe_idx);
            if (reserved == 0U) {
                nixl_gpunetio_dev_fail_request(progress_state, pos);
            }
        }
    }
    __syncthreads();
    if (reserved == 0U) {
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

    if (threadIdx.x == 0) {
        const uint64_t last_wqe_idx = base_wqe_idx + tot_wqe - 1;
        doca_gpu_dev_verbs_mark_wqes_ready(qp, base_wqe_idx, last_wqe_idx);
        doca_gpu_dev_verbs_submit(qp, last_wqe_idx + 1);
        xferReqRing[pos].last_wqe = last_wqe_idx;
        doca_gpu_dev_verbs_unlock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &xferReqRing[pos].qp_progress->data_producer_lock);
        nixl_gpunetio_dev_publish_data(&xferReqRing[pos], progress_state, pos);
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
kernel_write(doca_gpu_dev_verbs_qp *qp,
             struct docaXferReqGpu *xferReqRing,
             docaProgressState *progress_state,
             uint32_t *exit_flag,
             uint32_t pos) {
    nixl_gpunetio_dev_write_chunk(qp, xferReqRing, progress_state, exit_flag, pos);
}

__global__ void
kernel_write_striped(struct docaXferReqGpu *xferReqRing,
                     docaProgressState *progress_state,
                     uint32_t *exit_flag,
                     docaStripedWriteLaunch launch) {
    if (xferReqRing == nullptr) {
        return;
    }
    __shared__ bool final_owner;
    const uint32_t final_pos = launch.positions[launch.count - 1];

    for (uint32_t ordinal = blockIdx.x; ordinal + 1 < launch.count; ordinal += gridDim.x) {
        const uint32_t pos = launch.positions[ordinal];
        nixl_gpunetio_dev_write_chunk(
            xferReqRing[pos].qp_data, xferReqRing, progress_state, exit_flag, pos);
        // Reuse the helper's CTA-local state only after the posting lane finishes.
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        final_owner = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                          xferReqRing[final_pos].posting_ctas_done)
                          .fetch_add(1U, cuda::std::memory_order_acq_rel) == gridDim.x - 1;
    }
    __syncthreads();
    if (final_owner) {
        // The last arrival acquired every CTA's completed posting publications.
        nixl_gpunetio_dev_write_chunk(
            xferReqRing[final_pos].qp_data, xferReqRing, progress_state, exit_flag, final_pos);
    }
}

__global__ void
kernel_progress(struct docaXferReqGpu *xfer_req_ring,
                struct docaProgressState *progress_state,
                struct docaNotif *notif_fill,
                struct docaNotif *notif_progress,
                struct docaNotif *notif_send_gpu,
                uint32_t *exit_flag) {
    if (xfer_req_ring == nullptr) {
        return;
    }

    // Only the data scheduler uses a warp; notification blocks retain one worker.
    if (blockIdx.x != 0 && threadIdx.x != 0) {
        return;
    }

    doca_gpu_dev_verbs_ticket_t out_ticket;

    // Data QPs progress independently; notification SEND remains synchronous.
    if (blockIdx.x == 0) {
        static_assert(DOCA_XFER_REQ_MAX == 32, "One warp owns the data request slots");
        const uint32_t pos = threadIdx.x;
        while (__all_sync(0xffffffffU, nixl_gpunetio_dev_load_host_state(*exit_flag) == 0U)) {
            uint32_t active = 0;
            if (pos == 0) {
                active = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                             progress_state->active_bitmap)
                             .load(cuda::std::memory_order_acquire);
            }
            active = __shfl_sync(0xffffffffU, active, 0);
            docaXferReqGpu *request = &xfer_req_ring[pos];
            bool send_notification = false;
            if ((active & (1U << pos)) != 0U) {
                const uint32_t request_state = nixl_gpunetio_dev_load_host_state(request->state);

                if (request_state == DOCA_XFER_STATE_DATA_POSTED) {
                    const int poll_status = nixl_gpunetio_dev_poll_one_cq_at<
                        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                        DOCA_GPUNETIO_VERBS_QP_SQ>(
                        doca_gpu_dev_verbs_qp_get_cq_sq(request->qp_data), request->last_wqe);
                    if (poll_status == 0) {
                        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(request->data_state)
                            .store(DOCA_XFER_DATA_COMPLETE, cuda::std::memory_order_release);
                        if (DOCA_GPUNETIO_VOLATILE(request->has_notif_msg_idx) != DOCA_NOTIF_NULL) {
                            send_notification = true;
                        } else {
                            nixl_gpunetio_dev_complete_request(request, progress_state, pos);
                        }
                    } else if (poll_status != EBUSY) {
                        nixl_gpunetio_dev_fail_request(progress_state, pos);
                    }
                }
            }

            // Keep the baseline's synchronous notification and shared-QP ordering.
            uint32_t senders = __ballot_sync(0xffffffffU, send_notification);
            while (senders != 0U) {
                if (pos == static_cast<uint32_t>(__ffs(senders) - 1)) {
#if ENABLE_DEBUG == 1
                    printf("Notif after completion at %d id %d sz %d\n",
                           pos,
                           DOCA_GPUNETIO_VOLATILE(request->has_notif_msg_idx),
                           (int)request->msg_sz);
#endif
                    doca_gpu_dev_verbs_send(
                        request->qp_notif,
                        doca_gpu_dev_verbs_addr{.addr = (uint64_t)(request->lbuf_notif),
                                                .key = request->lkey_notif},
                        request->msg_sz,
                        &out_ticket);
                    int notif_status;
                    do {
                        notif_status = nixl_gpunetio_dev_poll_one_cq_at<
                            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                            DOCA_GPUNETIO_VERBS_QP_SQ>(
                            doca_gpu_dev_verbs_qp_get_cq_sq(request->qp_notif), out_ticket);
                    } while (notif_status == EBUSY && DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0 &&
                             nixl_gpunetio_dev_load_host_state(progress_state->failed) == 0U);
                    if (notif_status != 0) {
                        nixl_gpunetio_dev_mark_failed(progress_state);
                        printf("kernel_progress: notification CQ error %d wqe %ld\n",
                               notif_status,
                               out_ticket);
                        nixl_gpunetio_dev_terminal_error(progress_state, pos);
                    } else {
                        nixl_gpunetio_dev_complete_request(request, progress_state, pos);
                    }
#if ENABLE_DEBUG == 1
                    printf("Notif correctly sent %ld\n", out_ticket);
#endif
                }
                senders &= senders - 1;
            }
        }
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
                        nixl_gpunetio_dev_mark_failed(progress_state);
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
                } while (notif_status == EBUSY && DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0 &&
                         nixl_gpunetio_dev_load_host_state(progress_state->failed) == 0U);
                if (notif_status != 0) {
                    nixl_gpunetio_dev_mark_failed(progress_state);
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
doca_kernel_write_striped(cudaStream_t stream,
                          struct docaXferReqGpu *xferReqRing,
                          struct docaProgressState *progress_state,
                          uint32_t *exit_flag,
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
    kernel_write_striped<<<cta_count, DOCA_XFER_REQ_SIZE, 0, stream>>>(
        xferReqRing, progress_state, exit_flag, launch);
    result = cudaGetLastError();
    return result == cudaSuccess ? DOCA_SUCCESS : DOCA_ERROR_BAD_STATE;
}

doca_error_t
doca_kernel_read(cudaStream_t stream,
                 doca_gpu_dev_verbs_qp *qp,
                 struct docaXferReqGpu *xferReqRing,
                 struct docaProgressState *progress_state,
                 uint32_t *exit_flag,
                 uint32_t pos,
                 bool wait_for_completion) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_read<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(
        qp, xferReqRing, progress_state, exit_flag, pos, wait_for_completion);
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

    kernel_progress<<<3, 32, 0, stream>>>(
        xfer_req_ring, progress_state, notif_fill, notif_progress, notif_send_gpu, exit_flag);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(
            stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
