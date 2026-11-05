/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_BATCH_CUH
#define NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_BATCH_CUH

#include <gpu/nixl_device.cuh>

namespace nixl::gpu {

#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) && NIXL_ENABLE_GPUNETIO_DEVICE_API
/**
 * Experimental split-phase GPU read session. One CTA owns a session at a time.
 * The caller synchronizes the CTA before publishing and before retiring a batch,
 * polls every ticket, and retires batches in FIFO order. A failed or abandoned
 * reserved batch must be failed, not published or reused. Views, registrations,
 * buffers and the session remain alive until all accepted work has retired.
 */
class readSession {
    using View = impl::gpunetio::GpunetioDeviceView;
    using Elem = impl::gpunetio::GpunetioViewElem;
    using Lane = impl::gpunetio::GpunetioDeviceLane;
    View const *local_ = nullptr;
    View const *remote_ = nullptr;
    Lane *lane_ = nullptr;
    doca_gpu_dev_verbs_qp *qp_ = nullptr;
    doca_gpu_dev_verbs_cq *cq_ = nullptr;
    uint64_t generation_ = 0;
    uint64_t next_retire_ = 0;
    uint64_t reserved_first_ = 0;
    uint64_t dump_ticket_ = 0;
    uint32_t capacity_ = 0;
    uint32_t outstanding_ = 0;
    uint32_t reserved_count_ = 0;
    bool acquired_ = false;
    bool reserved_ = false;
    bool consistency_pending_ = false;
    bool dump_pending_ = false;

public:
    readSession() = default;
    readSession(const readSession &) = delete;
    readSession &
    operator=(const readSession &) = delete;

    /** Bind prepared views from the same engine; a failed bind clears the prior binding. */
    __device__ nixl_status_t
    bind(nixlMemViewH local, nixlMemViewH remote) {
        using namespace impl::gpunetio;
        if (acquired_) {
            return NIXL_ERR_INVALID_PARAM;
        }
        local_ = nullptr;
        remote_ = nullptr;
        lane_ = nullptr;
        qp_ = nullptr;
        cq_ = nullptr;
        if (!local || !remote) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const auto *lw = static_cast<const memViewWrapper *>(local);
        const auto *rw = static_cast<const memViewWrapper *>(remote);
        if (lw->execution_mode != exec_mode_t::GPUNETIO_DIRECT ||
            rw->execution_mode != exec_mode_t::GPUNETIO_DIRECT || !lw->backend_memview ||
            !rw->backend_memview) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
        const auto *l = static_cast<const View *>(lw->backend_memview);
        const auto *r = static_cast<const View *>(rw->backend_memview);
        if (l->abi_version != gpunetio_device_view_abi_version ||
            r->abi_version != gpunetio_device_view_abi_version ||
            l->role != gpunetio_device_view_role_local ||
            r->role != gpunetio_device_view_role_remote || !l->context ||
            l->context != r->context || !l->context->lane || !l->context_cookie ||
            l->context_cookie != r->context_cookie || l->execution_gpu != r->execution_gpu ||
            !l->elems || !r->elems) {
            return NIXL_ERR_INVALID_PARAM;
        }
        local_ = l;
        remote_ = r;
        lane_ = l->context->lane;
        return NIXL_SUCCESS;
    }

    __device__ size_t
    sourceLength(size_t index) const {
        return remote_ && index < remote_->count && remote_->elems[index].valid ?
            remote_->elems[index].length :
            0;
    }

    __device__ size_t
    destinationLength(size_t index) const {
        return local_ && index < local_->count && local_->elems[index].valid ?
            local_->elems[index].length :
            0;
    }

    __device__ bool
    needsConsistency() const {
        return qp_ && impl::gpunetio::compat::needsDump(qp_);
    }

    __device__ uint32_t
    capacity() const {
        return capacity_;
    }

    __device__ uint32_t
    sendQueueCapacity() const {
        return qp_ ? qp_->sq_wqe_num : 0;
    }

    /** Claim an idle lane; a failure poisons the lane until engine teardown. */
    __device__ nixl_status_t
    acquire() {
        using namespace impl::gpunetio;
        if (!lane_ || acquired_) {
            return NIXL_ERR_INVALID_PARAM;
        }
        auto phase = deviceAtomic(lane_->phase);
        uint32_t old = gpunetio_lane_phase_idle;
        if (!phase.compare_exchange_strong(old,
                                           gpunetio_lane_phase_posting,
                                           cuda::memory_order_acq_rel,
                                           cuda::memory_order_acquire)) {
            return old == gpunetio_lane_phase_failed ? NIXL_ERR_BACKEND : NIXL_ERR_NOT_ALLOWED;
        }
        qp_ = lane_->qp;
        if (deviceAtomic(lane_->failed).load(cuda::memory_order_relaxed) || !qp_ ||
            !compat::usesGpuSmDoorbell(qp_)) {
            deviceAtomic(lane_->failed).store(1, cuda::memory_order_relaxed);
            phase.store(gpunetio_lane_phase_failed, cuda::memory_order_release);
            return NIXL_ERR_BACKEND;
        }
        cq_ = doca_gpu_dev_verbs_qp_get_cq_sq(qp_);
        if (!cq_ || qp_->sq_wqe_num < 2 || cq_->cqe_num < 2 ||
            deviceAtomic(lane_->generation).load(cuda::memory_order_relaxed) == ~uint64_t{0}) {
            phase.store(gpunetio_lane_phase_idle, cuda::memory_order_release);
            return NIXL_ERR_INVALID_PARAM;
        }
        capacity_ = qp_->sq_wqe_num < cq_->cqe_num ? qp_->sq_wqe_num : cq_->cqe_num;
        generation_ = deviceAtomic(lane_->generation).fetch_add(1, cuda::memory_order_relaxed) + 1;
        outstanding_ = 0;
        reserved_ = false;
        acquired_ = true;
        consistency_pending_ = false;
        dump_pending_ = false;
        return NIXL_SUCCESS;
    }

    /** Reserve at most capacity-1 entries. No concurrent reservation in this session. */
    __device__ nixl_status_t
    reserve(uint32_t count, uint64_t &first) {
        if (!acquired_ || reserved_ || dump_pending_ || !count || count >= capacity_ ||
            (needsConsistency() && outstanding_) || outstanding_ >= capacity_ - count) {
            return NIXL_ERR_INVALID_PARAM;
        }
        first = doca_gpu_dev_verbs_reserve_wq_slots<>(qp_, count);
        if (!outstanding_) {
            next_retire_ = first;
        }
        outstanding_ += count;
        reserved_ = true;
        reserved_first_ = first;
        reserved_count_ = count;
        return NIXL_SUCCESS;
    }

    /** Prepare one read in a previously reserved slot; CTA threads may call in parallel. */
    __device__ nixl_status_t
    prepare(uint64_t ticket,
            size_t remote_index,
            size_t remote_offset,
            size_t local_index,
            size_t local_offset,
            size_t bytes) {
        using namespace impl::gpunetio;
        if (!acquired_ || !reserved_ || ticket - reserved_first_ >= reserved_count_ ||
            remote_index >= remote_->count || local_index >= local_->count || !bytes ||
            bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const Elem &r = remote_->elems[remote_index], &l = local_->elems[local_index];
        uint64_t ra, la;
        if (!r.valid || !l.valid || r.peer_slot != gpunetio_remote_peer_slot ||
            l.peer_slot != gpunetio_local_peer_slot ||
            !rangeIsValid(r.base, r.length, remote_offset, bytes, &ra) ||
            !rangeIsValid(l.base, l.length, local_offset, bytes, &la)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        doca_gpu_dev_verbs_wqe_prepare_read(qp_,
                                            doca_gpu_dev_verbs_get_wqe_ptr(qp_, ticket),
                                            ticket,
                                            DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                                            ra,
                                            r.key,
                                            la,
                                            l.key,
                                            bytes);
        return NIXL_SUCCESS;
    }

    /** Caller must synchronize all WQE builders before publishing. */
    __device__ nixl_status_t
    publish(uint64_t first, uint32_t count) {
        if (!acquired_ || !reserved_ || first != reserved_first_ || count != reserved_count_) {
            return NIXL_ERR_INVALID_PARAM;
        }
        doca_gpu_dev_verbs_mark_wqes_ready<>(qp_, first, first + count - 1);
        doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                  DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                  DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp_, first + count);
        reserved_ = false;
        reserved_count_ = 0;
        if (needsConsistency()) {
            consistency_pending_ = true;
        }
        return NIXL_IN_PROG;
    }

    /** Post a registered, disjoint scratch byte to make prior READ data visible. */
    __device__ nixl_status_t
    postConsistency(size_t local_index, size_t offset, uint64_t &ticket) {
        using namespace impl::gpunetio;
        if (!acquired_ || reserved_ || outstanding_ || dump_pending_ || !consistency_pending_ ||
            local_index >= local_->count) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const Elem &scratch = local_->elems[local_index];
        uint64_t address;
        if (!scratch.valid || scratch.peer_slot != gpunetio_local_peer_slot ||
            !rangeIsValid(scratch.base, scratch.length, offset, 1, &address)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        ticket = doca_gpu_dev_verbs_reserve_wq_slots<>(qp_, 1);
        doca_gpu_dev_verbs_wqe_prepare_dump(qp_,
                                            doca_gpu_dev_verbs_get_wqe_ptr(qp_, ticket),
                                            ticket,
                                            DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                                            address,
                                            scratch.key,
                                            1);
        doca_gpu_dev_verbs_mark_wqes_ready<>(qp_, ticket, ticket);
        doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                  DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                  DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp_, ticket + 1);
        dump_ticket_ = ticket;
        next_retire_ = ticket;
        outstanding_ = 1;
        dump_pending_ = true;
        return NIXL_IN_PROG;
    }

    /** Poll without advancing the CQ; individual CTA threads may poll in parallel. */
    __device__ nixl_status_t
    poll(uint64_t ticket) const {
        if (!acquired_ ||
            ticket - next_retire_ >= outstanding_ - (reserved_ ? reserved_count_ : 0)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const int rc = impl::gpunetio::compat::nixl_gpunetio_dev_priv_poll_one_cq_at<>(cq_, ticket);
        if (rc == 0) {
            return NIXL_SUCCESS;
        }
        if (rc == EBUSY) {
            return NIXL_IN_PROG;
        }
        impl::gpunetio::deviceAtomic(lane_->failed).store(1, cuda::memory_order_relaxed);
        return NIXL_ERR_BACKEND;
    }

    /** Retire an entirely successful FIFO range after all pollers synchronize. */
    __device__ nixl_status_t
    retire(uint64_t first, uint32_t count) {
        if (acquired_ &&
            impl::gpunetio::deviceAtomic(lane_->failed).load(cuda::memory_order_relaxed)) {
            return NIXL_ERR_BACKEND;
        }
        if (!acquired_ || !count || first != next_retire_ ||
            count > outstanding_ - (reserved_ ? reserved_count_ : 0)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &cq_->cqe_ci, first + count);
        next_retire_ += count;
        outstanding_ -= count;
        if (dump_pending_ && first == dump_ticket_ && count == 1) {
            dump_pending_ = false;
            consistency_pending_ = false;
        }
        return NIXL_SUCCESS;
    }

    /** Finish only after all reads and any required consistency operation retire. */
    __device__ nixl_status_t
    finish() {
        using namespace impl::gpunetio;
        if (acquired_ && deviceAtomic(lane_->failed).load(cuda::memory_order_relaxed)) {
            return NIXL_ERR_BACKEND;
        }
        if (!acquired_ || reserved_ || outstanding_ || consistency_pending_ || dump_pending_) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        deviceAtomic(lane_->last_success_generation).store(generation_, cuda::memory_order_relaxed);
        deviceAtomic(lane_->phase).store(gpunetio_lane_phase_idle, cuda::memory_order_release);
        acquired_ = false;
        return NIXL_SUCCESS;
    }

    /** Preserve the lane after any failure; release is not cancellation. */
    __device__ void
    fail() {
        using namespace impl::gpunetio;
        if (!acquired_) {
            return;
        }
        deviceAtomic(lane_->failed).store(1, cuda::memory_order_relaxed);
        deviceAtomic(lane_->phase).store(gpunetio_lane_phase_failed, cuda::memory_order_release);
        acquired_ = false;
    }
};
#else
class readSession {
public:
    readSession() = default;
    readSession(const readSession &) = delete;
    readSession &
    operator=(const readSession &) = delete;

    __device__ nixl_status_t
    bind(nixlMemViewH, nixlMemViewH) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ size_t
    sourceLength(size_t) const {
        return 0;
    }

    __device__ size_t
    destinationLength(size_t) const {
        return 0;
    }

    __device__ bool
    needsConsistency() const {
        return false;
    }

    __device__ uint32_t
    capacity() const {
        return 0;
    }

    __device__ uint32_t
    sendQueueCapacity() const {
        return 0;
    }

    __device__ nixl_status_t
    acquire() {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ nixl_status_t
    reserve(uint32_t, uint64_t &) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ nixl_status_t
    prepare(uint64_t, size_t, size_t, size_t, size_t, size_t) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ nixl_status_t
    publish(uint64_t, uint32_t) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ nixl_status_t
    postConsistency(size_t, size_t, uint64_t &) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ nixl_status_t
    poll(uint64_t) const {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ nixl_status_t
    retire(uint64_t, uint32_t) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ nixl_status_t
    finish() {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ void
    fail() {}
};
#endif
} // namespace nixl::gpu
#endif
