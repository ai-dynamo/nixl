/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH

#include <gpu/device_types.cuh>
#include <gpu/impl/gpunetio/device_gpunetio_compat.cuh>

#include <cstdint>
#include <cuda/atomic>

namespace nixl::gpu::impl::gpunetio {

static_assert(sizeof(nixl_status_t) <= sizeof(int32_t));
static_assert(sizeof(GpunetioStatusPayload) <= 60);
static_assert(xfer_status_payload_size >= 60);

__device__ inline GpunetioStatusPayload *
statusPayload(xferStatusH *xfer_status) {
    return reinterpret_cast<GpunetioStatusPayload *>(xfer_status->storage);
}

__device__ inline const GpunetioStatusPayload *
statusPayload(const xferStatusH *xfer_status) {
    return reinterpret_cast<const GpunetioStatusPayload *>(xfer_status->storage);
}

template<typename T> using device_atomic_ref = cuda::atomic_ref<T, cuda::thread_scope_device>;

template<typename T>
__device__ inline device_atomic_ref<T>
deviceAtomic(T &value) {
    return device_atomic_ref<T>(value);
}

__device__ inline void
failLane(GpunetioStatusPayload *payload) {
    payload->cached_status = NIXL_ERR_BACKEND;
    deviceAtomic(payload->lane->failed).store(1U, cuda::memory_order_relaxed);
    deviceAtomic(payload->lane->phase)
        .store(gpunetio_lane_phase_failed, cuda::memory_order_release);
}

// Only inspect a status after the dispatcher establishes its mode tag.
__device__ inline bool
isStatusInProgress(const xferStatusH &xfer_status) {
    const auto *payload = statusPayload(&xfer_status);
    return payload->initialized != 0 && payload->cached_status == NIXL_IN_PROG;
}

__device__ inline bool
rangeIsValid(uint64_t base, uint64_t length, size_t offset, size_t bytes, uint64_t *address) {
    if (offset > length || bytes > length - offset) {
        return false;
    }
    const uint64_t offset64 = static_cast<uint64_t>(offset);
    const uint64_t bytes64 = static_cast<uint64_t>(bytes);
    if (static_cast<size_t>(offset64) != offset || static_cast<size_t>(bytes64) != bytes) {
        return false;
    }
    if (base > ~uint64_t{0} - offset64) {
        return false;
    }
    *address = base + offset64;
    return *address <= ~uint64_t{0} - bytes64;
}

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
#if !NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    if constexpr (level != level_t::THREAD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    auto *payload = statusPayload(&xfer_status);
    if (payload->initialized == 0 || payload->lane == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (payload->cached_status != NIXL_IN_PROG) {
        return static_cast<nixl_status_t>(payload->cached_status);
    }

    auto *lane = payload->lane;
    if (deviceAtomic(lane->phase).load(cuda::memory_order_acquire) != gpunetio_lane_phase_posted ||
        deviceAtomic(lane->generation).load(cuda::memory_order_relaxed) != payload->generation ||
        deviceAtomic(lane->active_ticket).load(cuda::memory_order_relaxed) != payload->ticket) {
        failLane(payload);
        return NIXL_ERR_BACKEND;
    }

    if (payload->read_pending) {
        const int read_completion = compat::pollOne(lane->qp, payload->ticket - 1);
        if (read_completion == EBUSY) {
            return NIXL_IN_PROG;
        }
        if (read_completion != 0) {
            failLane(payload);
            return NIXL_ERR_BACKEND;
        }
        payload->read_pending = 0;
    }
    const int completion = compat::pollOne(lane->qp, payload->ticket);
    if (completion == EBUSY) {
        return NIXL_IN_PROG;
    }
    if (completion != 0) {
        failLane(payload);
        return NIXL_ERR_BACKEND;
    }

    payload->cached_status = NIXL_SUCCESS;
    deviceAtomic(lane->last_success_generation)
        .store(payload->generation, cuda::memory_order_relaxed);
    // IDLE is the last release: a new submitter may now replace active_ticket.
    deviceAtomic(lane->phase).store(gpunetio_lane_phase_idle, cuda::memory_order_release);
    return NIXL_SUCCESS;
#endif
}

template<level_t level>
__device__ nixl_status_t
get(const memViewElem &local_dst,
    const memViewElem &remote_src,
    size_t bytes,
    const memViewElem &local_dump_scratch,
    xferStatusH *xfer_status) noexcept {
    if constexpr (level != level_t::THREAD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

#if !defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) || !NIXL_ENABLE_GPUNETIO_DEVICE_API
    (void)local_dst;
    (void)remote_src;
    (void)bytes;
    (void)local_dump_scratch;
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    if (xfer_status == nullptr || bytes == 0 ||
        bytes > static_cast<size_t>(DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto *dst_view = static_cast<const GpunetioDeviceView *>(local_dst.mvh);
    const auto *src_view = static_cast<const GpunetioDeviceView *>(remote_src.mvh);
    const auto *dump_view = static_cast<const GpunetioDeviceView *>(local_dump_scratch.mvh);
    if (dst_view == nullptr || src_view == nullptr || dump_view == nullptr ||
        dst_view->abi_version != gpunetio_device_view_abi_version ||
        src_view->abi_version != gpunetio_device_view_abi_version ||
        dump_view->abi_version != gpunetio_device_view_abi_version ||
        dst_view->role != gpunetio_device_view_role_local ||
        dump_view->role != gpunetio_device_view_role_local ||
        src_view->role != gpunetio_device_view_role_remote || dst_view->context == nullptr ||
        dst_view->context != src_view->context || dst_view->context != dump_view->context ||
        dst_view->context_cookie == 0 || dst_view->context_cookie != src_view->context_cookie ||
        dst_view->context_cookie != dump_view->context_cookie ||
        dst_view->execution_gpu != src_view->execution_gpu ||
        dst_view->execution_gpu != dump_view->execution_gpu || dst_view->context->lane == nullptr ||
        dst_view->elems == nullptr || src_view->elems == nullptr || dump_view->elems == nullptr ||
        local_dst.index >= dst_view->count || remote_src.index >= src_view->count ||
        local_dump_scratch.index >= dump_view->count) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &dst = dst_view->elems[local_dst.index];
    const auto &src = src_view->elems[remote_src.index];
    const auto &dump = dump_view->elems[local_dump_scratch.index];
    uint64_t dst_address, src_address, dump_address;
    if (dst.valid == 0 || src.valid == 0 || dump.valid == 0 ||
        dst.peer_slot != gpunetio_local_peer_slot || dump.peer_slot != gpunetio_local_peer_slot ||
        src.peer_slot != gpunetio_remote_peer_slot ||
        !rangeIsValid(dst.base, dst.length, local_dst.offset, bytes, &dst_address) ||
        !rangeIsValid(src.base, src.length, remote_src.offset, bytes, &src_address) ||
        !rangeIsValid(dump.base, dump.length, local_dump_scratch.offset, 1, &dump_address) ||
        (dump_address >= dst_address && dump_address - dst_address < bytes)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *lane = dst_view->context->lane;
    auto phase = deviceAtomic(lane->phase);
    uint32_t observed = gpunetio_lane_phase_idle;
    if (!phase.compare_exchange_strong(observed,
                                       gpunetio_lane_phase_posting,
                                       cuda::memory_order_acq_rel,
                                       cuda::memory_order_acquire)) {
        return observed == gpunetio_lane_phase_failed ||
                deviceAtomic(lane->failed).load(cuda::memory_order_relaxed) != 0 ?
            NIXL_ERR_BACKEND :
            NIXL_ERR_NOT_ALLOWED;
    }
    if (deviceAtomic(lane->failed).load(cuda::memory_order_relaxed) != 0 || lane->qp == nullptr ||
        static_cast<enum doca_gpu_dev_verbs_nic_handler>(__ldg(reinterpret_cast<const int *>(
            &lane->qp->nic_handler))) != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
        deviceAtomic(lane->failed).store(1U, cuda::memory_order_relaxed);
        phase.store(gpunetio_lane_phase_failed, cuda::memory_order_release);
        return NIXL_ERR_BACKEND;
    }

    auto generation_ref = deviceAtomic(lane->generation);
    if (generation_ref.load(cuda::memory_order_relaxed) == ~uint64_t{0}) {
        phase.store(gpunetio_lane_phase_idle, cuda::memory_order_release);
        return NIXL_ERR_BACKEND;
    }
    const uint64_t generation = generation_ref.fetch_add(1U, cuda::memory_order_relaxed) + 1U;
    doca_gpu_dev_verbs_ticket_t ticket{};
    compat::releaseFence();
    if (compat::needsDump(lane->qp)) {
        doca_gpu_dev_verbs_get_thread<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                      DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                      DOCA_GPUNETIO_VERBS_MCST_ENABLED>(lane->qp,
                                                                        {src_address, src.key},
                                                                        {dst_address, dst.key},
                                                                        bytes,
                                                                        {dump_address, dump.key},
                                                                        &ticket);
    } else {
        doca_gpu_dev_verbs_get_thread<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                      DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                      DOCA_GPUNETIO_VERBS_MCST_DISABLED>(lane->qp,
                                                                         {src_address, src.key},
                                                                         {dst_address, dst.key},
                                                                         bytes,
                                                                         {dump_address, dump.key},
                                                                         &ticket);
    }

    const uint64_t ticket_bits = compat::ticketToBits(ticket);
    auto *payload = statusPayload(xfer_status);
    payload->lane = lane;
    payload->generation = generation;
    payload->ticket = ticket_bits;
    payload->cached_status = NIXL_IN_PROG;
    payload->initialized = 1;
    payload->read_pending = compat::needsDump(lane->qp);
    deviceAtomic(lane->active_ticket).store(ticket_bits, cuda::memory_order_relaxed);
    phase.store(gpunetio_lane_phase_posted, cuda::memory_order_release);
    return NIXL_IN_PROG;
#endif
}

__device__ inline void *
getPtr(void *, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::impl::gpunetio

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH
