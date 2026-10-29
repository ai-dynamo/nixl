# Legacy DOCA 3.1 GET reference

Inspection-only C++ reference, not a compilation target or supported current-main
API. The fenced text below preserves `legacy_doca31_get_reference.cuh` byte-for-byte, including its
original copyright and author attribution. SDK/private-view dependencies remain
unresolved; publishing it as documentation does not make it runnable.

```cpp
// SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Author: Yunxiao Ning
//
// Reference-only snapshot of the validated legacy DOCA 3.1 GET primitive.
// This file is not included by Meson and is not a current NIXL public API.
// It deliberately preserves the old internal-view and DUMP_AUTO dependencies
// so the current-main porting boundary is inspectable. Do not build or include
// this header in the default/current public Device API profile.

#ifndef NIXL_DEVICE_API_EXPERIMENTAL_GET_CUH
#define NIXL_DEVICE_API_EXPERIMENTAL_GET_CUH

#include <gpu/nixl_device.cuh>

namespace nixl::gpu::experimental {

// Private single-caller prototype: retire this GET before any next operation on
// the QP. In particular this does not permit mixing READs into the queued-PUT experiment.

// SINGLE-CALLER prototype: do not issue another GET or PUT on this QP until this GET retires.
template<level_t level = level_t::THREAD>
__device__ nixl_status_t
get(const memViewElem &local_dst,
    const memViewElem &remote_src,
    size_t bytes,
    const memViewElem &local_dump_scratch,
    xferStatusH *xfer_status) noexcept {
    if constexpr (level != level_t::THREAD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

// Deliberately exclude the public and MCST profiles; DUMP_AUTO is legacy 3.1 only.
#if !defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) || \
    !defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31) || defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
    (void)local_dst;
    (void)remote_src;
    (void)bytes;
    (void)local_dump_scratch;
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    const nixlDeviceMemViewWrapper *remote_wrapper = nullptr;
    const nixlDeviceMemViewWrapper *local_wrapper = nullptr;
    nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::NONE;
    if (!impl::unwrapViews(remote_src, local_dst, remote_wrapper, local_wrapper, mode) ||
        !impl::prepareStatus(xfer_status, mode)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (mode != nixl_device_exec_mode_t::GPUNETIO_DIRECT) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (impl::gpunetio::isStatusInProgress(*xfer_status)) {
        return NIXL_ERR_REPOST_ACTIVE;
    }

    const nixlDeviceMemViewWrapper *scratch_wrapper = nullptr;
    const nixlDeviceMemViewWrapper *scratch_local_wrapper = nullptr;
    nixl_device_exec_mode_t scratch_mode = nixl_device_exec_mode_t::NONE;
    if (!impl::unwrapViews(
            local_dump_scratch, local_dst, scratch_wrapper, scratch_local_wrapper, scratch_mode) ||
        scratch_mode != mode || bytes == 0 ||
        bytes > static_cast<size_t>(DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto *dst_view =
        static_cast<const impl::gpunetio::GpunetioDeviceView *>(local_wrapper->backend_memview);
    const auto *src_view =
        static_cast<const impl::gpunetio::GpunetioDeviceView *>(remote_wrapper->backend_memview);
    const auto *dump_view =
        static_cast<const impl::gpunetio::GpunetioDeviceView *>(scratch_wrapper->backend_memview);
    if (dst_view == nullptr || src_view == nullptr || dump_view == nullptr ||
        dst_view->abi_version != impl::gpunetio::gpunetio_device_view_abi_version ||
        src_view->abi_version != impl::gpunetio::gpunetio_device_view_abi_version ||
        dump_view->abi_version != impl::gpunetio::gpunetio_device_view_abi_version ||
        dst_view->role != impl::gpunetio::gpunetio_device_view_role_local ||
        dump_view->role != impl::gpunetio::gpunetio_device_view_role_local ||
        src_view->role != impl::gpunetio::gpunetio_device_view_role_remote ||
        dst_view->context == nullptr || dst_view->context != src_view->context ||
        dst_view->context != dump_view->context || dst_view->context_cookie == 0 ||
        dst_view->context_cookie != src_view->context_cookie ||
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
        dst.peer_slot != impl::gpunetio::gpunetio_local_peer_slot ||
        dump.peer_slot != impl::gpunetio::gpunetio_local_peer_slot ||
        src.peer_slot != impl::gpunetio::gpunetio_remote_peer_slot ||
        !impl::gpunetio::rangeIsValid(
            dst.base, dst.length, local_dst.offset, bytes, &dst_address) ||
        !impl::gpunetio::rangeIsValid(
            src.base, src.length, remote_src.offset, bytes, &src_address) ||
        !impl::gpunetio::rangeIsValid(
            dump.base, dump.length, local_dump_scratch.offset, 1, &dump_address) ||
        (dump_address >= dst_address && dump_address - dst_address < bytes)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *lane = dst_view->context->lane;
    auto phase = impl::gpunetio::deviceAtomic(lane->phase);
    uint32_t observed = impl::gpunetio::gpunetio_lane_phase_idle;
    if (!phase.compare_exchange_strong(observed,
                                       impl::gpunetio::gpunetio_lane_phase_posting,
                                       cuda::memory_order_acq_rel,
                                       cuda::memory_order_acquire)) {
        return observed == impl::gpunetio::gpunetio_lane_phase_failed ||
                impl::gpunetio::deviceAtomic(lane->failed).load(cuda::memory_order_relaxed) != 0 ?
            NIXL_ERR_BACKEND :
            NIXL_ERR_NOT_ALLOWED;
    }
    if (impl::gpunetio::deviceAtomic(lane->failed).load(cuda::memory_order_relaxed) != 0 ||
        lane->qp == nullptr ||
        static_cast<enum doca_gpu_dev_verbs_nic_handler>(__ldg(reinterpret_cast<const int *>(
            &lane->qp->nic_handler))) != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
        impl::gpunetio::deviceAtomic(lane->failed).store(1U, cuda::memory_order_relaxed);
        phase.store(impl::gpunetio::gpunetio_lane_phase_failed, cuda::memory_order_release);
        return NIXL_ERR_BACKEND;
    }

    auto generation_ref = impl::gpunetio::deviceAtomic(lane->generation);
    if (generation_ref.load(cuda::memory_order_relaxed) == ~uint64_t{0}) {
        phase.store(impl::gpunetio::gpunetio_lane_phase_idle, cuda::memory_order_release);
        return NIXL_ERR_BACKEND;
    }
    const uint64_t generation = generation_ref.fetch_add(1U, cuda::memory_order_relaxed) + 1U;
    doca_gpu_dev_verbs_ticket_t ticket{};
    impl::gpunetio::compat::releaseFence();
    doca_gpu_dev_verbs_get_thread<DOCA_GPUNETIO_VERBS_DUMP_AUTO,
                                  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                  DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(lane->qp,
                                                                        {src_address, src.key},
                                                                        {dst_address, dst.key},
                                                                        bytes,
                                                                        {dump_address, dump.key},
                                                                        &ticket);

    const uint64_t ticket_bits = impl::gpunetio::compat::ticketToBits(ticket);
    auto *payload = impl::gpunetio::statusPayload(xfer_status);
    payload->lane = lane;
    payload->generation = generation;
    payload->ticket = ticket_bits;
    payload->cached_status = NIXL_IN_PROG;
    payload->initialized = 1;
    impl::gpunetio::deviceAtomic(lane->active_ticket)
        .store(ticket_bits, cuda::memory_order_relaxed);
    phase.store(impl::gpunetio::gpunetio_lane_phase_posted, cuda::memory_order_release);
    impl::setStatusModeTag(*xfer_status, mode);
    return NIXL_IN_PROG;
#endif
}

} // namespace nixl::gpu::experimental

#endif // NIXL_DEVICE_API_EXPERIMENTAL_GET_CUH
```
