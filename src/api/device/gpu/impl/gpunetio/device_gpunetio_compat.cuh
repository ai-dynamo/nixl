/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH

#include <gpu/impl/gpunetio/device_gpunetio_types.h>

#if defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31)
#error "GPUNETIO native Device API requires the public DOCA 3.5+ profile"
#endif
#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) && NIXL_ENABLE_GPUNETIO_DEVICE_API
#if !defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
#error "SDK_CONTRACT_UNVERIFIED: select the public DOCA 3.5+ profile"
#endif
#define NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE 1
#include <doca_version.h>
#if DOCA_VERSION_MAJOR < 3 || (DOCA_VERSION_MAJOR == 3 && DOCA_VERSION_MINOR < 5)
#error "GPUNETIO native Device API requires DOCA 3.5 or newer"
#endif
#include <cerrno>
#include <doca_gpunetio_dev_verbs_cq.cuh>
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#include <doca_gpunetio_dev_verbs_qp.cuh>
#else
#define NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE 0
#endif

namespace nixl::gpu::impl::gpunetio::compat {
#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
static_assert(sizeof(doca_gpu_dev_verbs_ticket_t) == sizeof(uint64_t));

__device__ inline uint64_t
ticketToBits(doca_gpu_dev_verbs_ticket_t ticket) {
    uint64_t bits;
    __builtin_memcpy(&bits, &ticket, sizeof(bits));
    return bits;
}

__device__ inline doca_gpu_dev_verbs_ticket_t
ticketFromBits(uint64_t bits) {
    doca_gpu_dev_verbs_ticket_t ticket;
    __builtin_memcpy(&ticket, &bits, sizeof(ticket));
    return ticket;
}

__device__ inline void
releaseFence() {
    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
}

__device__ inline bool
usesGpuSmDoorbell(const doca_gpu_dev_verbs_qp *qp) {
    return static_cast<enum doca_gpu_dev_verbs_nic_handler>(__ldg(reinterpret_cast<const int *>(
               &qp->nic_handler))) == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
}

__device__ inline void
put(doca_gpu_dev_verbs_qp *qp,
    uint64_t remote_address,
    uint32_t remote_key,
    uint64_t local_address,
    uint32_t local_key,
    size_t bytes,
    uint64_t *ticket_bits) {
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                           DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                           DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>(
        qp,
        doca_gpu_dev_verbs_addr{remote_address, remote_key},
        doca_gpu_dev_verbs_addr{local_address, local_key},
        bytes,
        &ticket);
    *ticket_bits = ticketToBits(ticket);
}

__device__ inline int
pollOne(doca_gpu_dev_verbs_qp *qp, uint64_t ticket_bits) {
    return doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                             DOCA_GPUNETIO_VERBS_QP_SQ>(
        qp, ticketFromBits(ticket_bits));
}
#endif
} // namespace nixl::gpu::impl::gpunetio::compat
#endif
