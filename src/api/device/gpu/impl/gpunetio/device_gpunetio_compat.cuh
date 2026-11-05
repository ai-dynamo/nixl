/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH

#include <gpu/impl/gpunetio/device_gpunetio_types.h>

// The direct device API is enabled explicitly and limited to the tested SDK.
#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) && NIXL_ENABLE_GPUNETIO_DEVICE_API
#define NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE 1
#include <cerrno>
#include <doca_gpunetio_dev_verbs_cq.cuh>
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#include <doca_version.h>
#if DOCA_VERSION_MAJOR != 3 || DOCA_VERSION_MINOR != 5
#error \
    "GPUNETIO device API profile requires DOCA 3.5; revalidate CQ and GET semantics for other versions"
#endif
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

// This internal SDK helper does not advance the CQ; independent CTA threads
// check every CQE before the leader advances the consumer index.
template<enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
             DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
         enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ int
nixl_gpunetio_dev_priv_poll_one_cq_at(doca_gpu_dev_verbs_cq *cq, uint64_t index) {
    return doca_priv_gpu_dev_verbs_poll_one_cq_device_at<resource_sharing_mode, qp_type>(cq, index);
}

__device__ inline bool
needsDump(const doca_gpu_dev_verbs_qp *qp) {
    return qp->need_mcst;
}

__device__ inline int
pollOne(doca_gpu_dev_verbs_qp *qp, uint64_t ticket_bits) {
    const auto ticket = ticketFromBits(ticket_bits);
    return doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                             DOCA_GPUNETIO_VERBS_QP_SQ>(qp, ticket);
}

#endif // NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE

} // namespace nixl::gpu::impl::gpunetio::compat

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH
