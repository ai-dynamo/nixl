/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH

#include <gpu/impl/gpunetio/device_gpunetio_types.h>

// A device consumer must explicitly opt in and select a profile proven by the
// build probes.  There is deliberately no SDK-version-derived default.
#if defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31) && defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
#error "GPUNetIO device API profiles are mutually exclusive"
#endif

#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) && \
    !defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31) && !defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
#error \
    "SDK_CONTRACT_UNVERIFIED: NIXL_ENABLE_GPUNETIO_DEVICE_API requires an explicitly proven SDK profile"
#endif

#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) && \
    (defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31) || defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC))
#define NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE 1
#include <cerrno>
#include <cstdio>
#include <doca_gpunetio_dev_verbs_cq.cuh>
#include <doca_gpunetio_dev_verbs_onesided.cuh>
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

#if defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31)

// Extracted from foraxe/nixl 292249b8 gpunetio_kernels.cu.  This is the
// verified DOCA 3.1 ordinary-CQ parser, not the installed same-name helper
// that can report success before inspecting the CQE.
__device__ inline void
nixl_gpunetio_dev_cq_print_cqe_err(struct mlx5_cqe64 *cqe64) {
    struct mlx5_err_cqe_ex *err_cqe = (struct mlx5_err_cqe_ex *)cqe64;
    printf("got completion with err: syndrome=%#x, vendor_err_synd=%#x, "
           "hw_err_synd=%#x, hw_synd_type=%#x, wqe_counter=%u wqe_qpn=%x\\n",
           err_cqe->syndrome,
           err_cqe->vendor_err_synd,
           err_cqe->hw_err_synd,
           err_cqe->hw_synd_type,
           err_cqe->wqe_counter,
           err_cqe->s_wqe_opcode_qpn);
}

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

#endif // NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31

#if defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31)
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
#endif

__device__ inline int
pollOne(doca_gpu_dev_verbs_qp *qp, uint64_t ticket_bits) {
    const auto ticket = ticketFromBits(ticket_bits);
#if defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31)
    auto *cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
    return nixl_gpunetio_dev_poll_one_cq_at<>(cq, ticket);
#else
    return doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                             DOCA_GPUNETIO_VERBS_QP_SQ>(qp, ticket);
#endif
}

#endif // NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE

} // namespace nixl::gpu::impl::gpunetio::compat

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_COMPAT_CUH
