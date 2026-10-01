/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COMPLETION_CUH
#define NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COMPLETION_CUH

#include <doca_gpunetio_dev_verbs_cq.cuh>
#include <infiniband/mlx5dv.h>

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

// NIXL creates ordinary GPU-resident 64-byte CQs. Let the SDK account for
// reserved CQEs, owner-bit wrap and completions already consumed by kernel_read.
template<enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
             DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
         enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ inline int
nixl_gpunetio_dev_poll_one_cq_at(doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    doca_gpu_dev_verbs_cqe64 *cqe = nullptr;
    const int status = doca_gpu_dev_verbs_poll_one_cq_device_at<resource_sharing_mode, qp_type>(
        cq, cons_index, &cqe);
    if (status < 0 && cqe != nullptr) {
        nixl_gpunetio_dev_cq_print_cqe_err(reinterpret_cast<mlx5_cqe64 *>(cqe));
    }
    return status;
}

#endif // NIXL_SRC_PLUGINS_GPUNETIO_GPUNETIO_COMPLETION_CUH
