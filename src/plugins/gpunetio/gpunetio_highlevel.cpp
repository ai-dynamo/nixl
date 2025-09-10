/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <doca_log.h>
#include <doca_error.h>
#include <doca_uar.h>
#include <doca_umem.h>

#include "gpunetio_backend_aux.h"

#define VERBS_TEST_MAX_SEND_SEGS (1)
#define VERBS_TEST_MAX_RECEIVE_SEGS (1)
#define VERBS_TEST_DBR_SIZE (8)
#define ROUND_UP(unaligned_mapping_size, align_val) \
    ((unaligned_mapping_size) + (align_val)-1) & (~((align_val)-1))

DOCA_LOG_REGISTER(GPUVERBS::HIGH_LEVEL);

static uint32_t
align_up_uint32(uint32_t value, uint32_t alignment) {
    uint64_t remainder = (value % alignment);
    if (remainder == 0) return value;
    return (uint32_t)(value + (alignment - remainder));
}

static uint64_t
next_power_of_two(uint64_t x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

static size_t
get_page_size(void) {
    long ret = sysconf(_SC_PAGESIZE);
    if (ret == -1) return 4096; // 4KB, default Linux page size

    return (size_t)ret;
}

static doca_error_t
create_uar(doca_dev *dev,
           doca_gpu_dev_verbs_nic_handler nic_handler,
           doca_uar **external_uar,
           bool bf_supported) {
    doca_error_t status = DOCA_SUCCESS;

    if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to doca_uar_create NC DEDICATED");
#if CUDA_VERSION >= 12020
            status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, external_uar);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to doca_uar_create NC");
            } else {
                DOCA_LOG_ERR("UAR created with DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED");
            }
#endif
        } else
            return DOCA_SUCCESS;
    }

    if (bf_supported &&
        (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF ||
         (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO && status != DOCA_SUCCESS))) {
        status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_BLUEFLAME, external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to doca_uar_create NC");
            return status;
        }
    } else
        return DOCA_ERROR_DRIVER;

    return status;
}

static uint32_t
calc_cq_external_umem_size(uint32_t queue_size) {
    uint32_t cqe_buf_size = 0;

    if (queue_size != 0) cqe_buf_size = (uint32_t)(queue_size * sizeof(struct mlx5_cqe64));

    return align_up_uint32(cqe_buf_size + VERBS_TEST_DBR_SIZE, get_page_size());
}

static void
mlx5_init_cqes(struct mlx5_cqe64 *cqes, uint32_t nb_cqes) {
    for (uint32_t cqe_idx = 0; cqe_idx < nb_cqes; cqe_idx++)
        cqes[cqe_idx].op_own =
            (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) | MLX5_CQE_OWNER_MASK;
}

static doca_error_t
create_cq(doca_gpu *gpu_dev,
          doca_dev *dev,
          doca_verbs_context *verbs_ctx,
          uint32_t ncqes,
          void **gpu_umem_dev_ptr,
          doca_umem **gpu_umem,
          doca_uar *external_uar,
          doca_verbs_cq **verbs_cq) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    cudaError_t status_cuda = cudaSuccess;
    doca_verbs_cq_attr *verbs_cq_attr = NULL;
    doca_verbs_cq *new_cq = NULL;
    struct mlx5_cqe64 *cq_ring_haddr = NULL;
    uint32_t external_umem_size = 0;

    status = doca_verbs_cq_attr_create(&verbs_cq_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs cq attributes");
        return status;
    }

    status = doca_verbs_cq_attr_set_external_datapath_en(verbs_cq_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs cq external datapath en: %s",
                     doca_error_get_descr(status));
        goto destroy_resources;
    }

    external_umem_size = calc_cq_external_umem_size(ncqes);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to calc external umem size");
        goto destroy_resources;
    }

    status = doca_gpu_mem_alloc(gpu_dev,
                                external_umem_size,
                                get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU,
                                (void **)gpu_umem_dev_ptr,
                                NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to alloc gpu memory for external umem cq");
        goto destroy_resources;
    }

    cq_ring_haddr = (struct mlx5_cqe64 *)(calloc(external_umem_size, sizeof(uint8_t)));
    if (cq_ring_haddr == NULL) {
        DOCA_LOG_ERR("Failed to allocate cq host ring buffer memory for initialization");
        status = DOCA_ERROR_NO_MEMORY;
        goto destroy_resources;
    }

    mlx5_init_cqes(cq_ring_haddr, ncqes);

    DOCA_LOG_DBG("Create CQ memcpy cq_ring_haddr %p into gpu_umem_dev_ptr %p size %d\n",
                 (void *)(cq_ring_haddr),
                 (*gpu_umem_dev_ptr),
                 external_umem_size);

    status_cuda = cudaMemcpy(
        (*gpu_umem_dev_ptr), (void *)(cq_ring_haddr), external_umem_size, cudaMemcpyDefault);
    if (status_cuda != cudaSuccess) {
        DOCA_LOG_ERR("Failed to cudaMempy gpu cq cq ring buffer ret %d", status_cuda);
        goto destroy_resources;
    }

    free(cq_ring_haddr);
    cq_ring_haddr = NULL;

    status = doca_umem_gpu_create(gpu_dev,
                                  dev,
                                  (*gpu_umem_dev_ptr),
                                  external_umem_size,
                                  DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                  gpu_umem);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create gpu umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_set_external_umem(verbs_cq_attr, *gpu_umem, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs cq external umem");
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_set_cq_size(verbs_cq_attr, ncqes);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs cq size");
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_set_cq_overrun(verbs_cq_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs cq size");
        goto destroy_resources;
    }

    if (external_uar != NULL) {
        status = doca_verbs_cq_attr_set_external_uar(verbs_cq_attr, external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set doca verbs cq external uar");
            goto destroy_resources;
        }
    }

    status = doca_verbs_cq_create(verbs_ctx, verbs_cq_attr, &new_cq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs cq");
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_destroy(verbs_cq_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca verbs cq attributes");
        goto destroy_resources;
    }

    *verbs_cq = new_cq;

    return DOCA_SUCCESS;

destroy_resources:
    if (new_cq != NULL) {
        tmp_status = doca_verbs_cq_destroy(new_cq);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs cq");
    }

    if (verbs_cq_attr != NULL) {
        tmp_status = doca_verbs_cq_attr_destroy(verbs_cq_attr);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs cq attributes");
    }

    if (*gpu_umem != NULL) {
        tmp_status = doca_umem_destroy(*gpu_umem);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy gpu ring buffer umem");
    }

    if (cq_ring_haddr) {
        free(cq_ring_haddr);
    }

    if ((*gpu_umem_dev_ptr) != 0) {
        tmp_status = doca_gpu_mem_free(gpu_dev, (*gpu_umem_dev_ptr));
        if (tmp_status != DOCA_SUCCESS)
            DOCA_LOG_ERR("Failed to destroy gpu memory of cq umem buffer");
    }

    return status;
}

static uint32_t
calc_qp_external_umem_size(uint32_t rq_nwqes, uint32_t sq_nwqes) {
    uint32_t rq_ring_size = 0;
    uint32_t sq_ring_size = 0;

    if (rq_nwqes != 0) rq_ring_size = (uint32_t)(rq_nwqes * sizeof(struct mlx5_wqe_data_seg));
    if (sq_nwqes != 0) sq_ring_size = (uint32_t)(sq_nwqes * sizeof(doca_gpu_dev_verbs_wqe));

    return align_up_uint32(rq_ring_size + sq_ring_size, get_page_size());
}

static doca_error_t
create_qp(doca_gpu *gpu_dev,
          doca_dev *dev,
          doca_verbs_context *verbs_ctx,
          doca_verbs_pd *verbs_pd,
          doca_verbs_cq *cq_sq,
          uint32_t sq_nwqe,
          doca_verbs_cq *cq_rq,
          uint32_t rq_nwqe,
          void **gpu_umem_dev_ptr,
          doca_umem **gpu_umem,
          void **gpu_umem_dbr_dev_ptr,
          doca_umem **gpu_umem_dbr,
          doca_uar *external_uar,
          doca_gpu_dev_verbs_nic_handler nic_handler,
          bool set_core_direct,
          doca_verbs_qp **verbs_qp) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    doca_verbs_qp_init_attr *verbs_qp_init_attr = NULL;
    doca_verbs_qp *new_qp = NULL;
    uint32_t external_umem_size = 0;
    size_t dbr_umem_align_sz = ROUND_UP(VERBS_TEST_DBR_SIZE, get_page_size());

    status = doca_verbs_qp_init_attr_create(&verbs_qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs qp attributes");
        return status;
    }

    status = doca_verbs_qp_init_attr_set_external_datapath_en(verbs_qp_init_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs external datapath en: %s",
                     doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_uar(verbs_qp_init_attr, external_uar);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set receive_max_sges");
        goto destroy_resources;
    }

    external_umem_size = calc_qp_external_umem_size(rq_nwqe, sq_nwqe);

    status = doca_gpu_mem_alloc(gpu_dev,
                                external_umem_size,
                                get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU,
                                (void **)gpu_umem_dev_ptr,
                                NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to alloc gpu memory for external umem qp");
        goto destroy_resources;
    }

    status = doca_umem_gpu_create(gpu_dev,
                                  dev,
                                  (*gpu_umem_dev_ptr),
                                  external_umem_size,
                                  DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                  gpu_umem);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create gpu umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_umem(verbs_qp_init_attr, *gpu_umem, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs qp external umem");
        goto destroy_resources;
    }

    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        *gpu_umem_dbr_dev_ptr = calloc(dbr_umem_align_sz, sizeof(uint8_t));
        if (*gpu_umem_dbr_dev_ptr == NULL) {
            DOCA_LOG_ERR("Failed to alloc gpu memory for external umem qp");
            goto destroy_resources;
        }
    } else {
        status = doca_gpu_mem_alloc(gpu_dev,
                                    dbr_umem_align_sz,
                                    get_page_size(),
                                    DOCA_GPU_MEM_TYPE_GPU,
                                    (void **)gpu_umem_dbr_dev_ptr,
                                    NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to alloc gpu memory for external umem qp");
            goto destroy_resources;
        }
    }

    status = doca_umem_gpu_create(gpu_dev,
                                  dev,
                                  (*gpu_umem_dbr_dev_ptr),
                                  dbr_umem_align_sz,
                                  DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                  gpu_umem_dbr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create gpu umem: %s", doca_error_get_descr(status));
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_dbr_umem(verbs_qp_init_attr, *gpu_umem_dbr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs qp external dbr umem");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_pd(verbs_qp_init_attr, verbs_pd);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs PD");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_sq_wr(verbs_qp_init_attr, sq_nwqe);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set SQ size");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_rq_wr(verbs_qp_init_attr, rq_nwqe);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set RQ size");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_qp_type(verbs_qp_init_attr, DOCA_VERBS_QP_TYPE_RC);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set QP type");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_send_cq(verbs_qp_init_attr, cq_sq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs CQ");
        goto destroy_resources;
    }

    status =
        doca_verbs_qp_init_attr_set_send_max_sges(verbs_qp_init_attr, VERBS_TEST_MAX_SEND_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set send_max_sges");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_receive_max_sges(verbs_qp_init_attr,
                                                          VERBS_TEST_MAX_RECEIVE_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set receive_max_sges");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_receive_cq(verbs_qp_init_attr, cq_rq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca verbs CQ");
        goto destroy_resources;
    }

    if (set_core_direct) doca_verbs_qp_init_attr_set_core_direct_master(verbs_qp_init_attr, 1);

    status = doca_verbs_qp_create(verbs_ctx, verbs_qp_init_attr, &new_qp);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs QP");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy doca verbs QP attributes");
        goto destroy_resources;
    }

    *verbs_qp = new_qp;

    // Immediately close dmabuf_fd after registration.
    // if (dmabuf_fd > 0) close(dmabuf_fd);

    return DOCA_SUCCESS;

destroy_resources:
    if (new_qp != NULL) {
        tmp_status = doca_verbs_qp_destroy(new_qp);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs QP");
    }

    if (verbs_qp_init_attr != NULL) {
        tmp_status = doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs QP attributes");
    }

    if (*gpu_umem != NULL) {
        tmp_status = doca_umem_destroy(*gpu_umem);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy gpu umem");
    }

    if ((*gpu_umem_dev_ptr) != 0) {
        tmp_status = doca_gpu_mem_free(gpu_dev, (*gpu_umem_dev_ptr));
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy gpu memory of umem");
    }

    return status;
}

doca_error_t
doca_gpu_verbs_create_qp_hl(doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                            doca_gpu_verbs_qp_hl **qp) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;

    if (qp_init_attr == NULL || qp == NULL) {
        DOCA_LOG_ERR("Invalid input value");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (qp_init_attr->gpu_dev == NULL || qp_init_attr->dev == NULL ||
        qp_init_attr->verbs_context == NULL ||
        (qp_init_attr->sq_nwqe == 0 && qp_init_attr->rq_nwqe == 0)) {
        DOCA_LOG_ERR("Invalid input value: gpu_dev %p sq_nwqe %d rq_nwqe %d",
                     (void *)qp_init_attr->gpu_dev,
                     qp_init_attr->sq_nwqe,
                     qp_init_attr->rq_nwqe);
        return DOCA_ERROR_INVALID_VALUE;
    }

    doca_gpu_verbs_qp_hl *qp_ =
        (doca_gpu_verbs_qp_hl *)calloc(1, sizeof(doca_gpu_verbs_qp_hl));
    if (qp_ == NULL) {
        DOCA_LOG_ERR("Failed alloc memory for high-level qp");
        return DOCA_ERROR_NO_MEMORY;
    }

    qp_->gpu_dev = qp_init_attr->gpu_dev;

    if (qp_init_attr->sq_nwqe > 0) {
        qp_init_attr->sq_nwqe = (uint32_t)next_power_of_two(qp_init_attr->sq_nwqe);
        status = create_cq(qp_->gpu_dev,
                           qp_init_attr->dev,
                           qp_init_attr->verbs_context,
                           qp_init_attr->sq_nwqe,
                           &qp_->cq_sq_umem_gpu_ptr,
                           &qp_->cq_sq_umem,
                           NULL,
                           &qp_->cq_sq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    if (qp_init_attr->rq_nwqe > 0) {
        qp_init_attr->rq_nwqe = (uint32_t)next_power_of_two(qp_init_attr->rq_nwqe);
        status = create_cq(qp_->gpu_dev,
                           qp_init_attr->dev,
                           qp_init_attr->verbs_context,
                           qp_init_attr->rq_nwqe,
                           &qp_->cq_rq_umem_gpu_ptr,
                           &qp_->cq_rq_umem,
                           NULL,
                           &qp_->cq_rq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    qp_->nic_handler = qp_init_attr->nic_handler;

    status = create_uar(qp_init_attr->dev, qp_->nic_handler, &qp_->external_uar, true);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs uar: %s", doca_error_get_descr(status));
        goto exit_error;
    }

    status = create_qp(qp_->gpu_dev,
                       qp_init_attr->dev,
                       qp_init_attr->verbs_context,
                       qp_init_attr->verbs_pd,
                       qp_->cq_sq,
                       qp_init_attr->sq_nwqe,
                       qp_->cq_rq,
                       qp_init_attr->rq_nwqe,
                       &qp_->qp_umem_gpu_ptr,
                       &qp_->qp_umem,
                       &qp_->qp_umem_dbr_gpu_ptr,
                       &qp_->qp_umem_dbr,
                       qp_->external_uar,
                       qp_->nic_handler,
                       false,
                       &qp_->qp);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca verbs qp");
        goto exit_error;
    }

    status = doca_gpu_verbs_export_qp(qp_->gpu_dev,
                                      qp_init_attr->dev,
                                      qp_->qp,
                                      qp_->nic_handler,
                                      qp_->qp_umem_gpu_ptr,
                                      qp_->cq_sq,
                                      qp_->cq_rq,
                                      &qp_->qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create GPU verbs QP");
        return status;
    }

    *qp = qp_;

    return DOCA_SUCCESS;

exit_error:
    if (qp_->external_uar != NULL) {
        tmp_status = doca_uar_destroy(qp_->external_uar);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs UAR");
    }
    free(qp_);
    return status;
}

doca_error_t
doca_gpu_verbs_destroy_qp_hl(doca_gpu_verbs_qp_hl *qp) {
    doca_error_t status;

    if (qp == NULL) return DOCA_ERROR_INVALID_VALUE;

    status = doca_gpu_verbs_unexport_qp(qp->gpu_dev, qp->qp_gverbs);
    if (status != DOCA_SUCCESS)
        DOCA_LOG_ERR("Failed to destroy doca gpu thread argument cq memory");

    status = doca_verbs_qp_destroy(qp->qp);
    if (status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs QP");

    if (qp->qp_umem != NULL) {
        status = doca_umem_destroy(qp->qp_umem);
        if (status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy gpu qp umem");
    }

    if (qp->qp_umem_gpu_ptr != 0) {
        status = doca_gpu_mem_free(qp->gpu_dev, qp->qp_umem_gpu_ptr);
        if (status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy gpu memory of qp ring buffer");
    }

    if (qp->cq_rq) {
        status = doca_verbs_cq_destroy(qp->cq_rq);
        if (status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs CQ");

        if (qp->cq_rq_umem != NULL) {
            status = doca_umem_destroy(qp->cq_rq_umem);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu rq cq ring buffer umem");
        }

        if (qp->cq_rq_umem_gpu_ptr != 0) {
            status = doca_gpu_mem_free(qp->gpu_dev, qp->cq_rq_umem_gpu_ptr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu memory of rq cq ring buffer");
        }

        if (qp->cq_rq_umem_dbr != NULL) {
            status = doca_umem_destroy(qp->cq_rq_umem_dbr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu rq cq ring buffer umem");
        }

        if (qp->cq_rq_umem_dbr_gpu_ptr != 0) {
            status = doca_gpu_mem_free(qp->gpu_dev, qp->cq_rq_umem_dbr_gpu_ptr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu memory of rq cq umem dbr buffer");
        }
    }

    if (qp->cq_sq) {
        status = doca_verbs_cq_destroy(qp->cq_sq);
        if (status != DOCA_SUCCESS) DOCA_LOG_ERR("Failed to destroy doca verbs CQ");

        if (qp->cq_sq_umem != NULL) {
            status = doca_umem_destroy(qp->cq_sq_umem);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu sq cq ring buffer umem");
        }

        if (qp->cq_sq_umem_gpu_ptr != 0) {
            status = doca_gpu_mem_free(qp->gpu_dev, qp->cq_sq_umem_gpu_ptr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu memory of sq cq ring buffer");
        }

        if (qp->cq_sq_umem_dbr != NULL) {
            status = doca_umem_destroy(qp->cq_sq_umem_dbr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu sq cq ring buffer umem");
        }

        if (qp->cq_sq_umem_dbr_gpu_ptr != 0) {
            status = doca_gpu_mem_free(qp->gpu_dev, qp->cq_sq_umem_dbr_gpu_ptr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG_ERR("Failed to destroy gpu memory of sq cq umem dbr buffer");
        }
    }

    return DOCA_SUCCESS;
}
