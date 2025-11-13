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

#include <unistd.h>

#include "verbs.h"
#include "gpunetio_backend_aux.h"

#define VERBS_TEST_MAX_SEND_SEGS (1)
#define VERBS_TEST_MAX_RECEIVE_SEGS (1)
#define VERBS_TEST_DBR_SIZE (8)
#define ROUND_UP(unaligned_mapping_size, align_val) \
    ((unaligned_mapping_size) + (align_val) - 1) & (~((align_val) - 1))

static uint32_t
align_up_uint32(uint32_t value, uint32_t alignment) {
    uint64_t remainder = (value % alignment);
    if (remainder == 0) return value;
    return (uint32_t)(value + (alignment - remainder));
}

static size_t
get_page_size(void) {
    long ret = sysconf(_SC_PAGESIZE);
    if (ret == -1) return 4096; // 4KB, default Linux page size

    return (size_t)ret;
}

static uint32_t
calc_qp_external_umem_size(uint32_t rq_nwqes, uint32_t sq_nwqes) {
    uint32_t rq_ring_size = 0;
    uint32_t sq_ring_size = 0;

    if (rq_nwqes != 0) rq_ring_size = (uint32_t)(rq_nwqes * sizeof(struct mlx5_wqe_data_seg));
    if (sq_nwqes != 0) sq_ring_size = (uint32_t)(sq_nwqes * sizeof(doca_gpu_dev_verbs_wqe));

    return align_up_uint32(rq_ring_size + sq_ring_size, get_page_size());
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

namespace nixl::doca::verbs {

cq::cq(doca_gpu *gpu_dev_,
       doca_dev *dev_,
       doca_verbs_context *verbs_ctx_,
       doca_verbs_pd *verbs_pd_,
       uint16_t ncqe_)
    : gpu_dev(gpu_dev_),
      dev(dev_),
      verbs_ctx(verbs_ctx_),
      verbs_pd(verbs_pd_),
      ncqe(ncqe_) {
    cq_verbs = createCq();
}

cq::~cq() {
    destroyCq();
}

doca_verbs_cq *
cq::createCq() {
    doca_error_t status = DOCA_SUCCESS;
    cudaError_t status_cuda = cudaSuccess;
    doca_verbs_cq_attr *verbs_cq_attr = nullptr;
    doca_verbs_cq *new_cq = nullptr;
    struct mlx5_cqe64 *cq_ring_haddr = nullptr;
    uint32_t external_umem_size = 0;
    external_uar = nullptr;

    status = doca_verbs_cq_attr_create(&verbs_cq_attr);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create doca verbs cq attributes");

    status = doca_verbs_cq_attr_set_external_datapath_en(verbs_cq_attr, 1);
    if (status != DOCA_SUCCESS) {
        doca_verbs_cq_attr_destroy(verbs_cq_attr);
        throw std::runtime_error("Failed to set doca verbs cq external datapath en");
    }

    external_umem_size = calc_cq_external_umem_size(ncqe);
    if (status != DOCA_SUCCESS) {
        doca_verbs_cq_attr_destroy(verbs_cq_attr);
        throw std::runtime_error("Failed to calc external umem size");
    }

    status = doca_gpu_mem_alloc(gpu_dev,
                                external_umem_size,
                                get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU,
                                (void **)&cq_umem_gpu_ptr,
                                nullptr);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to alloc gpu memory for external umem cq");
    }

    cq_ring_haddr = (struct mlx5_cqe64 *)(calloc(external_umem_size, sizeof(uint8_t)));
    if (cq_ring_haddr == nullptr) {
        destroyCq();
        throw std::runtime_error(
            "Failed to allocate cq host ring buffer memory for initialization");
    }

    mlx5_init_cqes(cq_ring_haddr, ncqe);

    status_cuda =
        cudaMemcpy(cq_umem_gpu_ptr, (void *)(cq_ring_haddr), external_umem_size, cudaMemcpyDefault);
    if (status_cuda != cudaSuccess) {
        destroyCq();
        throw std::runtime_error("Failed to cudaMempy gpu cq cq ring buffer");
    }

    free(cq_ring_haddr);
    cq_ring_haddr = nullptr;

    status = doca_umem_gpu_create(gpu_dev,
                                  dev,
                                  cq_umem_gpu_ptr,
                                  external_umem_size,
                                  DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                  &cq_umem);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to create gpu umem");
    }

    status = doca_verbs_cq_attr_set_external_umem(verbs_cq_attr, cq_umem, 0);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to set doca verbs cq external umem");
    }

    status = doca_verbs_cq_attr_set_cq_size(verbs_cq_attr, ncqe);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to set doca verbs cq size");
    }

    status = doca_verbs_cq_attr_set_cq_overrun(verbs_cq_attr, 1);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to set doca verbs cq size");
    }

    status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &external_uar);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to doca_uar_create NC DEDICATED");
    }

    status = doca_verbs_cq_attr_set_external_uar(verbs_cq_attr, external_uar);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to set doca verbs cq external uar");
    }

    status = doca_verbs_cq_create(verbs_ctx, verbs_cq_attr, &new_cq);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to create doca verbs cq");
    }

    status = doca_verbs_cq_attr_destroy(verbs_cq_attr);
    if (status != DOCA_SUCCESS) {
        destroyCq();
        throw std::runtime_error("Failed to destroy doca verbs cq attributes");
    }

    return new_cq;
}

void
cq::destroyCq() {
    doca_error_t status;

    status = doca_verbs_cq_destroy(cq_verbs);
    if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy doca verbs CQ";

    if (external_uar != nullptr) {
        status = doca_uar_destroy(external_uar);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu external_uar";
    }

    if (cq_umem != nullptr) {
        status = doca_umem_destroy(cq_umem);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu cq_umem";
    }

    if (cq_umem_gpu_ptr != 0) {
        status = doca_gpu_mem_free(gpu_dev, cq_umem_gpu_ptr);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu memory of cq_umem_gpu_ptr";
    }

    if (cq_umem_dbr != nullptr) {
        status = doca_umem_destroy(cq_umem_dbr);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu cq_umem_dbr";
    }

    if (cq_umem_dbr_gpu_ptr != 0) {
        status = doca_gpu_mem_free(gpu_dev, cq_umem_dbr_gpu_ptr);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu cq_umem_dbr_gpu_ptr";
    }
}

qp::qp(doca_gpu *gpu_dev_,
       doca_dev *dev_,
       doca_verbs_context *verbs_ctx_,
       doca_verbs_pd *verbs_pd_,
       uint16_t sq_nwqe_,
       uint16_t rq_nwqe_,
       doca_gpu_dev_verbs_nic_handler nic_handler_)
    : gpu_dev(gpu_dev_),
      dev(dev_),
      verbs_ctx(verbs_ctx_),
      verbs_pd(verbs_pd_),
      sq_nwqe(sq_nwqe_),
      rq_nwqe(rq_nwqe_),
      nic_handler(nic_handler_) {
    doca_error_t status;

    cq_rq = std::make_unique<nixl::doca::verbs::cq>(gpu_dev, dev, verbs_ctx, verbs_pd, rq_nwqe);
    cq_sq = std::make_unique<nixl::doca::verbs::cq>(gpu_dev, dev, verbs_ctx, verbs_pd, sq_nwqe);
    qp_verbs = createQp();

    status = doca_gpu_verbs_export_qp(gpu_dev,
                                      dev,
                                      qp_verbs,
                                      nic_handler,
                                      qp_umem_gpu_ptr,
                                      cq_sq->get_cq(),
                                      cq_rq->get_cq(),
                                      &qp_gverbs);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to create GPU verbs QP");

    status = doca_gpu_verbs_get_qp_dev(qp_gverbs, &qp_gdev_verbs);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to create GPU verbs QP");
}

qp::~qp() {
    doca_error_t status;

    status = doca_gpu_verbs_unexport_qp(gpu_dev, qp_gverbs);
    if (status != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to destroy doca gpu thread argument cq memory";

    destroyQp();
}

doca_verbs_qp *
qp::createQp() {
    doca_error_t status = DOCA_SUCCESS;
    doca_verbs_qp_init_attr *verbs_qp_init_attr = nullptr;
    doca_verbs_qp *new_qp = nullptr;
    uint32_t external_umem_size = 0;
    size_t dbr_umem_align_sz = ROUND_UP(VERBS_TEST_DBR_SIZE, get_page_size());

    status = doca_verbs_qp_init_attr_create(&verbs_qp_init_attr);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create doca verbs qp attributes");

    status = doca_verbs_qp_init_attr_set_external_datapath_en(verbs_qp_init_attr, 1);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to set doca verbs external datapath en");

    status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &external_uar);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to doca_uar_create NC DEDICATED");

    status = doca_verbs_qp_init_attr_set_external_uar(verbs_qp_init_attr, external_uar);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set external_uar");

    external_umem_size = calc_qp_external_umem_size(rq_nwqe, sq_nwqe);

    status = doca_gpu_mem_alloc(gpu_dev,
                                external_umem_size,
                                get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU,
                                (void **)&qp_umem_gpu_ptr,
                                nullptr);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to alloc gpu memory for external umem qp");

    status = doca_umem_gpu_create(gpu_dev,
                                  dev,
                                  qp_umem_gpu_ptr,
                                  external_umem_size,
                                  DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                  &qp_umem);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to create gpu umem");

    status = doca_verbs_qp_init_attr_set_external_umem(verbs_qp_init_attr, qp_umem, 0);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to set doca verbs qp external umem");

    status = doca_gpu_mem_alloc(gpu_dev,
                                dbr_umem_align_sz,
                                get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU,
                                (void **)&qp_umem_dbr_gpu_ptr,
                                nullptr);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to alloc gpu memory for external umem qp");

    status = doca_umem_gpu_create(gpu_dev,
                                  dev,
                                  qp_umem_dbr_gpu_ptr,
                                  dbr_umem_align_sz,
                                  DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                      DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                  &qp_umem_dbr);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to create gpu umem");

    status = doca_verbs_qp_init_attr_set_external_dbr_umem(verbs_qp_init_attr, qp_umem_dbr, 0);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to set doca verbs qp external dbr umem");

    status = doca_verbs_qp_init_attr_set_pd(verbs_qp_init_attr, verbs_pd);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set doca verbs PD");

    status = doca_verbs_qp_init_attr_set_sq_wr(verbs_qp_init_attr, sq_nwqe);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set SQ size");

    status = doca_verbs_qp_init_attr_set_rq_wr(verbs_qp_init_attr, rq_nwqe);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set RQ size");

    status = doca_verbs_qp_init_attr_set_qp_type(verbs_qp_init_attr, DOCA_VERBS_QP_TYPE_RC);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set QP type");

    status = doca_verbs_qp_init_attr_set_send_cq(verbs_qp_init_attr, cq_sq->get_cq());
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set doca verbs CQ");

    status =
        doca_verbs_qp_init_attr_set_send_max_sges(verbs_qp_init_attr, VERBS_TEST_MAX_SEND_SEGS);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set send_max_sges");

    status = doca_verbs_qp_init_attr_set_receive_max_sges(verbs_qp_init_attr,
                                                          VERBS_TEST_MAX_RECEIVE_SEGS);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set receive_max_sges");

    status = doca_verbs_qp_init_attr_set_receive_cq(verbs_qp_init_attr, cq_rq->get_cq());
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to set doca verbs CQ");

    status = doca_verbs_qp_create(verbs_ctx, verbs_qp_init_attr, &new_qp);
    if (status != DOCA_SUCCESS) throw std::runtime_error("Failed to create doca verbs QP");

    status = doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to destroy doca verbs QP attributes");

    return new_qp;
}

void
qp::destroyQp() {
    doca_error_t status;

    status = doca_verbs_qp_destroy(qp_verbs);
    if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy doca verbs QP";

    if (qp_umem != nullptr) {
        status = doca_umem_destroy(qp_umem);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu qp umem";
    }

    if (qp_umem_gpu_ptr != 0) {
        status = doca_gpu_mem_free(gpu_dev, qp_umem_gpu_ptr);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu memory of qp ring buffer";
    }

    if (qp_umem_dbr != nullptr) {
        status = doca_umem_destroy(qp_umem_dbr);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu qp umem dbr";
    }

    if (qp_umem_dbr_gpu_ptr != 0) {
        status = doca_gpu_mem_free(gpu_dev, qp_umem_dbr_gpu_ptr);
        if (status != DOCA_SUCCESS) NIXL_ERROR << "Failed to destroy gpu memory of qp dbr";
    }
}

mr::mr(doca_gpu *gpu_dev_, void *addr_, uint32_t elem_num_, size_t elem_size_, struct ibv_pd *pd_)
    : gpu_dev(gpu_dev_),
      addr(addr_),
      elem_num(elem_num_),
      elem_size(elem_size_),
      pd(pd_) {
    if (gpu_dev == nullptr || addr == nullptr || elem_num == 0 || elem_size == 0 || pd == nullptr)
        throw std::invalid_argument("Invalid mr input values");

    doca_error_t status;
    static size_t host_page_size = sysconf(_SC_PAGESIZE);

    gpu_dev = gpu_dev_;
    dmabuf_fd = -1;
    addr = addr_;
    elem_num = elem_num_;
    elem_size = elem_size_;
    pd = pd_;
    tot_size = elem_num * elem_size;
    remote = false;
    ibmr = nullptr;

    /* Try to map GPU memory with dmabuf.
     * Input size and address should be aliegned to host page size.
     */
    if ((tot_size % host_page_size) == 0 &&
        ((reinterpret_cast<uintptr_t>(addr) % host_page_size) == 0)) {
        status = doca_gpu_dmabuf_fd(gpu_dev, addr, tot_size, &dmabuf_fd);
        if (status == DOCA_SUCCESS) {
            ibmr = ibv_reg_dmabuf_mr(pd,
                                     0,
                                     tot_size,
                                     (uint64_t)addr,
                                     dmabuf_fd,
                                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                         IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        }
    }

    /* Possible failure due to:
     * - GPU not supporting dmabuf mapping
     * - memory address or size not aligned to host page size
     * - linux kernel doesn't have the dmabuf capability
     * Fallback mechanism using legacy mode with nvidia-peermem module and ibv_reg_mr.
     */
    if (ibmr == nullptr) {
        ibmr = ibv_reg_mr(pd,
                          addr,
                          tot_size,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if (ibmr == nullptr) throw std::invalid_argument("Failed to create mr");
    }

    lkey = htobe32(ibmr->lkey);
    rkey = htobe32(ibmr->rkey);
}

mr::mr(void *addr_, size_t tot_size_, uint32_t rkey_)
    : addr(addr_),
      tot_size(tot_size_),
      rkey(rkey_) {
    remote = true;
    ibmr = nullptr;
}

mr::~mr() {
    int ret = ibv_dereg_mr(ibmr);
    if (ret != 0) NIXL_ERROR << "ibv_dereg_mr failed with error " << ret;
}

} // namespace nixl::doca::verbs
