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

#ifndef GPUNETIO_BACKEND_QP_H
#define GPUNETIO_BACKEND_QP_H

#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sys/types.h>
#include <thread>
#include <vector>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>
#include <doca_rdma_bridge.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_verbs_def.h>

#include "common/nixl_log.h"

namespace nixl::doca::verbs {
class cq {
public:
    cq(doca_gpu *gpu_dev,
       doca_dev *dev,
       doca_verbs_context *verbs_context,
       doca_verbs_pd *verbs_pd,
       uint16_t ncqe);
    ~cq();

    [[nodiscard]] doca_verbs_cq *
    get_cq() const {
        return cq_verbs;
    }

private:
    [[nodiscard]] doca_verbs_cq *
    createCq();

    void
    destroyCq();

    doca_gpu *gpu_dev;
    doca_dev *dev;
    doca_verbs_context *verbs_ctx;
    doca_verbs_pd *verbs_pd;
    doca_uar *external_uar;
    uint16_t ncqe;

    doca_verbs_cq *cq_verbs;
    void *cq_umem_gpu_ptr;
    doca_umem *cq_umem;
    void *cq_umem_dbr_gpu_ptr;
    doca_umem *cq_umem_dbr;
};

class qp {
public:
    qp(doca_gpu *gpu_dev,
       doca_dev *dev,
       doca_verbs_context *verbs_context,
       doca_verbs_pd *verbs_pd,
       uint16_t sq_nwqe,
       uint16_t rq_nwqe,
       doca_gpu_dev_verbs_nic_handler nic_handler);

    ~qp();

    [[nodiscard]] doca_verbs_qp *
    get_qp() const {
        return qp_verbs;
    }

    [[nodiscard]] doca_gpu_verbs_qp *
    get_qp_gpu() const {
        return qp_gverbs;
    }

    [[nodiscard]] doca_gpu_dev_verbs_qp *
    get_qp_gpu_dev() const {
        return qp_gdev_verbs;
    }

private:
    [[nodiscard]] doca_verbs_qp *
    createQp();

    void
    destroyQp();

    doca_gpu *gpu_dev;
    doca_dev *dev;
    doca_verbs_context *verbs_ctx;
    doca_verbs_pd *verbs_pd;
    uint16_t sq_nwqe;
    uint16_t rq_nwqe;
    doca_gpu_dev_verbs_nic_handler nic_handler;

    doca_verbs_qp *qp_verbs;
    void *qp_umem_gpu_ptr;
    doca_umem *qp_umem;
    void *qp_umem_dbr_gpu_ptr;
    doca_umem *qp_umem_dbr;
    doca_uar *external_uar;
    doca_gpu_verbs_qp *qp_gverbs;
    doca_gpu_dev_verbs_qp *qp_gdev_verbs;

    std::unique_ptr<nixl::doca::verbs::cq> cq_rq;
    std::unique_ptr<nixl::doca::verbs::cq> cq_sq;
};

class mr {
public:
    mr(doca_gpu *gpu_dev, void *addr, uint32_t elem_num, size_t elem_size, struct ibv_pd *pd);
    mr(void *addr, size_t tot_size, uint32_t rkey);
    ~mr();

    [[nodiscard]] struct ibv_mr *
    get_mr() const {
        return ibmr;
    }

    [[nodiscard]] uint32_t
    get_lkey() const {
        return lkey;
    }

    [[nodiscard]] uint32_t
    get_rkey() const {
        return rkey;
    }

    [[nodiscard]] size_t
    get_tot_size() const {
        return tot_size;
    }

    [[nodiscard]] void *
    get_addr() const {
        return addr;
    }

private:
    doca_gpu *gpu_dev;
    void *addr;
    uint32_t elem_num;
    size_t elem_size;
    size_t tot_size;
    struct ibv_pd *pd;
    struct ibv_mr *ibmr;
    uint32_t lkey;
    uint32_t rkey;
    bool remote;
    int dmabuf_fd;
};

} // namespace nixl::doca::verbs

#endif
