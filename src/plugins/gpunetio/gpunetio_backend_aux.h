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

#ifndef GPUNETIO_BACKEND_AUX_H
#define GPUNETIO_BACKEND_AUX_H

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
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>
#include <doca_rdma_bridge.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_verbs_def.h>

#include "backend/backend_engine.h"
#include "common/str_tools.h"
#include "nixl.h"

// Local includes
#include "common/nixl_time.h"

constexpr uint32_t DOCA_MAX_COMPLETION_INFLIGHT = 128;
constexpr uint32_t DOCA_MAX_COMPLETION_INFLIGHT_MASK = (DOCA_MAX_COMPLETION_INFLIGHT - 1);
constexpr uint32_t RDMA_SEND_QUEUE_SIZE = 2048;
constexpr uint32_t RDMA_RECV_QUEUE_SIZE = (RDMA_SEND_QUEUE_SIZE * 2);
constexpr uint32_t DOCA_POST_STREAM_NUM = 4;
constexpr uint32_t DOCA_XFER_REQ_SIZE = 512;
constexpr uint32_t DOCA_XFER_REQ_MAX = 32;
constexpr uint32_t DOCA_XFER_REQ_MASK = (DOCA_XFER_REQ_MAX - 1);
constexpr uint32_t DOCA_ENG_MAX_CONN = 20;
constexpr uint32_t DOCA_RDMA_CM_LOCAL_PORT_SERVER = 6544;
constexpr uint32_t VERBS_TEST_HOP_LIMIT = 255;

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DOCA_RDMA_SERVER_ADDR_LEN \
    (MAX(MAX(DOCA_DEVINFO_IPV4_ADDR_SIZE, DOCA_DEVINFO_IPV6_ADDR_SIZE), DOCA_GID_BYTE_LENGTH))
// Pre-fill the whole recv queue with notif once
constexpr uint32_t DOCA_MAX_NOTIF_INFLIGHT = RDMA_RECV_QUEUE_SIZE;
constexpr uint32_t DOCA_MAX_NOTIF_MESSAGE_SIZE = 8192;
constexpr uint32_t DOCA_NOTIF_NULL = 0xFFFFFFFF;

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile uint8_t *)&(x))
#endif

/* High-level API */
struct doca_gpu_verbs_qp_init_attr_hl {
    doca_gpu *gpu_dev;
    doca_dev *dev;
    doca_verbs_context *verbs_context;
    doca_verbs_pd *verbs_pd;
    uint16_t sq_nwqe;
    uint16_t rq_nwqe;
    doca_gpu_dev_verbs_nic_handler nic_handler;
};

struct doca_gpu_verbs_qp_hl {
    doca_gpu *gpu_dev; /* DOCA GPU device to use */

    // CQ
    doca_verbs_cq *cq_rq;
    doca_verbs_cq *cq_sq;
    void *cq_sq_umem_gpu_ptr;
    void *cq_rq_umem_gpu_ptr;
    doca_umem *cq_sq_umem;
    doca_umem *cq_rq_umem;
    void *cq_sq_umem_dbr_gpu_ptr;
    void *cq_rq_umem_dbr_gpu_ptr;
    doca_umem *cq_sq_umem_dbr;
    doca_umem *cq_rq_umem_dbr;

    // QP
    doca_verbs_qp *qp;
    void *qp_umem_gpu_ptr;
    doca_umem *qp_umem;
    void *qp_umem_dbr_gpu_ptr;
    doca_umem *qp_umem_dbr;
    doca_uar *external_uar;

    doca_gpu_dev_verbs_nic_handler nic_handler;

    // QP GPUNetIO Object
    doca_gpu_verbs_qp *qp_gverbs;
};

struct nixlDocaMr {
    nixlDocaMr(doca_gpu *gpu_dev, void *addr_, uint32_t elem_num_, size_t elem_size_, struct ibv_pd *pd_);

    nixlDocaMr(void *addr_, size_t tot_size_, uint32_t rkey_);

    ~nixlDocaMr();

    doca_gpu *gpu_dev;
    void *addr;
    uint32_t elem_num;
    size_t elem_size;
    size_t tot_size;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    uint32_t lkey;
    uint32_t rkey;
    bool remote;
    int dmabuf_fd;
};

struct docaXferReqGpu {
    uint32_t id;
    uintptr_t lbuf[DOCA_XFER_REQ_SIZE];
    uintptr_t rbuf[DOCA_XFER_REQ_SIZE];
    uint32_t size[DOCA_XFER_REQ_SIZE];
    uint32_t lkey[DOCA_XFER_REQ_SIZE];
    uint32_t rkey[DOCA_XFER_REQ_SIZE];
    uint16_t num;
    uint8_t in_use;
    uint32_t conn_idx;
    uint32_t has_notif_msg_idx;
    uint32_t msg_sz;
    uint64_t last_wqe;
    // doca_gpu_buf_arr *notif_barr_gpu;
    uintptr_t lbuf_notif;
    uint32_t lkey_notif;
    uint64_t *last_rsvd;
    uint64_t *last_posted;
    nixl_xfer_op_t backendOp; /* Needed only in case of GPU device transfer */
    doca_gpu_dev_verbs_qp *qp_data;
    doca_gpu_dev_verbs_qp *qp_notif;
    // doca_gpu_dev_rdma *rdma_gpu_data; /* DOCA RDMA instance GPU handler */
    // doca_gpu_dev_rdma *rdma_gpu_notif; /* DOCA RDMA instance GPU handler */
};

struct nixlDocaNotif {
    uint32_t elems_num;
    uint32_t elems_size;
    uint8_t *send_addr;
    std::atomic<uint32_t> send_pi;
    struct nixlDocaMr *send_mr;
    uint8_t *recv_addr;
    std::atomic<uint32_t> recv_pi;
    struct nixlDocaMr *recv_mr;
};

struct docaXferCompletion {
    uint8_t completed;
    struct docaXferReqGpu *xferReqRingGpu;
};

struct docaNotif {
    doca_gpu_dev_verbs_qp *qp_gpu;
    uint32_t msg_lkey;
    uintptr_t msg_buf;
    uint32_t msg_size;
    uint32_t msg_num;
    uint32_t msg_last;
};

class nixlDocaConnection : public nixlBackendConnMD {
private:
    std::string remoteAgent;
    volatile bool connected;

public:
    friend class nixlDocaEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlDocaPrivateMetadata : public nixlBackendMD {
private:
    struct nixlDocaMr *mr;
    uint32_t devId;
    nixl_blob_t remoteMrStr;

public:
    nixlDocaPrivateMetadata() : nixlBackendMD(true) {}

    ~nixlDocaPrivateMetadata() {}

    std::string
    get() const {
        return remoteMrStr;
    }

    friend class nixlDocaEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlDocaPublicMetadata : public nixlBackendMD {

public:
    struct nixlDocaMr *mr;
    nixlDocaConnection conn;

    nixlDocaPublicMetadata() : nixlBackendMD(false) {}

    ~nixlDocaPublicMetadata() {}
};

struct nixlDocaRdmaQp {
    doca_gpu_verbs_qp_hl *qp_data_hl;
    doca_gpu_dev_verbs_qp *qp_data_gpu;
    uint32_t qpn_data;
    uint32_t rqpn_data;
    uint32_t remote_gid_data;

    doca_gpu_verbs_qp_hl *qp_notif_hl;
    doca_gpu_dev_verbs_qp *qp_notif_gpu;
    uint32_t qpn_notif;
    uint32_t rqpn_notif;
    uint32_t remote_gid_notif;
};

struct nixlDocaEngine;

void
nixlDocaEngineCheckCudaError(cudaError_t result, const char *message);
void
nixlDocaEngineCheckCuError(CUresult result, const char *message);
int
oob_connection_client_setup(const char *server_ip, int *oob_sock_fd);
void
oob_connection_client_close(int oob_sock_fd);
void
oob_connection_server_close(int oob_sock_fd);
doca_verbs_context *
open_ib_device(char *name);
doca_error_t
create_verbs_ah_attr(doca_verbs_context *verbs_context,
                     uint32_t gid_index,
                     enum doca_verbs_addr_type addr_type,
                     doca_verbs_ah_attr **verbs_ah_attr);
doca_error_t
connect_verbs_qp(nixlDocaEngine *eng, doca_verbs_qp *qp, uint32_t rqpn, uint32_t remote_gid);
void *
threadProgressFunc(void *arg);

doca_error_t
doca_gpu_verbs_destroy_qp_hl(doca_gpu_verbs_qp_hl *qp);
doca_error_t
doca_gpu_verbs_create_qp_hl(doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                            doca_gpu_verbs_qp_hl **qp);
doca_error_t
doca_kernel_write(cudaStream_t stream,
                  doca_gpu_dev_verbs_qp *qp_gpu,
                  struct docaXferReqGpu *xferReqRing,
                  uint32_t pos);
doca_error_t
doca_kernel_read(cudaStream_t stream,
                 doca_gpu_dev_verbs_qp *qp_gpu,
                 struct docaXferReqGpu *xferReqRing,
                 uint32_t pos);
doca_error_t
doca_kernel_progress(cudaStream_t stream,
                     struct docaXferCompletion *completion_list,
                     struct docaNotif *notif_fill,
                     struct docaNotif *notif_progress,
                     struct docaNotif *notif_send_gpu,
                     uint32_t *exit_flag);

#endif /* GPUNETIO_BACKEND_AUX_H */
