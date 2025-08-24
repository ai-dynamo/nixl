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

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_buf_array.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_mmap.h>
#include <doca_rdma.h>
#include <doca_rdma_bridge.h>

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
constexpr uint32_t DOCA_RDMA_CM_LOCAL_PORT_CLIENT = 6543;
constexpr uint32_t DOCA_RDMA_CM_LOCAL_PORT_SERVER = 6544;
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DOCA_RDMA_SERVER_ADDR_LEN \
    (MAX (MAX (DOCA_DEVINFO_IPV4_ADDR_SIZE, DOCA_DEVINFO_IPV6_ADDR_SIZE), DOCA_GID_BYTE_LENGTH))
// Pre-fill the whole recv queue with notif once
constexpr uint32_t DOCA_MAX_NOTIF_INFLIGHT = RDMA_RECV_QUEUE_SIZE;
constexpr uint32_t DOCA_MAX_NOTIF_MESSAGE_SIZE = 8192;
constexpr uint32_t DOCA_NOTIF_NULL = 0xFFFFFFFF;

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile uint8_t *)&(x))
#endif

struct nixlDocaMmap {
    nixlDocaMmap(void *addr,
                  uint32_t elem_num,
                  size_t elem_size,
                  struct doca_dev *dev);
    nixlDocaMmap();

    ~nixlDocaMmap();

    void *addr;
    uint32_t elem_num;
    size_t elem_size;
    struct doca_dev *dev;
    struct doca_mmap *mmap;
};

struct nixlDocaBarr {
    nixlDocaBarr(struct doca_mmap *mmap,
                     uint32_t elem_num,
                     size_t elem_size,
                     struct doca_gpu *gpu);
    ~nixlDocaBarr();

    struct doca_mmap *mmap;
    uint32_t elem_num;
    size_t elem_size;
    struct doca_gpu *gpu;
    struct doca_buf_arr *barr;
    struct doca_gpu_buf_arr *barr_gpu;
};

struct docaXferReqGpu {
    uint32_t id;
    uintptr_t larr[DOCA_XFER_REQ_SIZE];
    uintptr_t rarr[DOCA_XFER_REQ_SIZE];
    size_t size[DOCA_XFER_REQ_SIZE];
    uint16_t num;
    uint8_t in_use;
    uint32_t conn_idx;
    uint32_t has_notif_msg_idx;
    size_t msg_sz;
    struct doca_gpu_buf_arr *notif_barr_gpu;
    uint64_t *last_rsvd;
    uint64_t *last_posted;
    nixl_xfer_op_t backendOp; /* Needed only in case of GPU device transfer */
    struct doca_gpu_dev_rdma *rdma_gpu_data; /* DOCA RDMA instance GPU handler */
    struct doca_gpu_dev_rdma *rdma_gpu_notif; /* DOCA RDMA instance GPU handler */
};

struct nixlDocaMem {
    void *addr;
    uint32_t len;
    struct nixlDocaMmap *mmap;
    struct nixlDocaBarr *barr;
    void *export_mmap;
    size_t export_len;
    uint32_t devId;
};

struct nixlDocaNotif {
    uint32_t elems_num;
    uint32_t elems_size;
    uint8_t *send_addr;
    std::atomic<uint32_t> send_pi;
    struct nixlDocaMmap *send_mmap;
    struct nixlDocaBarr *send_barr;
    uint8_t *recv_addr;
    std::atomic<uint32_t> recv_pi;
    struct nixlDocaMmap *recv_mmap;
    struct nixlDocaBarr *recv_barr;
};

struct docaXferCompletion {
    uint8_t completed;
    struct docaXferReqGpu *xferReqRingGpu;
};

struct docaNotifRecv {
    struct doca_gpu_dev_rdma *rdma_qp;
    struct doca_gpu_buf_arr *barr_gpu;
    int num_msg;
};

struct docaNotifSend {
    struct doca_gpu_dev_rdma *rdma_qp;
    struct doca_gpu_buf_arr *barr_gpu;
    int buf_idx;
    size_t msg_sz;
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
    nixlDocaMem mem;
    nixl_blob_t remoteMmapStr;

public:
    nixlDocaPrivateMetadata() : nixlBackendMD (true) {}

    ~nixlDocaPrivateMetadata() {}

    std::string
    get() const {
        return remoteMmapStr;
    }

    friend class nixlDocaEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlDocaPublicMetadata : public nixlBackendMD {

public:
    nixlDocaMem mem;
    nixlDocaConnection conn;

    nixlDocaPublicMetadata() : nixlBackendMD (false) {}

    ~nixlDocaPublicMetadata() {}
};

struct nixlDocaRdmaQp {
    struct doca_dev *dev; /* DOCA device handler associated to queues */
    struct doca_gpu *gpu; /* DOCA device handler associated to queues */
    struct doca_rdma *rdma_data; /* DOCA RDMA instance */
    struct doca_gpu_dev_rdma *rdma_gpu_data; /* DOCA RDMA instance GPU handler */
    struct doca_ctx *rdma_ctx_data; /* DOCA context to be used with DOCA RDMA */
    const void *connection_details_data; /* Remote peer connection details */
    size_t conn_det_len_data; /* Remote peer connection details data length */
    struct doca_rdma_connection *connection_data; /* The RDMA_CM connection instance */

    struct doca_rdma *rdma_notif; /* DOCA RDMA instance */
    struct doca_gpu_dev_rdma *rdma_gpu_notif; /* DOCA RDMA instance GPU handler */
    struct doca_ctx *rdma_ctx_notif; /* DOCA context to be used with DOCA RDMA */
    const void *connection_details_notif; /* Remote peer connection details */
    size_t conn_det_len_notif; /* Remote peer connection details data length */
    struct doca_rdma_connection *connection_notif; /* The RDMA_CM connection instance */
};


void nixlDocaEngineCheckCudaError (cudaError_t result, const char *message);
void nixlDocaEngineCheckCuError (CUresult result, const char *message);
int oob_connection_client_setup (const char *server_ip, int *oob_sock_fd);
void oob_connection_client_close (int oob_sock_fd);
void oob_connection_server_close (int oob_sock_fd);
doca_error_t open_doca_device_with_ibdev_name (const uint8_t *value, size_t val_size, struct doca_dev **retval);
void * threadProgressFunc (void *arg);

doca_error_t
doca_kernel_write (cudaStream_t stream,
                   struct doca_gpu_dev_rdma *rdma_gpu,
                   struct docaXferReqGpu *xferReqRing,
                   uint32_t pos);
doca_error_t
doca_kernel_read (cudaStream_t stream,
                  struct doca_gpu_dev_rdma *rdma_gpu,
                  struct docaXferReqGpu *xferReqRing,
                  uint32_t pos);
doca_error_t
doca_kernel_progress (cudaStream_t stream,
                      struct docaXferCompletion *completion_list,
                      struct docaNotifRecv *notif_fill,
                      struct docaNotifRecv *notif_progress,
                      struct docaNotifSend *notif_send_gpu,
                      uint32_t *exit_flag);

#endif /* GPUNETIO_BACKEND_AUX_H */
