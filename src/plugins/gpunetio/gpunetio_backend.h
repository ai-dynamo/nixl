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

#ifndef GPUNETIO_BACKEND_H
#define GPUNETIO_BACKEND_H

#include "gpunetio_backend_aux.h"

class nixlDocaEngine : public nixlBackendEngine {
public:
    CUcontext main_cuda_ctx;
    int oob_sock_server;
    std::mutex notifLock;
    std::mutex qpLock;
    std::mutex connectLock;
    std::vector<std::pair<uint32_t, doca_gpu *>> gdevs; /* List of DOCA GPUNetIO device handlers */
    doca_dev *ddev; /* DOCA device handler associated to queues */
    doca_verbs_context *verbs_context; /* DOCA Verbs Context */
    doca_verbs_pd *verbs_pd; /* DOCA Verbs Protection Domain */
    doca_verbs_ah_attr *verbs_ah_attr; /* DOCA Verbs address handle */
    struct ibv_pd *pd; /* local protection domain */
    doca_verbs_gid gid; /* local gid address */
    doca_verbs_gid remote_gid; /* remote gid address */
    int lid; /* IB: local ID */
    int dlid; /* IB: destination ID */
    int gid_index;
    struct ibv_port_attr port_attr;
    std::vector<std::string> oobdev;

    nixl_status_t
    addRdmaQp(const std::string &remote_agent);
    nixl_status_t
    connectServerRdmaQp(int oob_sock_client, const std::string &remote_agent);
    nixl_status_t
    nixlDocaInitNotif(const std::string &remote_agent, doca_dev *dev, doca_gpu *gpu);

    volatile uint32_t *pthrStop;
    nixlDocaEngine(const nixlBackendInitParams *init_params);
    ~nixlDocaEngine();

    bool
    supportsRemote() const {
        return true;
    }

    bool
    supportsLocal() const {
        return false;
    }

    bool
    supportsNotif() const {
        return true;
    }

    bool
    supportsProgTh() const {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const;

    /* Object management */
    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const override;
    nixl_status_t
    getConnInfo(std::string &str) const override;
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent,
                       const std::string &remote_conn_info) override;

    nixl_status_t
    connect(const std::string &remote_agent) override;
    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    loadRemoteMD(const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output) override;
    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    // Data transfer
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    nixl_status_t
    getNotifs(notif_list_t &notif_list);
    nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;

    void
    addConnection(struct doca_rdma_connection *connection);
    uint32_t
    getConnectionLast();
    void
    removeConnection(uint32_t connection_idx);
    uint32_t
    getGpuCudaId();

    nixl_status_t
    sendLocalAgentName(int oob_sock_client);
    nixl_status_t
    recvRemoteAgentName(int oob_sock_client, std::string &remote_agent);

private:
    struct doca_log_backend *sdk_log;
    std::string msg_tag_start = "DOCAS";
    std::string msg_tag_end = "DOCAE";
    std::vector<struct nixlDocaRdmaQp> rdma_qp_v;
    int nstreams;

    uint32_t local_port;
    int noSyncIters;
    uint8_t ipv4_addr[4];
    struct sockaddr oob_saddr;
    struct sockaddr oob_netmask;
    std::thread pthr;
    uint64_t *last_rsvd_flags;
    uint64_t *last_posted_flags;
    cudaStream_t post_stream[DOCA_POST_STREAM_NUM];
    cudaStream_t wait_stream;
    mutable std::atomic<uint32_t> xferStream;
    mutable std::atomic<uint32_t> lastPostedReq;

    struct docaXferReqGpu *xferReqRingGpu;
    struct docaXferReqGpu *xferReqRingCpu;
    mutable std::atomic<uint32_t> xferRingPos;

    struct docaXferCompletion *completion_list_gpu;
    struct docaXferCompletion *completion_list_cpu;
    uint32_t *wait_exit_gpu;
    uint32_t *wait_exit_cpu;
    struct docaNotif *notif_fill_gpu;
    struct docaNotif *notif_fill_cpu;
    struct docaNotif *notif_progress_gpu;
    struct docaNotif *notif_progress_cpu;

    struct docaNotif *notif_send_gpu;
    struct docaNotif *notif_send_cpu;

    // Map of agent name to saved nixlDocaConnection info
    std::unordered_map<std::string, nixlDocaConnection> remoteConnMap;
    std::unordered_map<std::string, struct nixlDocaRdmaQp *> qpMap;
    std::unordered_map<std::string, int> connMap;
    std::unordered_map<std::string, struct nixlDocaNotif *> notifMap;

    pthread_t server_thread_id;

    class nixlDocaBckndReq : public nixlBackendReqH {
    private:
    public:
        cudaStream_t stream;
        uint32_t devId;
        uint32_t start_pos;
        uint32_t end_pos;
        uintptr_t backendHandleGpu;

        nixlDocaBckndReq() : nixlBackendReqH() {}

        ~nixlDocaBckndReq() {}
    };

    nixl_status_t
    progressThreadStart();
    void
    progressThreadStop();


    nixl_status_t
    connectClientRdmaQp(int oob_sock_client, const std::string &remote_agent);
    nixl_status_t
    nixlDocaDestroyNotif(doca_gpu *gpu, struct nixlDocaNotif *notif);

    mutable std::mutex notifSendLock;
};

#endif
