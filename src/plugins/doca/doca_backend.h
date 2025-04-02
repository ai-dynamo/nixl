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
#ifndef __DOCA_BACKEND_H
#define __DOCA_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_gpunetio.h>
#include <doca_rdma.h>
#include <doca_mmap.h>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "common/str_tools.h"

// Local includes
#include "common/nixl_time.h"
#include "common/list_elem.h"

#define DOCA_DEVINFO_IBDEV_NAME_SIZE 64
#define RDMA_RECV_QUEUE_SIZE 2048
#define RDMA_SEND_QUEUE_SIZE 2048

typedef enum {CONN_CHECK, NOTIF_STR, DISCONNECT} ucx_cb_op_t;

typedef void * nixlDocaReq;

struct nixl_ucx_am_hdr {
    ucx_cb_op_t op;
};

struct nixlDocaMem {
    void *addr;
    uint32_t len;
    struct doca_mmap *mmap;
    void *export_mmap;
    size_t export_len;
};
    
class nixlDocaConnection : public nixlBackendConnMD {
    private:
        std::string remoteAgent;
        // rdma qp
        // nixlDocaEp ep;
        volatile bool connected;

    public:
        // Extra information required for UCX connections

    friend class nixlDocaEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlDocaPrivateMetadata : public nixlBackendMD {
    private:
        nixlDocaMem mem;
        uint32_t rkey;
        nixl_blob_t rkeyStr;

    public:
        nixlDocaPrivateMetadata() : nixlBackendMD(true) {
        }

        ~nixlDocaPrivateMetadata(){
        }

        std::string get() const {
            return rkeyStr;
        }

    friend class nixlDocaEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlDocaPublicMetadata : public nixlBackendMD {

    public:
        uint32_t rkey;
        nixlDocaConnection conn;

        nixlDocaPublicMetadata() : nixlBackendMD(false) {}

        ~nixlDocaPublicMetadata(){
        }
};

class nixlDocaEngine : public nixlBackendEngine {
    private:

        struct doca_gpu *gdev; /* GPUNetio handler associated to queues */
	    struct doca_dev *ddev;	  /* DOCA device handler associated to queues */
        struct doca_log_backend *sdk_log;
        struct doca_rdma *rdma;		    /* DOCA RDMA instance */
        struct doca_gpu_dev_rdma *gpu_rdma; /* DOCA RDMA instance GPU handler */
        struct doca_ctx *rdma_ctx;	    /* DOCA context to be used with DOCA RDMA */
        const void *connection_details;	    /* Remote peer connection details */
        size_t conn_det_len;		    /* Remote peer connection details data length */    

        /* Progress thread data */
        volatile bool pthrStop, pthrActive, pthrOn;
        int noSyncIters;
        std::thread pthr;
        nixlTime::us_t pthrDelay;

        /* Notifications */
        notif_list_t notifMainList;
        std::mutex  notifMtx;
        notif_list_t notifPthrPriv, notifPthr;

        // Map of agent name to saved nixlDocaConnection info
        std::unordered_map<std::string, nixlDocaConnection,
                           std::hash<std::string>, strEqual> remoteConnMap;

        class nixlDocaBckndReq : public nixlLinkElem<nixlDocaBckndReq>, public nixlBackendReqH {
            private:
                int _completed;
            public:
                std::string *amBuffer;

                nixlDocaBckndReq() : nixlLinkElem(), nixlBackendReqH() {
                    _completed = 0;
                    amBuffer = NULL;
                }

                ~nixlDocaBckndReq() {
                    _completed = 0;
                    if (amBuffer) {
                        delete amBuffer;
                    }
                }

                bool is_complete() { return _completed; }
                void completed() { _completed = 1; }
        };
#if 0
        void vramInitCtx();
        void vramFiniCtx();
        int vramUpdateCtx(void *address, uint32_t  devId, bool &restart_reqd);
        int vramApplyCtx();

        // Threading infrastructure
        //   TODO: move the thread management one outside of NIXL common infra
        void progressFunc();
        void progressThreadStart();
        void progressThreadStop();
        void progressThreadRestart();
        bool isProgressThread(){
            return (std::this_thread::get_id() == pthr.get_id());
        }
#endif
        // Request management
        static void _requestInit(void *request);
        static void _requestFini(void *request);
        void requestReset(nixlDocaBckndReq *req) {
            _requestInit((void *)req);
        }

        // Memory management helpers
        nixl_status_t internalMDHelper (const nixl_blob_t &blob,
                                        const std::string &agent,
                                        nixlBackendMD* &output);

        // Notifications
        nixl_status_t notifSendPriv(const std::string &remote_agent,
                                    const std::string &msg, nixlDocaReq &req);
        void notifProgress();
        void notifCombineHelper(notif_list_t &src, notif_list_t &tgt);
        void notifProgressCombineHelper(notif_list_t &src, notif_list_t &tgt);


        // Data transfer (priv)
        nixl_status_t retHelper(nixl_status_t ret, nixlDocaBckndReq *head, nixlDocaReq &req);

    public:
        nixlDocaEngine(const nixlBackendInitParams* init_params);
        ~nixlDocaEngine();

        bool supportsRemote () const { return true; }
        bool supportsLocal () const { return true; }
        bool supportsNotif () const { return true; }
        bool supportsProgTh () const { return pthrOn; }

        nixl_mem_list_t getSupportedMems () const;

        /* Object management */
        nixl_status_t getPublicData (const nixlBackendMD* meta,
                                     std::string &str) const;
        nixl_status_t getConnInfo(std::string &str) const;
        nixl_status_t loadRemoteConnInfo (const std::string &remote_agent,
                                          const std::string &remote_conn_info);

        nixl_status_t connect(const std::string &remote_agent);
        nixl_status_t disconnect(const std::string &remote_agent);

        nixl_status_t registerMem (const nixlBlobDesc &mem,
                                   const nixl_mem_t &nixl_mem,
                                   nixlBackendMD* &out);
        nixl_status_t deregisterMem (nixlBackendMD* meta);

        nixl_status_t loadLocalMD (nixlBackendMD* input,
                                   nixlBackendMD* &output);

        nixl_status_t loadRemoteMD (const nixlBlobDesc &input,
                                    const nixl_mem_t &nixl_mem,
                                    const std::string &remote_agent,
                                    nixlBackendMD* &output);
        nixl_status_t unloadMD (nixlBackendMD* input);

        // Data transfer
        nixl_status_t prepXfer (const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr);

        nixl_status_t postXfer (const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr);

        nixl_status_t checkXfer (nixlBackendReqH* handle);
        nixl_status_t releaseReqH(nixlBackendReqH* handle);

        int progress();

        nixl_status_t getNotifs(notif_list_t &notif_list);
        nixl_status_t genNotif(const std::string &remote_agent, const std::string &msg);

        //public function for UCX worker to mark connections as connected
        nixl_status_t checkConn(const std::string &remote_agent);
        nixl_status_t endConn(const std::string &remote_agent);
};

doca_error_t doca_util_map_and_export(struct doca_dev *dev, uint32_t permissions, void *addr, uint32_t size, nixlDocaMem *mem);
#endif
