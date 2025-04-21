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
#ifndef __UCX_BACKEND_H
#define __UCX_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "common/str_tools.h"

// Local includes
#include "common/nixl_time.h"
#include "ucx/ucx_utils.h"
#include "common/list_elem.h"

typedef enum {CONN_CHECK, NOTIF_STR, DISCONNECT} ucx_cb_op_t;

struct nixl_ucx_am_hdr {
    ucx_cb_op_t op;
};

class nixlUcxConnection : public nixlBackendConnMD {
    private:
        std::string remoteAgent;
        nixlUcxEp ep;
        volatile bool connected;

    public:
        // Extra information required for UCX connections

    friend class nixlUcxEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlUcxPrivateMetadata : public nixlBackendMD {
    private:
        nixlUcxMem mem;
        nixl_blob_t rkeyStr;

    public:
        nixlUcxPrivateMetadata() : nixlBackendMD(true) {
        }

        ~nixlUcxPrivateMetadata(){
        }

        std::string get() const {
            return rkeyStr;
        }

    friend class nixlUcxEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxPublicMetadata : public nixlBackendMD {

    public:
        nixlUcxRkey rkey;
        nixlUcxConnection conn;

        nixlUcxPublicMetadata() : nixlBackendMD(false) {}

        ~nixlUcxPublicMetadata(){
        }
};

// Forward declaration of CUDA context
// It is only visible in ucx_backend.cpp to ensure that
// HAVE_CUDA works properly
// Once we will introduce static config (i.e. config.h) that
// will be part of NIXL installation - we can have
// HAVE_CUDA in h-files
class nixlUcxCudaCtx;
class nixlUcxEngine : public nixlBackendEngine {
    private:

        /* UCX data */
        nixlUcxContext* uc;
        nixlUcxWorker* uw;
        void* workerAddr;
        size_t workerSize;

        /* Progress thread data */
        volatile bool pthrStop, pthrActive, pthrOn;
        int noSyncIters;
        std::thread pthr;
        nixlTime::us_t pthrDelay;

        /* CUDA data*/
        nixlUcxCudaCtx *cudaCtx;
        bool cuda_addr_wa;

        /* Notifications */
        notif_list_t notifMainList;
        std::mutex  notifMtx;
        notif_list_t notifPthrPriv, notifPthr;

        // Map of agent name to saved nixlUcxConnection info
        std::unordered_map<std::string, nixlUcxConnection,
                           std::hash<std::string>, strEqual> remoteConnMap;

        class nixlUcxBckndReq : public nixlLinkElem<nixlUcxBckndReq>, public nixlBackendReqH {
            private:
                int _completed;
            public:
                std::string *amBuffer;

                nixlUcxBckndReq() : nixlLinkElem(), nixlBackendReqH() {
                    _completed = 0;
                    amBuffer = NULL;
                }

                ~nixlUcxBckndReq() {
                    _completed = 0;
                    if (amBuffer) {
                        delete amBuffer;
                    }
                }

                bool is_complete() { return _completed; }
                void completed() { _completed = 1; }
        };

        void vramInitCtx();
        void vramFiniCtx();
        int vramUpdateCtx(void *address, uint64_t devId, bool &restart_reqd);
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

        // Request management
        static void _requestInit(void *request);
        static void _requestFini(void *request);
        void requestReset(nixlUcxBckndReq *req) {
            _requestInit((void *)req);
        }

        // Connection helper
        static ucs_status_t
        connectionCheckAmCb(void *arg, const void *header,
                            size_t header_length, void *data,
                            size_t length,
                            const ucp_am_recv_param_t *param);

        static ucs_status_t
        connectionTermAmCb(void *arg, const void *header,
                           size_t header_length, void *data,
                           size_t length,
                           const ucp_am_recv_param_t *param);

        // Memory management helpers
        nixl_status_t internalMDHelper (const nixl_blob_t &blob,
                                        const std::string &agent,
                                        nixlBackendMD* &output);

        // Notifications
        static ucs_status_t notifAmCb(void *arg, const void *header,
                                      size_t header_length, void *data,
                                      size_t length,
                                      const ucp_am_recv_param_t *param);
        nixl_status_t notifSendPriv(const std::string &remote_agent,
                                    const std::string &msg, nixlUcxReq &req);
        void notifProgress();
        void notifCombineHelper(notif_list_t &src, notif_list_t &tgt);
        void notifProgressCombineHelper(notif_list_t &src, notif_list_t &tgt);


        // Data transfer (priv)
        nixl_status_t retHelper(nixl_status_t ret, nixlUcxBckndReq *head, nixlUcxReq &req);

    public:
        nixlUcxEngine(const nixlBackendInitParams* init_params);
        ~nixlUcxEngine();

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

#endif
