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
#ifndef __UCX_MO_BACKEND_H
#define __UCX_MO_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <cassert>

#include "nixl.h"
#include "ucx_backend.h"

// Local includes
#include <common/nixl_time.h>
#include <common/list_elem.h>
#include <ucx/ucx_utils.h>

class nixlUcxMoConnection : public nixlBackendConnMD {
    private:
        std::string remoteAgent;
        uint32_t num_engines;

    public:
        // Extra information required for UCX connections

    friend class nixlUcxMoEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlUcxMoPrivateMetadata : public nixlBackendMD
{
private:
    uint32_t eidx;
    nixlBackendMD *md;
    nixl_mem_t  memType;
    nixl_blob_t rkeyStr;
public:
    nixlUcxMoPrivateMetadata() : nixlBackendMD(true) {
    }

    ~nixlUcxMoPrivateMetadata(){
    }

    std::string get() const {
        return rkeyStr;
    }

    friend class nixlUcxMoEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxMoPublicMetadata : public nixlBackendMD
{
    uint32_t eidx;
    nixlUcxMoConnection conn;
    std::vector<nixlBackendMD*> int_mds;

public:
    nixlUcxMoPublicMetadata() : nixlBackendMD(false) {}

    ~nixlUcxMoPublicMetadata(){
    }

    friend class nixlUcxMoEngine;
};


class nixlUcxMoRequestH : public nixlBackendReqH {
private:
    typedef std::pair<nixlBackendEngine *,nixlBackendReqH *> req_pair_t;
    typedef std::vector<req_pair_t> req_list_t;
    typedef req_list_t::iterator req_list_it_t;

    typedef std::pair<nixl_meta_dlist_t *,nixl_meta_dlist_t *> dl_pair_t;
    typedef std::vector<std::vector<dl_pair_t>> dl_matrix_t;

    dl_matrix_t dlMatrix;
    req_list_t reqs;

    std::string remoteAgent;
    bool notifNeed;
    std::string notifMsg;
public:
    nixlUcxMoRequestH(size_t l_eng_cnt, size_t r_eng_cnt) :
        dlMatrix(l_eng_cnt, std::vector<dl_pair_t>(r_eng_cnt, dl_pair_t{ NULL, NULL }))
    {
        notifNeed = false;
    }

    ~nixlUcxMoRequestH()
    {
        for (auto &row : dlMatrix) {
            for (auto &p : row) {
                if (p.first) {
                    delete p.first;
                }
                if (p.second) {
                    delete p.second;
                }
            }
        }
    }

    friend class nixlUcxMoEngine;

};

class nixlUcxMoEngine : public nixlBackendEngine {
private:
    uint32_t _engineCnt;
    uint32_t _gpuCnt;
    int setEngCnt(uint32_t host_engines);
    uint32_t getEngCnt();
    int32_t getEngIdx(nixl_mem_t type, uint32_t devId);
    std::string getEngName(const std::string &baseName, uint32_t eidx);
    std::string getEngBase(const std::string &engName);
    bool pthrOn;

    // UCX backends data
    std::vector<nixlBackendEngine*> engines;
    // Map of agent name to saved nixlUcxConnection info
    typedef std::map<std::string, nixlUcxMoConnection> remote_conn_map_t;
    typedef remote_conn_map_t::iterator remote_comm_it_t;
    remote_conn_map_t remoteConnMap;

    class nixlUcxMoBckndReq : public nixlBackendReqH {
        private:
            int engIdx;
            nixlBackendReqH *req;
        public:

            nixlUcxMoBckndReq() : nixlBackendReqH() {
            }

            ~nixlUcxMoBckndReq() {
            }
    };

    // Memory helper
    nixl_status_t internalMDHelper (const nixl_blob_t &blob,
                                    const nixl_mem_t &nixl_mem,
                                    const std::string &agent,
                                    nixlBackendMD* &output);

    // Data transfer
    nixl_status_t retHelper(nixl_status_t ret, nixlBackendEngine *eng,
                            nixlUcxMoRequestH *req, nixlBackendReqH *&int_req);
    void cancelRequests(nixlUcxMoRequestH *req);
public:
    nixlUcxMoEngine(const nixlBackendInitParams* init_params);
    ~nixlUcxMoEngine();

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
