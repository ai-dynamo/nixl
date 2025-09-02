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
#include <stdlib.h>
#include <cassert>


// Local includes
#include <nixl.h>
#include <common/nixl_time.h>
#include <serdes/serdes.h>
#include <ucx_mo_backend.h>

using namespace std;

/****************************************
 * CUDA related code
*****************************************/

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

static uint32_t _getNumVramDevices()
{
    cudaError_t result;
    int n_vram_dev;
    result = cudaGetDeviceCount(&n_vram_dev);
    if (result != cudaSuccess) {
        return 0;
    } else {
        return n_vram_dev;
    }
}


#else

static uint32_t _getNumVramDevices(){
    return 0;
}

#endif

/****************************************
 * UCX/MO Request management
*****************************************/

class nixlUcxMoRequestH : public nixlBackendReqH {
private:

    class dlMatrixElem {
    public:
        bool in_use, in_progress;
        nixl_meta_dlist_t *ldescs;
        nixl_meta_dlist_t *rdescs;
        nixlBackendReqH *ucx_req;

        dlMatrixElem() {
            in_use = false;
            in_progress = false;
            ldescs = nullptr;
            rdescs = nullptr;
            ucx_req = nullptr;
        }
    };

    using dl_matrix_t = std::vector<std::vector<dlMatrixElem>>;

    dl_matrix_t dlMatrix;

    std::string remoteAgent;
    bool notifNeed;
    std::string notifMsg;
public:
    nixlUcxMoRequestH(size_t l_eng_cnt, size_t r_eng_cnt) :
        dlMatrix(l_eng_cnt, std::vector<dlMatrixElem>(r_eng_cnt, dlMatrixElem()))
    {
        notifNeed = false;
    }

    ~nixlUcxMoRequestH()
    {
        for (auto &row : dlMatrix) {
            for (auto &p : row) {
                if (p.ldescs) {
                    delete p.ldescs;
                }
                if (p.rdescs) {
                    delete p.rdescs;
                }
            }
        }
    }

    friend class nixlUcxMoEngine;
};


/****************************************
 * UCX Engine management
*****************************************/


int
nixlUcxMoEngine::setEngCnt(uint32_t num_host)
{
    _gpuCnt = _getNumVramDevices();
    _engineCnt = (_gpuCnt > num_host) ? _gpuCnt : num_host;
    return 0;
}

uint32_t
nixlUcxMoEngine::getEngCnt()
{
    return _engineCnt;
}

int32_t
nixlUcxMoEngine::getEngIdx(nixl_mem_t type, uint64_t devId)
{
    switch (type) {
    case VRAM_SEG:
        assert(devId < _gpuCnt);
        if (!(devId < _gpuCnt)) {
            return -1;
        }
    case DRAM_SEG:
        break;
    default:
        return -1;
    }
    assert(devId < _engineCnt);
    return (devId < _engineCnt) ? devId : -1;
}

string
nixlUcxMoEngine::getEngName(const string &baseName, uint32_t eidx) const
{
    return baseName + ":" + to_string(eidx);
}

string
nixlUcxMoEngine::getEngBase(const string &engName)
{
    // find the last occurrence (agent name may have colon in its name)
    if (string::npos == engName.find_last_of(":")) {
        assert(engName.find_last_of(":") != string::npos);
        return engName;
    }
    return engName.substr(0, engName.find_last_of(":"));
}

/****************************************
 * Constructor/Destructor
*****************************************/

nixlUcxMoEngine::nixlUcxMoEngine(const nixlBackendInitParams* init_params):
                                 nixlBackendEngine(init_params)
{
    nixl_b_params_t* custom_params = init_params->customParams;
    uint32_t num_ucx_engines = 1;
    if (custom_params->count("num_ucx_engines")) {
        const char *cptr = (*custom_params)["num_ucx_engines"].c_str();
        char *eptr;
        uint32_t tmp = strtoul(cptr, &eptr, 0);
        if ( (size_t)(eptr - cptr) == (*custom_params)["num_ucx_engines"].length()) {
            num_ucx_engines = tmp;
        } else {
            this->initErr = true;
            // TODO: Log error
            return;
        }
    }

    setEngCnt(num_ucx_engines);
    // Initialize required number of engines
    for (uint32_t i = 0; i < getEngCnt(); i++) {
        auto e = nixlUcxEngine::create(*init_params);
        if (e->getInitErr()) {
            this->initErr = true;
            // TODO: Log error
            return;
        }
        engines.push_back(std::move(e));
    }
}

nixl_mem_list_t
nixlUcxMoEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t
nixlUcxMoEngine::getConnInfo(std::string &str) const
{
    nixlSerDes sd;
    nixl_status_t status;

    // Serialize the number of engines
    size_t sz = engines.size();
    sd.addBuf("Count", &sz, sizeof(sz));

    for( auto &e : engines ) {
        string s;
        status = e->getConnInfo(s);
        if (NIXL_SUCCESS != status) {
            return status;
        }
        sd.addStr("Value", s);
    }

    str = sd.exportStr();
    return NIXL_SUCCESS;
}


nixl_status_t
nixlUcxMoEngine::loadRemoteConnInfo (const string  &remote_agent,
                                     const string &remote_conn_info)
{
    nixlSerDes sd;
    nixlUcxMoConnection conn;
    nixl_status_t status;
    size_t sz;
    remote_comm_it_t it = remoteConnMap.find(remote_agent);

    if(it != remoteConnMap.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    conn.remoteAgent = remote_agent;

    status = sd.importStr(remote_conn_info);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ssize_t ret = sd.getBufLen("Count");
    if (ret != sizeof(sz)) {
        return NIXL_ERR_MISMATCH;
    }
    status = sd.getBuf("Count", &sz, ret);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    conn.num_engines = sz;

    for(size_t idx = 0; idx < sz; idx++) {
        string cinfo;
        cinfo = sd.getStr("Value");
        for (auto &e : engines) {
            status = e->loadRemoteConnInfo(getEngName(remote_agent, idx), cinfo);
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
    }

    remoteConnMap[remote_agent] = conn;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxMoEngine::connect(const string &remote_agent)
{
    remote_comm_it_t it = remoteConnMap.find(remote_agent);
    nixl_status_t status;

    if(it == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxMoConnection &conn = it->second;

    for (auto &e : engines) {
        for (uint32_t idx = 0; idx < conn.num_engines; idx++) {
            status = e->connect(getEngName(remote_agent, idx));
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxMoEngine::disconnect(const string &remote_agent)
{
    nixl_status_t status;
    remote_comm_it_t it = remoteConnMap.find(remote_agent);

    if(it == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxMoConnection &conn = it->second;

    for (auto &e : engines) {
        for (uint32_t idx = 0; idx < conn.num_engines; idx++) {
            status = e->disconnect(getEngName(remote_agent, idx));
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
    }

    remoteConnMap.erase(remote_agent);

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/


nixl_status_t
nixlUcxMoEngine::registerMem (const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD* &out)
{
    auto priv = std::make_unique<nixlUcxMoPrivateMetadata>();
    int32_t eidx = getEngIdx(nixl_mem, mem.devId);
    nixlSerDes sd;
    string str;
    nixl_status_t status;

    if (eidx < 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    priv->memType = nixl_mem;
    priv->eidx = eidx;
    status = engines[eidx]->registerMem(mem, nixl_mem, priv->md);
    if (NIXL_SUCCESS != status) {
        return status;
    }

    sd.addBuf("EngIdx", &eidx, sizeof(eidx));
    status = engines[eidx]->getPublicData(priv->md, str);
    if (NIXL_SUCCESS != status) {
        return status;
    }
    sd.addStr("RkeyStr", str);
    priv->rkeyStr = sd.exportStr();
    out = priv.release();

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxMoEngine::getPublicData (const nixlBackendMD* meta,
                                std::string &str) const
{
    const nixlUcxMoPrivateMetadata *priv = (nixlUcxMoPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxMoEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlUcxMoPrivateMetadata *priv = (nixlUcxMoPrivateMetadata*) meta;

    engines[priv->eidx]->deregisterMem(priv->md);
    delete priv;
    return NIXL_SUCCESS;
}

// To be cleaned up
nixl_status_t
nixlUcxMoEngine::internalMDHelper (const nixl_blob_t &blob,
                                   const nixl_mem_t &nixl_mem,
                                   const std::string &agent,
                                   nixlBackendMD* &output)
{
    nixlUcxMoConnection conn;
    nixlSerDes sd;
    nixl_blob_t ucx_blob;
    nixl_status_t status;
    nixlBlobDesc input_int;

    auto md = std::make_unique<nixlUcxMoPublicMetadata>();

    auto search = remoteConnMap.find(agent);

    if(search == remoteConnMap.end()) {
        //TODO: err: remote connection not found
        return NIXL_ERR_NOT_FOUND;
    }
    conn = search->second;

    status = sd.importStr(blob);

    ssize_t ret = sd.getBufLen("EngIdx");
    if (ret != sizeof(md->eidx)) {
        return NIXL_ERR_MISMATCH;
    }

    status = sd.getBuf("EngIdx", &md->eidx, ret);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ucx_blob = sd.getStr("RkeyStr");
    if (status != NIXL_SUCCESS) {
        return status;
    }

    for (auto &e : engines) {
        nixlBackendMD *int_md;
        input_int.metaInfo = ucx_blob;
        status = e->loadRemoteMD(input_int, nixl_mem,
                                 getEngName(agent, md->eidx),
                                 int_md);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        md->int_mds.push_back(int_md);
    }

    output = md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxMoEngine::loadLocalMD(nixlBackendMD* input,
                             nixlBackendMD* &output)
{
    nixlUcxMoPrivateMetadata* input_md = (nixlUcxMoPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, input_md->memType, localAgent, output);
}

nixl_status_t
nixlUcxMoEngine::loadRemoteMD (const nixlBlobDesc &input,
                               const nixl_mem_t &nixl_mem,
                               const string &remote_agent,
                               nixlBackendMD* &output)
{
    return internalMDHelper(input.metaInfo, nixl_mem, remote_agent, output);
}

nixl_status_t
nixlUcxMoEngine::unloadMD (nixlBackendMD* input)
{
    nixl_status_t status;

    nixlUcxMoPublicMetadata *md = (nixlUcxMoPublicMetadata *)input;
    for (size_t i = 0; i < md->int_mds.size(); i++) {
        status = engines[i]->unloadMD(md->int_mds[i]);
        if (NIXL_SUCCESS != status) {
            return status;
        }
    }
    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

nixl_status_t
nixlUcxMoEngine::prepXfer (const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t *opt_args) const
{
    size_t lidx, ridx;
    size_t lidx_max, ridx_max;

    // Number of local and remote descriptors must match
    int des_cnt = local.descCount();
    if (des_cnt != remote.descCount()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check operation type
    switch(operation) {
        case NIXL_READ:
        case NIXL_WRITE:
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
    }

    // Check that remote agent is known
    const auto it = remoteConnMap.find(remote_agent);
    if(it == remoteConnMap.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const nixlUcxMoConnection &conn = it->second;

    /* Allocate request and fill communication distribution matrix */
    size_t l_eng_cnt = engines.size();
    size_t r_eng_cnt = conn.num_engines;

    auto req = std::make_unique<nixlUcxMoRequestH>(l_eng_cnt, r_eng_cnt);

    /* Go over all input */
    for(int i = 0; i < des_cnt; i++) {
        size_t lsize = local[i].len;
        size_t rsize = remote[i].len;
        nixlUcxMoPrivateMetadata *lmd;
        lmd = (nixlUcxMoPrivateMetadata *)local[i].metadataP;
        nixlUcxMoPublicMetadata *rmd;
        rmd = (nixlUcxMoPublicMetadata *)remote[i].metadataP;
        size_t lidx = lmd->eidx;
        size_t ridx = rmd->eidx;

        if (!((lidx < l_eng_cnt) && (ridx < r_eng_cnt))) {
            // TODO: err output
            goto err_clean_req;
        }
        if (lsize != rsize) {
            // TODO: err output
            goto err_clean_req;
        }

        /* Allocate internal dlists if needed */
        if (!req->dlMatrix[lidx][ridx].in_use) {
            req->dlMatrix[lidx][ridx].in_use = true;
            req->dlMatrix[lidx][ridx].ldescs = new nixl_meta_dlist_t(local.getType());

            req->dlMatrix[lidx][ridx].rdescs = new nixl_meta_dlist_t(remote.getType());
        }

        nixlMetaDesc ldesc = local[i];
        ldesc.metadataP = lmd->md;
        req->dlMatrix[lidx][ridx].ldescs->addDesc(ldesc);

        nixlMetaDesc rdesc = remote[i];
        rdesc.metadataP = rmd->int_mds[lidx];
        req->dlMatrix[lidx][ridx].rdescs->addDesc(rdesc);
    }

    // Prepare UCX requests!
    for(lidx = 0; lidx < req->dlMatrix.size(); lidx++) {
        for(ridx = 0; ridx < req->dlMatrix[lidx].size(); ridx++) {
            nixl_status_t ret;

            if (!req->dlMatrix[lidx][ridx].in_use) {
                // Skip unused matrix elements
                continue;
            }
            ret = engines[lidx]->prepXfer(operation,
                                          *req->dlMatrix[lidx][ridx].ldescs,
                                          *req->dlMatrix[lidx][ridx].rdescs,
                                          getEngName(remote_agent, ridx),
                                          req->dlMatrix[lidx][ridx].ucx_req);
            if (NIXL_SUCCESS != ret) {
                goto err_clean_sub_req;
            }
        }
    }

    handle = req.release();

    return NIXL_SUCCESS;

err_clean_sub_req:
    /* Release only allocated requests */
    lidx_max = lidx + 1;
    ridx_max = ridx;
    for(lidx = 0; lidx < lidx_max; lidx++) {
        for(ridx = 0; ridx < ridx_max; ridx++) {
            nixl_status_t ret;

            if (!req->dlMatrix[lidx][ridx].in_use) {
                // Skip unused matrix elements
                continue;
            }

            engines[lidx]->releaseReqH(req->dlMatrix[lidx][ridx].ucx_req);
            if (NIXL_SUCCESS != ret) {
                // TODO: Output error, but still continue trying to fix others
            }
        }
    }

err_clean_req:
    return NIXL_ERR_INVALID_PARAM;
}


// Data transfer
nixl_status_t
nixlUcxMoEngine::postXfer (const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t *opt_args) const
{
    nixlUcxMoRequestH *req = (nixlUcxMoRequestH *)handle;
    bool in_progress = false;

    for(size_t lidx = 0; lidx < req->dlMatrix.size(); lidx++) {
        for(size_t ridx = 0; ridx < req->dlMatrix[lidx].size(); ridx++) {
            nixl_status_t ret;

            if (!req->dlMatrix[lidx][ridx].in_use) {
                // Skip unused matrix elements
                continue;
            }

            ret = engines[lidx]->postXfer(operation,
                                          *req->dlMatrix[lidx][ridx].ldescs,
                                          *req->dlMatrix[lidx][ridx].rdescs,
                                          getEngName(remote_agent, ridx),
                                          req->dlMatrix[lidx][ridx].ucx_req);

            /* if transfer wasn't immediately completed */
            switch(ret) {
            case NIXL_IN_PROG:
                req->dlMatrix[lidx][ridx].in_progress = true;
                in_progress = true;
            case NIXL_SUCCESS:
                // Nothing to do
                break;
            default:
                // Error.
                return ret;
            }
        }
    }

    if (in_progress) {
        // The transfers are performed via parallel UCX workers (meaning QPs).
        // This doesn't allow piggybacking the notification in postXfer. We
        // can only send it after all workers are flushed, in checkXfer().
        if (opt_args->hasNotif) {
            req->notifNeed = true;
            req->notifMsg = opt_args->notifMsg;
            req->remoteAgent = remote_agent;
        }

        return NIXL_IN_PROG;
    }

    if (opt_args->hasNotif) {
        auto ret = engines[0]->genNotif(getEngName(remote_agent, 0),
                                        opt_args->notifMsg);
        if (NIXL_SUCCESS != ret) {
            /* Return error, TODO: add output */
            return ret;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxMoEngine::checkXfer (nixlBackendReqH *handle) const
{
    nixlUcxMoRequestH *req = (nixlUcxMoRequestH *)handle;
    nixl_status_t out_ret = NIXL_SUCCESS;

    for(size_t lidx = 0; lidx < req->dlMatrix.size(); lidx++) {
        for(size_t ridx = 0; ridx < req->dlMatrix[lidx].size(); ridx++) {
            nixl_status_t ret;

            if (!req->dlMatrix[lidx][ridx].in_progress) {
                // Skip not-in-progress matrix elements
                continue;
            }

            ret = engines[lidx]->checkXfer(req->dlMatrix[lidx][ridx].ucx_req);
            switch (ret) {
            case NIXL_SUCCESS:
                /* Mark as completed */
                req->dlMatrix[lidx][ridx].in_progress = false;
                break;
            case NIXL_IN_PROG:
                out_ret = NIXL_IN_PROG;
                break;
            default:
                /* Any other ret value is unexpected */
                return ret;
            }
        }
    }

    if ((NIXL_SUCCESS == out_ret) && req->notifNeed) {
        nixl_status_t ret;

        // Now as all UCX backends (workers) have been flushed,
        // it is safe to send Notification
        ret = engines[0]->genNotif(getEngName(req->remoteAgent, 0), req->notifMsg);
        if (NIXL_SUCCESS != ret) {
            /* Return error, TODO: add output */
            return ret;
        }
    }

    return out_ret;
}

nixl_status_t
nixlUcxMoEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlUcxMoRequestH *req = (nixlUcxMoRequestH *)handle;
    nixl_status_t out_ret = NIXL_SUCCESS;

    for(size_t lidx = 0; lidx < req->dlMatrix.size(); lidx++) {
        for(size_t ridx = 0; ridx < req->dlMatrix[lidx].size(); ridx++) {
            nixl_status_t ret;

            if (!req->dlMatrix[lidx][ridx].in_use) {
                // Skip unused matrix elements
                continue;
            }

            ret = engines[lidx]->releaseReqH(req->dlMatrix[lidx][ridx].ucx_req);
            if (NIXL_SUCCESS != ret) {
                // TODO: Output error, but still continue trying to fix others
                out_ret = ret;
            }
        }
    }

    return out_ret;
}

int
nixlUcxMoEngine::progress()
{
    int ret = 0;
    // Iterate over all elements cancelling each one
    for ( auto &e : engines ) {
        ret += e->progress();
    }
    return ret;
}

nixl_status_t
nixlUcxMoEngine::getNotifs(notif_list_t &notif_list)
{
    return engines[0]->getNotifs(notif_list);
}

nixl_status_t
nixlUcxMoEngine::genNotif(const string &remote_agent, const string &msg) const
{
    return engines[0]->genNotif(getEngName(remote_agent, 0), msg);
}
