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
#include "nixl.h"
#include "ucx_backend.h"
#include "utils/serdes/serdes.h"
#include "backend/backend_engine.h"
#include "internal/transfer_request.h"
#include "internal/agent_data.h"
#include "internal/plugin_manager.h"

#ifndef DISABLE_GDS_BACKEND
#include "gds_backend.h"
#endif

nixlAgentData::nixlAgentData(const std::string &name,
                             const nixlAgentConfig &cfg) :
                             name(name), config(cfg) {}

nixlAgentData::~nixlAgentData() {
    for (auto & elm: remoteSections)
        delete elm.second;

    for (auto & elm: backendEngines) {
        auto& plugin_manager = nixlPluginManager::getInstance();
        auto plugin_handle = plugin_manager.getPlugin(elm.second->getType());

        if (plugin_handle) {
            // If we have a plugin handle, use it to destroy the engine
            plugin_handle->destroyEngine(elm.second);
        }
    }

    for (auto & elm: backendHandles)
        delete elm.second;
}

nixlAgent::nixlAgent(const std::string &name,
                     const nixlAgentConfig &cfg) {
    data = new nixlAgentData(name, cfg);
}

nixlAgent::~nixlAgent() {
    delete data;
}

nixl_b_params_t nixlAgent::getBackendOptions (const nixl_backend_t &type) {
    nixl_b_params_t params;

    // First try to get options from a loaded plugin
    auto& plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.getPlugin(type);

    if (plugin_handle) {
        // If the plugin is already loaded, get options directly
        return plugin_handle->getBackendOptions();
    }

    // If plugin isn't loaded yet, try to load it temporarily
    plugin_handle = plugin_manager.loadPlugin(type);
    if (plugin_handle) {
        params = plugin_handle->getBackendOptions();
        // We don't keep the plugin loaded if we didn't have it before
        if (data->backendEngines.count(type) == 0) {
            plugin_manager.unloadPlugin(type);
        }
        return params;
    }

    return params;
}

nixlBackendH* nixlAgent::createBackend(const nixl_backend_t &type,
                                       const nixl_b_params_t &params) {

    nixlBackendInitParams init_params;
    nixlBackendEngine* backend = nullptr;
    nixlBackendH* handle;
    nixl_status_t ret;
    std::string str;

    // Registering same type of backend is not supported, unlikely and prob error
    if (data->backendEngines.count(type)!=0)
        return nullptr;

    init_params.localAgent   = data->name;
    init_params.type         = type;
    init_params.customParams = const_cast<nixl_b_params_t*>(&params);
    init_params.enableProgTh = data->config.useProgThread;
    init_params.pthrDelay    = data->config.pthrDelay;

    // First, try to load the backend as a plugin
    auto& plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.loadPlugin(type);

    if (plugin_handle) {
        // Plugin found, use it to create the backend
        backend = plugin_handle->createEngine(&init_params);
    } else {
        // Fallback to built-in backends
        std::cout << "Unsupported backend: " << type << std::endl;
        return nullptr;
    }

    if (backend!=nullptr) {
        if (backend->getInitErr()) {
            delete backend;
            return nullptr;
        }

        if (backend->supportsRemote()) {
            str = backend->getConnInfo();
            if (str=="") {
                delete backend;
                return nullptr;
            }
            data->connMD[type] = str;
        }

        if (backend->supportsLocal()) {
            ret = backend->connect(data->name);

            if (NIXL_SUCCESS != ret) {
                delete backend;
                return nullptr;
            }
        }

        handle = new nixlBackendH(backend);
        if (handle == nullptr) {
            delete backend;
            return nullptr;
        }

        data->backendEngines[type] = backend;
        data->memorySection.addBackendHandler(backend);
        data->backendHandles[type] = handle;

        // TODO: Check if backend supports ProgThread when threading is in agent
    }

    return handle; // nullptr in case of error
}

nixl_status_t nixlAgent::registerMem(const nixl_reg_dlist_t &descs,
                                     nixlBackendH* backend) {
    nixl_status_t ret;
    nixl_meta_dlist_t remote_self(descs.getType(), descs.isUnifiedAddr(), false);
    ret = data->memorySection.addDescList(descs, backend->engine, remote_self);
    if (ret!=NIXL_SUCCESS)
        return ret;

    if (backend->supportsLocal()) {
        if (data->remoteSections.count(data->name)==0)
            data->remoteSections[data->name] = new nixlRemoteSection(
                                data->name, data->backendEngines);

        ret = data->remoteSections[data->name]->loadLocalData(remote_self,
                                                              backend->engine);
    }

    return ret;
}

nixl_status_t nixlAgent::deregisterMem(const nixl_reg_dlist_t &descs,
                                       nixlBackendH* backend) {
    nixl_status_t ret;
    nixl_meta_dlist_t resp(descs.getType(),
                           descs.isUnifiedAddr(),
                           descs.isSorted());
    nixl_xfer_dlist_t trimmed = descs.trim();
    // TODO: can use getIndex for exact match instead of populate
    ret = data->memorySection.populate(trimmed, backend->getType(), resp);
    if (ret != NIXL_SUCCESS)
        return ret;
    return (data->memorySection.remDescList(resp, backend->engine));
}

nixl_status_t nixlAgent::makeConnection(const std::string &remote_agent) {
    nixlBackendEngine* eng;
    nixl_status_t ret;
    int count = 0;

    if (data->remoteBackends.count(remote_agent)==0)
        return NIXL_ERR_NOT_FOUND;

    // For now making all the possible connections, later might take hints
    for (auto & r_eng: data->remoteBackends[remote_agent]) {
        if (data->backendEngines.count(r_eng)!=0) {
            eng = data->backendEngines[r_eng];
            ret = eng->connect(remote_agent);
            if (ret)
                return ret;
            count++;
        }
    }

    if (count == 0) // No common backend
        return NIXL_ERR_BACKEND;
    return NIXL_SUCCESS;
}

nixl_status_t nixlAgent::createXferReq(const nixl_xfer_dlist_t &local_descs,
                                       const nixl_xfer_dlist_t &remote_descs,
                                       const std::string &remote_agent,
                                       const std::string &notif_msg,
                                       const nixl_xfer_op_t &operation,
                                       nixlXferReqH* &req_handle,
                                       const nixlBackendH* backend) const {
    nixl_status_t ret;
    req_handle = nullptr;

    // Check the correspondence between descriptor lists
    if (local_descs.descCount() != remote_descs.descCount())
        return NIXL_ERR_INVALID_PARAM;
    for (int i=0; i<local_descs.descCount(); ++i)
        if (local_descs[i].len != remote_descs[i].len)
            return NIXL_ERR_INVALID_PARAM;

    if ((notif_msg.size()==0) &&
        ((operation==NIXL_WR_NOTIF) || (operation==NIXL_RD_NOTIF)))
        return NIXL_ERR_INVALID_PARAM;

    if (data->remoteSections.count(remote_agent)==0)
        return NIXL_ERR_NOT_FOUND;

    // TODO: when central KV is supported, add a call to fetchRemoteMD
    // TODO: merge descriptors back to back in memory (like makeXferReq).
    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    nixlXferReqH *handle = new nixlXferReqH;
    handle->initiatorDescs = new nixl_meta_dlist_t (
                                     local_descs.getType(),
                                     local_descs.isUnifiedAddr(),
                                     local_descs.isSorted());

    if (backend==nullptr) {
        handle->engine = data->memorySection.findQuery(local_descs,
                              remote_descs.getType(),
                              data->remoteBackends[remote_agent],
                              *handle->initiatorDescs);
        if (handle->engine==nullptr) {
            delete handle;
            return NIXL_ERR_NOT_FOUND;
        }
    } else {
        ret = data->memorySection.populate(local_descs,
                                           backend->getType(),
                                           *handle->initiatorDescs);
       if (ret!=NIXL_SUCCESS) {
            delete handle;
            return NIXL_ERR_BACKEND;
       }
       handle->engine = backend->engine;
    }

    if ((notif_msg.size()!=0) && (!handle->engine->supportsNotif())) {
        delete handle;
        return NIXL_ERR_BACKEND;
    }

    handle->targetDescs = new nixl_meta_dlist_t (
                                  remote_descs.getType(),
                                  remote_descs.isUnifiedAddr(),
                                  remote_descs.isSorted());

    // Based on the decided local backend, we check the remote counterpart
    ret = data->remoteSections[remote_agent]->populate(remote_descs,
               handle->engine->getType(), *handle->targetDescs);
    if (ret!=NIXL_SUCCESS) {
        delete handle;
        return ret;
    }

    handle->remoteAgent = remote_agent;
    handle->notifMsg    = notif_msg;
    handle->backendOp   = operation;
    handle->state       = NIXL_XFER_INIT;

    req_handle = handle;

    return NIXL_SUCCESS;
}

void nixlAgent::invalidateXferReq(nixlXferReqH *req) {
    //destructor will call release to abort transfer if necessary
    delete req;
}

nixl_xfer_state_t nixlAgent::postXferReq(nixlXferReqH *req) {
    nixl_xfer_state_t ret;

    if (req==nullptr)
        return NIXL_XFER_ERR;

    // We can't repost while a request is in progress
    if (req->state == NIXL_XFER_PROC) {
        req->state = req->engine->checkXfer(req->backendHandle);
        if (req->state == NIXL_XFER_PROC) {
            delete req;
            return NIXL_XFER_ERR;
        }
    }

    // // The remote was invalidated
    // if (data->remoteBackends.count(req->remoteAgent)==0)
    //     delete req;
    //     return NIXL_ERR_BAD;
    // }

    // If state is NIXL_XFER_INIT or NIXL_XFER_DONE we can repost,
    ret = (req->engine->postXfer (*req->initiatorDescs,
                                   *req->targetDescs,
                                   req->backendOp,
                                   req->remoteAgent,
                                   req->notifMsg,
                                   req->backendHandle));
    req->state = ret;
    return ret;
}

nixl_xfer_state_t nixlAgent::getXferStatus (nixlXferReqH *req) {
    // // The remote was invalidated
    // if (data->remoteBackends.count(req->remoteAgent)==0)
    //     delete req;
    //     return NIXL_ERR_BAD;
    // }

    // If the state is done, no need to recheck.
    if (req->state != NIXL_XFER_DONE)
        req->state = req->engine->checkXfer(req->backendHandle);

    return req->state;
}


nixlBackendH* nixlAgent::getXferBackend(const nixlXferReqH* req) const {
    return data->backendHandles[req->engine->getType()];
}

nixl_status_t nixlAgent::prepXferSide (const nixl_xfer_dlist_t &descs,
                                       const std::string &remote_agent,
                                       const nixlBackendH* backend,
                                       nixlXferSideH* &side_handle) const {
    nixl_status_t ret;

    if (backend==nullptr)
        return NIXL_ERR_NOT_FOUND;

    if (remote_agent.size()!=0)
        if (data->remoteSections.count(remote_agent)==0)
            return NIXL_ERR_NOT_FOUND;

    // TODO: when central KV is supported, add a call to fetchRemoteMD
    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    nixlXferSideH *handle = new nixlXferSideH;

    // This function is const regarding the backend, when transfer handle is
    // generated, there the backend can change upong post.
    handle->engine = backend->engine;
    handle->descs = new nixl_meta_dlist_t (descs.getType(),
                                           descs.isUnifiedAddr(),
                                           descs.isSorted());

    if (remote_agent.size()==0) { // Local descriptor list
        handle->isLocal = true;
        handle->remoteAgent = "";
        ret = data->memorySection.populate(
                   descs, backend->getType(), *handle->descs);
    } else {
        handle->isLocal = false;
        handle->remoteAgent = remote_agent;
        ret = data->remoteSections[remote_agent]->populate(
                   descs, backend->getType(), *handle->descs);
    }

    if (ret<0) {
        delete handle;
        return ret;
    }


    side_handle = handle;

    return NIXL_SUCCESS;
}

nixl_status_t nixlAgent::makeXferReq (const nixlXferSideH* local_side,
                                      const std::vector<int> &local_indices,
                                      const nixlXferSideH* remote_side,
                                      const std::vector<int> &remote_indices,
                                      const std::string &notif_msg,
                                      const nixl_xfer_op_t &operation,
                                      nixlXferReqH* &req_handle) const {
    req_handle     = nullptr;
    int desc_count = (int) local_indices.size();

    if ((!local_side->isLocal) || (remote_side->isLocal))
        return NIXL_ERR_INVALID_PARAM;

    if ((local_side->engine == nullptr) || (remote_side->engine == nullptr) ||
        (local_side->engine != remote_side->engine))
        return NIXL_ERR_INVALID_PARAM;

    if ((desc_count==0) || (remote_indices.size()==0) ||
        (desc_count != (int) remote_indices.size()))
        return NIXL_ERR_INVALID_PARAM;

    for (int i=0; i<desc_count; ++i) {
        if ((local_indices[i] >= local_side->descs->descCount())
               || (local_indices[i]<0))
            return NIXL_ERR_INVALID_PARAM;
        if ((remote_indices[i] >= remote_side->descs->descCount())
               || (remote_indices[i]<0))
            return NIXL_ERR_INVALID_PARAM;
        if ((*local_side->descs )[local_indices [i]].len !=
            (*remote_side->descs)[remote_indices[i]].len)
            return NIXL_ERR_INVALID_PARAM;
    }

    if ((notif_msg.size()==0) &&
        ((operation==NIXL_WR_NOTIF) || (operation==NIXL_RD_NOTIF)))
        return NIXL_ERR_INVALID_PARAM;

    if ((notif_msg.size()!=0) && (!local_side->engine->supportsNotif())) {
        return NIXL_ERR_BACKEND;
    }

    // // The remote was invalidated
    // if (data->remoteBackends.count(remote_side->remoteAgent)==0)
    //     delete req_handle;
    //     return NIXL_ERR_BAD;
    // }

    // Populate has been already done, no benefit in having sorted descriptors
    // which will be overwritten by [] assignment operator.
    nixlXferReqH *handle = new nixlXferReqH;
    handle->initiatorDescs = new nixl_meta_dlist_t (
                                     local_side->descs->getType(),
                                     local_side->descs->isUnifiedAddr(),
                                     false, desc_count);

    handle->targetDescs = new nixl_meta_dlist_t (
                                  remote_side->descs->getType(),
                                  remote_side->descs->isUnifiedAddr(),
                                  false, desc_count);

    int i = 0, j = 0; //final list size
    while(i<(desc_count)) {
        nixlMetaDesc local_desc1 = (*local_side->descs)[local_indices[i]];
        nixlMetaDesc remote_desc1 = (*remote_side->descs)[remote_indices[i]];

        if(i != (desc_count-1) ) {
            nixlMetaDesc local_desc2 = (*local_side->descs)[local_indices[i+1]];
            nixlMetaDesc remote_desc2 = (*remote_side->descs)[remote_indices[i+1]];

          while(((local_desc1.addr + local_desc1.len) == local_desc2.addr)
             && ((remote_desc1.addr + remote_desc1.len) == remote_desc2.addr)
             && (local_desc1.metadataP == local_desc2.metadataP)
             && (remote_desc1.metadataP == remote_desc2.metadataP)
             && (local_desc1.devId == local_desc2.devId)
             && (remote_desc1.devId == remote_desc2.devId))
            {
                local_desc1.len += local_desc2.len;
                remote_desc1.len += remote_desc2.len;

                i++;
                if(i == (desc_count-1)) break;

                local_desc2 = (*local_side->descs)[local_indices[i+1]];
                remote_desc2 = (*remote_side->descs)[remote_indices[i+1]];
            }
        }

        (*handle->initiatorDescs)[j] = local_desc1;
        (*handle->targetDescs)[j] = remote_desc1;
        j++;
        i++;
    }

    handle->initiatorDescs->resize(j);
    handle->targetDescs->resize(j);

    // To be added to logging
    //std::cout << "reqH descList size down to " << j << "\n";

    handle->engine      = local_side->engine;
    handle->remoteAgent = remote_side->remoteAgent;
    handle->notifMsg    = notif_msg;
    handle->backendOp   = operation;
    handle->state       = NIXL_XFER_INIT;

    req_handle = handle;
    return NIXL_SUCCESS;
}

void nixlAgent::invalidateXferSide(nixlXferSideH* side_handle) const {
    delete side_handle;
}

nixl_status_t nixlAgent::genNotif(const std::string &remote_agent,
                                  const std::string &msg,
                                  nixlBackendH* backend) {
    if (backend!=nullptr)
        return backend->engine->genNotif(remote_agent, msg);

    // TODO: add logic to choose between backends if multiple support it
    for (auto & eng: data->backendEngines) {
        if (eng.second->supportsNotif()) {
            if (data->remoteBackends[remote_agent].count(
                                    eng.second->getType()) != 0)
                return eng.second->genNotif(remote_agent, msg);
        }
    }
    return NIXL_ERR_NOT_FOUND;
}

int nixlAgent::getNotifs(nixl_notifs_t &notif_map) {
    notif_list_t backend_list;
    int ret, bad_ret=0, tot=0;
    bool any_backend = false;

    // Doing best effort, if any backend errors out we return
    // error but proceed with the rest. We can add metadata about
    // the backend to the msg, but user could put it themselves.
    for (auto & eng: data->backendEngines) {
        if (eng.second->supportsNotif()) {
            any_backend = true;
            backend_list.clear();
            ret = eng.second->getNotifs(backend_list);
            if (ret<0)
                bad_ret=ret;

            if (backend_list.size()==0)
                continue;

            for (auto & elm: backend_list) {
                if (notif_map.count(elm.first)==0)
                    notif_map[elm.first] = std::vector<std::string>();

                notif_map[elm.first].push_back(elm.second);
                tot++;
            }
        }
    }

    if (bad_ret)
        return bad_ret;
    else if (!any_backend)
        return -1;
    else
        return tot;
}

std::string nixlAgent::getLocalMD () const {
    // data->connMD was populated when the backend was created
    size_t conn_cnt = data->connMD.size();
    nixl_backend_t nixl_backend;

    if (conn_cnt == 0) // Error, no backend supports remote
        return "";

    nixlSerDes sd;
    if (sd.addStr("Agent", data->name)<0)
        return "";

    if (sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt)))
        return "";

    for (auto &c : data->connMD) {
        nixl_backend = c.first;
        if (sd.addStr("t", nixl_backend)<0)
            return "";
        if (sd.addStr("c", c.second)<0)
            return "";
    }

    if (sd.addStr("", "MemSection")<0)
        return "";

    if (data->memorySection.serialize(&sd)<0)
        return "";

    return sd.exportStr();
}

std::string nixlAgent::loadRemoteMD (const std::string &remote_metadata) {
    int count = 0;
    nixlSerDes sd;
    size_t conn_cnt;
    std::string conn_info;
    nixl_backend_t nixl_backend;
    nixlBackendEngine* eng;

    if (sd.importStr(remote_metadata)<0)
        return "";

    std::string remote_agent = sd.getStr("Agent");
    if (remote_agent.size()==0)
        return "";

    if (remote_agent == data->name)
        return "";

    if (sd.getBuf("Conns", &conn_cnt, sizeof(conn_cnt)))
        return "";

    if (conn_cnt<1)
        return "";

    for (size_t i=0; i<conn_cnt; ++i) {
        nixl_backend = sd.getStr("t");
        if (nixl_backend.size()==0)
            return "";
        conn_info = sd.getStr("c");
        if (conn_info.size()==0) // Fine if doing marginal updates
            return "";

        // Current agent might not support a remote backend
        if (data->backendEngines.count(nixl_backend)!=0) {

            // No need to reload same conn info, (TODO to cache the old val?)
            if (data->remoteBackends.count(remote_agent)!=0)
                if (data->remoteBackends[remote_agent].count(nixl_backend)!=0) {
                    count++;
                    continue;
                }

            eng = data->backendEngines[nixl_backend];
            if (eng->supportsRemote()) {
                if (eng->loadRemoteConnInfo(remote_agent, conn_info)
                                            != NIXL_SUCCESS)
                    return ""; // Error in load
                count++;
                data->remoteBackends[remote_agent].insert(nixl_backend);
            } else {
                return ""; // This is an erroneous case
            }
        }
    }

    // No common backend, no point in loading the rest, unexpected
    if (count == 0)
        return "";

    // If there was an issue and we return -1 while some connections
    // are loaded, they will be deleted in backend destructor.
    // the backend connection list for this agent will be empty.

    // It's just a check, not introducing section_info
    conn_info = sd.getStr("");
    if (conn_info != "MemSection")
        return "";

    if (data->remoteSections.count(remote_agent) == 0)
        data->remoteSections[remote_agent] = new nixlRemoteSection(
                            remote_agent, data->backendEngines);

    if (data->remoteSections[remote_agent]->loadRemoteData(&sd)<0) {
        delete data->remoteSections[remote_agent];
        data->remoteSections.erase(remote_agent);
        return "";
    }

    return remote_agent;
}

nixl_status_t nixlAgent::invalidateRemoteMD(const std::string &remote_agent) {
    if (remote_agent == data->name)
        return NIXL_ERR_BAD;

    nixl_status_t ret = NIXL_ERR_NOT_FOUND;
    if (data->remoteSections.count(remote_agent)!=0) {
        delete data->remoteSections[remote_agent];
        data->remoteSections.erase(remote_agent);
        ret = NIXL_SUCCESS;
    }

    if (data->remoteBackends.count(remote_agent)!=0) {
        for (auto & elm: data->remoteBackends[remote_agent])
            data->backendEngines[elm]->disconnect(remote_agent);
        data->remoteBackends.erase(remote_agent);
        ret = NIXL_SUCCESS;
    }

    return ret;
}
