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

#include <iostream>
#include "nixl.h"
#include "serdes/serdes.h"
#include "backend/backend_engine.h"
#include "transfer_request.h"
#include "agent_data.h"
#include "plugin_manager.h"

/*** nixlEnumStrings namespace implementation in API ***/
std::string nixlEnumStrings::memTypeStr(const nixl_mem_t &mem) {
    static std::array<std::string, FILE_SEG+1> nixl_mem_str = {
           "DRAM_SEG", "VRAM_SEG", "BLK_SEG", "OBJ_SEG", "FILE_SEG"};
    if (mem<DRAM_SEG || mem>FILE_SEG)
        return "BAD_SEG";
    return nixl_mem_str[mem];
}

std::string nixlEnumStrings::xferOpStr (const nixl_xfer_op_t &op) {
    static std::array<std::string, 2> nixl_op_str = {"READ", "WRITE"};
    if (op<NIXL_READ || op>NIXL_WRITE)
        return "BAD_OP";
    return nixl_op_str[op];

}

std::string nixlEnumStrings::statusStr (const nixl_status_t &status) {
    switch (status) {
        case NIXL_IN_PROG:           return "NIXL_IN_PROG";
        case NIXL_SUCCESS:           return "NIXL_SUCCESS";
        case NIXL_ERR_NOT_POSTED:    return "NIXL_ERR_NOT_POSTED";
        case NIXL_ERR_INVALID_PARAM: return "NIXL_ERR_INVALID_PARAM";
        case NIXL_ERR_BACKEND:       return "NIXL_ERR_BACKEND";
        case NIXL_ERR_NOT_FOUND:     return "NIXL_ERR_NOT_FOUND";
        case NIXL_ERR_MISMATCH:      return "NIXL_ERR_MISMATCH";
        case NIXL_ERR_NOT_ALLOWED:   return "NIXL_ERR_NOT_ALLOWED";
        case NIXL_ERR_REPOST_ACTIVE: return "NIXL_ERR_REPOST_ACTIVE";
        case NIXL_ERR_UNKNOWN:       return "NIXL_ERR_UNKNOWN";
        case NIXL_ERR_NOT_SUPPORTED: return "NIXL_ERR_NOT_SUPPORTED";
        default:                     return "BAD_STATUS";
    }
}


/*** nixlAgentData constructor/destructor, as part of nixlAgent's ***/
nixlAgentData::nixlAgentData(const std::string &name,
                             const nixlAgentConfig &cfg) :
                                   name(name), config(cfg) {
        memorySection = new nixlLocalSection();
}

nixlAgentData::~nixlAgentData() {
    delete memorySection;

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


/*** nixlAgent implementation ***/
nixlAgent::nixlAgent(const std::string &name,
                     const nixlAgentConfig &cfg) {
    if (name.size() == 0)
        throw std::invalid_argument("Agent needs a name");
    data = new nixlAgentData(name, cfg);
}

nixlAgent::~nixlAgent() {
    delete data;
}

nixl_status_t
nixlAgent::getAvailPlugins (std::vector<nixl_backend_t> &plugins) {
    auto& plugin_manager = nixlPluginManager::getInstance();
    plugins = plugin_manager.getLoadedPluginNames();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getPluginParams (const nixl_backend_t &type,
                            nixl_mem_list_t &mems,
                            nixl_b_params_t &params) const {

    // TODO: unify to uppercase/lowercase and do ltrim/rtrim for type

    // First try to get options from a loaded plugin
    auto& plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.getPlugin(type);

    if (plugin_handle) {
      // If the plugin is already loaded, get options directly
        params = plugin_handle->getBackendOptions();
        mems   = plugin_handle->getBackendMems();
        return NIXL_SUCCESS;
    }

    // If plugin isn't loaded yet, try to load it temporarily
    plugin_handle = plugin_manager.loadPlugin(type);
    if (plugin_handle) {
        params = plugin_handle->getBackendOptions();
        mems   = plugin_handle->getBackendMems();

        // We don't keep the plugin loaded if we didn't have it before
        if (data->backendEngines.count(type) == 0) {
            plugin_manager.unloadPlugin(type);
        }
        return NIXL_SUCCESS;
    }

    return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
nixlAgent::getBackendParams (const nixlBackendH* backend,
                             nixl_mem_list_t &mems,
                             nixl_b_params_t &params) const {
    if (!backend)
        return NIXL_ERR_INVALID_PARAM;

    mems   = backend->engine->getSupportedMems();
    params = backend->engine->getCustomParams();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::createBackend(const nixl_backend_t &type,
                         const nixl_b_params_t &params,
                         nixlBackendH* &bknd_hndl) {

    nixlBackendEngine*    backend = nullptr;
    nixlBackendInitParams init_params;
    nixl_mem_list_t       mems;
    nixl_status_t         ret;
    std::string           str;
    backend_list_t*       backend_list;

    // Registering same type of backend is not supported, unlikely and prob error
    if (data->backendEngines.count(type)!=0)
        return NIXL_ERR_INVALID_PARAM;

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
        std::cout << "Unsupported backend: " << type << std::endl;
        return NIXL_ERR_NOT_FOUND;
    }

    if (backend) {
        if (backend->getInitErr()) {
            delete backend;
            return NIXL_ERR_BACKEND;
        }

        if (backend->supportsRemote()) {
            ret = backend->getConnInfo(str);
            if (ret != NIXL_SUCCESS) {
                delete backend;
                return ret;
            }
            data->connMD[type] = str;
        }

        if (backend->supportsLocal()) {
            ret = backend->connect(data->name);

            if (NIXL_SUCCESS != ret) {
                delete backend;
                return ret;
            }
        }

        bknd_hndl = new nixlBackendH(backend);
        if (!bknd_hndl) {
            delete backend;
            return NIXL_ERR_BACKEND;
        }

        data->backendEngines[type] = backend;
        data->backendHandles[type] = bknd_hndl;
        mems = backend->getSupportedMems();
        for (auto & elm : mems) {
            backend_list = &data->memToBackend[elm];
            // First time creating this backend handle, so unique
            // The order of creation sets the preference order
            backend_list->push_back(backend);
        }

        if (backend->supportsRemote())
            data->notifEngines.push_back(backend);

        // TODO: Check if backend supports ProgThread
        //       when threading is in agent
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::registerMem(const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t* extra_params) {

    backend_list_t* backend_list;
    nixl_status_t   ret;
    unsigned int    count = 0;

    if (!extra_params || extra_params->backends.size() == 0) {
        backend_list = &data->memToBackend[descs.getType()];
        if (backend_list->empty())
            return NIXL_ERR_NOT_FOUND;
    } else {
        backend_list = new backend_list_t();
        for (auto & elm : extra_params->backends)
            backend_list->push_back(elm->engine);
    }

    // Best effort, if at least one succeeds NIXL_SUCCESS is returned
    // Can become more sophisticated to have a soft error case
    for (size_t i=0; i<backend_list->size(); ++i) {
        nixlBackendEngine* backend = (*backend_list)[i];
        // meta_descs use to be passed to loadLocalData
        nixl_meta_dlist_t meta_descs(descs.getType(), false);
        ret = data->memorySection->addDescList(descs, backend, meta_descs);
        if (ret == NIXL_SUCCESS) {
            if (backend->supportsLocal()) {
                if (data->remoteSections.count(data->name) == 0)
                    data->remoteSections[data->name] =
                          new nixlRemoteSection(data->name);

                ret = data->remoteSections[data->name]->loadLocalData(
                                                        meta_descs, backend);
                if (ret == NIXL_SUCCESS)
                    count++;
                else
                    data->memorySection->remDescList(meta_descs, backend);
            } else {
                count++;
            }
        } // a bad_ret can be saved in an else
    }

    if (extra_params && extra_params->backends.size() > 0)
        delete backend_list;

    if (count > 0)
        return NIXL_SUCCESS;
    else
        return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlAgent::deregisterMem(const nixl_reg_dlist_t &descs,
                         const nixl_opt_args_t* extra_params) {


    backend_set_t     backend_set;
    nixl_status_t     ret, bad_ret=NIXL_SUCCESS;
    nixl_xfer_dlist_t trimmed = descs.trim();

    if (!extra_params || extra_params->backends.size() == 0) {
        backend_set_t* avail_backends;
        avail_backends = data->memorySection->queryBackends(
                                              descs.getType());
        if (!avail_backends || avail_backends->empty())
            return NIXL_ERR_NOT_FOUND;
        // Make a copy as we might change it in remDescList
        backend_set = *avail_backends;
    } else {
        for (auto & elm : extra_params->backends)
            backend_set.insert(elm->engine);
    }

    // Doing best effort, and returning err if any
    for (auto & backend : backend_set) {
        nixl_meta_dlist_t resp(descs.getType(),
                               descs.isSorted());

        // Now just populating the metadata part, inside remDescList,
        // getIndex is used to make sure exact match
        ret = data->memorySection->populate(trimmed, backend, resp);
        if (ret != NIXL_SUCCESS)
            bad_ret = ret;
        ret = data->memorySection->remDescList(resp, backend);
        if (ret != NIXL_SUCCESS)
            bad_ret = ret;
    }

    return bad_ret;
}

nixl_status_t
nixlAgent::makeConnection(const std::string &remote_agent,
                          const nixl_opt_args_t* extra_params) {
    nixlBackendEngine* eng;
    nixl_status_t ret;
    std::set<nixl_backend_t>* backend_set;
    int count = 0;

    if (data->remoteBackends.count(remote_agent) == 0)
        return NIXL_ERR_NOT_FOUND;

    if (!extra_params || extra_params->backends.size() == 0) {
        backend_set = &data->remoteBackends[remote_agent];
        if (backend_set->empty())
            return NIXL_ERR_NOT_FOUND;
    } else {
        backend_set = new std::set<nixl_backend_t>();
        for (auto & elm : extra_params->backends)
            backend_set->insert(elm->engine->getType());
    }

    // For now trying to make all the connections, can become best effort,
    for (auto & backend: *backend_set) {
        if (data->backendEngines.count(backend)!=0) {
            eng = data->backendEngines[backend];
            ret = eng->connect(remote_agent);
            if (ret)
                break;
            count++;
        }
    }

    if (extra_params && extra_params->backends.size() > 0)
        delete backend_set;

    if (ret)
        return ret;
    else if (count == 0) // No common backend
        return NIXL_ERR_BACKEND;
    else
        return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::prepXferDlist (const std::string &agent_name,
                          const nixl_xfer_dlist_t &descs,
                          nixlDlistH* &dlist_hndl,
                          const nixl_opt_args_t* extra_params) const {

    // Using a set as order is not important to revert the operation
    backend_set_t* backend_set;
    nixl_status_t  ret;
    int            count = 0;
    bool           init_side = (agent_name == NIXL_INIT_AGENT);

    // When central KV is supported, still it should return error,
    // just we can add a call to fetchRemoteMD for next time
    if (!init_side && (data->remoteSections.count(agent_name) == 0))
        return NIXL_ERR_NOT_FOUND;

    if (!extra_params || extra_params->backends.size() == 0) {
        if (!init_side)
            backend_set = data->remoteSections[agent_name]->
                                queryBackends(descs.getType());
        else
            backend_set = data->memorySection->
                                queryBackends(descs.getType());

        if (!backend_set || backend_set->empty())
            return NIXL_ERR_NOT_FOUND;
    } else {
        backend_set = new backend_set_t();
        for (auto & elm : extra_params->backends)
            backend_set->insert(elm->engine);
    }

    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    nixlDlistH *handle = new nixlDlistH;
    if (init_side) {
        handle->isLocal     = true;
        handle->remoteAgent = "";
    } else {
        handle->isLocal     = false;
        handle->remoteAgent = agent_name;
    }

    for (auto & backend : *backend_set) {
        handle->descs[backend] = new nixl_meta_dlist_t (
                                         descs.getType(),
                                         descs.isSorted());
        if (init_side)
            ret = data->memorySection->populate(
                       descs, backend, *(handle->descs[backend]));
        else
            ret = data->remoteSections[agent_name]->populate(
                       descs, backend, *(handle->descs[backend]));
        if (ret == NIXL_SUCCESS) {
            count++;
        } else {
            delete handle->descs[backend];
            handle->descs.erase(backend);
        }
    }

    if (extra_params && extra_params->backends.size() > 0)
        delete backend_set;

    if (count == 0) {
        delete handle;
        dlist_hndl = nullptr;
        return NIXL_ERR_NOT_FOUND;
    } else {
        dlist_hndl = handle;
        return NIXL_SUCCESS;
    }
}

nixl_status_t
nixlAgent::makeXferReq (const nixl_xfer_op_t &operation,
                        const nixlDlistH* local_side,
                        const std::vector<int> &local_indices,
                        const nixlDlistH* remote_side,
                        const std::vector<int> &remote_indices,
                        nixlXferReqH* &req_hndl,
                        const nixl_opt_args_t* extra_params) const {

    nixl_opt_b_args_t  opt_args;
    nixl_status_t      ret;
    int                desc_count = (int) local_indices.size();
    nixlBackendEngine* backend    = nullptr;

    req_hndl = nullptr;

    if (!local_side || !remote_side)
        return NIXL_ERR_INVALID_PARAM;

    if ((!local_side->isLocal) || (remote_side->isLocal))
        return NIXL_ERR_INVALID_PARAM;

    // The remote was invalidated in between prepXferDlist and this call
    if (data->remoteSections.count(remote_side->remoteAgent) == 0) {
        delete req_hndl;
        return NIXL_ERR_NOT_FOUND;
    }

    if (extra_params && extra_params->backends.size() > 0) {
        for (auto & elm : extra_params->backends) {
            if ((local_side->descs.count(elm->engine) > 0) &&
                (remote_side->descs.count(elm->engine) > 0)) {
                backend = elm->engine;
                break;
            }
        }
    } else {
        for (auto & loc_bknd : local_side->descs) {
            for (auto & rem_bknd : remote_side->descs) {
                if (loc_bknd.first == rem_bknd.first) {
                    backend = loc_bknd.first;
                    break;
                }
            }
            if (backend)
                break;
        }
    }

    if (!backend)
        return NIXL_ERR_INVALID_PARAM;

    nixl_meta_dlist_t* local_descs  = local_side->descs.at(backend);
    nixl_meta_dlist_t* remote_descs = remote_side->descs.at(backend);

    if ((desc_count == 0) || (remote_indices.size() == 0) ||
        (desc_count != (int) remote_indices.size()))
        return NIXL_ERR_INVALID_PARAM;

    for (int i=0; i<desc_count; ++i) {
        if ((local_indices[i] >= local_descs->descCount())
               || (local_indices[i]<0))
            return NIXL_ERR_INVALID_PARAM;
        if ((remote_indices[i] >= remote_descs->descCount())
               || (remote_indices[i]<0))
            return NIXL_ERR_INVALID_PARAM;
        if ((*local_descs )[local_indices [i]].len !=
            (*remote_descs)[remote_indices[i]].len)
            return NIXL_ERR_INVALID_PARAM;
    }

    if (extra_params && extra_params->hasNotif) {
        opt_args.notifMsg = extra_params->notifMsg;
        opt_args.hasNotif = true;
    }

    if ((opt_args.hasNotif) && (!backend->supportsNotif())) {
        return NIXL_ERR_BACKEND;
    }

    // Populate has been already done, no benefit in having sorted descriptors
    // which will be overwritten by [] assignment operator.
    nixlXferReqH* handle   = new nixlXferReqH;
    handle->initiatorDescs = new nixl_meta_dlist_t (
                                     local_descs->getType(),
                                     false, desc_count);

    handle->targetDescs    = new nixl_meta_dlist_t (
                                     remote_descs->getType(),
                                     false, desc_count);

    if (extra_params && extra_params->skipDescMerge) {
        for (int i=0; i<desc_count; ++i) {
            (*handle->initiatorDescs)[i] =
                                     (*local_descs)[local_indices[i]];
            (*handle->targetDescs)[i] =
                                     (*remote_descs)[remote_indices[i]];
        }
    } else {
        int i = 0, j = 0; //final list size
        while (i<(desc_count)) {
            nixlMetaDesc local_desc1  = (*local_descs) [local_indices[i]];
            nixlMetaDesc remote_desc1 = (*remote_descs)[remote_indices[i]];

            if(i != (desc_count-1) ) {
                nixlMetaDesc* local_desc2  = &((*local_descs) [local_indices[i+1]]);
                nixlMetaDesc* remote_desc2 = &((*remote_descs)[remote_indices[i+1]]);

              while (((local_desc1.addr + local_desc1.len) == local_desc2->addr)
                  && ((remote_desc1.addr + remote_desc1.len) == remote_desc2->addr)
                  && (local_desc1.metadataP == local_desc2->metadataP)
                  && (remote_desc1.metadataP == remote_desc2->metadataP)
                  && (local_desc1.devId == local_desc2->devId)
                  && (remote_desc1.devId == remote_desc2->devId)) {

                    local_desc1.len += local_desc2->len;
                    remote_desc1.len += remote_desc2->len;

                    i++;
                    if(i == (desc_count-1)) break;

                    local_desc2  = &((*local_descs) [local_indices[i+1]]);
                    remote_desc2 = &((*remote_descs)[remote_indices[i+1]]);
                }
            }

            (*handle->initiatorDescs)[j] = local_desc1;
            (*handle->targetDescs)   [j] = remote_desc1;
            j++;
            i++;
        }

        handle->initiatorDescs->resize(j);
        handle->targetDescs->resize(j);
    }

    // To be added to logging
    //std::cout << "reqH descList size down to " << j << "\n";

    handle->engine      = backend;
    handle->remoteAgent = remote_side->remoteAgent;
    handle->notifMsg    = opt_args.notifMsg;
    handle->hasNotif    = opt_args.hasNotif;
    handle->backendOp   = operation;
    handle->status      = NIXL_ERR_NOT_POSTED;

    ret = handle->engine->prepXfer (handle->backendOp,
                                    *handle->initiatorDescs,
                                    *handle->targetDescs,
                                    handle->remoteAgent,
                                    handle->backendHandle,
                                    &opt_args);
    if (ret != NIXL_SUCCESS) {
        delete handle;
        return ret;
    }

    req_hndl = handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::createXferReq(const nixl_xfer_op_t &operation,
                         const nixl_xfer_dlist_t &local_descs,
                         const nixl_xfer_dlist_t &remote_descs,
                         const std::string &remote_agent,
                         nixlXferReqH* &req_hndl,
                         const nixl_opt_args_t* extra_params) const {
    nixl_status_t     ret1, ret2;
    nixl_opt_b_args_t opt_args;
    backend_set_t*    backend_set = new backend_set_t();

    req_hndl = nullptr;

    if (data->remoteSections.count(remote_agent) == 0)
        return NIXL_ERR_NOT_FOUND;

    // Check the correspondence between descriptor lists
    if (local_descs.descCount() != remote_descs.descCount())
        return NIXL_ERR_INVALID_PARAM;
    for (int i=0; i<local_descs.descCount(); ++i)
        if (local_descs[i].len != remote_descs[i].len)
            return NIXL_ERR_INVALID_PARAM;

    if (!extra_params || extra_params->backends.size() == 0) {
        // Finding backends that support the corresponding memories
        // locally and remotely, and find the common ones.
        backend_set_t* local_set =
            data->memorySection->queryBackends(local_descs.getType());
        backend_set_t* remote_set =
            data->remoteSections[remote_agent]->queryBackends(
                                                remote_descs.getType());
        if (!local_set || !remote_set) {
            delete backend_set;
            return NIXL_ERR_NOT_FOUND;
        }

        for (auto & elm : *local_set)
            if (remote_set->count(elm) != 0)
                backend_set->insert(elm);

        if (backend_set->empty()) {
            delete backend_set;
            return NIXL_ERR_NOT_FOUND;
        }
    } else {
        for (auto & elm : extra_params->backends)
            backend_set->insert(elm->engine);
    }

    // TODO: when central KV is supported, add a call to fetchRemoteMD
    // TODO: merge descriptors back to back in memory (like makeXferReq).
    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    nixlXferReqH *handle   = new nixlXferReqH;
    handle->initiatorDescs = new nixl_meta_dlist_t (
                                     local_descs.getType(),
                                     local_descs.isSorted());

    handle->targetDescs    = new nixl_meta_dlist_t (
                                     remote_descs.getType(),
                                     remote_descs.isSorted());

    // Currently we loop through and find first local match. Can use a
    // preference list or more exhaustive search.
    for (auto & backend : *backend_set) {
        // If populate fails, it clears the resp before return
        ret1 = data->memorySection->populate(
                     local_descs, backend, *handle->initiatorDescs);
        ret2 = data->remoteSections[remote_agent]->populate(
                     remote_descs, backend, *handle->targetDescs);

        if ((ret1 == NIXL_SUCCESS) && (ret2 == NIXL_SUCCESS)) {
            // For Logging:
            // std::cout << "Selected backend: " << backend->getType() << "\n";
            handle->engine = backend;
            break;
        }
    }

    delete backend_set;

    if (!handle->engine) {
        delete handle;
        return NIXL_ERR_NOT_FOUND;
    }

    if (extra_params && extra_params->hasNotif) {
        opt_args.notifMsg = extra_params->notifMsg;
        opt_args.hasNotif = true;
    }

    if (opt_args.hasNotif && (!handle->engine->supportsNotif())) {
        delete handle;
        return NIXL_ERR_BACKEND;
    }

    handle->remoteAgent = remote_agent;
    handle->backendOp   = operation;
    handle->status      = NIXL_ERR_NOT_POSTED;
    handle->notifMsg    = opt_args.notifMsg;
    handle->hasNotif    = opt_args.hasNotif;

    ret1 = handle->engine->prepXfer (handle->backendOp,
                                     *handle->initiatorDescs,
                                     *handle->targetDescs,
                                     handle->remoteAgent,
                                     handle->backendHandle,
                                     &opt_args);
    if (ret1 != NIXL_SUCCESS) {
        delete handle;
        return ret1;
    }

    req_hndl = handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::postXferReq(nixlXferReqH *req_hndl,
                       const nixl_opt_args_t* extra_params) const {
    nixl_status_t ret;
    nixl_opt_b_args_t opt_args;

    opt_args.hasNotif = false;

    if (!req_hndl)
        return NIXL_ERR_INVALID_PARAM;

    // Check if the remote was invalidated before post/repost
    if (data->remoteSections.count(req_hndl->remoteAgent) == 0) {
        delete req_hndl;
        return NIXL_ERR_NOT_FOUND;
    }

    // We can't repost while a request is in progress
    if (req_hndl->status == NIXL_IN_PROG) {
        req_hndl->status = req_hndl->engine->checkXfer(
                                     req_hndl->backendHandle);
        if (req_hndl->status == NIXL_IN_PROG) {
            delete req_hndl;
            return NIXL_ERR_REPOST_ACTIVE;
        }
    }

    // We CAN repost a previous request that is completed
    if (req_hndl->status == NIXL_SUCCESS && req_hndl->backendHandle)
        req_hndl->status = req_hndl->engine->releaseReqH(
                                     req_hndl->backendHandle);

    // Carrying over notification from xfer handle creation time
    if (req_hndl->hasNotif) {
        opt_args.notifMsg = req_hndl->notifMsg;
        opt_args.hasNotif = true;
    }

    // Updating the notification based on opt_args
    if (extra_params) {
        if (extra_params->hasNotif) {
            req_hndl->notifMsg = extra_params->notifMsg;
            opt_args.notifMsg  = extra_params->notifMsg;
            req_hndl->hasNotif = true;
            opt_args.hasNotif  = true;
        } else {
            req_hndl->hasNotif = false;
            opt_args.hasNotif  = false;
        }
    }

    if (opt_args.hasNotif && (!req_hndl->engine->supportsNotif())) {
        delete req_hndl;
        return NIXL_ERR_BACKEND;
    }

    // If status is not NIXL_IN_PROG we can repost,
    ret = req_hndl->engine->postXfer (req_hndl->backendOp,
                                     *req_hndl->initiatorDescs,
                                     *req_hndl->targetDescs,
                                      req_hndl->remoteAgent,
                                      req_hndl->backendHandle,
                                      &opt_args);
    req_hndl->status = ret;
    return ret;
}

nixl_status_t
nixlAgent::getXferStatus (nixlXferReqH *req_hndl) {

    // If the status is done, no need to recheck.
    if (req_hndl->status != NIXL_SUCCESS) {
        // Check if the remote was invalidated before completion
        if (data->remoteSections.count(req_hndl->remoteAgent) == 0) {
            delete req_hndl;
            return NIXL_ERR_NOT_FOUND;
        }
        req_hndl->status = req_hndl->engine->checkXfer(
                                     req_hndl->backendHandle);
    }

    return req_hndl->status;
}


nixl_status_t
nixlAgent::queryXferBackend(const nixlXferReqH* req_hndl,
                            nixlBackendH* &backend) const {
    backend = data->backendHandles[req_hndl->engine->getType()];
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::releaseXferReq(nixlXferReqH *req_hndl) {

    //attempt to cancel request
    if(req_hndl->status == NIXL_IN_PROG) {
        req_hndl->status = req_hndl->engine->checkXfer(
                                     req_hndl->backendHandle);

        if(req_hndl->status == NIXL_IN_PROG) {

            req_hndl->status = req_hndl->engine->releaseReqH(
                                         req_hndl->backendHandle);

            if(req_hndl->status < 0)
                return NIXL_ERR_REPOST_ACTIVE;

            // just in case the backend doesn't set to NULL on success
            // this will prevent calling releaseReqH again in destructor
            req_hndl->backendHandle = nullptr;
        }
    }
    delete req_hndl;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::releasedDlistH (nixlDlistH* dlist_hndl) const {
    delete dlist_hndl;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getNotifs(nixl_notifs_t &notif_map,
                     const nixl_opt_args_t* extra_params) {
    notif_list_t    bknd_notif_list;
    nixl_status_t   ret, bad_ret=NIXL_SUCCESS;
    backend_list_t* backend_list;

    if (!extra_params || extra_params->backends.size() == 0) {
        backend_list = &data->notifEngines;
        if (backend_list->empty())
            return NIXL_ERR_BACKEND;
    } else {
        backend_list = new backend_list_t();
        for (auto & elm : extra_params->backends)
            if (elm->engine->supportsNotif())
                backend_list->push_back(elm->engine);

        if (backend_list->empty()) {
            delete backend_list;
            return NIXL_ERR_BACKEND;
        }
    }

    // Doing best effort, if any backend errors out we return
    // error but proceed with the rest. We can add metadata about
    // the backend to the msg, but user could put it themselves.
    for (auto & eng: *backend_list) {
        bknd_notif_list.clear();
        ret = eng->getNotifs(bknd_notif_list);
        if (ret < 0)
            bad_ret=ret;

        if (bknd_notif_list.size() == 0)
            continue;

        for (auto & elm: bknd_notif_list) {
            if (notif_map.count(elm.first) == 0)
                notif_map[elm.first] = std::vector<nixl_blob_t>();

            notif_map[elm.first].push_back(elm.second);
        }
    }

    if (extra_params && extra_params->backends.size() > 0)
        delete backend_list;

    return bad_ret;
}

nixl_status_t
nixlAgent::genNotif(const std::string &remote_agent,
                    const nixl_blob_t &msg,
                    const nixl_opt_args_t* extra_params) {

    nixlBackendEngine* backend = nullptr;
    backend_list_t*    backend_list;

    if (!extra_params || extra_params->backends.size() == 0) {
        backend_list = &data->notifEngines;
        if (backend_list->empty())
            return NIXL_ERR_BACKEND;
    } else {
        backend_list = new backend_list_t();
        for (auto & elm : extra_params->backends)
            if (elm->engine->supportsNotif())
                backend_list->push_back(elm->engine);

        if (backend_list->empty()) {
            delete backend_list;
            return NIXL_ERR_BACKEND;
        }
    }

    for (auto & eng: *backend_list) {
        if (data->remoteBackends[remote_agent].count(
                                 eng->getType()) != 0) {
            backend = eng;
            break;
        }
    }

    if (extra_params && extra_params->backends.size() > 0)
        delete backend_list;

    if (backend)
        return backend->genNotif(remote_agent, msg);
    else
        return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
nixlAgent::getLocalMD (nixl_blob_t &str) const {
    // data->connMD was populated when the backend was created
    size_t conn_cnt = data->connMD.size();
    nixl_backend_t nixl_backend;
    nixl_status_t ret;

    if (conn_cnt == 0) // Error, no backend supports remote
        return NIXL_ERR_INVALID_PARAM;

    nixlSerDes sd;
    ret = sd.addStr("Agent", data->name);
    if(ret)
        return ret;

    ret = sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if(ret)
        return ret;

    for (auto &c : data->connMD) {
        nixl_backend = c.first;
        ret = sd.addStr("t", nixl_backend);
        if(ret)
            return ret;
        ret = sd.addStr("c", c.second);
        if(ret)
            return ret;
    }

    ret = sd.addStr("", "MemSection");
    if(ret)
        return ret;

    ret = data->memorySection->serialize(&sd);
    if(ret)
        return ret;

    str = sd.exportStr();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::loadRemoteMD (const nixl_blob_t &remote_metadata,
                         std::string &agent_name) {
    int count = 0;
    nixlSerDes sd;
    size_t conn_cnt;
    std::string conn_info;
    nixl_backend_t nixl_backend;
    nixlBackendEngine* eng;
    nixl_status_t ret;

    ret = sd.importStr(remote_metadata);
    if(ret)
        return ret;

    std::string remote_agent = sd.getStr("Agent");
    if (remote_agent.size() == 0)
        return NIXL_ERR_MISMATCH;

    if (remote_agent == data->name)
        return NIXL_ERR_INVALID_PARAM;

    ret = sd.getBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if(ret)
        return ret;

    // TODO: add support for marginal updates, then conn_cnt might be 0
    if (conn_cnt<1)
        return NIXL_ERR_INVALID_PARAM;

    for (size_t i=0; i<conn_cnt; ++i) {
        nixl_backend = sd.getStr("t");
        if (nixl_backend.size() == 0)
            return NIXL_ERR_MISMATCH;
        conn_info = sd.getStr("c");
        if (conn_info.size() == 0)
            return NIXL_ERR_MISMATCH;

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
                ret = eng->loadRemoteConnInfo(remote_agent, conn_info);
                if (ret)
                    return ret; // Error in load
                count++;
                data->remoteBackends[remote_agent].insert(nixl_backend);
            } else {
                // If there was an issue and we return error while some connections
                // are loaded, they will be deleted in the backend destructor.
                return NIXL_ERR_UNKNOWN; // This is an erroneous case
            }
        }
    }

    // No common backend, no point in loading the rest, unexpected
    if (count == 0)
        return NIXL_ERR_BACKEND;

    if (sd.getStr("") != "MemSection")
        return NIXL_ERR_MISMATCH;

    if (data->remoteSections.count(remote_agent) == 0)
        data->remoteSections[remote_agent] = new nixlRemoteSection(
                                                  remote_agent);

    ret = data->remoteSections[remote_agent]->loadRemoteData(&sd,
                                                  data->backendEngines);

    // TODO: can be more graceful, if just the new MD blob was improper
    if (ret) {
        delete data->remoteSections[remote_agent];
        data->remoteSections.erase(remote_agent);
        return ret;
    }

    agent_name = remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::invalidateRemoteMD(const std::string &remote_agent) {
    if (remote_agent == data->name)
        return NIXL_ERR_INVALID_PARAM;

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
