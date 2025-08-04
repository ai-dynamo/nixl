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
#include <chrono>
#include <iostream>
#include "nixl.h"
#include "serdes/serdes.h"
#include "backend/backend_engine.h"
#include "transfer_request.h"
#include "agent_data.h"
#include "plugin_manager.h"
#include "common/nixl_log.h"

static const std::vector<std::vector<std::string>> illegal_plugin_combinations = {
    {"GDS", "GDS_MT"},
};

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
        case NIXL_IN_PROG:               return "NIXL_IN_PROG";
        case NIXL_SUCCESS:               return "NIXL_SUCCESS";
        case NIXL_ERR_NOT_POSTED:        return "NIXL_ERR_NOT_POSTED";
        case NIXL_ERR_INVALID_PARAM:     return "NIXL_ERR_INVALID_PARAM";
        case NIXL_ERR_BACKEND:           return "NIXL_ERR_BACKEND";
        case NIXL_ERR_NOT_FOUND:         return "NIXL_ERR_NOT_FOUND";
        case NIXL_ERR_MISMATCH:          return "NIXL_ERR_MISMATCH";
        case NIXL_ERR_NOT_ALLOWED:       return "NIXL_ERR_NOT_ALLOWED";
        case NIXL_ERR_REPOST_ACTIVE:     return "NIXL_ERR_REPOST_ACTIVE";
        case NIXL_ERR_UNKNOWN:           return "NIXL_ERR_UNKNOWN";
        case NIXL_ERR_NOT_SUPPORTED:     return "NIXL_ERR_NOT_SUPPORTED";
        case NIXL_ERR_REMOTE_DISCONNECT: return "NIXL_ERR_REMOTE_DISCONNECT";
        default:                         return "BAD_STATUS";
    }
}

/*** nixlXferReqH telemetry update method, used mainly in the nixlAgent ***/
void
nixlXferReqH::updateRequestStats(const std::string &dbg_msg_type) {
    const auto xfer_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - telemetry.startTime);
    // If endTime needs to be recorded per Xfer, now() value here can be returned

    // To be replaced with NIXL_DEBUG when full telemetry is added
    std::cout << "[NIXL TELEMETRY]: From backend " << engine->getType() << " " << dbg_msg_type
              << " Xfer with " << initiatorDescs->descCount() << " descriptors of total size "
              << telemetry.totalBytes << "B in " << xfer_time.count() << "us." << std::endl;
}

/*** nixlAgentData constructor/destructor, as part of nixlAgent's ***/
nixlAgentData::nixlAgentData(const std::string &name,
                             const nixlAgentConfig &cfg) :
                                   name(name), config(cfg), lock(cfg.syncMode)
{
#if HAVE_ETCD
    if (getenv("NIXL_ETCD_ENDPOINTS")) {
        useEtcd = true;
        NIXL_DEBUG << "NIXL ETCD is enabled";
    } else {
        useEtcd = false;
        NIXL_DEBUG << "NIXL ETCD is disabled";
    }
#endif // HAVE_ETCD
    if (name.empty())
        throw std::invalid_argument("Agent needs a name");

    memorySection = new nixlLocalSection();

    const char *telemetry = std::getenv("NIXL_TELEMETRY_ENABLE");
    if (telemetry != nullptr) {
        if (!strcasecmp(telemetry, "y"))
            telemetryEnabled = true;
        else if (!strcasecmp(telemetry, "n"))
            telemetryEnabled = false;
        else
            NIXL_WARN
                << "Invalid NIXL_TELEMETRY_ENABLE environment variable, not enabling telemetry.";
    }
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
nixlAgent::nixlAgent(const std::string &name, const nixlAgentConfig &cfg) :
    data(std::make_unique<nixlAgentData>(name, cfg))
{
    if(cfg.useListenThread) {
        int my_port = cfg.listenPort;
        if(my_port == 0) my_port = default_comm_port;
        data->listener = new nixlMDStreamListener(my_port);
        data->listener->setupListener();
    }

    if (data->useEtcd || cfg.useListenThread) {
        data->commThreadStop = false;
        data->commThread =
            std::thread(&nixlAgentData::commWorker, data.get(), this);
    }
}

nixlAgent::~nixlAgent() {
    if (data && (data->useEtcd || data->config.useListenThread)) {
        data->commThreadStop = true;
        if(data->commThread.joinable()) data->commThread.join();

        // Close remaining connections from comm thread
        for (auto &[remote, fd] : data->remoteSockets) {
            shutdown(fd, SHUT_RDWR);
            close(fd);
        }

        if(data->config.useListenThread) {
            if(data->listener) delete data->listener;
        }
    }
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

        NIXL_LOCK_GUARD(data->lock);

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

    NIXL_LOCK_GUARD(data->lock);
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

    NIXL_LOCK_GUARD(data->lock);
    // Registering same type of backend is not supported, unlikely and prob error
    if (data->backendEngines.count(type)!=0)
        return NIXL_ERR_INVALID_PARAM;

    // Check if the plugin is in an illegal combination with another plugin backend already created
    for (const auto &combination : illegal_plugin_combinations) {
        if (std::find(combination.begin(), combination.end(), type) != combination.end()) {
            for (const auto &plugin_name : combination) {
                if (plugin_name != type &&
                    data->backendEngines.find(plugin_name) != data->backendEngines.end()) {
                    NIXL_ERROR << "Plugin backend " << type << " is in illegal combination with "
                               << plugin_name;
                    return NIXL_ERR_NOT_ALLOWED;
                }
            }
        }
    }

    init_params.localAgent   = data->name;
    init_params.type         = type;
    init_params.customParams = const_cast<nixl_b_params_t*>(&params);
    init_params.enableProgTh = data->config.useProgThread;
    init_params.pthrDelay    = data->config.pthrDelay;
    init_params.syncMode     = data->config.syncMode;

    // First, try to load the backend as a plugin
    auto& plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.loadPlugin(type);

    if (plugin_handle) {
        // Plugin found, use it to create the backend
        backend = plugin_handle->createEngine(&init_params);
    } else {
        NIXL_ERROR << "Unsupported backend: " << type;
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

        NIXL_DEBUG << "Created backend: " << type;

        return NIXL_SUCCESS;
    }

    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlAgent::queryMem(const nixl_reg_dlist_t &descs,
                    std::vector<nixl_query_resp_t> &resp,
                    const nixl_opt_args_t *extra_params) const {

    if (!extra_params || extra_params->backends.size() != 1) {
        return NIXL_ERR_INVALID_PARAM;
    }

    return extra_params->backends[0]->engine->queryMem(descs, resp);
}

nixl_status_t
nixlAgent::registerMem(const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t* extra_params) {

    backend_list_t* backend_list;
    nixl_status_t   ret;
    unsigned int    count = 0;

    NIXL_LOCK_GUARD(data->lock);
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
        nixl_sec_dlist_t sec_descs(descs.getType(), false);
        ret = data->memorySection->addDescList(descs, backend, sec_descs);
        if (ret == NIXL_SUCCESS) {
            if (backend->supportsLocal()) {
                if (data->remoteSections.count(data->name) == 0)
                    data->remoteSections[data->name] =
                          new nixlRemoteSection(data->name);

                ret = data->remoteSections[data->name]->loadLocalData(
                                                        sec_descs, backend);
                if (ret == NIXL_SUCCESS)
                    count++;
                else
                    data->memorySection->remDescList(descs, backend);
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

    NIXL_LOCK_GUARD(data->lock);
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
        ret = data->memorySection->remDescList(descs, backend);
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
    std::set<nixl_backend_t> backend_set;
    int count = 0;

    NIXL_LOCK_GUARD(data->lock);
    if (data->remoteBackends.count(remote_agent) == 0)
        return NIXL_ERR_NOT_FOUND;

    if (!extra_params || extra_params->backends.size() == 0) {
        if (data->remoteBackends[remote_agent].empty())
            return NIXL_ERR_NOT_FOUND;
        for (auto & [r_bknd, conn_info] : data->remoteBackends[remote_agent])
            backend_set.insert(r_bknd);
    } else {
        for (auto & elm : extra_params->backends)
            backend_set.insert(elm->engine->getType());
    }

    // For now trying to make all the connections, can become best effort,
    for (auto & backend: backend_set) {
        if (data->backendEngines.count(backend)!=0) {
            eng = data->backendEngines[backend];
            ret = eng->connect(remote_agent);
            if (ret)
                break;
            count++;
        }
    }

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

    NIXL_LOCK_GUARD(data->lock);
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

    NIXL_LOCK_GUARD(data->lock);
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
    size_t totalBytes = 0;

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
        totalBytes += (*local_descs)[local_indices[i]].len;
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
        NIXL_DEBUG << "reqH descList size down to " << j;
        handle->initiatorDescs->resize(j);
        handle->targetDescs->resize(j);
    }

    handle->engine = backend;
    handle->remoteAgent = remote_side->remoteAgent;
    handle->notifMsg = opt_args.notifMsg;
    handle->hasNotif = opt_args.hasNotif;
    handle->backendOp = operation;
    handle->status = NIXL_ERR_NOT_POSTED;
    handle->telemetry.totalBytes = totalBytes;

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

    NIXL_SHARED_LOCK_GUARD(data->lock);
    if (data->remoteSections.count(remote_agent) == 0)
    {
        delete backend_set;
        return NIXL_ERR_NOT_FOUND;
    }

    // Check the correspondence between descriptor lists
    size_t totalBytes = 0;
    if (local_descs.descCount() != remote_descs.descCount())
        return NIXL_ERR_INVALID_PARAM;
    for (int i = 0; i < local_descs.descCount(); ++i) {
        if (local_descs[i].len != remote_descs[i].len)
            return NIXL_ERR_INVALID_PARAM;
        totalBytes += local_descs[i].len;
    }

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
            NIXL_INFO << "Selected backend: " << backend->getType();
            handle->engine = backend;
            break;
        }
    }

    delete backend_set;

    if (!handle->engine) {
        delete handle;
        return NIXL_ERR_NOT_FOUND;
    }

    if (extra_params) {
        if (extra_params->hasNotif) {
            opt_args.notifMsg = extra_params->notifMsg;
            opt_args.hasNotif = true;
        }

        if (extra_params->customParam.length() > 0)
            opt_args.customParam = extra_params->customParam;
    }

    if (opt_args.hasNotif && (!handle->engine->supportsNotif())) {
        delete handle;
        return NIXL_ERR_BACKEND;
    }

    handle->remoteAgent = remote_agent;
    handle->backendOp = operation;
    handle->status = NIXL_ERR_NOT_POSTED;
    handle->notifMsg = opt_args.notifMsg;
    handle->hasNotif = opt_args.hasNotif;
    handle->telemetry.totalBytes = totalBytes;

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
nixlAgent::estimateXferCost(const nixlXferReqH *req_hndl,
                            std::chrono::microseconds &duration,
                            std::chrono::microseconds &err_margin,
                            nixl_cost_t &method,
                            const nixl_opt_args_t* extra_params) const
{
    NIXL_SHARED_LOCK_GUARD(data->lock);

    // Check if the remote agent connection info is still valid
    // (assuming cost estimation requires connection info like transfers)
    if (!req_hndl->remoteAgent.empty() &&
        (data->remoteSections.count(req_hndl->remoteAgent) == 0)) {
        NIXL_ERROR << "Invalid request handle: remote agent not found";
        return NIXL_ERR_NOT_FOUND;
    }

    if (!req_hndl->engine) {
        NIXL_ERROR << "Invalid request handle: engine is null";
        return NIXL_ERR_UNKNOWN;
    }

    return req_hndl->engine->estimateXferCost(req_hndl->backendOp,
                                              *req_hndl->initiatorDescs,
                                              *req_hndl->targetDescs,
                                              req_hndl->remoteAgent,
                                              req_hndl->backendHandle,
                                              duration,
                                              err_margin,
                                              method,
                                              extra_params);
}

nixl_status_t
nixlAgent::postXferReq(nixlXferReqH *req_hndl,
                       const nixl_opt_args_t* extra_params) const {
    nixl_status_t ret;
    nixl_opt_b_args_t opt_args;

    opt_args.hasNotif = false;

    if (!req_hndl)
        return NIXL_ERR_INVALID_PARAM;

    // The initial checks should be fast if post succeeds, including them in the overall time
    if (data->telemetryEnabled) {
        req_hndl->telemetry.startTime = std::chrono::high_resolution_clock::now();
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);
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

    if (data->telemetryEnabled) {
        if (req_hndl->status == NIXL_SUCCESS)
            req_hndl->updateRequestStats("Posted and Completed");
        else if (req_hndl->status == NIXL_IN_PROG)
            req_hndl->updateRequestStats("Posted");
        // Errors should show up in debug log separately, not adding a print here
    }

    return ret;
}

nixl_status_t
nixlAgent::getXferStatus (nixlXferReqH *req_hndl) const {

    NIXL_SHARED_LOCK_GUARD(data->lock);
    // If the status is done, no need to recheck.
    if (req_hndl->status == NIXL_IN_PROG) {
        // Check if the remote was invalidated before completion
        if (data->remoteSections.count(req_hndl->remoteAgent) == 0) {
            delete req_hndl;
            return NIXL_ERR_NOT_FOUND;
        }
        req_hndl->status = req_hndl->engine->checkXfer(
                                     req_hndl->backendHandle);
    }

    if (data->telemetryEnabled && req_hndl->status == NIXL_SUCCESS)
        req_hndl->updateRequestStats("Completed");

    return req_hndl->status;
}


nixl_status_t
nixlAgent::queryXferBackend(const nixlXferReqH* req_hndl,
                            nixlBackendH* &backend) const {
    NIXL_LOCK_GUARD(data->lock);
    backend = data->backendHandles[req_hndl->engine->getType()];
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::releaseXferReq(nixlXferReqH *req_hndl) const {

    NIXL_SHARED_LOCK_GUARD(data->lock);
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
    NIXL_LOCK_GUARD(data->lock);
    delete dlist_hndl;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getNotifs(nixl_notifs_t &notif_map,
                     const nixl_opt_args_t* extra_params) {
    notif_list_t    bknd_notif_list;
    nixl_status_t   ret, bad_ret=NIXL_SUCCESS;
    backend_list_t* backend_list;

    NIXL_LOCK_GUARD(data->lock);
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
                    const nixl_opt_args_t *extra_params) const {

    backend_list_t backend_list_value;
    backend_list_t *backend_list;

    if (!extra_params || extra_params->backends.empty()) {
        backend_list = &data->notifEngines;
    } else {
        backend_list = &backend_list_value;
        for (auto &elm : extra_params->backends) {
            if (elm->engine->supportsNotif()) {
                backend_list->push_back(elm->engine);
            }
        }
    }

    if (backend_list->empty()) {
        return NIXL_ERR_BACKEND;
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);

    if (data->name == remote_agent) {
        for (const auto &eng : *backend_list) {
            if (eng->supportsLocal()) {
                return eng->genNotif(remote_agent, msg);
            }
        }
        return NIXL_ERR_NOT_FOUND;
    }
    const auto iter = data->remoteBackends.find(remote_agent);

    if (iter != data->remoteBackends.end()) {
        for (const auto &eng : *backend_list) {
            if (iter->second.count(eng->getType()) != 0) {
                return eng->genNotif(remote_agent, msg);
            }
        }
    }
    return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
nixlAgent::getLocalMD (nixl_blob_t &str) const {
    size_t conn_cnt;
    nixl_backend_t nixl_backend;
    nixl_status_t ret;

    NIXL_LOCK_GUARD(data->lock);
    // data->connMD was populated when the backend was created
    conn_cnt = data->connMD.size();

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
nixlAgent::getLocalPartialMD(const nixl_reg_dlist_t &descs,
                             nixl_blob_t &str,
                             const nixl_opt_args_t* extra_params) const {
    backend_list_t tmp_list;
    backend_list_t *backend_list;
    nixl_status_t ret;

    NIXL_LOCK_GUARD(data->lock);

    if (!extra_params || extra_params->backends.size() == 0) {
        if (descs.descCount() != 0) {
            // Non-empty dlist, return backends that support the memory type
            backend_list = &data->memToBackend[descs.getType()];
            if (backend_list->empty())
                return NIXL_ERR_NOT_FOUND;
        } else {
            // Empty dlist, return all backends
            backend_list = &tmp_list;
            for (const auto & elm : data->backendEngines)
                backend_list->push_back(elm.second);
        }
    } else {
        backend_list = &tmp_list;
        for (const auto & elm : extra_params->backends)
            backend_list->push_back(elm->engine);
    }

    // First find all relevant engines and their conn info.
    // Best effort, ignore if no conn info (meaning backend doesn't support remote).
    backend_set_t selected_engines;
    std::vector<typename decltype(data->connMD)::iterator> found_iters;
    for (const auto &backend : *backend_list) {
        auto it = data->connMD.find(backend->getType());
        if (it == data->connMD.end())
            continue;
        found_iters.push_back(it);
        selected_engines.insert(backend);
    }

    nixlSerDes sd;
    ret = sd.addStr("Agent", data->name);
    if(ret)
        return ret;

    // Only add connection info if requested via extra_params or empty dlist
    size_t conn_cnt = ((extra_params && extra_params->includeConnInfo) || descs.descCount() == 0) ?
                      found_iters.size() : 0;
    ret = sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if(ret)
        return ret;

    for (size_t i = 0; i < conn_cnt; i++) {
        ret = sd.addStr("t", found_iters[i]->first);
        if(ret)
            return ret;
        ret = sd.addStr("c", found_iters[i]->second);
        if(ret)
            return ret;
    }

    // No engines found, but there are descs, this is an error
    if (selected_engines.size() == 0 && descs.descCount() > 0)
        return NIXL_ERR_BACKEND;

    ret = sd.addStr("", "MemSection");
    if(ret)
        return ret;

    ret = data->memorySection->serializePartial(&sd, selected_engines, descs);
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
    nixl_blob_t conn_info;
    nixl_backend_t nixl_backend;
    nixlBackendEngine* eng;
    nixl_status_t ret;

    NIXL_LOCK_GUARD(data->lock);
    ret = sd.importStr(remote_metadata);
    if(ret)
        return ret;

    std::string remote_agent = sd.getStr("Agent");
    if (remote_agent.size() == 0)
        return NIXL_ERR_MISMATCH;

    if (remote_agent == data->name)
        return NIXL_ERR_INVALID_PARAM;

    NIXL_DEBUG << "Loading remote metadata for agent: " << remote_agent;

    ret = sd.getBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if(ret) {
        NIXL_ERROR << "Error getting connection count: " << nixlEnumStrings::statusStr(ret);
        return ret;
    }

    for (size_t i=0; i<conn_cnt; ++i) {
        nixl_backend = sd.getStr("t");
        if (nixl_backend.size() == 0)
            return NIXL_ERR_MISMATCH;
        conn_info = sd.getStr("c");
        if (conn_info.size() == 0)
            return NIXL_ERR_MISMATCH;

        // Current agent might not support a remote backend
        if (data->backendEngines.count(nixl_backend)!=0) {

            // No need to reload same conn info, error if it changed
            if (data->remoteBackends.count(remote_agent) != 0 &&
                data->remoteBackends[remote_agent].count(nixl_backend) != 0) {
                if (data->remoteBackends[remote_agent][nixl_backend] != conn_info)
                    return NIXL_ERR_NOT_ALLOWED;
                count++;
                continue;
            }

            eng = data->backendEngines[nixl_backend];
            if (eng->supportsRemote()) {
                ret = eng->loadRemoteConnInfo(remote_agent, conn_info);
                if (ret)
                    return ret; // Error in load
                count++;
                data->remoteBackends[remote_agent].emplace(nixl_backend, conn_info);
            } else {
                // If there was an issue and we return error while some connections
                // are loaded, they will be deleted in the backend destructor.
                return NIXL_ERR_UNKNOWN; // This is an erroneous case
            }
        }
    }

    // No common backend, no point in loading the rest, unexpected
    if (count == 0 && conn_cnt > 0)
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
        data->remoteBackends.erase(remote_agent);
        return ret;
    }

    agent_name = remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::invalidateRemoteMD(const std::string &remote_agent) {
    NIXL_LOCK_GUARD(data->lock);

    if (remote_agent == data->name)
        return NIXL_ERR_INVALID_PARAM;

    nixl_status_t ret = NIXL_ERR_NOT_FOUND;
    if (data->remoteSections.count(remote_agent)!=0) {
        delete data->remoteSections[remote_agent];
        data->remoteSections.erase(remote_agent);
        ret = NIXL_SUCCESS;
    }

    if (data->remoteBackends.count(remote_agent)!=0) {
        for (auto & it: data->remoteBackends[remote_agent])
            data->backendEngines[it.first]->disconnect(remote_agent);
        data->remoteBackends.erase(remote_agent);
        ret = NIXL_SUCCESS;
    }

    return ret;
}

nixl_status_t
nixlAgent::sendLocalMD (const nixl_opt_args_t* extra_params) const {
    nixl_blob_t myMD;
    nixl_status_t ret = getLocalMD(myMD);
    if(ret < 0) return ret;

    // If IP is provided, use socket-based communication
    if (extra_params && !extra_params->ipAddr.empty()) {
        data->enqueueCommWork(std::make_tuple(SOCK_SEND, extra_params->ipAddr, extra_params->port, std::move(myMD)));
        return NIXL_SUCCESS;
    }

#if HAVE_ETCD
    // If no IP is provided, use etcd (now via thread)
    if (data->useEtcd) {
        data->enqueueCommWork(std::make_tuple(ETCD_SEND, default_metadata_label, 0, std::move(myMD)));
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_INVALID_PARAM;
#else
    return NIXL_ERR_NOT_SUPPORTED;
#endif // HAVE_ETCD
}

nixl_status_t
nixlAgent::sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                              const nixl_opt_args_t* extra_params) const {
    nixl_blob_t myMD;
    nixl_status_t ret = getLocalPartialMD(descs, myMD, extra_params);
    if(ret < 0) return ret;

    // If IP is provided, use socket-based communication
    if (extra_params && !extra_params->ipAddr.empty()) {
        data->enqueueCommWork(std::make_tuple(SOCK_SEND, extra_params->ipAddr, extra_params->port, std::move(myMD)));
        return NIXL_SUCCESS;
    }

#if HAVE_ETCD
    // If no IP is provided, use etcd (now via thread)
    if (data->useEtcd) {
        if (!extra_params || extra_params->metadataLabel.empty()) {
            NIXL_ERROR << "Metadata label is required for etcd send of local partial metadata";
            return NIXL_ERR_INVALID_PARAM;
        }
        data->enqueueCommWork(std::make_tuple(ETCD_SEND, extra_params->metadataLabel, 0, std::move(myMD)));
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_INVALID_PARAM;
#else
    return NIXL_ERR_NOT_SUPPORTED;
#endif // HAVE_ETCD
}

nixl_status_t
nixlAgent::fetchRemoteMD (const std::string remote_name,
                          const nixl_opt_args_t* extra_params) {
    // If IP is provided, use socket-based communication
    if (extra_params && !extra_params->ipAddr.empty()) {
        data->enqueueCommWork(std::make_tuple(SOCK_FETCH, extra_params->ipAddr, extra_params->port, ""));
        return NIXL_SUCCESS;
    }

#if HAVE_ETCD
    // If no IP is provided, use etcd via thread with watch capability
    if (data->useEtcd) {
        std::string metadata_label = extra_params && !extra_params->metadataLabel.empty() ?
                                     extra_params->metadataLabel :
                                     default_metadata_label;
        data->enqueueCommWork(std::make_tuple(ETCD_FETCH, std::move(metadata_label), 0, remote_name));
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_INVALID_PARAM;
#else
    return NIXL_ERR_NOT_SUPPORTED;
#endif // HAVE_ETCD
}

nixl_status_t
nixlAgent::invalidateLocalMD (const nixl_opt_args_t* extra_params) const {
    // If IP is provided, use socket-based communication
    if (extra_params && !extra_params->ipAddr.empty()) {
        data->enqueueCommWork(std::make_tuple(SOCK_INVAL, extra_params->ipAddr, extra_params->port, ""));
        return NIXL_SUCCESS;
    }

#if HAVE_ETCD
    // If no IP is provided, use etcd via thread
    if (data->useEtcd) {
        data->enqueueCommWork(std::make_tuple(ETCD_INVAL, "", 0, ""));
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_INVALID_PARAM;
#else
    return NIXL_ERR_NOT_SUPPORTED;
#endif // HAVE_ETCD
}

nixl_status_t
nixlAgent::checkRemoteMD (const std::string remote_name,
                          const nixl_xfer_dlist_t &descs) const {
    NIXL_LOCK_GUARD(data->lock);
    if (data->remoteSections.count(remote_name) != 0) {
        if (descs.descCount() == 0) {
            return NIXL_SUCCESS;
        } else {
            nixl_meta_dlist_t dummy(descs.getType(), descs.isSorted());
            // We only add to data->remoteBackends if data->backendEngines[backend] exists
            for (const auto& [backend, conn_info] : data->remoteBackends[remote_name])
                if (data->remoteSections[remote_name]->populate(
                          descs, data->backendEngines[backend], dummy) == NIXL_SUCCESS)
                    return NIXL_SUCCESS;
            dummy.clear();
        }
    }
    return NIXL_ERR_NOT_FOUND;
}
