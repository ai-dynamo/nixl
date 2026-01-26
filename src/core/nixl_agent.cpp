/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <numeric>

#include "nixl.h"
#include "serdes/serdes.h"
#include "backend/backend_engine.h"
#include "transfer_request.h"
#include "agent_data.h"
#include "plugin_manager.h"
#include "common/nixl_log.h"
#include "common/operators.h"
#include "telemetry.h"
#include "telemetry_event.h"

constexpr char TELEMETRY_ENABLED_VAR[] = "NIXL_TELEMETRY_ENABLE";
static const std::vector<std::vector<std::string>> illegal_plugin_combinations = {
    {"GDS", "GDS_MT"},
};
const std::string nixl_invalid_agent = "INVALID_AGENT";

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

std::string
nixlEnumStrings::statusStr(const nixl_status_t &status) {
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
        case NIXL_ERR_CANCELED:
            return "NIXL_ERR_CANCELED";
        case NIXL_ERR_NO_TELEMETRY:
            return "NIXL_ERR_NO_TELEMETRY";
        default:                         return "BAD_STATUS";
    }
}

inline void
nixlXferReqH::updateRequestStats(std::unique_ptr<nixlTelemetry> &telemetry_pub,
                                 nixl_telemetry_stat_status_t stat_status) {

    static const std::array<std::string, 3> nixl_post_status_str = {
        " Posted", " Posted and Completed", " Completed"};
    auto duration = std::chrono::duration_cast<chrono_period_us_t>(
        std::chrono::steady_clock::now() - telemetry.startTime);
    if (stat_status == NIXL_TELEMETRY_POST) {
        telemetry.postDuration = duration;
    } else if (stat_status == NIXL_TELEMETRY_POST_AND_FINISH) {
        telemetry.postDuration = duration;
        telemetry.xferDuration = duration;
    } else { // stat_status == NIXL_TELEMETRY_FINISH
        telemetry.xferDuration = duration;
    }

    if (telemetry_pub && (stat_status != NIXL_TELEMETRY_POST)) {
        telemetry_pub->addPostTime(telemetry.postDuration);
        telemetry_pub->addXferTime(duration, backendOp == NIXL_WRITE, telemetry.totalBytes);
    }

    NIXL_TRACE << "[NIXL TELEMETRY]: From backend " << engine->getType()
               << nixl_post_status_str[stat_status] << " Xfer with " << telemetry.descCount
               << " descriptors of total size " << telemetry.totalBytes << "B in "
               << duration.count() << "us.";
}

/*** nixlAgentData constructor/destructor, as part of nixlAgent's ***/
nixlAgentData::nixlAgentData(const std::string &name, const nixlAgentConfig &cfg)
    : name(name),
      config(cfg),
      lock(cfg.syncMode) {
#if HAVE_ETCD
    if (getenv("NIXL_ETCD_ENDPOINTS")) {
        useEtcd = true;
        NIXL_DEBUG << "NIXL ETCD is enabled";
    } else {
        useEtcd = false;
        NIXL_DEBUG << "NIXL ETCD is disabled";
    }
#else
    useEtcd = false;
    NIXL_DEBUG << "NIXL ETCD is excluded";
#endif // HAVE_ETCD
    if (name.empty())
        throw std::invalid_argument("Agent needs a name");

    memorySection = new nixlLocalSection();
    const char *telemetry_env_val = std::getenv(TELEMETRY_ENABLED_VAR);

    if (telemetry_env_val != nullptr) {
        if (!strcasecmp(telemetry_env_val, "y") || !strcasecmp(telemetry_env_val, "1") ||
            !strcasecmp(telemetry_env_val, "yes") || !strcasecmp(telemetry_env_val, "on")) {
            telemetryEnabled = true;
            telemetry_ = std::make_unique<nixlTelemetry>(name, backendEngines);
        } else if (cfg.captureTelemetry) {
            telemetryEnabled = true;
            NIXL_WARN << "NIXL telemetry is enabled through config, "
                         "ignoring the NIXL_TELEMETRY_ENABLE environment variable";
        } else if (!strcasecmp(telemetry_env_val, "n") || !strcasecmp(telemetry_env_val, "0") ||
                   !strcasecmp(telemetry_env_val, "no") || !strcasecmp(telemetry_env_val, "off")) {
            NIXL_DEBUG << "NIXL telemetry is disabled";
        } else {
            NIXL_WARN
                << "NIXL telemetry is disabled for invalid NIXL_TELEMETRY_ENABLE environment "
                   "variable -- valid are 'y', 'yes', '1', 'on', 'n', 'no', '0', 'off', any case";
        }
    } else if (cfg.captureTelemetry) {
        telemetryEnabled = true;
        NIXL_DEBUG << "Capturing NIXL telemetry based on config (without an output file)";
    }
}

nixlAgentData::~nixlAgentData() {
    delete memorySection;

    // explicitly reset telemetry so i can publish backend events before destroying backends
    telemetry_.reset();

    for (auto & elm: remoteSections)
        delete elm.second;

    for (auto & elm: backendEngines) {
        auto& plugin_manager = nixlPluginManager::getInstance();
        auto plugin_handle = plugin_manager.getBackendPlugin(elm.second->getType());

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
        data->agentShutdown = false;
        data->commThread = std::thread(&nixlAgentData::commWorker, data.get(), std::ref(*this));
    }
}

nixlAgent::~nixlAgent() {
    if (data && (data->useEtcd || data->config.useListenThread)) {
        data->agentShutdown = true;
        while (!data->commQueue.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        data->commThreadStop = true;
        if(data->commThread.joinable()) data->commThread.join();

        try {
            if (data->commThreadException_) {
                std::rethrow_exception(data->commThreadException_);
            }
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Communication thread has thrown an exception: " << e.what();
        }

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
    plugins = plugin_manager.getLoadedBackendPluginNames();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getPluginParams (const nixl_backend_t &type,
                            nixl_mem_list_t &mems,
                            nixl_b_params_t &params) const {

    // TODO: unify to uppercase/lowercase and do ltrim/rtrim for type

    // First try to get options from a loaded plugin
    auto& plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.getBackendPlugin(type);

    if (plugin_handle) {
      // If the plugin is already loaded, get options directly
        params = plugin_handle->getBackendOptions();
        mems   = plugin_handle->getBackendMems();
        return NIXL_SUCCESS;
    }

    // If plugin isn't loaded yet, try to load it temporarily
    plugin_handle = plugin_manager.loadBackendPlugin(type);
    if (plugin_handle) {
        params = plugin_handle->getBackendOptions();
        mems   = plugin_handle->getBackendMems();

        NIXL_LOCK_GUARD(data->lock);

        // We don't keep the plugin loaded if we didn't have it before
        if (data->backendEngines.count(type) == 0) {
            plugin_manager.unloadBackendPlugin(type);
        }
        return NIXL_SUCCESS;
    }

    NIXL_ERROR_FUNC << "backend '" << type << "' not found";
    return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
nixlAgent::getBackendParams (const nixlBackendH* backend,
                             nixl_mem_list_t &mems,
                             nixl_b_params_t &params) const {
    if (!backend) {
        NIXL_ERROR_FUNC << "backend handle is not provided";
        return NIXL_ERR_INVALID_PARAM;
    }

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
    if (data->backendEngines.count(type) != 0) {
        NIXL_ERROR_FUNC << "backend already created for type '" << type << "'";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check if the plugin is in an illegal combination with another plugin backend already created
    for (const auto &combination : illegal_plugin_combinations) {
        if (std::find(combination.begin(), combination.end(), type) != combination.end()) {
            for (const auto &plugin_name : combination) {
                if (plugin_name != type &&
                    data->backendEngines.find(plugin_name) != data->backendEngines.end()) {
                    NIXL_ERROR_FUNC << "Plugin backend " << type
                                    << " is in illegal combination with " << plugin_name;
                    return NIXL_ERR_NOT_ALLOWED;
                }
            }
        }
    }

    init_params.localAgent = data->name;
    init_params.type = type;
    init_params.customParams = const_cast<nixl_b_params_t *>(&params);
    init_params.enableProgTh = data->config.useProgThread;
    init_params.pthrDelay = data->config.pthrDelay;
    init_params.syncMode = data->config.syncMode;
    init_params.enableTelemetry_ = data->telemetry_ != nullptr;

    // First, try to load the backend as a plugin
    auto& plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.loadBackendPlugin(type);

    if (plugin_handle) {
        // Plugin found, use it to create the backend
        backend = plugin_handle->createEngine(&init_params);
    } else {
        NIXL_ERROR_FUNC << "unsupported backend '" << type << "'";
        return NIXL_ERR_NOT_FOUND;
    }

    if (backend) {
        if (backend->getInitErr()) {
            delete backend;
            NIXL_ERROR_FUNC << "backend initialization error for '" << type << "'";
            return NIXL_ERR_BACKEND;
        }

        if (backend->supportsRemote()) {
            if (!backend->supportsNotif()) {
                delete backend;
                NIXL_ERROR_FUNC << "backend '" << type << "' supportsRemote but not notifications";
                return NIXL_ERR_BACKEND;
            }

            ret = backend->getConnInfo(str);
            if (ret != NIXL_SUCCESS) {
                delete backend;
                NIXL_ERROR_FUNC << "failed to get connection info for '" << type << "' with status "
                                << ret;
                return ret;
            }
            data->connMD[type] = str;
        }

        if (backend->supportsLocal()) {
            ret = backend->connect(data->name);

            if (NIXL_SUCCESS != ret) {
                delete backend;
                NIXL_ERROR_FUNC
                    << "backend '" << type
                    << "' encountered error during intra-agent transfer setup with status " << ret;
                return ret;
            }
        }

        bknd_hndl = new nixlBackendH(backend);
        if (!bknd_hndl) {
            delete backend;
            NIXL_ERROR_FUNC << "allocation of backend handle failed for '" << type << "'";
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

    NIXL_ERROR_FUNC << "backend creation failed for '" << type << "'";
    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlAgent::queryMem(const nixl_reg_dlist_t &descs,
                    std::vector<nixl_query_resp_t> &resp,
                    const nixl_opt_args_t *extra_params) const {

    if (!extra_params || extra_params->backends.size() != 1) {
        NIXL_ERROR_FUNC << "this method requires exactly one backend to be passed";
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
        if (backend_list->empty()) {
            NIXL_ERROR_FUNC << "no available backends for mem type '" << descs.getType() << "'";
            return NIXL_ERR_NOT_FOUND;
        }
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
        nixl_sec_dlist_t sec_descs(descs.getType());
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

    if (count > 0) {
        // sum all the sizes of the descriptors using std::accumulate
        if (data->telemetry_) {
            uint64_t total_size = std::accumulate(
                descs.begin(),
                descs.end(),
                uint64_t{0},
                [](uint64_t sum, const nixlBlobDesc &desc) { return sum + desc.len; });
            data->telemetry_->updateMemoryRegistered(total_size);
        }
        return NIXL_SUCCESS;
    }
    NIXL_ERROR_FUNC << "registration failed for the specified or all potential backends";
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
        if (!avail_backends || avail_backends->empty()) {
            NIXL_ERROR_FUNC << "no available backends for mem type '" << descs.getType() << "'";
            return NIXL_ERR_NOT_FOUND;
        }
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
    if (bad_ret == NIXL_SUCCESS) {
        if (data->telemetry_) {
            uint64_t total_size = std::accumulate(
                descs.begin(),
                descs.end(),
                uint64_t{0},
                [](uint64_t sum, const nixlBlobDesc &desc) { return sum + desc.len; });
            data->telemetry_->updateMemoryDeregistered(total_size);
        }
    } else {
        NIXL_ERROR_FUNC << "deregistration failed on at least one backend with status " << bad_ret;
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
    if (data->remoteBackends.count(remote_agent) == 0) {
        NIXL_ERROR_FUNC << "metadata for remote agent '" << remote_agent << "' not found";
        return NIXL_ERR_NOT_FOUND;
    }

    if (!extra_params || extra_params->backends.size() == 0) {
        if (data->remoteBackends[remote_agent].empty()) {
            NIXL_ERROR_FUNC << "no backends are found in metadata for remote agent '"
                            << remote_agent << "'";
            return NIXL_ERR_NOT_FOUND;
        }
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
            if (ret) {
                NIXL_ERROR_FUNC << "connect('" << remote_agent << "') failed on backend '"
                                << eng->getType() << "' with status " << ret;
                break;
            }
            count++;
        }
    }

    if (ret) // Error is already logged
        return ret;

    if (count == 0) { // No common backend
        NIXL_ERROR_FUNC << "no common backend to connect with '" << remote_agent << "'";
        return NIXL_ERR_BACKEND;
    }

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
    if (!init_side && (data->remoteSections.count(agent_name) == 0)) {
        NIXL_ERROR_FUNC << "metadata for remote agent '" << agent_name << "' not found";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    if (!extra_params || extra_params->backends.size() == 0) {
        if (!init_side)
            backend_set = data->remoteSections[agent_name]->
                                queryBackends(descs.getType());
        else
            backend_set = data->memorySection->
                                queryBackends(descs.getType());

        if (!backend_set || backend_set->empty()) {
            NIXL_ERROR_FUNC << "no available backends for mem type '" << descs.getType() << "'";
            data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
            return NIXL_ERR_NOT_FOUND;
        }
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
        handle->descs[backend] = new nixl_meta_dlist_t(descs.getType());
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
        NIXL_ERROR_FUNC << "failed to prepare the descriptors for any of "
                           "the specified or potential backends for agent '"
                        << agent_name << "'";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
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

    if (!local_side || !remote_side) {
        NIXL_ERROR_FUNC << "local or remote side handle is null";
        data->addErrorTelemetry(NIXL_ERR_INVALID_PARAM);
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((!local_side->isLocal) || (remote_side->isLocal)) {
        NIXL_ERROR_FUNC << "invalid sides (local must be local, remote must be remote)";
        data->addErrorTelemetry(NIXL_ERR_INVALID_PARAM);
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_LOCK_GUARD(data->lock);
    // The remote was invalidated in between prepXferDlist and this call
    if (data->remoteSections.count(remote_side->remoteAgent) == 0) {
        NIXL_ERROR_FUNC << "remote agent '" << remote_side->remoteAgent
                        << "' was invalidated in between prepXferDlist and this call";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
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

    if (!backend) {
        NIXL_ERROR_FUNC << "could not find a common backend in the specified or "
                           "available list of backends for the prepped Dlists";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_meta_dlist_t* local_descs  = local_side->descs.at(backend);
    nixl_meta_dlist_t* remote_descs = remote_side->descs.at(backend);
    size_t total_bytes = 0;

    if ((desc_count == 0) || (remote_indices.size() == 0) ||
        (desc_count != (int)remote_indices.size())) {
        NIXL_ERROR_FUNC << "different number of indices for local (" << desc_count << "), remote ("
                        << remote_indices.size() << ")";
        return NIXL_ERR_INVALID_PARAM;
    }

    for (int i=0; i<desc_count; ++i) {
        if ((local_indices[i] >= local_descs->descCount()) || (local_indices[i] < 0)) {
            NIXL_ERROR_FUNC << "local index out of range at index " << i << " with value "
                            << local_indices[i];
            return NIXL_ERR_INVALID_PARAM;
        }
        if ((remote_indices[i] >= remote_descs->descCount()) || (remote_indices[i] < 0)) {
            NIXL_ERROR_FUNC << "remote index out of range at index " << i << " with value "
                            << remote_indices[i];
            return NIXL_ERR_INVALID_PARAM;
        }
        if ((*local_descs)[local_indices[i]].len != (*remote_descs)[remote_indices[i]].len) {
            NIXL_ERROR_FUNC << "length mismatch at index pair " << i << " with local index "
                            << local_indices[i] << " and remote index " << remote_indices[i];
            return NIXL_ERR_INVALID_PARAM;
        }
        total_bytes += (*local_descs)[local_indices[i]].len;
    }

    if (extra_params && extra_params->hasNotif) {
        opt_args.notifMsg = extra_params->notifMsg;
        opt_args.hasNotif = true;
    }

    if ((opt_args.hasNotif) && (!backend->supportsNotif())) {
        NIXL_ERROR_FUNC << "the selected backend '" << backend->getType()
                        << "' does not support notifications";
        return NIXL_ERR_BACKEND;
    }

    std::unique_ptr<nixlXferReqH> handle = std::make_unique<nixlXferReqH>();
    handle->initiatorDescs = new nixl_meta_dlist_t(local_descs->getType(), desc_count);

    handle->targetDescs = new nixl_meta_dlist_t(remote_descs->getType(), desc_count);

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

    if (data->telemetryEnabled) {
        handle->telemetry.totalBytes = total_bytes;
        handle->telemetry.descCount = handle->initiatorDescs->descCount();
    }

    ret = handle->engine->prepXfer (handle->backendOp,
                                    *handle->initiatorDescs,
                                    *handle->targetDescs,
                                    handle->remoteAgent,
                                    handle->backendHandle,
                                    &opt_args);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "backend '" << backend->getType()
                        << "' failed to prepare the transfer request with status " << ret;
        data->addErrorTelemetry(ret);
        return ret;
    }

    req_hndl = handle.release();
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

    std::unique_ptr<backend_set_t> backend_set = std::make_unique<backend_set_t>();

    req_hndl = nullptr;

    NIXL_SHARED_LOCK_GUARD(data->lock);
    if (data->remoteSections.count(remote_agent) == 0)
    {
        NIXL_ERROR_FUNC << "metadata for remote agent '" << remote_agent << "' not found";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    size_t total_bytes = 0;
    // Check the correspondence between descriptor lists
    if (local_descs.descCount() != remote_descs.descCount()) {
        NIXL_ERROR_FUNC << "different descriptor list sizes (local=" << local_descs.descCount()
                        << ", remote=" << remote_descs.descCount() << ")";
        return NIXL_ERR_INVALID_PARAM;
    }
    for (int i = 0; i < local_descs.descCount(); ++i) {
        if (local_descs[i].len != remote_descs[i].len) {
            NIXL_ERROR_FUNC << "length mismatch at index " << i;
            return NIXL_ERR_INVALID_PARAM;
        }
        total_bytes += local_descs[i].len;
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
            NIXL_ERROR_FUNC << "no backends found for local or remote for their "
                               "corresponding memory type";
            return NIXL_ERR_NOT_FOUND;
        }

        for (auto & elm : *local_set)
            if (remote_set->count(elm) != 0)
                backend_set->insert(elm);

        if (backend_set->empty()) {
            NIXL_ERROR_FUNC << "no potential backend found to be able to do the transfer";
            return NIXL_ERR_NOT_FOUND;
        }
    } else {
        for (auto & elm : extra_params->backends)
            backend_set->insert(elm->engine);
    }

    // TODO: when central KV is supported, add a call to fetchRemoteMD
    // TODO: merge descriptors back to back in memory (like makeXferReq).
    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    std::unique_ptr<nixlXferReqH> handle = std::make_unique<nixlXferReqH>();
    handle->initiatorDescs = new nixl_meta_dlist_t(local_descs.getType());

    handle->targetDescs = new nixl_meta_dlist_t(remote_descs.getType());

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

    if (!handle->engine) {
        NIXL_ERROR_FUNC << "no specified or potential backend had the required "
                           "registrations to be able to do the transfer";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
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
        NIXL_ERROR_FUNC << "the selected backend '" << handle->engine->getType()
                        << "' does not support notifications";
        data->addErrorTelemetry(NIXL_ERR_BACKEND);
        return NIXL_ERR_BACKEND;
    }

    handle->remoteAgent = remote_agent;
    handle->backendOp = operation;
    handle->status = NIXL_ERR_NOT_POSTED;
    handle->notifMsg = opt_args.notifMsg;
    handle->hasNotif = opt_args.hasNotif;

    if (data->telemetryEnabled) {
        handle->telemetry.totalBytes = total_bytes;
        handle->telemetry.descCount = handle->initiatorDescs->descCount();
    }

    ret1 = handle->engine->prepXfer (handle->backendOp,
                                     *handle->initiatorDescs,
                                     *handle->targetDescs,
                                     handle->remoteAgent,
                                     handle->backendHandle,
                                     &opt_args);
    if (ret1 != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "backend '" << handle->engine->getType()
                        << "' failed to prepare the transfer request with status " << ret1;
        data->addErrorTelemetry(ret1);
        return ret1;
    }

    req_hndl = handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::estimateXferCost(const nixlXferReqH *req_hndl,
                            std::chrono::microseconds &duration,
                            std::chrono::microseconds &err_margin,
                            nixl_cost_t &method,
                            const nixl_opt_args_t* extra_params) const
{
    nixl_status_t ret;
    NIXL_SHARED_LOCK_GUARD(data->lock);

    // Check if the remote agent connection info is still valid
    // (assuming cost estimation requires connection info like transfers)
    if (!req_hndl->remoteAgent.empty() &&
        (data->remoteSections.count(req_hndl->remoteAgent) == 0)) {
        NIXL_ERROR_FUNC << "invalid request handle, remote agent was invalidated "
                           "after transfer request creation";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    if (!req_hndl->engine) {
        NIXL_ERROR_FUNC << "invalid request handle: engine is null";
        data->addErrorTelemetry(NIXL_ERR_UNKNOWN);
        return NIXL_ERR_UNKNOWN;
    }

    ret = req_hndl->engine->estimateXferCost(req_hndl->backendOp,
                                             *req_hndl->initiatorDescs,
                                             *req_hndl->targetDescs,
                                             req_hndl->remoteAgent,
                                             req_hndl->backendHandle,
                                             duration,
                                             err_margin,
                                             method,
                                             extra_params);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "backend '" << req_hndl->engine->getType()
                        << "' failed to estimate the transfer cost with status " << ret;
    }
    return ret;
}

nixl_status_t
nixlAgent::postXferReq(nixlXferReqH *req_hndl,
                       const nixl_opt_args_t* extra_params) const {
    nixl_opt_b_args_t opt_args;

    opt_args.hasNotif = false;

    if (!req_hndl) {
        NIXL_ERROR_FUNC << "transfer request handle is null";
        data->addErrorTelemetry(NIXL_ERR_INVALID_PARAM);
        return NIXL_ERR_INVALID_PARAM;
    }

    if (data->telemetryEnabled) {
        req_hndl->telemetry.startTime = std::chrono::steady_clock::now();
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);
    // Check if the remote was invalidated before post/repost
    if (data->remoteSections.count(req_hndl->remoteAgent) == 0) {
        NIXL_ERROR_FUNC << "remote agent '" << req_hndl->remoteAgent
                        << "' was invalidated after transfer request creation";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    // We can't repost while a request is in progress
    if (req_hndl->status == NIXL_IN_PROG) {
        req_hndl->status = req_hndl->engine->checkXfer(
                                     req_hndl->backendHandle);
        if (req_hndl->status == NIXL_IN_PROG) {
            NIXL_ERROR_FUNC << "transfer request is still in progress and cannot be reposted";
            return NIXL_ERR_REPOST_ACTIVE;
        }

        if (req_hndl->status == NIXL_ERR_REMOTE_DISCONNECT) {
            data->invalidateRemoteData(req_hndl->remoteAgent);
            NIXL_ERROR_FUNC << "remote agent '" << req_hndl->remoteAgent
                            << "' was disconnected after transfer request creation";
            return NIXL_ERR_REMOTE_DISCONNECT;
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
        NIXL_ERROR_FUNC << "the selected backend '" << req_hndl->engine->getType()
                        << "' does not support notifications";
        data->addErrorTelemetry(NIXL_ERR_BACKEND);
        return NIXL_ERR_BACKEND;
    }

    // If status is not NIXL_IN_PROG we can repost,
    req_hndl->status = req_hndl->engine->postXfer(req_hndl->backendOp,
                                                  *req_hndl->initiatorDescs,
                                                  *req_hndl->targetDescs,
                                                  req_hndl->remoteAgent,
                                                  req_hndl->backendHandle,
                                                  &opt_args);

    if (req_hndl->status < 0) {
        if (req_hndl->status == NIXL_ERR_REMOTE_DISCONNECT) {
            NIXL_ERROR_FUNC << "remote agent '" << req_hndl->remoteAgent
                            << "' was disconnected after transfer request creation";
            data->invalidateRemoteData(req_hndl->remoteAgent);
            return NIXL_ERR_REMOTE_DISCONNECT;
        } else {
            NIXL_ERROR_FUNC << "backend '" << req_hndl->engine->getType()
                            << "' failed to post the transfer request with status "
                            << req_hndl->status;
        }
    }

    if (data->telemetryEnabled) {
        if (req_hndl->status < 0) {
            data->addErrorTelemetry(req_hndl->status);
        } else if (req_hndl->status == NIXL_IN_PROG) {
            req_hndl->updateRequestStats(data->telemetry_, NIXL_TELEMETRY_POST);
        } else {
            req_hndl->updateRequestStats(data->telemetry_, NIXL_TELEMETRY_POST_AND_FINISH);
        }
    }

    return req_hndl->status;
}

nixl_status_t
nixlAgent::getXferStatus (nixlXferReqH *req_hndl) const {

    NIXL_SHARED_LOCK_GUARD(data->lock);
    // If the status is done, no need to recheck and no state changes.
    // Same for users incorrectly recalling this method in error/done.
    if (req_hndl->status == NIXL_IN_PROG) {
        // Check if the remote was invalidated before completion
        if (data->remoteSections.count(req_hndl->remoteAgent) == 0) {
            NIXL_ERROR_FUNC << "remote agent '" << req_hndl->remoteAgent
                            << "' was invalidated during transfer";
            return NIXL_ERR_NOT_FOUND;
        }

        req_hndl->status = req_hndl->engine->checkXfer(req_hndl->backendHandle);
        if (req_hndl->status < 0) {
            if (req_hndl->status == NIXL_ERR_REMOTE_DISCONNECT) {
                data->invalidateRemoteData(req_hndl->remoteAgent);
                return NIXL_ERR_REMOTE_DISCONNECT;
            } else {
                NIXL_ERROR_FUNC << "backend '" << req_hndl->engine->getType()
                                << "' returned error status " << req_hndl->status;
            }
        }
        if (data->telemetryEnabled) {
            if (req_hndl->status == NIXL_SUCCESS) {
                req_hndl->updateRequestStats(data->telemetry_, NIXL_TELEMETRY_FINISH);
            } else if (req_hndl->status < 0) {
                data->addErrorTelemetry(req_hndl->status);
            }
        }
    }

    // If the status is error when entering this method, it was already logged
    return req_hndl->status;
}

nixl_status_t
nixlAgent::getXferTelemetry(const nixlXferReqH *req_hndl, nixl_xfer_telem_t &telemetry) const {

    if (!data->telemetryEnabled) {
        NIXL_ERROR_FUNC << "cannot return values when telemetry is not enabled.";
        return NIXL_ERR_NO_TELEMETRY;
    }

    if (req_hndl->status != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "Transfer is not complete yet";
        return req_hndl->status;
    }

    telemetry = req_hndl->telemetry;
    return NIXL_SUCCESS;
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

            if (req_hndl->status < 0) {
                NIXL_ERROR_FUNC << "backend '" << req_hndl->engine->getType()
                                << "' could not release transfer request and returned error status "
                                << req_hndl->status;
                return NIXL_ERR_REPOST_ACTIVE; // Might need renaming
            }
            // just in case the backend doesn't set to NULL on success
            // this will prevent calling releaseReqH again in destructor
            req_hndl->backendHandle = nullptr;
        }
    }
    delete req_hndl;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::createGpuXferReq(const nixlXferReqH &req_hndl, nixlGpuXferReqH &gpu_req_hndl) const {
    if (!req_hndl.engine) {
        NIXL_ERROR_FUNC << "Invalid request handle[" << &req_hndl << "]: engine is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!req_hndl.backendHandle) {
        NIXL_ERROR_FUNC << "Invalid request handle[" << &req_hndl << "]: backendHandle is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);
    const auto status = req_hndl.engine->createGpuXferReq(
        *req_hndl.backendHandle, *req_hndl.initiatorDescs, *req_hndl.targetDescs, gpu_req_hndl);
    if (status == NIXL_SUCCESS) {
        data->gpuReqToEngine.emplace(gpu_req_hndl, req_hndl.engine);
    }

    return status;
}

void
nixlAgent::releaseGpuXferReq(nixlGpuXferReqH gpu_req_hndl) const {
    NIXL_SHARED_LOCK_GUARD(data->lock);
    auto it = data->gpuReqToEngine.find(gpu_req_hndl);
    if (it == data->gpuReqToEngine.end()) {
        NIXL_WARN << "Invalid gpu_req_hndl[" << gpu_req_hndl << "] ";
        return;
    }

    it->second->releaseGpuXferReq(gpu_req_hndl);

    data->gpuReqToEngine.erase(it);
}

nixl_status_t
nixlAgent::getGpuSignalSize(size_t &signal_size, const nixl_opt_args_t *extra_params) const {
    if (!extra_params || extra_params->backends.empty()) {
        NIXL_ERROR_FUNC << "backend must be specified in extra_params";
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);
    return extra_params->backends[0]->engine->getGpuSignalSize(signal_size);
}

nixl_status_t
nixlAgent::prepGpuSignal(const nixl_reg_dlist_t &signal_descs,
                         const nixl_opt_args_t *extra_params) const {
    if (signal_descs.descCount() == 0) {
        NIXL_ERROR_FUNC << "signal descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!extra_params || extra_params->backends.empty()) {
        NIXL_ERROR_FUNC << "backend must be specified in extra_params";
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);

    nixlBackendH *backend = extra_params->backends[0];

    // Get the size of individual GPU signals
    size_t signal_size;
    nixl_status_t ret = backend->engine->getGpuSignalSize(signal_size);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "failed to get GPU signal size with status: "
                        << nixlEnumStrings::statusStr(ret);
        return ret;
    }

    // Convert reg_dlist to xfer_dlist for populate call
    nixl_xfer_dlist_t xfer_descs = signal_descs.trim();

    nixl_meta_dlist_t result(signal_descs.getType());
    ret = data->memorySection->populate(xfer_descs, backend->engine, result);

    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "failed to populate signal metadata with specified backend";
        return ret;
    }

    for (size_t i = 0; i < static_cast<size_t>(result.descCount()); i++) {
        size_t desc_len = result[i].len;
        uintptr_t desc_addr = result[i].addr;

        size_t num_signals = desc_len / signal_size;

        if (num_signals == 0) {
            NIXL_ERROR_FUNC << "descriptor " << i << " is too small (length=" << desc_len
                            << ") to contain even one signal (signal_size=" << signal_size << ")";
            return NIXL_ERR_INVALID_PARAM;
        }

        for (size_t j = 0; j < num_signals; j++) {
            void *signal = reinterpret_cast<void *>(desc_addr + j * signal_size);
            ret = backend->engine->prepGpuSignal(*result[i].metadataP, signal);

            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR_FUNC << "failed to prepare GPU signal " << j << " in descriptor " << i
                                << " with status: " << nixlEnumStrings::statusStr(ret);
                return ret;
            }

            NIXL_DEBUG << "Successfully prepared GPU signal " << j << " in descriptor " << i
                       << " at address " << signal;
        }
    }

    NIXL_DEBUG << "Successfully prepared GPU signals for " << result.descCount() << " descriptors";
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
        if (backend_list->empty()) {
            NIXL_ERROR_FUNC << "no backends support notifications";
            return NIXL_ERR_BACKEND;
        }
    } else {
        backend_list = new backend_list_t();
        for (auto & elm : extra_params->backends)
            if (elm->engine->supportsNotif())
                backend_list->push_back(elm->engine);

        if (backend_list->empty()) {
            NIXL_ERROR_FUNC << "none of specified backends support notifications";
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
        if (ret < 0) {
            NIXL_ERROR_FUNC << "backend '" << eng->getType() << "' returned error status " << ret
                            << " while getting notifications";
            bad_ret=ret;
        }

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

    // If any backend had an error, it was already logged
    return bad_ret;
}

nixl_status_t
nixlAgent::genNotif(const std::string &remote_agent,
                    const nixl_blob_t &msg,
                    const nixl_opt_args_t *extra_params) const {

    backend_list_t backend_list_value;
    backend_list_t *backend_list;
    nixl_status_t ret;

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
        NIXL_ERROR_FUNC << "no specified or potential backend supports notifications";
        return NIXL_ERR_BACKEND;
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);

    if (data->name == remote_agent) {
        for (const auto &eng : *backend_list) {
            if (eng->supportsLocal()) {
                ret = eng->genNotif(remote_agent, msg);
                if (ret < 0) {
                    NIXL_ERROR_FUNC << "backend '" << eng->getType() << "' returned error status "
                                    << ret << " while sending intra-agent notifications";
                }
                return ret;
            }
        }
        NIXL_ERROR_FUNC << "no specified or potential backend can send intra-agent notifications";
        return NIXL_ERR_NOT_FOUND;
    }
    const auto iter = data->remoteBackends.find(remote_agent);

    if (iter != data->remoteBackends.end()) {
        for (const auto &eng : *backend_list) {
            if (iter->second.count(eng->getType()) != 0) {
                ret = eng->genNotif(remote_agent, msg);
                if (ret < 0) {
                    NIXL_ERROR_FUNC << "backend '" << eng->getType() << "' returned error status "
                                    << ret << " while sending notification to agent '"
                                    << remote_agent << "'";
                }
                return ret;
            }
        }
    }

    NIXL_ERROR_FUNC << "no specified or potential backend could send the inter-agent notifications";
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

    if (conn_cnt == 0) { // Error, no backend supports remote
        NIXL_ERROR_FUNC << "no backends support remote operations";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes sd;
    ret = sd.addStr("Agent", data->name);
    // Always returns SUCCESS, serdes class logs errors if necessary
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if (ret) return NIXL_ERR_UNKNOWN;

    for (auto &c : data->connMD) {
        nixl_backend = c.first;
        ret = sd.addStr("t", nixl_backend);
        if (ret) break;
        ret = sd.addStr("c", c.second);
        if (ret) break;
    }
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = sd.addStr("", "MemSection");
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = data->memorySection->serialize(&sd);
    if (ret) {
        NIXL_ERROR_FUNC << "serialization failed";
        return ret;
    }

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
            if (backend_list->empty()) {
                NIXL_ERROR_FUNC << "no available backends for mem type '" << descs.getType() << "'";
                return NIXL_ERR_NOT_FOUND;
            }
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

    if (selected_engines.size() == 0 && descs.descCount() > 0) {
        NIXL_ERROR_FUNC << "no backends support the requested descriptors";
        return NIXL_ERR_BACKEND;
    }

    nixlSerDes sd;
    ret = sd.addStr("Agent", data->name);
    // Always returns SUCCESS, serdes class logs errors if necessary
    if (ret) return NIXL_ERR_UNKNOWN;

    // Only add connection info if requested via extra_params or empty dlist
    size_t conn_cnt = ((extra_params && extra_params->includeConnInfo) || descs.descCount() == 0) ?
                      found_iters.size() : 0;
    ret = sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if (ret) return NIXL_ERR_UNKNOWN;

    for (size_t i = 0; i < conn_cnt; i++) {
        ret = sd.addStr("t", found_iters[i]->first);
        if (ret) break;
        ret = sd.addStr("c", found_iters[i]->second);
        if (ret) break;
    }
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = sd.addStr("", "MemSection");
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = data->memorySection->serializePartial(&sd, selected_engines, descs);
    if (ret) {
        NIXL_ERROR_FUNC << "serialization failed";
        return ret;
    }

    str = sd.exportStr();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::loadRemoteMD (const nixl_blob_t &remote_metadata,
                         std::string &agent_name) {
    nixlSerDes sd;
    nixl_blob_t conn_info;
    nixl_backend_t nixl_backend;
    nixl_status_t ret;

    NIXL_LOCK_GUARD(data->lock);
    ret = sd.importStr(remote_metadata);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "failed to deserialize remote metadata";
        return NIXL_ERR_MISMATCH;
    }

    std::string remote_agent = sd.getStr("Agent");
    if (remote_agent.empty()) {
        NIXL_ERROR_FUNC << "error in deserializing remote agent name";
        return NIXL_ERR_MISMATCH;
    }

    if (remote_agent == data->name) {
        NIXL_ERROR_FUNC << "remote agent name same as local agent, "
                           "no need to load metadata";
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_DEBUG << "Loading remote metadata for agent: " << remote_agent;

    size_t conn_cnt;
    ret = sd.getBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "error getting connection count: " << ret;
        return NIXL_ERR_MISMATCH;
    }

    int count = 0;
    for (size_t i = 0; i < conn_cnt; ++i) {
        nixl_backend = sd.getStr("t");
        conn_info = sd.getStr("c");

        if (nixl_backend.empty() || conn_info.empty()) {
            NIXL_ERROR_FUNC << "failed to deserialize remote metadata";
            return NIXL_ERR_MISMATCH;
        }

        ret = data->loadConnInfo(remote_agent, nixl_backend, conn_info);
        if (ret == NIXL_SUCCESS) {
            count++;
        } else if (ret != NIXL_ERR_NOT_SUPPORTED) {
            NIXL_ERROR_FUNC << "error loading connection info for backend '" << nixl_backend
                            << "' with status " << ret;
            return ret;
        }
    }

    if ((count == 0) && (conn_cnt > 0)) {
        NIXL_ERROR_FUNC << "no common backend found";
        return NIXL_ERR_BACKEND;
    }

    if (sd.getStr("") != "MemSection") {
        NIXL_ERROR_FUNC << "failed to deserialize remote metadata";
        return NIXL_ERR_MISMATCH;
    }

    ret = data->loadRemoteSections(remote_agent, sd);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "error loading remote metadata for agent '" << remote_agent
                        << "' with status " << ret;
        return ret;
    }

    agent_name = remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::invalidateRemoteMD(const std::string &remote_agent) {
    NIXL_LOCK_GUARD(data->lock);

    if (remote_agent == data->name) {
        NIXL_ERROR_FUNC << "remote agent same as local agent, cannot invalidate local metadata";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t ret = NIXL_ERR_NOT_FOUND;
    if (data->remoteSections.count(remote_agent) != 0) {
        delete data->remoteSections[remote_agent];
        data->remoteSections.erase(remote_agent);
        ret = NIXL_SUCCESS;
    }

    if (data->remoteBackends.count(remote_agent) != 0) {
        for (auto &it : data->remoteBackends[remote_agent]) {
            data->backendEngines[it.first]->disconnect(remote_agent);
        }

        data->remoteBackends.erase(remote_agent);
        ret = NIXL_SUCCESS;
    }

    if (ret == NIXL_ERR_NOT_FOUND)
        NIXL_INFO << __FUNCTION__ << ": remote metadata for agent '" << remote_agent
                  << "' not found.";
    else if (ret != NIXL_SUCCESS)
        NIXL_ERROR_FUNC << "error invalidating remote metadata for agent '" << remote_agent
                        << "' with status " << ret;
    return ret;
}

nixl_status_t
nixlAgent::sendLocalMD (const nixl_opt_args_t* extra_params) const {
    nixl_blob_t myMD;
    nixl_status_t ret = getLocalMD(myMD);
    if (ret < 0) {
        NIXL_ERROR_FUNC << "error getting local metadata with status " << ret;
        return ret;
    }

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
    NIXL_ERROR_FUNC << "invalid parameters to be used for either socket or ETCD";
    return NIXL_ERR_INVALID_PARAM;
#else
    NIXL_ERROR_FUNC
        << "sendLocalMD: ETCD is not supported and socket information was not provided either";
    return NIXL_ERR_NOT_SUPPORTED;
#endif // HAVE_ETCD
}

nixl_status_t
nixlAgent::sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                              const nixl_opt_args_t* extra_params) const {
    nixl_blob_t myMD;
    nixl_status_t ret = getLocalPartialMD(descs, myMD, extra_params);
    if (ret < 0) {
        NIXL_ERROR_FUNC << "error getting local partial metadata with status " << ret;
        return ret;
    }

    // If IP is provided, use socket-based communication
    if (extra_params && !extra_params->ipAddr.empty()) {
        data->enqueueCommWork(std::make_tuple(SOCK_SEND, extra_params->ipAddr, extra_params->port, std::move(myMD)));
        return NIXL_SUCCESS;
    }

#if HAVE_ETCD
    // If no IP is provided, use etcd (now via thread)
    if (data->useEtcd) {
        if (!extra_params || extra_params->metadataLabel.empty()) {
            NIXL_ERROR_FUNC << "metadata label is required for etcd send of local partial metadata";
            return NIXL_ERR_INVALID_PARAM;
        }
        data->enqueueCommWork(std::make_tuple(ETCD_SEND, extra_params->metadataLabel, 0, std::move(myMD)));
        return NIXL_SUCCESS;
    }
    NIXL_ERROR_FUNC << "invalid parameters to be used for either socket or ETCD";
    return NIXL_ERR_INVALID_PARAM;
#else
    NIXL_ERROR_FUNC << "ETCD is not supported and socket information was not provided either";
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
    NIXL_ERROR_FUNC << "invalid parameters to be used for either socket or ETCD";
    return NIXL_ERR_INVALID_PARAM;
#else
    NIXL_ERROR_FUNC << "ETCD is not supported and socket information was not provided either";
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
    NIXL_ERROR_FUNC << "invalid parameters to be used for either socket or ETCD";
    return NIXL_ERR_INVALID_PARAM;
#else
    NIXL_ERROR_FUNC << "ETCD is not supported and socket information was not provided either";
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
            nixl_meta_dlist_t dummy(descs.getType());
            // We only add to data->remoteBackends if data->backendEngines[backend] exists
            for (const auto& [backend, conn_info] : data->remoteBackends[remote_name])
                if (data->remoteSections[remote_name]->populate(
                          descs, data->backendEngines[backend], dummy) == NIXL_SUCCESS)
                    return NIXL_SUCCESS;
            dummy.clear();
        }
    }

    // This is a checker method, returning not found is not an error to be logged
    return NIXL_ERR_NOT_FOUND;
}

backend_set_t
nixlAgentData::getBackends(const nixl_opt_args_t *opt_args,
                           nixlMemSection *section,
                           nixl_mem_t mem_type) {
    if (opt_args && !opt_args->backends.empty()) {
        backend_set_t backends;
        for (const auto &backend : opt_args->backends) {
            backends.insert(backend->engine);
        }

        return backends;
    }

    const auto mem_type_backends = section->queryBackends(mem_type);
    return mem_type_backends ? *mem_type_backends : backend_set_t{};
}

nixl_status_t
nixlAgent::prepMemoryView(const nixl_remote_dlist_t &dlist,
                          nixlMemoryViewH &mvh,
                          const nixl_opt_args_t *extra_params) const {
    const auto desc_count = static_cast<size_t>(dlist.descCount());
    const auto mem_type = dlist.getType();
    nixl_remote_meta_dlist_t remote_meta_dlist{mem_type};
    nixlBackendEngine *engine{nullptr};

    NIXL_SHARED_LOCK_GUARD(data->lock);
    for (size_t i = 0; i < desc_count; ++i) {
        const auto &desc = dlist[i];
        if (desc.remoteAgent == nixl_invalid_agent) {
            remote_meta_dlist.addDesc(nixlRemoteMetaDesc(nixl_invalid_agent));
            continue;
        }

        const auto it = data->remoteSections.find(desc.remoteAgent);
        if (it == data->remoteSections.end()) {
            NIXL_ERROR_FUNC << "Metadata for remote agent '" << desc.remoteAgent << "' not found";
            return NIXL_ERR_NOT_FOUND;
        }

        if (engine) {
            // Engine has already been selected, add element to the remote metadata
            const auto status = it->second->addElement(desc, engine, remote_meta_dlist);
            if (status != NIXL_SUCCESS) {
                return status;
            }

            continue;
        }

        // Engine has not been selected yet, try to find a backend that can add an element to the
        // remote metadata
        const auto backends = data->getBackends(extra_params, it->second, mem_type);
        for (const auto &backend : backends) {
            const auto status = it->second->addElement(desc, backend, remote_meta_dlist);
            if (status == NIXL_SUCCESS) {
                NIXL_DEBUG << "Selected backend: " << backend->getType();
                engine = backend;
                break;
            }
        }

        // If no backend can add an element to the remote metadata, return an error
        if (!engine) {
            break;
        }
    }

    if (!engine) {
        NIXL_ERROR_FUNC
            << "A backend capable of creating a list of remote memory descriptors was not found";
        return NIXL_ERR_NOT_FOUND;
    }

    nixl_opt_b_args_t opt_args;
    if (extra_params) {
        opt_args.customParam = extra_params->customParam;
    }

    const auto status = engine->prepMemoryView(remote_meta_dlist, mvh, &opt_args);
    if (status == NIXL_SUCCESS) {
        data->mvhToEngine.emplace(mvh, *engine);
    }

    return status;
}

nixl_status_t
nixlAgent::prepMemoryView(const nixl_xfer_dlist_t &dlist,
                          nixlMemoryViewH &mvh,
                          const nixl_opt_args_t *extra_params) const {
    const auto mem_type = dlist.getType();
    nixl_meta_dlist_t meta_dlist{mem_type};
    nixlBackendEngine *engine{nullptr};

    NIXL_SHARED_LOCK_GUARD(data->lock);
    const auto backends = data->getBackends(extra_params, data->memorySection, mem_type);
    for (const auto &backend : backends) {
        const auto status = data->memorySection->populate(dlist, backend, meta_dlist);
        if (status == NIXL_SUCCESS) {
            NIXL_DEBUG << "Selected backend: " << backend->getType();
            engine = backend;
            break;
        }
    }

    if (!engine) {
        NIXL_ERROR_FUNC
            << "A backend capable of creating a list of local memory descriptors was not found";
        return NIXL_ERR_NOT_FOUND;
    }

    nixl_opt_b_args_t opt_args;
    if (extra_params) {
        opt_args.customParam = extra_params->customParam;
    }

    const auto status = engine->prepMemoryView(meta_dlist, mvh, &opt_args);
    if (status == NIXL_SUCCESS) {
        data->mvhToEngine.emplace(mvh, *engine);
    }

    return status;
}

void
nixlAgent::releaseMemoryView(nixlMemoryViewH mvh) const {
    NIXL_SHARED_LOCK_GUARD(data->lock);

    const auto it = data->mvhToEngine.find(mvh);
    if (it == data->mvhToEngine.end()) {
        NIXL_WARN << "Invalid memory view handle: " << mvh;
        return;
    }

    it->second.releaseMemoryView(mvh);
    data->mvhToEngine.erase(it);
}
