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
#include <optional>
#include <set>

#include <absl/strings/ascii.h>
#include <absl/strings/str_split.h>

#include "nixl.h"
#include "serdes/serdes.h"
#include "backend/backend_engine.h"
#include "transfer_request.h"
#include "agent_data.h"
#include "nixl_md_manager.h"
#include "plugin_manager.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include "common/operators.h"
#include "common/hw_info.h"
#include "telemetry.h"
#include "telemetry_event.h"
#include "tracing/trace.h"
#include "tracing/trace_macros.h"

namespace {

const std::vector<std::vector<std::string>> illegal_plugin_combinations = {
    {"GDS", "GDS_MT"},
};

} // namespace

void
nixlEngineDeleter::operator()(nixlBackendEngine *engine) const noexcept {
    auto &plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.getBackendPlugin(engine->getType());

    if (plugin_handle) {
        // If we have a plugin handle, use it to destroy the engine
        plugin_handle->destroyEngine(engine);
    }
    // TODO: Else delete engine?
}

nixlXferReqH::nixlXferReqH(const std::string &remote_agent,
                           const nixl_xfer_op_t backend_op,
                           const nixl_mem_t local_type,
                           const nixl_mem_t remote_type,
                           const uint64_t remote_generation,
                           const size_t desc_count)
    : initiatorDescs(local_type),
      targetDescs(remote_type),
      remoteAgent(remote_agent),
      remoteGeneration_(remote_generation),
      backendOp(backend_op) {
    initiatorDescs.reserve(desc_count);
    targetDescs.reserve(desc_count);
}

void
nixlXferReqH::updateRequestStats(nixlTelemetry *telemetry_pub,
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
        telemetry_pub->addXferStats(
            duration, backendOp == NIXL_WRITE, telemetry.totalBytes, telemetry.postDuration);
    }

    NIXL_TRACE << "[NIXL TELEMETRY]: From backend " << engine->getType()
               << nixl_post_status_str[stat_status] << " Xfer with " << telemetry.descCount
               << " descriptors of total size " << telemetry.totalBytes << "B in "
               << duration.count() << "us.";
}

nixlDlistH::nixlDlistH(const std::string &remote_agent, descs_t &&descs)
    : remoteAgent(remote_agent),
      descs(std::move(descs)) {}

/*** nixlAgentData constructor/destructor, as part of nixlAgent's ***/

namespace nixl::trace {

// True when the process runs under Nsight Systems: nsys injects
// NVTX_INJECTION64_PATH into the profiled process's environment (its presence
// means "running under nsys", not merely that nsys is installed).
[[nodiscard]] bool
runningUnderNsys() {
    return nixl::config::checkExistence("NVTX_INJECTION64_PATH");
}

// Backend-selection policy for the agent-wiring layer (kept out of the
// backend-agnostic facade so nsys/NVTX specifics never reach makeTracer()).
// Running under nsys auto-enables NVTX *in addition to* any explicitly requested
// backends; a set-but-empty NIXL_TRACE_BACKENDS is a hard "off" that beats it.
[[nodiscard]] std::vector<std::string>
resolveTraceBackends(const std::optional<std::string> &explicit_spec, bool under_nsys) {
    std::set<std::string> backends;
    bool explicit_off = false;
    if (explicit_spec) {
        // Trim entries so a padded value like "chakra, nvtx" matches backend
        // names; the set dedups them.
        for (const absl::string_view raw : absl::StrSplit(*explicit_spec, ',')) {
            const absl::string_view name = absl::StripAsciiWhitespace(raw);
            if (!name.empty()) {
                backends.emplace(name);
            }
        }
        // Set-but-empty (or all-blank) is an explicit "off" that must beat the
        // nsys auto-enable below.
        explicit_off = backends.empty();
    }

    if (under_nsys && !explicit_off) {
        backends.emplace("nvtx");
    }
    return {backends.begin(), backends.end()};
}

} // namespace nixl::trace

namespace {

[[nodiscard]] bool
detectEtcd() {
#if HAVE_ETCD
    return nixl::config::checkExistence("NIXL_ETCD_ENDPOINTS");
#else
    return false;
#endif
}

// Opt-in routing of the public metadata methods through the agent-owned
// nixlMDManager. Off by default; the inline path is used unless set.
[[nodiscard]] bool
detectMdManager() {
    return nixl::config::checkExistence("NIXL_USE_MD_MANAGER");
}

// The comm thread (used for etcd or listen-based metadata exchange) shares
// agent data structures (remoteSections_, remoteBackends_, …) with the caller.
// SYNC_NONE would leave those accesses unprotected, so upgrade to STRICT.
[[nodiscard]] nixl_thread_sync_t
effectiveSyncMode(nixl_thread_sync_t requested, bool needs_comm_thread) {
    if (needs_comm_thread && (requested == nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE)) {
        NIXL_INFO << "syncMode upgraded from NONE to STRICT "
                     "because a communication thread will be started";
        return nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
    }
    return requested;
}

// Build the agent's composite tracer. Backend selection follows
// nixl::trace::resolveTraceBackends (explicit NIXL_TRACE_BACKENDS wins; unset +
// running under nsys auto-enables NVTX). Returns null when nothing resolves, so
// the agent can hold the result in a const member and call sites take the no-op
// branch.
[[nodiscard]] std::unique_ptr<nixl::trace::Tracer>
makeAgentTracer(const std::string &name) {
    const auto trace_env = nixl::config::getValueOptional<std::string>("NIXL_TRACE_BACKENDS");
    auto requested_backends =
        nixl::trace::resolveTraceBackends(trace_env, nixl::trace::runningUnderNsys());
    if (requested_backends.empty()) {
        return nullptr;
    }
    return nixl::trace::makeTracer(nixl::trace::TracerConfig{name, std::move(requested_backends)});
}

// Build the manager (via the nixlMetadataContext interface) when metadata
// exchange is enabled; null otherwise, so nixlAgentData can hold it as const.
[[nodiscard]] std::unique_ptr<nixlMDManager>
makeMDManager(nixlMetadataContext &ctx, bool needs_comm_thread) {
    if (!needs_comm_thread) {
        return nullptr;
    }
    return std::make_unique<nixlMDManager>(ctx);
}

} // namespace

nixlAgentData::nixlAgentData(const std::string &name, const nixlAgentConfig &config)
    : name_(name),
      config_(config),
      useEtcd_(detectEtcd()),
      needsCommThread_(useEtcd_ || config.useListenThread),
      useMdManager_(detectMdManager()),
      lock(effectiveSyncMode(config.syncMode, needsCommThread_)),
      md_(makeMDManager(*this, needsCommThread_)),
      tracer_(makeAgentTracer(name)) {
#if HAVE_ETCD
    NIXL_DEBUG << "NIXL ETCD is " << (useEtcd_ ? "enabled" : "disabled");
#else
    NIXL_DEBUG << "NIXL ETCD is excluded";
#endif
    if (name.empty()) {
        throw std::invalid_argument("Agent needs a non-empty name");
    }

    const auto telemetry_enabled = nixl::config::getValueOptional<bool>("NIXL_TELEMETRY_ENABLE");

    if (telemetry_enabled) {
        if (*telemetry_enabled) {
            telemetry_ = nixlTelemetry::create(name);
        } else if (config.captureTelemetry) {
            NIXL_WARN << "NIXL telemetry is disabled; ignoring telemetry requested through agent "
                         "config";
        } else {
            NIXL_DEBUG << "NIXL telemetry is disabled";
        }
    } else if (config.captureTelemetry) {
        telemetry_ = nixlTelemetry::create(name);
    }
}

/*** nixlAgent implementation ***/
nixlAgent::nixlAgent(const std::string &name, const nixlAgentConfig &cfg)
    : data(std::make_unique<nixlAgentData>(name, cfg)) {
    if(cfg.useListenThread) {
        data->listener = std::make_unique<nixlMDStreamListener>(cfg.listenPort);
        data->listener->setupListener(); // throws on bind/listen failure
    }

    if (data->needsCommThread_) {
        data->commThreadStop = false;
        data->agentShutdown = false;
        data->commThread = std::thread(&nixlAgentData::commWorker, data.get(), std::ref(*this));
    }
}

nixlAgent::~nixlAgent() {
    if (data->needsCommThread_) {
        data->agentShutdown = true;
        // commQueue is guarded by commLock (see enqueueCommWork/getCommWork);
        // take the lock for the drain check to avoid racing the comm thread.
        while (true) {
            {
                const std::lock_guard<std::mutex> lock(data->commLock);
                if (data->commQueue.empty()) {
                    break;
                }
            }
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

        data->listener.reset();
    }
}

nixl_status_t
nixlAgent::getAvailPlugins (std::vector<nixl_backend_t> &plugins) {
    auto& plugin_manager = nixlPluginManager::getInstance();
    plugins = plugin_manager.getAvailBackendPluginNames();
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
        if (data->backendEngines_.count(type) == 0) {
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

void
nixlAgentData::warnAboutEfaHardwareMismatch() {
    // Atomic test-and-set so concurrent registerMem() calls warn at most once
    // without racing on the flag.
    if (efaWarningChecked.exchange(true)) {
        return;
    }

    if ((backendEngines_.count("UCX") != 0) && (backendEngines_.count("LIBFABRIC") == 0)) {
        const auto &hw_info = nixl::hwInfo::instance();
        if (hw_info.numEfaDevices > 0) {
            NIXL_WARN
                << hw_info.numEfaDevices
                << " Amazon EFA(s) were detected, but the UCX backend was configured."
                   " For best performance, it's recommended to use the LIBFABRIC backend instead.";
        }
    }
}

nixl_status_t
nixlAgent::createBackend(const nixl_backend_t &type,
                         const nixl_b_params_t &params,
                         nixlBackendH* &bknd_hndl) {

    NIXL_LOCK_GUARD(data->lock);
    // Registering same type of backend is not supported, unlikely and prob error
    if (data->backendEngines_.count(type) != 0) {
        NIXL_ERROR_FUNC << "backend already created for type '" << type << "'";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check if the plugin is in an illegal combination with another plugin backend already created
    for (const auto &combination : illegal_plugin_combinations) {
        if (std::find(combination.begin(), combination.end(), type) != combination.end()) {
            for (const auto &plugin_name : combination) {
                if (plugin_name != type &&
                    data->backendEngines_.find(plugin_name) != data->backendEngines_.end()) {
                    NIXL_ERROR_FUNC << "Plugin backend " << type
                                    << " is in illegal combination with " << plugin_name;
                    return NIXL_ERR_NOT_ALLOWED;
                }
            }
        }
    }

    nixlBackendInitParams init_params;
    init_params.localAgent = data->name_;
    init_params.type = type;
    init_params.customParams = const_cast<nixl_b_params_t *>(&params);
    init_params.enableProgTh = data->config_.useProgThread;
    init_params.pthrDelay = data->config_.pthrDelay;
    init_params.syncMode = data->config_.syncMode;
    init_params.enableTelemetry_ = (data->telemetry_ != nullptr);

    // First, try to load the backend as a plugin
    auto& plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.loadBackendPlugin(type);

    if (!plugin_handle) {
        NIXL_ERROR_FUNC << "unsupported backend '" << type << "'";
        return NIXL_ERR_NOT_FOUND;
    }

    // Plugin found, use it to create the backend
    backend_ptr_t backend(plugin_handle->createEngine(&init_params));

    if (!backend) {
        NIXL_ERROR_FUNC << "backend creation failed for '" << type << "'";
        return NIXL_ERR_BACKEND;
    }

    if (backend->getInitErr()) {
        NIXL_ERROR_FUNC << "backend initialization error for '" << type << "'";
        return NIXL_ERR_BACKEND;
    }

    std::string conn_info;

    if (backend->supportsRemote()) {
        if (!backend->supportsNotif()) {
            NIXL_ERROR_FUNC << "backend '" << type << "' supportsRemote but not notifications";
            return NIXL_ERR_BACKEND;
        }

        const nixl_status_t ret = backend->getConnInfo(conn_info);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR_FUNC << "failed to get connection info for '" << type << "' with status "
                            << ret;
            return ret;
        }
    }

    if (backend->supportsLocal()) {
        const nixl_status_t ret = backend->connect(data->name_);

        if (NIXL_SUCCESS != ret) {
            NIXL_ERROR_FUNC << "backend '" << type
                            << "' encountered error during intra-agent transfer setup with status "
                            << ret;
            return ret;
        }
    }

    for (auto &elm : backend->getSupportedMems()) {
        // First time creating this backend handle, so unique
        // The order of creation sets the preference order
        data->memToBackend[elm].push_back(backend.get());
    }

    if (backend->supportsRemote()) {
        data->notifEngines.push_back(backend.get());
        data->connMd_[type] = conn_info;
    }

    // TODO: Simplify, e.g. by making nixlBackendH's c'tor public?
    std::unique_ptr<nixlBackendH> bknd_temp(new nixlBackendH(backend.get()));
    const auto [it, inserted] = data->backendHandles_.try_emplace(type, std::move(bknd_temp));
    NIXL_ASSERT(inserted);
    bknd_hndl = it->second.get();

    data->backendEngines_.try_emplace(type, std::move(backend));

    // TODO: Check if backend supports ProgThread
    //       when threading is in agent

    NIXL_DEBUG << "Created backend: " << type;

    return NIXL_SUCCESS;
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
    NIXL_TRACE_SCOPE(
        trace_span, data->tracer_.get(), "nixl::registerMem", nixl::trace::Kind::MemoryW);
    NIXL_TRACE_ATTR(trace_span, "mem_type", static_cast<std::int64_t>(descs.getType()));

    backend_list_t* backend_list;
    unsigned int    count = 0;

    NIXL_LOCK_GUARD(data->lock);

    data->warnAboutEfaHardwareMismatch();

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
        nixl_status_t ret = data->localSection_.addDescList(descs, backend, sec_descs);
        if (ret == NIXL_SUCCESS) {
            if (backend->supportsLocal()) {
                const auto [it, inserted] =
                    data->remoteSections_.try_emplace(data->name_, data->name_);

                ret = it->second.loadLocalData(std::move(sec_descs), backend);
                if (ret == NIXL_SUCCESS) {
                    count++;
                } else {
                    data->localSection_.remDescList(descs, backend);
                }
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
    NIXL_TRACE_SCOPE(
        trace_span, data->tracer_.get(), "nixl::deregisterMem", nixl::trace::Kind::Generic);
    NIXL_TRACE_ATTR(trace_span, "mem_type", static_cast<std::int64_t>(descs.getType()));

    backend_set_t backend_set;
    nixl_status_t bad_ret = NIXL_SUCCESS;

    NIXL_LOCK_GUARD(data->lock);
    if (!extra_params || extra_params->backends.size() == 0) {
        backend_set_t *avail_backends = data->localSection_.queryBackends(descs.getType());
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
    for (auto &backend : backend_set) {
        if (backend->supportsLocal()) {
            const auto it = data->remoteSections_.find(data->name_);
            if (it != data->remoteSections_.end()) {
                it->second.removeLocalData(descs, *backend);
            }
        }

        const nixl_status_t ret = data->localSection_.remDescList(descs, backend);
        if (ret != NIXL_SUCCESS) {
            bad_ret = ret;
        }
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
    NIXL_TRACE_SCOPE(
        trace_span, data->tracer_.get(), "nixl::makeConnection", nixl::trace::Kind::Generic);
    NIXL_TRACE_ATTR(trace_span, "remote_agent", std::string_view{remote_agent});

    std::set<nixl_backend_t> backend_set;
    int count = 0;

    NIXL_LOCK_GUARD(data->lock);
    if (data->remoteBackends_.count(remote_agent) == 0) {
        NIXL_ERROR_FUNC << "metadata for remote agent '" << remote_agent << "' not found";
        return NIXL_ERR_NOT_FOUND;
    }

    if (!extra_params || extra_params->backends.size() == 0) {
        if (data->remoteBackends_[remote_agent].empty()) {
            NIXL_ERROR_FUNC << "no backends are found in metadata for remote agent '"
                            << remote_agent << "'";
            return NIXL_ERR_NOT_FOUND;
        }
        for (auto &[r_bknd, conn_info] : data->remoteBackends_[remote_agent])
            backend_set.insert(r_bknd);
    } else {
        for (auto & elm : extra_params->backends)
            backend_set.insert(elm->engine->getType());
    }

    // For now trying to make all the connections, can become best effort,
    nixl_status_t ret = NIXL_SUCCESS;
    for (auto & backend: backend_set) {
        const auto iter = data->backendEngines_.find(backend);
        if (iter != data->backendEngines_.end()) {
            nixlBackendEngine *eng = iter->second.get();
            ret = eng->connect(remote_agent);
            if (ret) {
                NIXL_ERROR_FUNC << "connect('" << remote_agent << "') failed on backend '"
                                << eng->getType() << "' with status " << ret;
                break;
            }
            count++;
        }
    }

    if (ret) { // Error is already logged
        return ret;
    }

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
    backend_set_t *backend_set;
    const bool init_side = (agent_name == NIXL_INIT_AGENT);

    NIXL_SHARED_LOCK_GUARD(data->lock);
    // When central KV is supported, still it should return error,
    // just we can add a call to fetchRemoteMD for next time
    const auto rem_sec_it = data->remoteSections_.find(agent_name);
    if (!init_side && (data->remoteSections_.end() == rem_sec_it)) {
        NIXL_ERROR_FUNC << "metadata for remote agent '" << agent_name << "' not found";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    nixlMemSection &section = init_side ? static_cast<nixlMemSection &>(data->localSection_) :
                                          static_cast<nixlMemSection &>(rem_sec_it->second);

    if (!extra_params || (extra_params->backends.size() == 0)) {
        backend_set = section.queryBackends(descs.getType());

        if (!backend_set || backend_set->empty()) {
            NIXL_ERROR_FUNC << "no available backends for mem type '" << descs.getType() << "'";
            data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
            return NIXL_ERR_NOT_FOUND;
        }
    } else {
        backend_set = new backend_set_t();
        for (auto &elm : extra_params->backends) {
            backend_set->insert(elm->engine);
        }
    }

    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    nixlDlistH::descs_t dlists;

    for (const auto &backend : *backend_set) {
        nixl_stride_dlist_t dlist(descs.getType());
        if (section.populate(descs, backend, dlist) == NIXL_SUCCESS) {
            NIXL_DEBUG << "backend " << backend->getType() << ": prepared " << descs.descCount()
                       << " descriptors into " << dlist.descCount() << " strides";

            dlists.try_emplace(backend, std::make_unique<nixl_stride_dlist_t>(std::move(dlist)));
        }
    }

    if (extra_params && (extra_params->backends.size() > 0)) {
        delete backend_set;
    }

    if (dlists.empty()) {
        dlist_hndl = nullptr;
        NIXL_ERROR_FUNC << "failed to prepare the descriptors for any of "
                           "the specified or potential backends for agent '"
                        << agent_name << "'";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    dlist_hndl = new nixlDlistH(agent_name, std::move(dlists));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::prepXferDlist(const nixl_xfer_dlist_t &descs,
                         nixlDlistH *&dlist_hndl,
                         const nixl_opt_args_t *extra_params) const {
    return prepXferDlist(NIXL_INIT_AGENT, descs, dlist_hndl, extra_params);
}

nixl_status_t
nixlAgent::makeXferReq (const nixl_xfer_op_t &operation,
                        const nixlDlistH* local_side,
                        const std::vector<int> &local_indices,
                        const nixlDlistH* remote_side,
                        const std::vector<int> &remote_indices,
                        nixlXferReqH* &req_hndl,
                        const nixl_opt_args_t* extra_params) const {
    NIXL_TRACE_SCOPE(
        trace_span, data->tracer_.get(), "nixl::makeXferReq", nixl::trace::Kind::Generic);
    NIXL_TRACE_ATTR(trace_span, "desc_count", static_cast<std::int64_t>(local_indices.size()));

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

    if ((!local_side->remoteAgent.empty()) || remote_side->remoteAgent.empty()) {
        NIXL_ERROR_FUNC << "invalid sides (local must be local, remote must be remote)";
        data->addErrorTelemetry(NIXL_ERR_INVALID_PARAM);
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((desc_count == 0) || (remote_indices.size() == 0) ||
        (desc_count != (int)remote_indices.size())) {
        NIXL_ERROR_FUNC << "different number of indices for local (" << desc_count << "), remote ("
                        << remote_indices.size() << ")";
        return NIXL_ERR_INVALID_PARAM;
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

    if (extra_params) {
        if (extra_params->notif) {
            opt_args.notifMsg = *extra_params->notif;
            opt_args.hasNotif = true;
        } else if (extra_params->hasNotif) {
            opt_args.notifMsg = extra_params->notifMsg;
            opt_args.hasNotif = true;
        }
    }

    if ((opt_args.hasNotif) && (!backend->supportsNotif())) {
        NIXL_ERROR_FUNC << "the selected backend '" << backend->getType()
                        << "' does not support notifications";
        return NIXL_ERR_BACKEND;
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);
    // The remote was invalidated in between prepXferDlist and this call
    const auto rem_sec_it = data->remoteSections_.find(remote_side->remoteAgent);
    if (rem_sec_it == data->remoteSections_.end()) {
        NIXL_ERROR_FUNC << "remote agent '" << remote_side->remoteAgent
                        << "' was invalidated in between prepXferDlist and this call";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    const nixl_stride_dlist_t &local_descs = *local_side->descs.at(backend);
    const nixl_stride_dlist_t &remote_descs = *remote_side->descs.at(backend);

    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    auto handle = std::make_unique<nixlXferReqH>(remote_side->remoteAgent,
                                                 operation,
                                                 local_descs.getType(),
                                                 remote_descs.getType(),
                                                 rem_sec_it->second.getGeneration(),
                                                 desc_count);

    size_t total_bytes = 0;
    const bool skip_desc_merge = extra_params && extra_params->skipDescMerge;
    const size_t local_size = local_descs.flatSize();
    const size_t remote_size = remote_descs.flatSize();
    // Ceiling division so that find()'s first probe (flat_idx / run_size) can never exceed
    // the last run index, letting find() skip a bounds clamp on its hot path.
    const size_t local_run_size =
        (local_size + local_descs.descCount() - 1) / local_descs.descCount();
    const size_t remote_run_size =
        (remote_size + remote_descs.descCount() - 1) / remote_descs.descCount();
    size_t seq_count = 1;

    for (size_t i = 0; i < static_cast<size_t>(desc_count); i += seq_count) {
        const size_t local_idx = static_cast<size_t>(local_indices[i]);
        const size_t remote_idx = static_cast<size_t>(remote_indices[i]);

        if (local_idx >= local_size) [[unlikely]] {
            NIXL_ERROR_FUNC << "local index out of range at index " << i << " with value "
                            << local_indices[i];
            return NIXL_ERR_INVALID_PARAM;
        }
        if (remote_idx >= remote_size) [[unlikely]] {
            NIXL_ERROR_FUNC << "remote index out of range at index " << i << " with value "
                            << remote_indices[i];
            return NIXL_ERR_INVALID_PARAM;
        }

        // Keep by value to avoid reloads and keep in registers
        const nixlStrideDesc local_stride = local_descs.find(local_idx, local_run_size);
        const nixlStrideDesc remote_stride = remote_descs.find(remote_idx, remote_run_size);

        if (local_stride.len != remote_stride.len) [[unlikely]] {
            NIXL_ERROR_FUNC << "length mismatch at index " << i << " with local index "
                            << local_indices[i] << " and remote index " << remote_indices[i];
            return NIXL_ERR_INVALID_PARAM;
        }

        seq_count = 1;
        // Merge only dense strides
        if (!skip_desc_merge && local_stride.stride == local_stride.len &&
            local_stride.stride == remote_stride.stride) [[likely]] {
            const size_t cap =
                std::min(std::min(local_stride.start_idx + local_stride.count - local_idx,
                                  remote_stride.start_idx + remote_stride.count - remote_idx),
                         static_cast<size_t>(desc_count) - i);

            auto *local_indices_ptr = reinterpret_cast<const unsigned *>(&local_indices[i]);
            auto *remote_indices_ptr = reinterpret_cast<const unsigned *>(&remote_indices[i]);
            while (seq_count < cap && local_indices_ptr[seq_count] == local_idx + seq_count &&
                   remote_indices_ptr[seq_count] == remote_idx + seq_count) {
                ++seq_count;
            }
        }

        const auto &local_desc =
            handle->initiatorDescs.emplace(local_stride.getMetaDesc(local_idx, seq_count));
        handle->targetDescs.emplace(remote_stride.getMetaDesc(remote_idx, seq_count));
        total_bytes += local_desc.len;
    }

    NIXL_DEBUG << "merged " << desc_count << " indices into " << handle->initiatorDescs.descCount()
               << " descriptors";

    handle->engine = backend;
    handle->notifMsg = opt_args.notifMsg;
    handle->hasNotif = opt_args.hasNotif;

    // Set unconditionally so the trace bytes/desc_count attributes are correct
    // even when telemetry is disabled; both telemetry and tracing read these
    // fields when active.
    handle->telemetry.totalBytes = total_bytes;
    handle->telemetry.descCount = handle->initiatorDescs.descCount();

    ret = handle->engine->prepXfer(handle->backendOp,
                                   handle->initiatorDescs,
                                   handle->targetDescs,
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
    NIXL_TRACE_SCOPE(
        trace_span, data->tracer_.get(), "nixl::createXferReq", nixl::trace::Kind::Generic);
    NIXL_TRACE_ATTR(trace_span, "remote_agent", std::string_view{remote_agent});
    NIXL_TRACE_ATTR(trace_span, "desc_count", static_cast<std::int64_t>(local_descs.descCount()));

    nixl_status_t ret1, ret2;
    nixl_opt_b_args_t opt_args;
    backend_set_t backend_set;

    req_hndl = nullptr;

    // Check the correspondence between descriptor lists
    if (local_descs.descCount() != remote_descs.descCount()) {
        NIXL_ERROR_FUNC << "different descriptor list sizes (local=" << local_descs.descCount()
                        << ", remote=" << remote_descs.descCount() << ")";
        return NIXL_ERR_INVALID_PARAM;
    }

    size_t total_bytes = 0;
    for (int i = 0; i < local_descs.descCount(); ++i) {
        if (local_descs[i].len != remote_descs[i].len) [[unlikely]] {
            NIXL_ERROR_FUNC << "length mismatch at index " << i;
            return NIXL_ERR_INVALID_PARAM;
        }
        total_bytes += local_descs[i].len;
    }

    NIXL_SHARED_LOCK_GUARD(data->lock);
    const auto rem_sec_it = data->remoteSections_.find(remote_agent);
    if (data->remoteSections_.end() == rem_sec_it) {
        NIXL_ERROR_FUNC << "metadata for remote agent '" << remote_agent << "' not found";
        data->addErrorTelemetry(NIXL_ERR_NOT_FOUND);
        return NIXL_ERR_NOT_FOUND;
    }

    if (!extra_params || extra_params->backends.size() == 0) {
        // Finding backends that support the corresponding memories
        // locally and remotely, and find the common ones.
        backend_set_t *local_set = data->localSection_.queryBackends(local_descs.getType());
        backend_set_t *remote_set = rem_sec_it->second.queryBackends(remote_descs.getType());
        if (!local_set || !remote_set) {
            NIXL_ERROR_FUNC << "no backends found for local or remote for their "
                               "corresponding memory type";
            return NIXL_ERR_NOT_FOUND;
        }

        for (auto & elm : *local_set)
            if (remote_set->count(elm) != 0) {
                backend_set.insert(elm);
            }

        if (backend_set.empty()) {
            NIXL_ERROR_FUNC << "no potential backend found to be able to do the transfer";
            return NIXL_ERR_NOT_FOUND;
        }
    } else {
        for (auto & elm : extra_params->backends)
            backend_set.insert(elm->engine);
    }

    // TODO: when central KV is supported, add a call to fetchRemoteMD
    // TODO: merge descriptors back to back in memory (like makeXferReq).
    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    auto handle = std::make_unique<nixlXferReqH>(remote_agent,
                                                 operation,
                                                 local_descs.getType(),
                                                 remote_descs.getType(),
                                                 rem_sec_it->second.getGeneration(),
                                                 local_descs.descCount());

    // Currently we loop through and find first local match. Can use a
    // preference list or more exhaustive search.
    for (auto &backend : backend_set) {
        // If populate fails, it clears the resp before return
        ret1 = data->localSection_.populate(local_descs, backend, handle->initiatorDescs);
        ret2 = rem_sec_it->second.populate(remote_descs, backend, handle->targetDescs);

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
        if (extra_params->notif) {
            opt_args.notifMsg = *extra_params->notif;
            opt_args.hasNotif = true;
        } else if (extra_params->hasNotif) {
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

    handle->notifMsg = opt_args.notifMsg;
    handle->hasNotif = opt_args.hasNotif;

    // Set unconditionally so the trace bytes/desc_count attributes are correct
    // even when telemetry is disabled; both telemetry and tracing read these
    // fields when active.
    handle->telemetry.totalBytes = total_bytes;
    handle->telemetry.descCount = handle->initiatorDescs.descCount();

    ret1 = handle->engine->prepXfer(handle->backendOp,
                                    handle->initiatorDescs,
                                    handle->targetDescs,
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
        (data->remoteSections_.count(req_hndl->remoteAgent) == 0)) {
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
                                             req_hndl->initiatorDescs,
                                             req_hndl->targetDescs,
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

    NIXL_TRACE_SCOPE(trace_span,
                     data->tracer_.get(),
                     req_hndl->backendOp == NIXL_WRITE ? "nixl::postXferReq.write" :
                                                         "nixl::postXferReq.read",
                     req_hndl->backendOp == NIXL_WRITE ? nixl::trace::Kind::CommSend :
                                                         nixl::trace::Kind::CommRecv);
    NIXL_TRACE_ATTR(trace_span, "remote_agent", std::string_view{req_hndl->remoteAgent});
    NIXL_TRACE_ATTR(trace_span, "bytes", static_cast<std::int64_t>(req_hndl->telemetry.totalBytes));

    if (data->telemetry_) {
        req_hndl->telemetry.startTime = std::chrono::steady_clock::now();
    }

    std::shared_lock<nixlLock> read_lock(data->lock);
    // Check if the remote was invalidated before post/repost
    if (data->remoteSections_.count(req_hndl->remoteAgent) == 0) {
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
            read_lock.unlock();
            NIXL_LOCK_GUARD(data->lock);
            data->invalidateRemoteData(req_hndl->remoteAgent, req_hndl->remoteGeneration_);
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
        if (extra_params->notif) {
            req_hndl->notifMsg = *extra_params->notif;
            opt_args.notifMsg = *extra_params->notif;
            req_hndl->hasNotif = true;
            opt_args.hasNotif = true;
        } else if (extra_params->hasNotif) {
            req_hndl->notifMsg = extra_params->notifMsg;
            opt_args.notifMsg = extra_params->notifMsg;
            req_hndl->hasNotif = true;
            opt_args.hasNotif = true;
        } else {
            req_hndl->hasNotif = false;
            opt_args.hasNotif = false;
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
                                                  req_hndl->initiatorDescs,
                                                  req_hndl->targetDescs,
                                                  req_hndl->remoteAgent,
                                                  req_hndl->backendHandle,
                                                  &opt_args);

    if (req_hndl->status < 0) {
        if (req_hndl->status == NIXL_ERR_REMOTE_DISCONNECT) {
            read_lock.unlock();
            NIXL_LOCK_GUARD(data->lock);
            NIXL_ERROR_FUNC << "remote agent '" << req_hndl->remoteAgent
                            << "' was disconnected after transfer request creation";
            data->invalidateRemoteData(req_hndl->remoteAgent, req_hndl->remoteGeneration_);
            return NIXL_ERR_REMOTE_DISCONNECT;
        } else {
            NIXL_ERROR_FUNC << "backend '" << req_hndl->engine->getType()
                            << "' failed to post the transfer request with status "
                            << req_hndl->status;
        }
    }

    if (data->telemetry_) {
        NIXL_DEBUG << req_hndl->initiatorDescs.to_string(true);

        if (req_hndl->status < 0) {
            data->addErrorTelemetry(req_hndl->status);
        } else if (req_hndl->status == NIXL_IN_PROG) {
            req_hndl->updateRequestStats(data->telemetry_.get(), NIXL_TELEMETRY_POST);
        } else {
            req_hndl->updateRequestStats(data->telemetry_.get(), NIXL_TELEMETRY_POST_AND_FINISH);
        }
    }

    return req_hndl->status;
}

nixl_status_t
nixlAgent::getXferStatus (nixlXferReqH *req_hndl) const {

    std::shared_lock<nixlLock> read_lock(data->lock);
    // If the status is done, no need to recheck and no state changes.
    // Same for users incorrectly recalling this method in error/done.
    if (req_hndl->status == NIXL_IN_PROG) {
        // Check if the remote was invalidated before completion
        if (data->remoteSections_.count(req_hndl->remoteAgent) == 0) {
            NIXL_ERROR_FUNC << "remote agent '" << req_hndl->remoteAgent
                            << "' was invalidated during transfer";
            return NIXL_ERR_NOT_FOUND;
        }

        req_hndl->status = req_hndl->engine->checkXfer(req_hndl->backendHandle);
        if (req_hndl->status < 0) {
            if (req_hndl->status == NIXL_ERR_REMOTE_DISCONNECT) {
                read_lock.unlock();
                NIXL_LOCK_GUARD(data->lock);
                data->invalidateRemoteData(req_hndl->remoteAgent, req_hndl->remoteGeneration_);
                return NIXL_ERR_REMOTE_DISCONNECT;
            } else {
                NIXL_ERROR_FUNC << "backend '" << req_hndl->engine->getType()
                                << "' returned error status " << req_hndl->status;
            }
        }
        if (req_hndl->status == NIXL_SUCCESS) {
            NIXL_TRACE_MARK(
                data->tracer_.get(), "nixl::xfer.complete", nixl::trace::Kind::Metadata);
        }
        if (data->telemetry_) {
            if (req_hndl->status == NIXL_SUCCESS) {
                req_hndl->updateRequestStats(data->telemetry_.get(), NIXL_TELEMETRY_FINISH);
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

    if (!data->telemetry_) {
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
    backend = data->backendHandles_[req_hndl->engine->getType()].get();
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
nixlAgent::releasedDlistH (nixlDlistH* dlist_hndl) const {
    NIXL_LOCK_GUARD(data->lock);
    delete dlist_hndl;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getNotifs(nixl_notifs_t &notif_map,
                     const nixl_opt_args_t* extra_params) {
    NIXL_TRACE_SCOPE(
        trace_span, data->tracer_.get(), "nixl::getNotifs", nixl::trace::Kind::Metadata);

    notif_list_t bknd_notif_list;
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
    NIXL_TRACE_SCOPE(
        trace_span, data->tracer_.get(), "nixl::genNotif", nixl::trace::Kind::Metadata);
    NIXL_TRACE_ATTR(trace_span, "remote_agent", std::string_view{remote_agent});

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

    if (data->name_ == remote_agent) {
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
    const auto iter = data->remoteBackends_.find(remote_agent);

    if (iter != data->remoteBackends_.end()) {
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
nixlAgentData::getLocalMD(nixl_blob_t &str) {
    nixl_backend_t nixl_backend;
    nixl_status_t ret;

    NIXL_LOCK_GUARD(lock);
    // connMd_ was populated when the backend was created
    const size_t conn_cnt = connMd_.size();

    if (conn_cnt == 0) { // Error, no backend supports remote
        NIXL_ERROR_FUNC << "no backends support remote operations";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes sd;
    ret = sd.addStr("Agent", name_);
    // Always returns SUCCESS, serdes class logs errors if necessary
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    ret = sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    for (auto &c : connMd_) {
        nixl_backend = c.first;
        ret = sd.addStr("t", nixl_backend);
        if (ret != NIXL_SUCCESS) {
            break;
        }
        ret = sd.addStr("c", c.second);
        if (ret != NIXL_SUCCESS) {
            break;
        }
    }
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    ret = sd.addStr("", "MemSection");
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    ret = localSection_.serialize(&sd);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "serialization failed";
        return ret;
    }

    str = sd.exportStr();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getLocalMD(nixl_blob_t &str) const {
    size_t conn_cnt;
    nixl_backend_t nixl_backend;
    nixl_status_t ret;

    NIXL_LOCK_GUARD(data->lock);
    // data->connMd_ was populated when the backend was created
    conn_cnt = data->connMd_.size();

    if (conn_cnt == 0) { // Error, no backend supports remote
        NIXL_ERROR_FUNC << "no backends support remote operations";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes sd;
    ret = sd.addStr("Agent", data->name_);
    // Always returns SUCCESS, serdes class logs errors if necessary
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if (ret) return NIXL_ERR_UNKNOWN;

    for (auto &c : data->connMd_) {
        nixl_backend = c.first;
        ret = sd.addStr("t", nixl_backend);
        if (ret) break;
        ret = sd.addStr("c", c.second);
        if (ret) break;
    }
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = sd.addStr("", "MemSection");
    if (ret) return NIXL_ERR_UNKNOWN;

    ret = data->localSection_.serialize(&sd);
    if (ret) {
        NIXL_ERROR_FUNC << "serialization failed";
        return ret;
    }

    str = sd.exportStr();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgentData::getLocalPartialMD(const nixl_reg_dlist_t &descs,
                                 nixl_blob_t &str,
                                 const nixl_opt_args_t *extra_params) {
    backend_list_t tmp_list;
    backend_list_t *backend_list;
    nixl_status_t ret;

    NIXL_LOCK_GUARD(lock);

    if (!extra_params || extra_params->backends.size() == 0) {
        if (!descs.isEmpty()) {
            // Non-empty dlist, return backends that support the memory type
            backend_list = &memToBackend[descs.getType()];
            if (backend_list->empty()) {
                NIXL_ERROR_FUNC << "no available backends for mem type '" << descs.getType() << "'";
                return NIXL_ERR_NOT_FOUND;
            }
        } else {
            // Empty dlist, return all backends
            backend_list = &tmp_list;
            for (const auto &elm : backendEngines_) {
                backend_list->push_back(elm.second.get());
            }
        }
    } else {
        backend_list = &tmp_list;
        for (const auto &elm : extra_params->backends) {
            backend_list->push_back(elm->engine);
        }
    }

    // First find all relevant engines and their conn info.
    // Best effort, ignore if no conn info (meaning backend doesn't support remote).
    backend_set_t selected_engines;
    std::vector<typename decltype(connMd_)::iterator> found_iters;
    for (const auto &backend : *backend_list) {
        auto it = connMd_.find(backend->getType());
        if (it == connMd_.end()) {
            continue;
        }
        found_iters.push_back(it);
        selected_engines.insert(backend);
    }

    if (selected_engines.size() == 0 && descs.descCount() > 0) {
        NIXL_ERROR_FUNC << "no backends support the requested descriptors";
        return NIXL_ERR_BACKEND;
    }

    nixlSerDes sd;
    ret = sd.addStr("Agent", name_);
    // Always returns SUCCESS, serdes class logs errors if necessary
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    // Only add connection info if requested via extra_params or empty dlist
    size_t conn_cnt = ((extra_params && extra_params->includeConnInfo) || descs.descCount() == 0) ?
        found_iters.size() :
        0;
    ret = sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    for (size_t i = 0; i < conn_cnt; i++) {
        ret = sd.addStr("t", found_iters[i]->first);
        if (ret != NIXL_SUCCESS) {
            break;
        }
        ret = sd.addStr("c", found_iters[i]->second);
        if (ret != NIXL_SUCCESS) {
            break;
        }
    }
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    ret = sd.addStr("", "MemSection");
    if (ret != NIXL_SUCCESS) {
        return NIXL_ERR_UNKNOWN;
    }

    ret = localSection_.serializePartial(&sd, selected_engines, descs);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "serialization failed";
        return ret;
    }

    str = sd.exportStr();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getLocalPartialMD(const nixl_reg_dlist_t &descs,
                             nixl_blob_t &str,
                             const nixl_opt_args_t *extra_params) const {
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
            for (const auto &elm : data->backendEngines_) {
                backend_list->push_back(elm.second.get());
            }
        }
    } else {
        backend_list = &tmp_list;
        for (const auto &elm : extra_params->backends) {
            backend_list->push_back(elm->engine);
        }
    }

    // First find all relevant engines and their conn info.
    // Best effort, ignore if no conn info (meaning backend doesn't support remote).
    backend_set_t selected_engines;
    std::vector<typename decltype(data->connMd_)::iterator> found_iters;
    for (const auto &backend : *backend_list) {
        auto it = data->connMd_.find(backend->getType());
        if (it == data->connMd_.end()) continue;
        found_iters.push_back(it);
        selected_engines.insert(backend);
    }

    if (selected_engines.size() == 0 && descs.descCount() > 0) {
        NIXL_ERROR_FUNC << "no backends support the requested descriptors";
        return NIXL_ERR_BACKEND;
    }

    nixlSerDes sd;
    ret = sd.addStr("Agent", data->name_);
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

    ret = data->localSection_.serializePartial(&sd, selected_engines, descs);
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

    if (remote_agent == data->name_) {
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

    if (remote_agent == data->name_) {
        NIXL_ERROR_FUNC << "remote agent same as local agent, cannot invalidate local metadata";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t ret = NIXL_ERR_NOT_FOUND;
    if (data->remoteSections_.erase(remote_agent) > 0) {
        ret = NIXL_SUCCESS;
    }

    if (data->remoteBackends_.count(remote_agent) != 0) {
        for (auto &it : data->remoteBackends_[remote_agent]) {
            data->backendEngines_[it.first]->disconnect(remote_agent);
        }

        data->remoteBackends_.erase(remote_agent);
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
    // Opt-in: route P2P sends through the agent-owned metadata manager.
    if (data->md_ && data->useMdManager_ && extra_params && !extra_params->ipAddr.empty()) {
        return data->md_->sendLocalMD(extra_params);
    }

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
    if (data->useEtcd_) {
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
    // Opt-in: route P2P sends through the agent-owned metadata manager.
    if (data->md_ && data->useMdManager_ && extra_params && !extra_params->ipAddr.empty()) {
        return data->md_->sendLocalPartialMD(descs, extra_params);
    }

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
    if (data->useEtcd_) {
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
    // Opt-in: route P2P fetches through the agent-owned metadata manager.
    if (data->md_ && data->useMdManager_ && extra_params && !extra_params->ipAddr.empty()) {
        return data->md_->fetchRemoteMD(remote_name, extra_params);
    }

    // If IP is provided, use socket-based communication
    if (extra_params && !extra_params->ipAddr.empty()) {
        data->enqueueCommWork(std::make_tuple(SOCK_FETCH, extra_params->ipAddr, extra_params->port, ""));
        return NIXL_SUCCESS;
    }

#if HAVE_ETCD
    // If no IP is provided, use etcd via thread with watch capability
    if (data->useEtcd_) {
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
    // Opt-in: route P2P invalidations through the agent-owned metadata manager.
    if (data->md_ && data->useMdManager_ && extra_params && !extra_params->ipAddr.empty()) {
        return data->md_->invalidateLocalMD(extra_params);
    }

    // If IP is provided, use socket-based communication
    if (extra_params && !extra_params->ipAddr.empty()) {
        data->enqueueCommWork(std::make_tuple(SOCK_INVAL, extra_params->ipAddr, extra_params->port, ""));
        return NIXL_SUCCESS;
    }

#if HAVE_ETCD
    // If no IP is provided, use etcd via thread
    if (data->useEtcd_) {
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
    const auto rem_sec_it = data->remoteSections_.find(remote_name);
    if (data->remoteSections_.end() == rem_sec_it) {
        // This is a checker method, returning not found is not an error to be logged
        return NIXL_ERR_NOT_FOUND;
    }

    if (descs.isEmpty()) {
        return NIXL_SUCCESS;
    }

    nixl_meta_dlist_t dummy(descs.getType());
    // We only add to data->remoteBackends_ if data->backendEngines_[backend] exists
    for (const auto &[backend, conn_info] : data->remoteBackends_[remote_name]) {
        if (rem_sec_it->second.populate(descs, data->backendEngines_[backend].get(), dummy) ==
            NIXL_SUCCESS) {
            return NIXL_SUCCESS;
        }
    }

    // This is a checker method, returning not found is not an error to be logged
    return NIXL_ERR_NOT_FOUND;
}

backend_set_t
nixlAgentData::getBackends(const nixl_opt_args_t *opt_args,
                           const nixlMemSection &section,
                           nixl_mem_t mem_type) {
    if (opt_args && !opt_args->backends.empty()) {
        backend_set_t backends;
        for (const auto &backend : opt_args->backends) {
            backends.insert(backend->engine);
        }

        return backends;
    }

    const auto mem_type_backends = section.queryBackends(mem_type);
    return mem_type_backends ? *mem_type_backends : backend_set_t{};
}

nixl_status_t
nixlAgent::prepMemView(const nixl_remote_dlist_t &dlist,
                       nixlMemViewH &mvh,
                       const nixl_opt_args_t *extra_params) const {
    const auto desc_count = static_cast<size_t>(dlist.descCount());
    const auto mem_type = dlist.getType();
    nixl_remote_meta_dlist_t remote_meta_dlist{mem_type};
    nixlBackendEngine *engine{nullptr};

    NIXL_SHARED_LOCK_GUARD(data->lock);
    for (size_t i = 0; i < desc_count; ++i) {
        const auto &desc = dlist[i];
        if (desc.remoteAgent == nixl_null_agent) {
            remote_meta_dlist.addDesc(nixlRemoteMetaDesc(nixl_null_agent));
            continue;
        }

        const auto it = data->remoteSections_.find(desc.remoteAgent);
        if (it == data->remoteSections_.end()) {
            NIXL_ERROR_FUNC << "Metadata for remote agent '" << desc.remoteAgent << "' not found";
            return NIXL_ERR_NOT_FOUND;
        }

        if (engine) {
            // Engine has already been selected, add element to the remote metadata
            const auto status = it->second.addElement(desc, engine, remote_meta_dlist);
            if (status != NIXL_SUCCESS) {
                return status;
            }

            continue;
        }

        // Engine has not been selected yet, try to find a backend that can add an element to the
        // remote metadata
        const auto backends = data->getBackends(extra_params, it->second, mem_type);
        for (const auto &backend : backends) {
            const auto status = it->second.addElement(desc, backend, remote_meta_dlist);
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

    const auto status = engine->prepMemView(remote_meta_dlist, mvh, &opt_args);
    if (status == NIXL_SUCCESS) {
        data->mvhToEngine.emplace(mvh, *engine);
    }

    return status;
}

nixl_status_t
nixlAgent::prepMemView(const nixl_local_dlist_t &dlist,
                       nixlMemViewH &mvh,
                       const nixl_opt_args_t *extra_params) const {
    const auto mem_type = dlist.getType();
    nixl_meta_dlist_t meta_dlist{mem_type};
    nixlBackendEngine *engine{nullptr};

    NIXL_SHARED_LOCK_GUARD(data->lock);
    const auto backends = data->getBackends(extra_params, data->localSection_, mem_type);
    for (const auto &backend : backends) {
        const auto status = data->localSection_.populate(dlist, backend, meta_dlist);
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

    const auto status = engine->prepMemView(meta_dlist, mvh, &opt_args);
    if (status == NIXL_SUCCESS) {
        data->mvhToEngine.emplace(mvh, *engine);
    }

    return status;
}

void
nixlAgent::releaseMemView(nixlMemViewH mvh) const {
    NIXL_SHARED_LOCK_GUARD(data->lock);

    const auto it = data->mvhToEngine.find(mvh);
    if (it == data->mvhToEngine.end()) {
        NIXL_WARN << "Invalid memory view handle: " << mvh;
        return;
    }

    it->second.releaseMemView(mvh);
    data->mvhToEngine.erase(it);
}
