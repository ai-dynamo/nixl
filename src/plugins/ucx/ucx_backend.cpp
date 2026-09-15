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

#include "ucx_backend.h"
#include "ucx_thread_engine.h"
#include "ucx_thread_pool_engine.h"
#include "ucx_backend_req.h"
#include "ucx_sgl.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include "common/backend.h"
#include "common/configuration.h"
#include "common/nixl_log.h"

#include <optional>
#include <string.h>
#include "absl/strings/str_split.h"

namespace {
[[nodiscard]] bool
sglEnabledFromConfig() {
    const bool enabled = nixl::config::getValueDefaulted("NIXL_UCX_SGL_ENABLE", false);
#ifdef HAVE_UCX_SGL_API
    NIXL_DEBUG << "UCX SGL offload " << (enabled ? "enabled" : "disabled");
    return enabled;
#else
    if (enabled) {
        NIXL_WARN << "NIXL_UCX_SGL_ENABLE is set but NIXL was built without UCX SGL support";
    }
    return false;
#endif
}
} // namespace

// A transfer to a single endpoint posts at most three requests:
// one data request, one flush request, and one notification request.
constexpr size_t single_ep_request_count = 3;

/****************************************
 * Constructor/Destructor
 *****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params) {
    nixlUcxEngine *engine;
    const size_t num_threads =
        nixl::getBackendParamDefaulted(init_params.customParams, "num_threads", 0u);
    if (num_threads > 0) {
        engine = new nixlUcxThreadPoolEngine(init_params, num_threads);
    } else if (init_params.enableProgTh) {
        engine = new nixlUcxThreadEngine(init_params);
    } else {
        engine = new nixlUcxEngine(init_params);
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams &init_params, size_t num_dedicated_workers)
    : nixlBackendEngine(&init_params),
      sharedWorkerIndex_(1),
      sglEnabled_(sglEnabledFromConfig()) {
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t *custom_params = init_params.customParams;

    if (const auto opt = nixl::getBackendParamOptional<std::string>(custom_params, "device_list")) {
        devs = absl::StrSplit(*opt, ", ");
    }

    size_t num_workers = nixl::getBackendParamDefaulted(custom_params, "num_workers", 1u);
    if (num_workers <= num_dedicated_workers) {
        num_workers = num_dedicated_workers + 1;
    }
    numSharedWorkers_ = num_workers - num_dedicated_workers;

    const size_t num_device_channels =
        nixl::getBackendParamDefaulted(custom_params, "ucx_num_device_channels", 4u);


    ucp_err_handling_mode_t err_handling_mode = UCP_ERR_HANDLING_MODE_PEER;
    if (const auto opt = nixl::getBackendParamOptional<std::string>(
            custom_params, std::string(nixl_ucx_err_handling_param_name))) {
        err_handling_mode = ucx_err_mode_from_string(*opt);
    }

    const auto engine_config =
        nixl::getBackendParamDefaulted(custom_params, "engine_config", std::string());

    uc = std::make_unique<nixlUcxContext>(devs,
                                          init_params.enableProgTh,
                                          num_workers,
                                          init_params.syncMode,
                                          num_device_channels,
                                          engine_config,
                                          localAgent);

    uc->warnAboutHardwareSupportMismatch();

    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; i++) {
        workers_.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode, i));
    }

    auto &worker = workers_.front();
    workerAddr = worker->epAddr();
    worker->regAmCallback(nixl::ucx::am_cb_op_t::NOTIF_STR, notifAmCb, this);
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

static std::unordered_map<const nixlUcxEngine *, size_t> &
tlsSharedWorkerMap() {
    static thread_local std::unordered_map<const nixlUcxEngine *, size_t> map;
    return map;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine() {
    tlsSharedWorkerMap().erase(this);
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    str = workerAddr;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    if(remote_agent == localAgent) {
        return loadRemoteConnInfo(remote_agent, workerAddr);
    }

    return (remoteConnMap.find(remote_agent) == remoteConnMap.end()) ? NIXL_ERR_NOT_FOUND :
                                                                       NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
    const auto it = remoteConnMap.find(remote_agent);

    if (it == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    // thread safety?
    remoteConnMap.erase(it);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                 const std::string &remote_conn_info)
{
    size_t size = remote_conn_info.size();
    std::vector<char> addr(size);

    if(remoteConnMap.count(remote_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes::_stringToBytes(addr.data(), remote_conn_info, size);
    std::shared_ptr<nixlUcxConnection> conn = std::make_shared<nixlUcxConnection>();
    for (const auto &uw : workers_) {
        std::unique_ptr<nixlUcxEp> ep = uw->connect(addr.data());
        if (!ep) {
            return NIXL_ERR_BACKEND;
        }
        conn->eps.push_back(std::move(ep));
    }

    remoteConnMap.insert({remote_agent, conn});

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlUcxEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

    // TODO: Add nixl_mem check?
    const int ret = uc->memReg((void*) mem.addr, mem.len, priv->mem, nixl_mem);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    priv->rkeyStr = uc->packRkey(priv->mem);

    if (priv->rkeyStr.empty()) {
        return NIXL_ERR_BACKEND;
    }
    out = priv.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}

namespace {

[[nodiscard]] std::vector<nixl::ucx::rkey>
makePublicMetadataRkeys(const ucx_connection_ptr_t &conn, const size_t count, const void *buffer) {
    std::vector<nixl::ucx::rkey> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(*conn->getEp(i), buffer);
    }
    return result;
}

} // namespace

nixlUcxPublicMetadata::nixlUcxPublicMetadata(const ucx_connection_ptr_t &conn,
                                             std::vector<nixl::ucx::rkey> &&rkeys)
    : nixlBackendMD(false),
      conn(conn),
      rkeys_(std::move(rkeys)) {}

nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    try {
        const auto it = remoteConnMap.find(agent);

        if (it == remoteConnMap.end()) {
            // TODO: err: remote connection not found
            return NIXL_ERR_NOT_FOUND;
        }
        for (size_t i = 0; i < workers_.size(); ++i) {
            const nixl_status_t status = it->second->getEp(i)->checkTxState();
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
        // nixlSerDes::_stringToBytes() was used to "unpack" blob here.
        output = new nixlUcxPublicMetadata(
            it->second, makePublicMetadataRkeys(it->second, workers_.size(), blob.data()));
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {

    nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata*) input; //typecast?
    delete md;

    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

size_t
nixlUcxEngine::getSharedWorkerId(const nixl_opt_b_args_t *opt_args) const noexcept {
    if (opt_args) {
        const std::optional<size_t> worker_id = getWorkerIdFromOptArgs(*opt_args);
        if (worker_id) {
            return *worker_id;
        }
    }

    auto it = tlsSharedWorkerMap().find(this);
    if (it == tlsSharedWorkerMap().end()) {
        const size_t index = sharedWorkerIndex_.fetch_add(1) % getSharedWorkersSize();
        it = tlsSharedWorkerMap().emplace(this, index).first;
        NIXL_DEBUG << "engine " << this << " bound shared worker " << index << " to thread "
                   << std::this_thread::get_id();
    }
    return it->second;
}

std::optional<size_t>
nixlUcxEngine::getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept {
    constexpr std::string_view worker_id_key = "worker_id=";
    size_t pos = opt_args.customParam.find(worker_id_key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }

    try {
        size_t worker_id = std::stoull(opt_args.customParam.substr(pos + worker_id_key.length()));

        if (worker_id >= getSharedWorkersSize()) {
            NIXL_WARN << "Invalid worker_id " << worker_id << " (must be < "
                      << getSharedWorkersSize() << ")";
            return std::nullopt;
        }

        return worker_id;
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Failed to parse worker_id from customParam: " << e.what();
        return std::nullopt;
    }
}

nixl_status_t nixlUcxEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    if (local.descCount() == 0 || remote.descCount() == 0) {
        NIXL_ERROR << "Local or remote descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t worker_id = getSharedWorkerId(opt_args);
    /* TODO: try to get from a pool first */
    handle = new nixlUcxBackendReqH(getSharedWorker(worker_id).get());

#ifdef HAVE_UCX_SGL_API
    if (sglEnabled_ && operation == NIXL_WRITE) {
        return prepXferSgl(local, remote, handle);
    }
#endif

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::estimateXferCost (const nixl_xfer_op_t &operation,
                                               const nixl_meta_dlist_t &local,
                                               const nixl_meta_dlist_t &remote,
                                               const std::string &remote_agent,
                                               nixlBackendReqH* const &handle,
                                               std::chrono::microseconds &duration,
                                               std::chrono::microseconds &err_margin,
                                               nixl_cost_t &method,
                                               const nixl_opt_args_t* opt_args) const
{
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Local (" << local.descCount() << ") and remote (" << remote.descCount()
                   << ") descriptor lists differ in size for cost estimation";
        return NIXL_ERR_MISMATCH;
    }

    duration = std::chrono::microseconds(0);
    err_margin = std::chrono::microseconds(0);

    if (local.descCount() == 0) {
        // Nothing to do, use a default value
        method = nixl_cost_t::ANALYTICAL_BACKEND;
        return NIXL_SUCCESS;
    }

    for (int i = 0; i < local.descCount(); i++) {
        const size_t lsize = local[i].len;
        const size_t rsize = remote[i].len;

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i << " during cost estimation";
        NIXL_ASSERT(lsize == rsize) << "Local size (" << lsize << ") != Remote size (" << rsize
                                    << ") at index " << i << " during cost estimation";

        std::chrono::microseconds msg_duration;
        std::chrono::microseconds msg_err_margin;
        nixl_cost_t msg_method;
        const nixl_status_t ret = rmd->conn->getEp(worker_id)->estimateCost(
            lsize, msg_duration, msg_err_margin, msg_method);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Worker failed to estimate cost for segment " << i << " status: " << ret;
            return ret;
        }

        duration += msg_duration;
        err_margin += msg_err_margin;
        method = msg_method;
    }

    return NIXL_SUCCESS;
}

#ifdef HAVE_UCX_SGL_API
nixl_status_t
nixlUcxEngine::prepXferSgl(const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           nixlBackendReqH *handle) const {
    NIXL_ASSERT(local.descCount() == remote.descCount());

    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    int_handle->sgl.emplace(local, remote, int_handle->getWorkerId(), 0, local.descCount());
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::sendXferSgl(nixlBackendReqH *handle) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    NIXL_ASSERT(int_handle->sgl);
    auto &sgl = *int_handle->sgl;

    const ucx_connection_ptr_t &conn = sgl.conn();

    auto &ep = conn->getEp(int_handle->getWorkerId());

    int_handle->reserve(single_ep_request_count);

    nixlUcxReq req;
    const nixl_status_t post_ret = sgl.post(*ep, req);
    if (int_handle->append(post_ret, req, conn) != NIXL_SUCCESS) {
        return post_ret;
    }

    nixlUcxReq flush_req;
    const nixl_status_t flush_ret = ep->flushEp(flush_req);
    if (int_handle->append(flush_ret, flush_req, conn) != NIXL_SUCCESS) {
        return flush_ret;
    }

    return NIXL_SUCCESS;
}
#endif

nixl_status_t
nixlUcxEngine::sendXferRange(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *handle,
                             size_t start_idx,
                             size_t end_idx) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();

    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        return NIXL_ERR_INVALID_PARAM;
    }

#ifdef HAVE_UCX_SGL_API
    if (sglEnabled_ && operation == NIXL_WRITE) {
        if (!int_handle->sgl) {
            int_handle->sgl.emplace(local, remote, worker_id, start_idx, end_idx);
        }
        return sendXferSgl(handle);
    }
#endif

    int_handle->reserve(single_ep_request_count);

    const ucx_connection_ptr_t &conn =
        static_cast<nixlUcxPublicMetadata *>(remote[start_idx].metadataP)->conn;
    auto &ep = conn->getEp(worker_id);

    nixl_status_t status = NIXL_SUCCESS;
    nixlUcxReq pending_req = nullptr;

    for (size_t i = start_idx; i < end_idx; ++i) {
        void *laddr = (void *)local[i].addr;
        size_t lsize = local[i].len;
        uint64_t raddr = static_cast<uint64_t>(remote[i].addr);
        NIXL_ASSERT(lsize == remote[i].len);

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        NIXL_ASSERT(rmd->conn->getEp(worker_id).get() == ep.get());

        nixlUcxReq req;
        const nixl_status_t ret = operation == NIXL_READ ?
            ep->read(raddr, rmd->getRkey(worker_id), laddr, lmd->mem, lsize, req) :
            ep->write(laddr, lmd->mem, raddr, rmd->getRkey(worker_id), lsize, req);

        if (ret == NIXL_IN_PROG) {
            if (pending_req != nullptr) [[likely]] {
                ucp_request_free(pending_req);
            }
            pending_req = req;
        } else if (ret != NIXL_SUCCESS) {
            status = ret;
            if (pending_req != nullptr) {
                ucp_request_free(pending_req);
                pending_req = nullptr;
            }
            break;
        }
    }

    if (status == NIXL_SUCCESS && pending_req) {
        status = NIXL_IN_PROG;
    }

    if (int_handle->append(status, pending_req, conn) != NIXL_SUCCESS) {
        return status;
    }

    /*
     * Flush keeps int_handle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     */
    nixlUcxReq flush_req;
    const nixl_status_t flush_ret = ep->flushEp(flush_req);
    if (int_handle->append(flush_ret, flush_req, conn) != NIXL_SUCCESS) {
        return flush_ret;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    const size_t lcnt = local.descCount();
    const size_t rcnt = remote.descCount();
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    nixl_status_t ret;

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    // TODO: assert that handle is empty/completed, as we can't post request before completion

    ret = sendXferRange(operation, local, remote, remote_agent, handle, 0, lcnt);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    ret = int_handle->status();
    if (opt_args && opt_args->hasNotif) {
        if (ret == NIXL_SUCCESS) {
            nixlUcxReq req;
            const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[0].metadataP);
            const nixlUcxEp &ep = *rmd->conn->getEp(int_handle->getWorkerId());
            ret = notifSendPriv(remote_agent, opt_args->notifMsg, ep, &req);
            if (int_handle->append(ret, req, rmd->conn) != NIXL_SUCCESS) {
                return ret;
            }

            ret = int_handle->status();
        } else if (ret == NIXL_IN_PROG) {
            int_handle->notif.emplace(remote_agent, buildNotif(opt_args->notifMsg));
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const nixl_status_t handle_status = int_handle->status();

    if ((handle_status == NIXL_IN_PROG) || !int_handle->notif) {
        return handle_status;
    }

    nixlUcxBackendReqH::Notif notif(std::move(int_handle->notif).value());
    int_handle->notif.reset();

    if (handle_status != NIXL_SUCCESS) [[unlikely]] {
        return handle_status;
    }

    const ucx_connection_ptr_t conn = getConnection(notif.agent);
    if (!conn) [[unlikely]] {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxReq req;
    const nixlUcxEp &ep = *conn->getEp(int_handle->getWorkerId());
    const nixl_status_t status = sendNotif(std::move(notif.msg), ep, &req);

    if (int_handle->append(status, req, conn) != NIXL_SUCCESS) {
        return status;
    }

    return int_handle->status();
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    int_handle->release();

    /* TODO: return to a pool instead. */
    delete int_handle;

    return NIXL_SUCCESS;
}

unsigned
nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    unsigned ret = 0;
    for (const auto &uw : getSharedWorkers()) {
        ret += uw->progress();
    }
    return ret;
}

void
nixlUcxEngine::progressLoop() {
    while (progress() != 0)
        ;
}

/****************************************
 * Notifications
*****************************************/

std::unique_ptr<std::string>
nixlUcxEngine::buildNotif(const std::string &msg) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance
    return std::make_unique<std::string>(ser_des.exportStr());
}

nixl_status_t
nixlUcxEngine::sendNotif(std::unique_ptr<std::string> &&msg, const nixlUcxEp &ep, nixlUcxReq *req) {
    std::string *buffer = msg.release();
    auto cleanup = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            /* Caller is not interested in the request, free it */
            ucp_request_free(completed_request);
        }
    };

    return ep.sendAm(nixl::ucx::am_cb_op_t::NOTIF_STR,
                     nullptr,
                     0,
                     buffer->data(),
                     buffer->size(),
                     UCP_AM_SEND_FLAG_EAGER,
                     req,
                     std::move(cleanup));
}

nixl_status_t
nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                             const std::string &msg,
                             const nixlUcxEp &ep,
                             nixlUcxReq *req) const {
    return sendNotif(buildNotif(msg), ep, req);
}

ucx_connection_ptr_t
nixlUcxEngine::getConnection(const std::string &remote_agent) const {
    const auto it = remoteConnMap.find(remote_agent);
    return (it != remoteConnMap.end()) ? it->second : nullptr;
}

void
nixlUcxEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg, const void *header,
                         size_t header_length, void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param)
{
    nixlSerDes ser_des;

    std::string ser_str( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    // send_am should be forcing EAGER protocol
    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    ser_des.importStr(ser_str);
    std::string remote_name = ser_des.getStr("name");
    std::string msg = ser_des.getStr("msg");

    engine->appendNotif(std::move(remote_name), std::move(msg));
    return UCS_OK;
}

nixl_status_t
nixlUcxEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    progressLoop();

    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    const auto conn = getConnection(remote_agent);
    if (!conn) {
        return NIXL_ERR_NOT_FOUND;
    }

    const nixlUcxEp &ep = *conn->getEp(getSharedWorkerId());
    const nixl_status_t ret = notifSendPriv(remote_agent, msg, ep);
    if (ret == NIXL_IN_PROG) {
        return NIXL_SUCCESS;
    }
    return ret;
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getSharedWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, *getSharedWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare remote memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getSharedWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, *getSharedWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare local memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlUcxEngine::releaseMemView(nixlMemViewH mem_view) const {
    nixl::ucx::releaseMemList(mem_view);
}
