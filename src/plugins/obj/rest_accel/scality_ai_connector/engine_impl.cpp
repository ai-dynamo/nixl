/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "engine_impl.h"
#include "client.h"
#include "common/config_traits.h"
#include "common/nixl_log.h"
#include "common/str_util.h"
#include <absl/strings/str_format.h>
#include "cufile_nics.h"
#include "dc_descriptor_provider.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <future>
#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <optional>

#include "obj_engine_registry.h"

namespace {

objAccelEngineRegistrar reg_scality(
    "scality_ai_connector",
    [](const nixlBackendInitParams *p) { return std::make_unique<ScalityObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client>, std::shared_ptr<iS3Client>) {
        return std::make_unique<ScalityObjEngineImpl>(p);
    });

std::string
objKeyFor(const std::string &meta_info, uint64_t dev_id) {
    return meta_info.empty() ? std::to_string(dev_id) : meta_info;
}

/// Look up a key in customParams, returning a default when absent/empty.
std::string
paramOr(const nixl_b_params_t *params, const std::string &key, const std::string &def) {
    if (params == nullptr) {
        return def;
    }
    auto it = params->find(key);
    return (it == params->end() || it->second.empty()) ? def : it->second;
}

/// Integer customParams value in [0, max], or def when absent. A malformed or
/// out-of-range value is reported and replaced by def.
unsigned long
paramInRange(const nixl_b_params_t *params,
             const std::string &key,
             unsigned long max,
             unsigned long def) {
    const std::string value = paramOr(params, key, "");
    if (value.empty()) {
        return def;
    }
    try {
        const unsigned long parsed = nixl::config::configTraits<unsigned long>::convert(value);
        if (parsed <= max) {
            return parsed;
        }
    }
    catch (const std::runtime_error &) {
    }
    NIXL_WARN << "Ignoring " << key << "=" << value << " (expected 0-" << max << "), using " << def;
    return def;
}

/// A DC access key in hex, with or without 0x ("0xffeeddcc" or "ffeeddcc").
std::optional<uint64_t>
parseDcKeyValue(const std::string &text) {
    // NIXL's converter reads hex only with the 0x prefix.
    const bool prefixed = text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X');
    try {
        return nixl::config::configTraits<uint64_t>::convert(prefixed ? text : "0x" + text);
    }
    catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

/// Path of cufile.json: CUFILE_ENV_PATH_JSON, else /etc/cufile.json.
std::string
cufileJsonPath() {
    const char *env = std::getenv("CUFILE_ENV_PATH_JSON");
    return env ? env : "/etc/cufile.json";
}

/// cufile.json content, or "" if unreadable. Used to resolve both the NIC list
/// and the DC key.
std::string
readCufileJson() {
    std::ifstream f(cufileJsonPath());
    if (!f) {
        return std::string();
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/**
 * Validate parameters for prepXfer operation.
 */
bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (remote_agent != local_agent) {
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);
    }

    if ((local.getType() != DRAM_SEG) && (local.getType() != VRAM_SEG)) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local memory type must be VRAM_SEG or DRAM_SEG, got %d", local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local and remote descriptor counts must match. Got %d local, %d remote",
            local.descCount(),
            remote.descCount());
        return false;
    }

    return true;
}

/**
 * One HTTP request of a transfer: a local byte range, the object range it maps
 * to, and the RDMA descriptor describing the local range.
 */
class scalityObjTransferRequestH {
public:
    uintptr_t addr = 0;
    size_t size = 0;
    size_t offset = 0;
    std::string rdmaDesc;
    std::string objKey;

    scalityObjTransferRequestH(uintptr_t a, size_t s, size_t off) : addr(a), size(s), offset(off) {}
};

/**
 * Backend request handle for Scality AI Connector RDMA operations.
 * Manages multiple transfer requests and their completion futures.
 */
class nixlScalityObjBackendReqH : public nixlBackendReqH {
public:
    std::vector<scalityObjTransferRequestH> reqs_;
    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixlScalityObjBackendReqH() = default;

    /// Waits for every request of this transfer to finish.
    ///
    /// A request writes straight into the caller's buffer, and the caller frees
    /// that buffer once it has released the handle. Destroying the handle while
    /// requests are running would leave them writing into freed memory, which
    /// surfaces as heap corruption rather than as an error.
    ///
    /// The wait always ends: every request completes, times out
    /// (request_timeout_ms) or is failed when the client shuts down. On the
    /// success path every future is already resolved, so this costs nothing.
    ~nixlScalityObjBackendReqH() {
        for (auto &future : statusFutures_) {
            if (future.valid()) {
                future.wait();
            }
        }
    }

    nixl_status_t
    getOverallStatus() {
        bool has_pending = false;
        auto it = statusFutures_.begin();
        while (it != statusFutures_.end()) {
            if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                const auto current_status = it->get();
                it = statusFutures_.erase(it);
                // Record the failure and keep the rest: the futures still pending
                // are the only handle on requests that are still writing, and the
                // destructor needs them. Dropping a ready future is safe.
                if (current_status != NIXL_SUCCESS && firstError_ == NIXL_SUCCESS) {
                    firstError_ = current_status;
                }
            } else {
                ++it;
                has_pending = true;
            }
        }
        // No final status while any request is still running, failure or not: the
        // caller may reuse the buffer once it sees one, and a running request still
        // writes into it. Then the first error, if any.
        return has_pending ? NIXL_IN_PROG : firstError_;
    }

    /// A prepared handle can be posted again; each post reports its own outcome.
    void
    resetStatus() {
        firstError_ = NIXL_SUCCESS;
    }

private:
    nixl_status_t firstError_ = NIXL_SUCCESS;
};

/**
 * Metadata for Scality AI Connector RDMA operations.
 */
class nixlScalityObjMetadata : public nixlBackendMD {
public:
    nixlScalityObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key),
          localAddr(0) {}

    nixlScalityObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr, uint64_t dev_id = 0)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(""),
          localAddr(addr) {}

    ~nixlScalityObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
    uintptr_t localAddr;
};

} // namespace

ScalityObjEngineImpl::ScalityObjEngineImpl(const nixlBackendInitParams *init_params)
    : ScalityObjEngineImpl(init_params, nullptr) {}

ScalityObjEngineImpl::ScalityObjEngineImpl(const nixlBackendInitParams *init_params,
                                           std::shared_ptr<iRestClient> connector_client) {
    if (connector_client) {
        connectorClient_ = connector_client;
    } else {
        // init_params is only optional when a client is injected (the test seam);
        // the production constructor always supplies a non-null init_params.
        assert(init_params != nullptr &&
               "init_params must be non-null when no restClient is injected");
        connectorClient_ = std::make_shared<restClient>(init_params->customParams);
    }

    NIXL_INFO << "Object storage backend initialized with Scality AI Connector RDMA client";

    // Resolve the NIC list and DC key now: explicit customParams first, else
    // cufile.json (rdma_dev_addr_list / rdma_dc_key), else the DC-key default.
    // The descriptor provider itself is built on the first DRAM/VRAM registration.
    const nixl_b_params_t *params = init_params ? init_params->customParams : nullptr;
    const std::string cufile = readCufileJson();

    const std::string nics_param = paramOr(params, "rdma_nics", "");
    if (!nics_param.empty()) {
        rdmaNics_ = nixl::str::splitStripped(nics_param);
    } else {
        rdmaNics_ = parseRdmaDevAddrList(cufile);
        if (!rdmaNics_.empty()) {
            NIXL_INFO << "Resolved " << rdmaNics_.size()
                      << " RDMA NIC(s) from cufile.json rdma_dev_addr_list";
        }
    }

    std::string dckey = paramOr(params, "rdma_dc_key", "");
    if (dckey.empty()) {
        dckey = parseRdmaDcKey(cufile); // "" if absent/commented
    }
    if (!dckey.empty()) {
        if (const std::optional<uint64_t> key = parseDcKeyValue(dckey)) {
            dcKey_ = *key;
        } else {
            NIXL_WARN << "Ignoring rdma_dc_key=" << dckey
                      << " (expected hex, e.g. 0xffeeddcc), using the default";
        }
    }

    rdmaSl_ = static_cast<uint8_t>(paramInRange(params, "rdma_sl", 15, default_rdma_sl));
    rdmaTrafficClass_ = static_cast<uint8_t>(
        paramInRange(params, "rdma_traffic_class", 255, default_rdma_traffic_class));
}

nixl_status_t
ScalityObjEngineImpl::ensureDescriptorProvider() {
    std::lock_guard<std::mutex> lk(descriptorProviderMu_);
    if (descriptorProvider_) {
        return NIXL_SUCCESS;
    }
    if (rdmaNics_.empty()) {
        NIXL_ERROR << "RDMA requires a NIC list: set the 'rdma_nics' parameter (IPv4 addresses "
                      "or device names like mlx5_1), or rdma_dev_addr_list in "
                   << cufileJsonPath();
        return NIXL_ERR_BACKEND;
    }
    auto client =
        std::make_shared<dcDescriptorProvider>(rdmaNics_, dcKey_, rdmaSl_, rdmaTrafficClass_);
    if (!client->isConnected()) {
        NIXL_ERROR << "Failed to initialize the DC descriptor provider";
        return NIXL_ERR_BACKEND;
    }
    descriptorProvider_ = std::move(client);
    NIXL_INFO << "RDMA transfers using DC across " << rdmaNics_.size() << " NIC(s)";
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD *&out) {
    auto supported_mems = {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlScalityObjMetadata> obj_md = std::make_unique<nixlScalityObjMetadata>(
            nixl_mem, mem.devId, objKeyFor(mem.metaInfo, mem.devId));
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
    } else if ((nixl_mem == DRAM_SEG) || (nixl_mem == VRAM_SEG)) {
        nixl_status_t st = ensureDescriptorProvider();
        if (st != NIXL_SUCCESS) {
            return st;
        }

        NIXL_DEBUG << absl::StrFormat("registerMem: addr=0x%016x, len=%zu, nixl_mem=%d, devId=%d",
                                      mem.addr,
                                      mem.len,
                                      nixl_mem,
                                      mem.devId);
        std::unique_ptr<nixlScalityObjMetadata> mem_md =
            std::make_unique<nixlScalityObjMetadata>(nixl_mem, mem.addr, mem.devId);

        // VRAM passes the GPU ordinal as the NIC-affinity hint; DRAM passes -1.
        const int affinity_dev = (nixl_mem == VRAM_SEG) ? (int)mem.devId : -1;
        st = descriptorProvider_->registerMemory((void *)(mem.addr), mem.len, affinity_dev);
        if (st != NIXL_SUCCESS) {
            NIXL_ERROR << "RDMA registration of " << mem.len << " bytes at 0x" << std::hex
                       << mem.addr << std::dec << " failed";
            return st;
        }
        out = mem_md.release();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    std::unique_ptr<nixlScalityObjMetadata> md(static_cast<nixlScalityObjMetadata *>(meta));
    if (!md) {
        return NIXL_SUCCESS;
    }
    if (md->nixlMem == OBJ_SEG) {
        devIdToObjKey_.erase(md->devId);
    } else if ((md->nixlMem == DRAM_SEG) || (md->nixlMem == VRAM_SEG)) {
        const nixl_status_t st = descriptorProvider_->deregisterMemory((void *)(md->localAddr));
        if (st != NIXL_SUCCESS) {
            NIXL_ERROR << "RDMA deregistration at 0x" << std::hex << md->localAddr << std::dec
                       << " failed";
            return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               const std::string &local_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!descriptorProvider_ || !descriptorProvider_->isConnected()) {
        NIXL_ERROR << "RDMA descriptor provider is not connected.";
        return NIXL_ERR_BACKEND;
    }

    auto req_h = std::make_unique<nixlScalityObjBackendReqH>();

    for (int i = 0; i < local.descCount(); ++i) {
        auto obj_key_search = devIdToObjKey_.find(remote[i].devId);
        if (obj_key_search == devIdToObjKey_.end()) {
            NIXL_ERROR << "The object segment key " << remote[i].devId
                       << " is not registered with the backend";
            return NIXL_ERR_INVALID_PARAM;
        }

        // A PUT writes the whole object, so a WRITE must start at object offset 0.
        if ((operation == NIXL_WRITE) && (remote[i].addr != 0)) {
            NIXL_ERROR << "WRITE at object offset " << remote[i].addr
                       << " is not supported: a PUT writes the whole object";
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // An empty READ has nothing to move. An empty WRITE would have to create an
        // empty object, and a PUT here always carries its data by RDMA.
        if (local[i].len == 0) {
            if (operation == NIXL_WRITE) {
                NIXL_ERROR << "WRITE of a zero-length descriptor is not supported";
                return NIXL_ERR_NOT_SUPPORTED;
            }
            continue;
        }

        scalityObjTransferRequestH req(local[i].addr, local[i].len, remote[i].addr);
        req.objKey = obj_key_search->second;
        req.rdmaDesc = descriptorProvider_->makeDescriptor((void *)req.addr, req.size);
        if (req.rdmaDesc.empty()) {
            return NIXL_ERR_BACKEND;
        }
        // Log only the RDMA descriptor length, never the raw descriptor: it grants buffer access.
        NIXL_DEBUG << absl::StrFormat(
            "prepXfer: addr=0x%016x, size=%zu, offset=%zu, rdma_desc_len=%zu",
            req.addr,
            req.size,
            req.offset,
            req.rdmaDesc.size());

        req_h->reqs_.push_back(std::move(req));
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);
    req_h->resetStatus();

    for (const auto &req : req_h->reqs_) {
        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());
        auto on_done = [status_promise](bool success) {
            status_promise->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
        };

        if (operation == NIXL_WRITE) {
            connectorClient_->putObjectRdmaAsync(
                req.objKey, req.addr, req.size, req.offset, req.rdmaDesc, on_done);
        } else {
            connectorClient_->getObjectRdmaAsync(
                req.objKey, req.addr, req.size, req.offset, req.rdmaDesc, on_done);
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
ScalityObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
ScalityObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::queryMem(const nixl_reg_dlist_t &descs,
                               std::vector<nixl_query_resp_t> &resp) const {
    resp.assign(descs.descCount(), std::nullopt);

    if (!connectorClient_) {
        NIXL_ERROR << "queryMem: REST client not available";
        return NIXL_ERR_BACKEND;
    }

    // Shared so a callback that fires after queryMem returns (a timed-out request)
    // never touches this stack frame. A descriptor is settled once, under mu,
    // either by its callback or by the deadline.
    struct queryState {
        std::mutex mu;
        std::vector<nixl_query_resp_t> resp;
        std::vector<bool> settled;
        bool hasError = false;
    };

    auto state = std::make_shared<queryState>();
    state->resp.assign(descs.descCount(), std::nullopt);
    state->settled.assign(descs.descCount(), false);

    std::vector<std::future<void>> futures;
    futures.reserve(descs.descCount());

    // One aggregate deadline for all checks: concurrent HEADs share a single
    // bound, so a hung endpoint can't make this block for 30s * descCount().
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);

    for (int i = 0; i < descs.descCount(); ++i) {
        const auto &desc = descs[i];
        // Mirror the key convention used by registerMem: metaInfo, or the
        // device id when no key was supplied.
        std::string key = objKeyFor(desc.metaInfo, desc.devId);

        auto promise = std::make_shared<std::promise<void>>();
        futures.push_back(promise->get_future());

        connectorClient_->checkObjectExistsAsync(
            key, [state, i, promise](std::optional<bool> exists) {
                {
                    std::lock_guard<std::mutex> lk(state->mu);
                    if (state->settled[i]) {
                        return;
                    }
                    state->settled[i] = true;
                    if (!exists.has_value()) {
                        state->hasError = true;
                    } else if (*exists) {
                        state->resp[i] = nixl_query_resp_t{nixl_b_params_t{}};
                    }
                }
                promise->set_value();
            });
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        if (futures[i].wait_until(deadline) == std::future_status::timeout) {
            std::lock_guard<std::mutex> lk(state->mu);
            if (!state->settled[i]) {
                state->settled[i] = true;
                state->hasError = true;
            }
        }
    }

    std::lock_guard<std::mutex> lk(state->mu);
    resp = state->resp;
    return state->hasError ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}
