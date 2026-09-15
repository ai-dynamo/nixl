/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "engine_utils.h"
#include "s3/client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <vector>

namespace {

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

    if (remote_agent != local_agent)
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    return true;
}

class nixlObjBackendReqH : public nixlBackendReqH {
public:
    nixlObjBackendReqH() = default;
    ~nixlObjBackendReqH() = default;

    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixl_status_t
    getOverallStatus() {
        // Iterate front-to-back to detect failures in earlier futures even if
        // later futures are not yet ready. This ensures we return errors as
        // soon as they occur rather than waiting for all futures to complete.
        auto it = statusFutures_.begin();
        while (it != statusFutures_.end()) {
            if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                auto current_status = it->get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    return current_status;
                }
                it = statusFutures_.erase(it);
            } else {
                return NIXL_IN_PROG;
            }
        }
        return NIXL_SUCCESS;
    }
};

class nixlObjMetadata : public nixlBackendMD {
public:
    nixlObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key) {}

    ~nixlObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
};

} // namespace

DefaultObjEngineImpl::DefaultObjEngineImpl(const nixlBackendInitParams *init_params)
    : executor_(std::make_shared<asioThreadPoolExecutor>(getNumThreads(init_params->customParams))),
      crtMinLimit_(getCrtMinLimit(init_params->customParams)) {
    s3Client_ = std::make_shared<awsS3Client>(init_params->customParams, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Standard client only";

    // Ensure at least one client was created
    if (!s3Client_) {
        throw std::runtime_error("Failed to create any S3 client");
    }
}

DefaultObjEngineImpl::DefaultObjEngineImpl(const nixlBackendInitParams *init_params,
                                           std::shared_ptr<iS3Client> s3_client,
                                           std::shared_ptr<iS3Client> s3_client_crt)
    : executor_(std::make_shared<asioThreadPoolExecutor>(std::thread::hardware_concurrency())),
      s3Client_(s3_client),
      crtMinLimit_(getCrtMinLimit(init_params->customParams)) {
    // DefaultObjEngineImpl only uses the standard S3 client, not the CRT client.
    // The s3_client_crt parameter is accepted for API consistency with derived
    // engine implementations (e.g., S3CrtObjEngineImpl) but is intentionally unused here.
    (void)s3_client_crt;
    if (s3Client_) s3Client_->setExecutor(executor_);
    NIXL_INFO << "Object storage backend initialized with injected S3 clients";
}

DefaultObjEngineImpl::~DefaultObjEngineImpl() {
    executor_->WaitUntilStopped();
}

nixl_status_t
DefaultObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD *&out) {
    nixl_mem_list_t supported_mems = {OBJ_SEG, DRAM_SEG};
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end())
        return NIXL_ERR_NOT_SUPPORTED;

    if (nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlObjMetadata> obj_md = std::make_unique<nixlObjMetadata>(
            nixl_mem, mem.devId, mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
    } else {
        out = nullptr;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
DefaultObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    nixlObjMetadata *obj_md = static_cast<nixlObjMetadata *>(meta);
    if (obj_md) {
        std::unique_ptr<nixlObjMetadata> obj_md_ptr = std::unique_ptr<nixlObjMetadata>(obj_md);
        devIdToObjKey_.erase(obj_md->devId);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
DefaultObjEngineImpl::queryMem(const nixl_reg_dlist_t &descs,
                               std::vector<nixl_query_resp_t> &resp) const {
    resp.assign(descs.descCount(), std::nullopt);

    iS3Client *client = getClient();
    if (!client) {
        NIXL_ERROR << "Failed to query memory: no client available";
        return NIXL_ERR_BACKEND;
    }

    struct PerDescriptorState {
        std::shared_ptr<std::promise<void>> promise;
        std::shared_ptr<std::atomic<bool>> completed;
    };

    std::vector<std::future<void>> futures;
    futures.reserve(descs.descCount());
    std::vector<PerDescriptorState> states;
    states.reserve(descs.descCount());
    std::atomic<bool> has_error{false};

    constexpr auto kQueryTimeout = std::chrono::seconds(30);

    try {
        // Launch async requests for each descriptor
        for (int i = 0; i < descs.descCount(); ++i) {
            auto &desc = descs[i];
            auto state = PerDescriptorState{std::make_shared<std::promise<void>>(),
                                            std::make_shared<std::atomic<bool>>(false)};

            client->checkObjectExistsAsync(
                desc.metaInfo,
                [&resp, &has_error, i, promise = state.promise, completed = state.completed](
                    std::optional<bool> exists) {
                    if (completed->exchange(true)) return;
                    if (!exists.has_value()) {
                        resp[i] = std::nullopt;
                        has_error.store(true, std::memory_order_relaxed);
                    } else {
                        resp[i] = *exists ? nixl_query_resp_t{nixl_b_params_t{}} : std::nullopt;
                    }
                    try {
                        promise->set_value();
                    }
                    catch (...) {
                    }
                });

            futures.push_back(state.promise->get_future());
            states.push_back(std::move(state));
        }
    }
    catch (const std::exception &e) {
        // Drain all already-launched async callbacks with timeout before returning
        for (size_t j = 0; j < futures.size(); ++j) {
            if (futures[j].wait_for(kQueryTimeout) == std::future_status::timeout) {
                if (!states[j].completed->exchange(true)) {
                    try {
                        states[j].promise->set_value();
                    }
                    catch (...) {
                    }
                }
            }
        }
        // Wait for all callbacks to complete their writes to resp/has_error
        // to avoid use-after-free if a callback passed the exchange check
        // but was preempted before writing.
        for (size_t j = 0; j < futures.size(); ++j)
            futures[j].wait();
        NIXL_ERROR << "Failed to query memory: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    // Wait for all async operations with timeout fallback
    for (size_t i = 0; i < futures.size(); ++i) {
        if (futures[i].wait_for(kQueryTimeout) == std::future_status::timeout) {
            // Watchdog: if callback never fired, mark as completed and resolve
            if (!states[i].completed->exchange(true)) {
                resp[i] = std::nullopt;
                has_error.store(true, std::memory_order_relaxed);
                try {
                    states[i].promise->set_value();
                }
                catch (...) {
                }
            }
            NIXL_ERROR << "checkObjectExistsAsync timed out for descriptor " << i;
        }
    }

    if (has_error.load(std::memory_order_relaxed)) {
        NIXL_ERROR << "Failed to query memory: one or more HeadObject requests failed";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
DefaultObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               const std::string &local_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent))
        return NIXL_ERR_INVALID_PARAM;

    auto req_h = std::make_unique<nixlObjBackendReqH>();
    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
DefaultObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *>(handle);

    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];

        auto obj_key_search = devIdToObjKey_.find(remote_desc.devId);
        if (obj_key_search == devIdToObjKey_.end()) {
            NIXL_ERROR << "The object segment key " << remote_desc.devId
                       << " is not registered with the backend";
            return NIXL_ERR_INVALID_PARAM;
        }

        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());

        uintptr_t data_ptr = local_desc.addr;
        size_t data_len = local_desc.len;
        size_t offset = remote_desc.addr;

        iS3Client *client = getClientForSize(data_len);
        if (!client) {
            NIXL_ERROR << "Failed to post transfer: no client available";
            return NIXL_ERR_BACKEND;
        }

        // S3 client interface signals completion via a callback, but NIXL API polls request handle
        // for the status code. Use future/promise pair to bridge the gap.
        auto status_callback = [status_promise](bool success) {
            status_promise->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
        };

        if (operation == NIXL_WRITE)
            client->putObjectAsync(
                obj_key_search->second, data_ptr, data_len, offset, status_callback);
        else
            client->getObjectAsync(
                obj_key_search->second, data_ptr, data_len, offset, status_callback);
    }

    return NIXL_IN_PROG;
}

nixl_status_t
DefaultObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
DefaultObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}

iS3Client *
DefaultObjEngineImpl::getClient() const {
    return s3Client_.get();
}

iS3Client *
DefaultObjEngineImpl::getClientForSize(size_t data_len) const {
    (void)data_len;
    return getClient();
}
