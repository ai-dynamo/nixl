/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Microsoft Corporation.
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

#include "azure_blob_backend.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include <asio.hpp>
#include <absl/strings/str_format.h>
#include <memory>
#include <future>
#include <optional>
#include <vector>
#include <chrono>
#include <algorithm>

namespace {

std::size_t
getNumThreads(nixl_b_params_t *custom_params) {
    return custom_params && custom_params->count("num_threads") > 0 ?
        std::stoul(custom_params->at("num_threads")) :
        std::max(1u, std::thread::hardware_concurrency() / 2);
}

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

class nixlAzureBlobBackendReqH : public nixlBackendReqH {
public:
    nixlAzureBlobBackendReqH() = default;
    ~nixlAzureBlobBackendReqH() = default;

    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixl_status_t
    getOverallStatus() {
        while (!statusFutures_.empty()) {
            if (statusFutures_.back().wait_for(std::chrono::seconds(0)) ==
                std::future_status::ready) {
                auto current_status = statusFutures_.back().get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    return current_status;
                }
                statusFutures_.pop_back();
            } else {
                return NIXL_IN_PROG;
            }
        }
        return NIXL_SUCCESS;
    }
};

class nixlAzureBlobMetadata : public nixlBackendMD {
public:
    nixlAzureBlobMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string blob_name)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          blobName(blob_name) {}

    ~nixlAzureBlobMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string blobName;
};

} // namespace

// -----------------------------------------------------------------------------
// Azure Blob Engine Implementation
// -----------------------------------------------------------------------------

nixlAzureBlobEngine::nixlAzureBlobEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      executor_(std::make_shared<asio::thread_pool>(getNumThreads(init_params->customParams))),
      blobClient_(std::make_shared<azureBlobClient>(init_params->customParams, executor_)) {
    NIXL_INFO << "Azure Blob backend initialized with Blob client wrapper";
}

// Used for testing to inject a mock Blob client dependency
nixlAzureBlobEngine::nixlAzureBlobEngine(const nixlBackendInitParams *init_params,
                                         std::shared_ptr<iBlobClient> blob_client)
    : nixlBackendEngine(init_params),
      executor_(std::make_shared<asio::thread_pool>(std::thread::hardware_concurrency())),
      blobClient_(blob_client) {
    blobClient_->setExecutor(executor_);
    NIXL_INFO << "Azure Blob backend initialized with injected Blob client";
}

nixlAzureBlobEngine::~nixlAzureBlobEngine() {
    executor_->wait();
}

nixl_status_t
nixlAzureBlobEngine::registerMem(const nixlBlobDesc &mem,
                                 const nixl_mem_t &nixl_mem,
                                 nixlBackendMD *&out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end())
        return NIXL_ERR_NOT_SUPPORTED;

    if (nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlAzureBlobMetadata> blob_md = std::make_unique<nixlAzureBlobMetadata>(
            nixl_mem, mem.devId, mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        devIdToBlobName_[mem.devId] = blob_md->blobName;
        out = blob_md.release();
    } else {
        out = nullptr;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlAzureBlobEngine::deregisterMem(nixlBackendMD *meta) {
    nixlAzureBlobMetadata *blob_md = static_cast<nixlAzureBlobMetadata *>(meta);
    if (blob_md) {
        std::unique_ptr<nixlAzureBlobMetadata> blob_md_ptr =
            std::unique_ptr<nixlAzureBlobMetadata>(blob_md);
        devIdToBlobName_.erase(blob_md->devId);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlAzureBlobEngine::queryMem(const nixl_reg_dlist_t &descs,
                              std::vector<nixl_query_resp_t> &resp) const {
    resp.reserve(descs.descCount());

    try {
        for (auto &desc : descs)
            resp.emplace_back(blobClient_->checkBlobExists(desc.metaInfo) ?
                                  nixl_query_resp_t{nixl_b_params_t{}} :
                                  std::nullopt);
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << "Failed to query memory: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlAzureBlobEngine::prepXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH *&handle,
                              const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent))
        return NIXL_ERR_INVALID_PARAM;

    auto req_h = std::make_unique<nixlAzureBlobBackendReqH>();
    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAzureBlobEngine::postXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH *&handle,
                              const nixl_opt_b_args_t *opt_args) const {
    nixlAzureBlobBackendReqH *req_h = static_cast<nixlAzureBlobBackendReqH *>(handle);

    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];

        auto blob_name_search = devIdToBlobName_.find(remote_desc.devId);
        if (blob_name_search == devIdToBlobName_.end()) {
            NIXL_ERROR << "The obj segment key " << remote_desc.devId
                       << " is not registered with the backend";
            return NIXL_ERR_INVALID_PARAM;
        }

        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());

        uintptr_t data_ptr = local_desc.addr;
        size_t data_len = local_desc.len;
        size_t offset = remote_desc.addr;

        if (operation == NIXL_WRITE)
            blobClient_->putBlobAsync(blob_name_search->second,
                                      data_ptr,
                                      data_len,
                                      offset,
                                      [status_promise](bool success) {
                                          status_promise->set_value(success ? NIXL_SUCCESS :
                                                                              NIXL_ERR_BACKEND);
                                      });
        else
            blobClient_->getBlobAsync(blob_name_search->second,
                                      data_ptr,
                                      data_len,
                                      offset,
                                      [status_promise](bool success) {
                                          status_promise->set_value(success ? NIXL_SUCCESS :
                                                                              NIXL_ERR_BACKEND);
                                      });
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlAzureBlobEngine::checkXfer(nixlBackendReqH *handle) const {
    nixlAzureBlobBackendReqH *req_h = static_cast<nixlAzureBlobBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
nixlAzureBlobEngine::releaseReqH(nixlBackendReqH *handle) const {
    nixlAzureBlobBackendReqH *req_h = static_cast<nixlAzureBlobBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}
