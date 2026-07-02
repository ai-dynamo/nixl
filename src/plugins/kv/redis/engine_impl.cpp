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

/**
 * @file engine_impl.cpp
 * @brief nixlRedisKVEngineImpl — postXfer/checkXfer over async Redis SET/GET.
 */

#include "engine_impl.h"
#include "client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

std::size_t
defaultNumThreads() {
    return std::max(1u, std::thread::hardware_concurrency() / 2);
}

std::size_t
getNumThreads(nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("num_threads") > 0) {
        try {
            const auto num_threads = std::stoul(custom_params->at("num_threads"));
            if (num_threads > 0) {
                return num_threads;
            }
            NIXL_WARN << "Invalid num_threads value 0, using default";
        }
        catch (const std::exception &) {
            NIXL_WARN << "Invalid num_threads value, using default";
        }
    }

    return defaultNumThreads();
}

nixl_b_params_t *
getCustomParams(const nixlBackendInitParams *init_params) {
    return init_params ? init_params->customParams : nullptr;
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

    if (local.descCount() == 0) {
        NIXL_ERROR << "Error: Transfer descriptor lists must not be empty";
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local and remote descriptor counts must match (%d != %d)",
            local.descCount(),
            remote.descCount());
        return false;
    }

    if (remote_agent != local_agent) {
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != DRAM_SEG && remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Remote memory type must be DRAM_SEG or OBJ_SEG, got %d", remote.getType());
        return false;
    }

    return true;
}

class nixlRedisBackendReqH : public nixlBackendReqH {
public:
    std::vector<std::shared_ptr<std::promise<nixl_status_t>>> statusPromises_;
    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixl_status_t
    getOverallStatus() {
        while (!statusFutures_.empty()) {
            if (statusFutures_.back().wait_for(std::chrono::seconds(0)) ==
                std::future_status::ready) {
                auto current_status = statusFutures_.back().get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    statusPromises_.clear();
                    return current_status;
                }
                statusFutures_.pop_back();
                statusPromises_.pop_back();
            } else {
                return NIXL_IN_PROG;
            }
        }
        return NIXL_SUCCESS;
    }
};

class nixlRedisMetadata : public nixlBackendMD {
public:
    nixlRedisMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string redis_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          redisKey(std::move(redis_key)) {}

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string redisKey;
};

} // namespace

nixlRedisKVEngineImpl::nixlRedisKVEngineImpl(const nixlBackendInitParams *init_params)
    : executor_(std::make_shared<redisThreadPoolExecutor>(
          getNumThreads(getCustomParams(init_params)))) {
    redisClient_ = std::make_shared<hiredisAsyncClient>(getCustomParams(init_params), executor_);
    NIXL_INFO << "Redis KV backend initialized";
}

nixlRedisKVEngineImpl::nixlRedisKVEngineImpl(const nixlBackendInitParams *init_params,
                                             std::shared_ptr<iRedisClient> redis_client)
    : executor_(std::make_shared<redisThreadPoolExecutor>(
          getNumThreads(getCustomParams(init_params)))),
      redisClient_(std::move(redis_client)) {
    if (redisClient_) {
        redisClient_->setExecutor(executor_);
    }
}

nixlRedisKVEngineImpl::~nixlRedisKVEngineImpl() {
    if (executor_) {
        executor_->waitUntilStopped();
    }
}

nixl_status_t
nixlRedisKVEngineImpl::registerMem(const nixlBlobDesc &mem,
                                   const nixl_mem_t &nixl_mem,
                                   nixlBackendMD *&out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        out = nullptr;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    std::string redis_key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;

    if (nixl_mem == OBJ_SEG || nixl_mem == DRAM_SEG) {
        auto redis_md = std::make_unique<nixlRedisMetadata>(nixl_mem, mem.devId, redis_key);
        devIdToRedisKey_[mem.devId] = redis_key;
        out = redis_md.release();
    } else {
        out = nullptr;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngineImpl::deregisterMem(nixlBackendMD *meta) {
    auto *redis_md = static_cast<nixlRedisMetadata *>(meta);
    if (redis_md) {
        std::unique_ptr<nixlRedisMetadata> redis_md_ptr(redis_md);
        devIdToRedisKey_.erase(redis_md->devId);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngineImpl::queryMem(const nixl_reg_dlist_t &descs,
                                std::vector<nixl_query_resp_t> &resp) const {
    iRedisClient *client = getClient();
    if (!client) {
        NIXL_ERROR << "Failed to query memory: no Redis client available";
        return NIXL_ERR_BACKEND;
    }

    resp.clear();
    resp.reserve(descs.descCount());

    bool has_error = false;

    try {
        for (int i = 0; i < descs.descCount(); ++i) {
            const auto &desc = descs[i];
            const std::string key =
                desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo;

            std::optional<bool> exists = client->checkKeyExistsSync(key);
            if (!exists.has_value()) {
                resp.emplace_back(std::nullopt);
                has_error = true;
            } else {
                resp.emplace_back(*exists ? nixl_query_resp_t{nixl_b_params_t{}} : std::nullopt);
            }
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to query memory: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    if (has_error) {
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                const std::string &local_agent,
                                nixlBackendReqH *&handle,
                                const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlRedisBackendReqH>();
    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH *&handle,
                                const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (local.descCount() == 0 || local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Invalid transfer descriptor counts for Redis postXfer (%d local, %d remote)",
            local.descCount(),
            remote.descCount());
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *req_h = static_cast<nixlRedisBackendReqH *>(handle);
    iRedisClient *client = getClient();
    if (!client) {
        return NIXL_ERR_BACKEND;
    }

    req_h->statusFutures_.clear();
    req_h->statusPromises_.clear();

    std::vector<std::string> redis_keys;
    redis_keys.reserve(local.descCount());

    for (int i = 0; i < remote.descCount(); ++i) {
        const auto &remote_desc = remote[i];

        std::string redis_key;
        if (remote_desc.metadataP) {
            auto *redis_md = dynamic_cast<nixlRedisMetadata *>(remote_desc.metadataP);
            if (redis_md) {
                redis_key = redis_md->redisKey;
            } else {
                auto it = devIdToRedisKey_.find(remote_desc.devId);
                if (it == devIdToRedisKey_.end()) {
                    return NIXL_ERR_INVALID_PARAM;
                }
                redis_key = it->second;
            }
        } else {
            auto it = devIdToRedisKey_.find(remote_desc.devId);
            if (it == devIdToRedisKey_.end()) {
                return NIXL_ERR_INVALID_PARAM;
            }
            redis_key = it->second;
        }
        redis_keys.push_back(std::move(redis_key));
    }

    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &redis_key = redis_keys[i];

        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusPromises_.push_back(status_promise);
        req_h->statusFutures_.push_back(status_promise->get_future());

        uintptr_t data_ptr = local_desc.addr;
        size_t data_len = local_desc.len;

        if (operation == NIXL_WRITE) {
            client->putKeyAsync(redis_key, data_ptr, data_len, status_promise);
        } else {
            client->getKeyAsync(redis_key, data_ptr, data_len, status_promise);
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlRedisKVEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return static_cast<nixlRedisBackendReqH *>(handle)->getOverallStatus();
}

nixl_status_t
nixlRedisKVEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    delete static_cast<nixlRedisBackendReqH *>(handle);
    return NIXL_SUCCESS;
}
