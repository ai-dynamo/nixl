/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "redis_backend.h"

#include "common/nixl_log.h"

#include <absl/strings/str_format.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

nixl_b_params_t *
getInitCustomParams(const nixlBackendInitParams *init_params) {
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
    // Redis callbacks write into (GET) or read from (SET) the caller's buffers through raw
    // pointers, so the handle must not go away while any of them can still run.
    ~nixlRedisBackendReqH() override {
        waitForOutstanding();
    }

    void
    addOperation(std::future<nixl_status_t> future) {
        statusFutures_.push_back(std::move(future));
    }

    void
    reset() {
        waitForOutstanding();
        statusFutures_.clear();
        overallStatus_ = NIXL_SUCCESS;
    }

    nixl_status_t
    getOverallStatus() {
        for (const auto &future : statusFutures_) {
            if (future.valid() &&
                future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                return NIXL_IN_PROG;
            }
        }

        for (auto &future : statusFutures_) {
            if (!future.valid()) {
                continue;
            }
            nixl_status_t status = NIXL_ERR_BACKEND;
            try {
                status = future.get();
            }
            catch (const std::future_error &e) {
                NIXL_ERROR << "Redis operation completed without a status: " << e.what();
            }
            if (status != NIXL_SUCCESS && overallStatus_ == NIXL_SUCCESS) {
                overallStatus_ = status;
            }
        }
        statusFutures_.clear();
        return overallStatus_;
    }

private:
    void
    waitForOutstanding() const {
        for (const auto &future : statusFutures_) {
            if (future.valid()) {
                future.wait();
            }
        }
    }

    // Promises are owned only by the client callbacks; if one is dropped without a value the
    // future becomes ready with broken_promise instead of blocking the destructor forever.
    std::vector<std::future<nixl_status_t>> statusFutures_;
    nixl_status_t overallStatus_ = NIXL_SUCCESS;
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

nixlRedisKVEngine::nixlRedisKVEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    auto config = RedisConfig::fromBackendParams(getInitCustomParams(init_params));
    redisClient_ = std::make_shared<RedisConnectionPool>(std::move(config));
    NIXL_INFO << "Redis backend initialized";
}

nixlRedisKVEngine::nixlRedisKVEngine(const nixlBackendInitParams *init_params,
                                     std::shared_ptr<iRedisClient> redis_client)
    : nixlBackendEngine(init_params),
      redisClient_(std::move(redis_client)) {}

nixl_status_t
nixlRedisKVEngine::registerMem(const nixlBlobDesc &mem,
                               const nixl_mem_t &nixl_mem,
                               nixlBackendMD *&out) {
    const auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        out = nullptr;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    std::string redis_key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
    auto redis_md = std::make_unique<nixlRedisMetadata>(nixl_mem, mem.devId, redis_key);
    devIdToRedisKey_[mem.devId] = redis_key;
    out = redis_md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngine::deregisterMem(nixlBackendMD *meta) {
    auto *redis_md = static_cast<nixlRedisMetadata *>(meta);
    if (redis_md) {
        std::unique_ptr<nixlRedisMetadata> redis_md_ptr(redis_md);
        devIdToRedisKey_.erase(redis_md->devId);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngine::queryMem(const nixl_reg_dlist_t &descs,
                            std::vector<nixl_query_resp_t> &resp) const {
    if (!redisClient_) {
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
            const auto exists = redisClient_->checkKeyExistsSync(key);
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

    return has_error ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngine::prepXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    handle = new nixlRedisBackendReqH();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRedisKVEngine::postXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *) const {
    if (!handle || (operation != NIXL_WRITE && operation != NIXL_READ)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (local.descCount() == 0 || local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Invalid transfer descriptor counts for Redis postXfer (%d local, %d remote)",
            local.descCount(),
            remote.descCount());
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!redisClient_) {
        return NIXL_ERR_BACKEND;
    }

    auto *req_h = static_cast<nixlRedisBackendReqH *>(handle);
    req_h->reset();

    // Resolve every key before dispatching any command so invalid descriptors cannot
    // produce a partially submitted Redis transfer.
    std::vector<std::string> redis_keys;
    redis_keys.reserve(remote.descCount());
    for (int i = 0; i < remote.descCount(); ++i) {
        const auto &remote_desc = remote[i];
        std::string redis_key;

        if (remote_desc.metadataP) {
            redis_key = static_cast<nixlRedisMetadata *>(remote_desc.metadataP)->redisKey;
        }

        if (redis_key.empty()) {
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
        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->addOperation(status_promise->get_future());

        if (operation == NIXL_WRITE) {
            redisClient_->putKeyAsync(
                redis_keys[i], local_desc.addr, local_desc.len, status_promise);
        } else {
            redisClient_->getKeyAsync(
                redis_keys[i], local_desc.addr, local_desc.len, status_promise);
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlRedisKVEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return static_cast<nixlRedisBackendReqH *>(handle)->getOverallStatus();
}

nixl_status_t
nixlRedisKVEngine::releaseReqH(nixlBackendReqH *handle) const {
    delete static_cast<nixlRedisBackendReqH *>(handle);
    return NIXL_SUCCESS;
}
