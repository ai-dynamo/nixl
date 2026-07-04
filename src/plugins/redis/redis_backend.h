/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_REDIS_REDIS_BACKEND_H
#define NIXL_SRC_PLUGINS_REDIS_REDIS_BACKEND_H

#include "backend/backend_engine.h"
#include "redis_client.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

inline constexpr const char *REDIS_PLUGIN_NAME = "REDIS";
inline constexpr const char *REDIS_PLUGIN_VERSION = "0.1.0";

/** NIXL backend engine for Redis key-value storage. */
class nixlRedisKVEngine : public nixlBackendEngine {
public:
    explicit nixlRedisKVEngine(const nixlBackendInitParams *init_params);
    nixlRedisKVEngine(const nixlBackendInitParams *init_params,
                      std::shared_ptr<iRedisClient> redis_client);
    ~nixlRedisKVEngine() override = default;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        if (!input) {
            output = nullptr;
            return NIXL_ERR_INVALID_PARAM;
        }
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

private:
    std::shared_ptr<iRedisClient> redisClient_;
    std::unordered_map<uint64_t, std::string> devIdToRedisKey_;
};

#endif // NIXL_SRC_PLUGINS_REDIS_REDIS_BACKEND_H
