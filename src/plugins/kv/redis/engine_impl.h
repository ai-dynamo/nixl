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
 * @file engine_impl.h
 * @brief nixlRedisKVEngineImpl — NIXL protocol layer over iRedisClient.
 */

#ifndef KV_PLUGIN_REDIS_ENGINE_IMPL_H
#define KV_PLUGIN_REDIS_ENGINE_IMPL_H

#include "../kv_engine_impl.h"
#include "redis_engine.h"
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @class nixlRedisKVEngineImpl
 * @brief REDIS KV engine implementation — NIXL transfer protocol over iRedisClient.
 */
class nixlRedisKVEngineImpl : public nixlKVEngineImpl {
public:
    explicit nixlRedisKVEngineImpl(const nixlBackendInitParams *init_params);

    nixlRedisKVEngineImpl(const nixlBackendInitParams *init_params,
                          std::shared_ptr<iRedisClient> redis_client);

    ~nixlRedisKVEngineImpl() override;

    nixl_mem_list_t getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    nixl_status_t checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t releaseReqH(nixlBackendReqH *handle) const override;

protected:
    virtual iRedisClient *
    getClient() const {
        return redisClient_.get();
    }

    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::shared_ptr<iRedisClient> redisClient_;
    std::unordered_map<uint64_t, std::string> devIdToRedisKey_;
};

#endif // KV_PLUGIN_REDIS_ENGINE_IMPL_H
