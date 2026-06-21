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
 * @file inmemkv_engine_impl.h
 * @brief Concrete nixlKVEngineImpl for the INMEMKV example plugin.
 *
 * This class owns the in-memory iKVStore and implements the full KV backend
 * data path (registerMem through releaseReqH).
 */

#ifndef NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_ENGINE_IMPL_H
#define NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_ENGINE_IMPL_H

#include "inmemkv_store.h"
#include "kv_engine_impl.h"
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @class nixlInMemKVEngineImpl
 * @brief INMEMKV-specific implementation of nixlKVEngineImpl.
 */
class nixlInMemKVEngineImpl : public nixlKVEngineImpl {
public:
    /**
     * @brief Construct with default InMemKVStore.
     * @param init_params Used for logging local agent name at initialization.
     */
    explicit nixlInMemKVEngineImpl(const nixlBackendInitParams *init_params);

    /**
     * @brief Construct with an injected store (useful for unit tests).
     * @param init_params NIXL initialization parameters.
     * @param store Ownership of the iKVStore instance is transferred.
     */
    nixlInMemKVEngineImpl(const nixlBackendInitParams *init_params,
                          std::unique_ptr<iKVStore> store);

    ~nixlInMemKVEngineImpl() override = default;

    nixl_mem_list_t getSupportedMems() const override {
        return {DRAM_SEG};
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

private:
    std::unique_ptr<iKVStore> store_;
    std::unordered_map<uint64_t, std::string> dev_id_to_key_;
    std::string local_agent_;
};

#endif // NIXL_EXAMPLES_PLUGINS_INMEMKV_INMEMKV_ENGINE_IMPL_H
