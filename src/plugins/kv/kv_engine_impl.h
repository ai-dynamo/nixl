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
 * @file kv_engine_impl.h
 * @brief Abstract implementation interface for KV-style NIXL backend engines.
 *
 * Architecture:
 *
 *   nixlBackendEngine          NIXL agent-facing protocol (registerMem, prepXfer, ...)
 *        ^
 *        |  nixlKVEngine        Thin wrapper; delegates lifecycle calls to impl_
 *        |
 *   nixlKVEngineImpl           Vendor/backend-specific logic (this header)
 *        ^
 *        |
 *   nixlInMemKVEngineImpl      Example: in-process map via iKVStore
 *   nixlRedisKVEngineImpl      Future: Redis via hiredis implementing iKVStore
 *
 * Storage is further factored through iKVStore (kv_store.h) so impl classes focus
 * on NIXL descriptor/key mapping while iKVStore handles put/get/exists.
 */

#ifndef NIXL_SRC_PLUGINS_KV_KV_ENGINE_IMPL_H
#define NIXL_SRC_PLUGINS_KV_KV_ENGINE_IMPL_H

#include "backend/backend_engine.h"
#include <string>
#include <vector>

/**
 * @class nixlKVEngineImpl
 * @brief Abstract implementation interface for KV-style backend engines.
 *
 * Each KV plugin (INMEMKV example, REDIS src plugin, etc.) provides a concrete
 * subclass that implements register/deregister, query, and synchronous transfer
 * operations against its chosen iKVStore backend.
 */
class nixlKVEngineImpl {
public:
    virtual ~nixlKVEngineImpl() = default;

    /** @return Memory segment types supported by this KV backend (typically DRAM_SEG). */
    virtual nixl_mem_list_t
    getSupportedMems() const = 0;

    virtual nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) = 0;

    virtual nixl_status_t
    deregisterMem(nixlBackendMD *meta) = 0;

    virtual nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const = 0;

    /**
     * @brief Prepare a transfer request.
     *
     * @param local_agent Agent name from nixlKVEngine::localAgent (passed explicitly
     *                    so impl does not depend on nixlBackendEngine protected members).
     */
    virtual nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const = 0;

    virtual nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const = 0;

    virtual nixl_status_t
    checkXfer(nixlBackendReqH *handle) const = 0;

    virtual nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const = 0;
};

#endif // NIXL_SRC_PLUGINS_KV_KV_ENGINE_IMPL_H
