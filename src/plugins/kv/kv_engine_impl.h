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

    /** @brief Memory segment types supported by this KV backend. @return Segment list (typically DRAM_SEG). */
    virtual nixl_mem_list_t
    getSupportedMems() const = 0;

    /**
     * @brief Registers a local memory descriptor as a KV key.
     *
     * @param mem Descriptor blob including key metadata (metaInfo or devId).
     * @param nixl_mem Memory segment type; must be supported by getSupportedMems().
     * @param out On success, receives a newly allocated nixlBackendMD* owned by the caller
     *            until deregisterMem().
     * @return NIXL_SUCCESS on success, or an error status on invalid input or backend failure.
     */
    virtual nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) = 0;

    /**
     * @brief Deregisters backend metadata created by registerMem().
     *
     * @param meta Metadata pointer returned by registerMem(); must be non-null.
     * @return NIXL_SUCCESS on success, or an error status if meta is invalid.
     */
    virtual nixl_status_t
    deregisterMem(nixlBackendMD *meta) = 0;

    /**
     * @brief Queries whether registered keys exist in the backing store.
     *
     * @param descs Registered descriptors to query.
     * @param resp Output vector; each entry is set when the key exists, or std::nullopt otherwise.
     * @return NIXL_SUCCESS on success, or an error status on backend failure.
     */
    virtual nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const = 0;

    /**
     * @brief Prepares a transfer request and allocates a backend request handle.
     *
     * @param operation Transfer operation (NIXL_WRITE or NIXL_READ).
     * @param local Local descriptor metadata list.
     * @param remote Remote descriptor metadata list.
     * @param remote_agent Remote agent name (unused for local-only KV backends).
     * @param local_agent Local agent name from nixlKVEngine::localAgent (passed explicitly
     *                    so impl does not depend on nixlBackendEngine protected members).
     * @param handle On success, receives a newly allocated nixlBackendReqH* owned by the caller
     *               until releaseReqH().
     * @param opt_args Optional backend arguments; may be nullptr.
     * @return NIXL_SUCCESS on success, or an error status on invalid parameters.
     */
    virtual nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const = 0;

    /**
     * @brief Executes a prepared transfer request.
     *
     * @param operation Transfer operation (NIXL_WRITE or NIXL_READ).
     * @param local Local descriptor metadata list.
     * @param remote Remote descriptor metadata list.
     * @param remote_agent Remote agent name (unused for local-only KV backends).
     * @param local_agent Local agent name from nixlKVEngine::localAgent (passed explicitly
     *                    so impl does not depend on nixlBackendEngine protected members).
     * @param handle Request handle allocated by prepXfer(); must be non-null.
     * @param opt_args Optional backend arguments; may be nullptr.
     * @return NIXL_SUCCESS on success, or an error status on transfer or backend failure.
     */
    virtual nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const = 0;

    /**
     * @brief Checks transfer completion status for a request handle.
     *
     * @param handle Request handle from prepXfer(); must be non-null.
     * @return NIXL_SUCCESS when the transfer is complete, or an in-progress/error status.
     */
    virtual nixl_status_t
    checkXfer(nixlBackendReqH *handle) const = 0;

    /**
     * @brief Releases a request handle allocated by prepXfer().
     *
     * @param handle Request handle to release; must be non-null.
     * @return NIXL_SUCCESS on success.
     */
    virtual nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const = 0;
};

#endif // NIXL_SRC_PLUGINS_KV_KV_ENGINE_IMPL_H
