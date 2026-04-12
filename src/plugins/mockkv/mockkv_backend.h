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
 * @file mockkv_backend.h
 * @brief MOCKKV Backend Header - Simple in-memory key-value store for learning NIXL
 *
 * Architecture:
 * - Apps (e.g. nixlbench) create nixlAgent with backend "MOCKKV" and call
 *   registerMem / prepXfer / postXfer / checkXfer / releaseReqH / deregisterMem.
 * - nixlAgent routes by memory type (DRAM_SEG, ...) to this backend.
 * - This engine implements nixlBackendEngine: register/deregister, prep/exec/check transfer,
 *   release handles; if supportsRemote, also conn info, metadata serialization, connect/unloadMD.
 *
 * Lifecycle: createBackend -> registerMem -> prepXfer -> postXfer
 * -> checkXfer -> releaseReqH -> deregisterMem; unloadMD is no-op (same as Obj — no free there).
 *
 * See MOCKKV_ARCHITECTURE.md in this directory for diagrams and log meanings.
 *
 * Key Features:
 * - Synchronous operations (simple to understand)
 * - In-memory storage (no external dependencies)
 * - Full NIXL backend interface implementation
 */

#ifndef MOCKKV_BACKEND_H
#define MOCKKV_BACKEND_H

#include "backend/backend_engine.h"
#include "nixl_types.h"
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class nixlMockKVEngine
 * @brief Simple in-memory key-value store backend for NIXL
 * 
 * This backend implements a synchronous, in-memory key-value store using std::map.
 * It's designed to be simple and easy to understand, making it perfect for learning
 * how NIXL plugins work.
 * 
 * Architecture:
 * - Storage: std::unordered_map<std::string, std::vector<uint8_t>>
 * - Operations: Synchronous (no async complexity)
 * - Thread Safety: Like nixlObjEngine, no internal locking; assume serialized backend calls
 * 
 * Supported Operations:
 * - registerMem: Register memory descriptors with keys
 * - prepXfer: Prepare transfer operations
 * - postXfer: Execute PUT/GET operations (synchronous)
 * - checkXfer: Check operation status (always returns SUCCESS immediately)
 * - queryMem: Check if a key exists
 */
class nixlMockKVEngine : public nixlBackendEngine {
public:
    /**
     * @brief Construct MOCKKV backend engine
     * @param init_params Initialization parameters from NIXL
     */
    explicit nixlMockKVEngine(const nixlBackendInitParams *init_params);

    /**
     * @brief Destructor
     */
    ~nixlMockKVEngine() override = default;

    // ========================================================================
    // Phase 2: register / deregister (registerMem / deregisterMem)
    // ========================================================================

    /**
     * @brief Register memory descriptor with the backend
     *
     * Steps: (1) Key from metaInfo or stringified devId; (2) allocate nixlMockKVMetadata(devId, key);
     * (3) devIdToKey_[devId]=key for postXfer; (4) return meta, ownership to caller.
     * Logs: registerMem: type=..., devId=..., metaInfo=... ; registered devId=... -> key=...
     *
     * @param mem Memory descriptor (devId, metaInfo, etc.)
     * @param nixl_mem Memory type (must be DRAM_SEG)
     * @param out Output metadata pointer (caller takes ownership)
     * @return NIXL_SUCCESS on success
     */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem,
                const nixl_mem_t &nixl_mem,
                nixlBackendMD *&out) override;

    /**
     * @brief Deregister memory descriptor
     *
     * Same as obj_backend (S3): erase devIdToKey_ and free metadata with unique_ptr.
     * Log: deregisterMem: removing devId=..., key=...
     *
     * @param meta Metadata pointer from registerMem
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    /**
     * @brief Query if memory descriptors exist
     * For each descriptor, key = metaInfo or devId string; check kv_store_.
     * @param descs Descriptors to query
     * @param resp Output responses (exists -> nixl_b_params_t{}, not exists -> nullopt)
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    // ========================================================================
    // Phase 3: prep / post / check / release (prepXfer -> postXfer -> checkXfer -> releaseReqH)
    // ========================================================================

    /**
     * @brief Prepare transfer operation
     *
     * Steps: (1) Validate WRITE/READ and DRAM_SEG on both sides; (2) allocate placeholder handle.
     * MOCKKV is synchronous — handle is a stub. Log: prepXfer: operation=..., local_count=..., remote_count=...
     *
     * @param operation Transfer operation (NIXL_WRITE or NIXL_READ)
     * @param local Local memory descriptors
     * @param remote Remote memory descriptors
     * @param remote_agent Remote agent name
     * @param handle Output request handle
     * @param opt_args Optional arguments
     * @return NIXL_SUCCESS on success
     */
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    /**
     * @brief Post transfer operation (execute PUT/GET)
     *
     * Steps: (1) In-memory only (no filesystem); (2) resolve key from remote metadata or devIdToKey_;
     * (3) WRITE: memcpy to kv_store_[key]; (4) READ: memcpy from kv_store_[key] to local buffer.
     * Logs: postXfer: Starting WRITE/READ ... ; Local operation - using in-memory store ; Found key ...
     *
     * @param operation Transfer operation (NIXL_WRITE = PUT, NIXL_READ = GET)
     * @param local Local memory descriptors
     * @param remote Remote memory descriptors
     * @param remote_agent Remote agent name
     * @param handle Request handle from prepXfer
     * @param opt_args Optional arguments
     * @return NIXL_SUCCESS (synchronous, complete immediately)
     */
    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    /**
     * @brief Check transfer operation status
     * MOCKKV is synchronous; always SUCCESS (work done in postXfer).
     * @param handle Request handle
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    /**
     * @brief Release request handle
     * Frees handle from prepXfer. Log: releaseReqH: Releasing request handle
     * @param handle Request handle to release
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    /**
     * @brief Get supported memory types
     * @return Vector of supported memory types (DRAM_SEG only)
     */
    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG};
    }

    /**
     * @brief Check if backend supports remote operations
     * @return false (MOCKKV is local in-memory only)
     */
    bool
    supportsRemote() const override {
        return false;
    }

    /**
     * @brief Check if backend supports local operations
     * @return true (MOCKKV supports local operations)
     */
    bool
    supportsLocal() const override {
        return true;
    }

    /**
     * @brief Check if backend supports notifications
     * @return false (not needed for local-only backend)
     */
    bool
    supportsNotif() const override {
        return false;
    }

    /**
     * @brief Connect (local-only no-op)
     * @param remote_agent Remote agent name
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    connect(const std::string &remote_agent) override;

    /**
     * @brief Disconnect (local-only no-op)
     * @param remote_agent Remote agent name
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    /**
     * @brief Unload metadata (same as nixlObjEngine: no free here; free only in deregisterMem)
     *
     * @param input Metadata pointer (ignored)
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    /**
     * @brief Load local metadata for local transfers
     * Pass-through: output = input.
     * @param input Local metadata
     * @param output Output metadata pointer (same as input)
     * @return NIXL_SUCCESS
     */
    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

private:
    /**
     * @brief Get the local agent name
     * @return Local agent name
     */
    std::string
    getLocalAgent() const {
        return localAgent;
    }

    // ========================================================================
    // Internal Storage and State
    // ========================================================================

    /**
     * @brief In-memory key-value store
     * 
     * Key: std::string (from metaInfo or devId)
     * Value: std::vector<uint8_t> (binary data)
     */
    mutable std::unordered_map<std::string, std::vector<uint8_t>> kv_store_;

    /**
     * @brief Mapping from devId to key
     * Used to look up keys when only devId is available
     */
    std::unordered_map<uint64_t, std::string> devIdToKey_;

    /**
     * @brief Local agent name
     */
    std::string localAgent;

};

#endif // MOCKKV_BACKEND_H
