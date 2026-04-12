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
 * @file mockkv_backend.cpp
 * @brief MOCKKV Backend Implementation - Simple in-memory key-value store
 *
 * Architecture and lifecycle: see MOCKKV_ARCHITECTURE.md in this directory. Call order:
 * createBackend -> registerMem -> prepXfer -> postXfer -> checkXfer -> releaseReqH -> deregisterMem.
 * Core may call unloadMD on teardown; same as Obj (S3): unloadMD does not free — only deregisterMem does
 * (see nixlLocalSection teardown path).
 *
 * Implementation:
 * - Storage: kv_store_[key] = vector<uint8_t> (in-process only)
 * - Transfer: synchronous PUT/GET in postXfer; prepXfer validates and allocates a placeholder handle
 */

#include "mockkv_backend.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include <absl/strings/str_format.h>
#include <memory>
#include <cstring>
#include <algorithm>

namespace {

/**
 * @brief Simple request handle for MOCKKV
 * 
 * Since MOCKKV operations are synchronous, we don't need to track
 * async operations. This handle is minimal - just a placeholder.
 */
class nixlMockKVBackendReqH : public nixlBackendReqH {
public:
    nixlMockKVBackendReqH() = default;
    ~nixlMockKVBackendReqH() = default;
};

/**
 * @brief Metadata for MOCKKV memory descriptors
 * 
 * Stores the mapping between devId and key for later lookups.
 */
class nixlMockKVMetadata : public nixlBackendMD {
public:
    nixlMockKVMetadata(uint64_t dev_id, std::string key)
        : nixlBackendMD(true),
          devId(dev_id),
          key(key) {}

    ~nixlMockKVMetadata() = default;

    uint64_t devId;      // Device ID from NIXL
    std::string key;     // Key in the map
};

/**
 * @brief Validate transfer preparation parameters
 * 
 * Checks that:
 * - Operation is valid (WRITE or READ)
 * - Local memory is DRAM_SEG
 * - Remote memory is DRAM_SEG (for MOCKKV)
 * 
 * @param operation Transfer operation type
 * @param local Local memory descriptors
 * @param remote Remote memory descriptors
 * @param remote_agent Remote agent name (not used for MOCKKV)
 * @param local_agent Local agent name
 * @return true if valid, false otherwise
 */
bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    (void)remote_agent;
    (void)local_agent;
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    // MOCKKV uses DRAM_SEG for both local and remote
    if (remote.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be DRAM_SEG, got %d",
                                      remote.getType());
        return false;
    }

    return true;
}

} // namespace

// ============================================================================
// MOCKKV Engine Implementation
// ============================================================================

/**
 * @brief Construct MOCKKV backend engine
 *
 * Step 1: Read localAgent from init_params.
 * Step 2: In-memory store only (no filesystem).
 * Logs: MOCKKV backend initialized ; Local agent =
 */
nixlMockKVEngine::nixlMockKVEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {

    // 1. Local agent name
    localAgent = init_params->localAgent;

    // 2. Values live only in kv_store_ (no file persistence).

    NIXL_INFO << "MOCKKV backend initialized (in-memory only)";
    NIXL_INFO << "MOCKKV: Local agent = " << localAgent;
}

/**
 * @brief Register memory descriptor with MOCKKV backend
 *
 * Step 1: Require DRAM_SEG.
 * Step 2: Key from metaInfo or stringified devId.
 * Step 3: Allocate nixlMockKVMetadata(devId, key), set devIdToKey_[devId]=key.
 * Step 4: Return meta; caller owns it (out = release()).
 * Logs: registerMem: type=..., devId=..., metaInfo=... ; registerMem: registered devId=... -> key=...
 */
nixl_status_t
nixlMockKVEngine::registerMem(const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD *&out) {
    NIXL_INFO << "registerMem: type=" << nixl_mem << ", devId=" << mem.devId
              << ", metaInfo=" << (mem.metaInfo.empty() ? "<empty>" : mem.metaInfo);

    // 1. DRAM_SEG only
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        NIXL_ERROR << "registerMem: unsupported memory type " << nixl_mem;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // 2. key = metaInfo or to_string(devId)
    std::string key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
    NIXL_DEBUG << "registerMem: using key=" << key;

    // 3. Metadata and devId -> key for postXfer lookups
    std::unique_ptr<nixlMockKVMetadata> mockkv_md = std::make_unique<nixlMockKVMetadata>(
        mem.devId, key);
    devIdToKey_[mem.devId] = key;
    NIXL_INFO << "registerMem: registered devId=" << mem.devId << " -> key=" << key;

    // 4. Ownership to NIXL
    out = mockkv_md.release();
    return NIXL_SUCCESS;
}

/**
 * @brief Deregister memory descriptor
 *
 * Same as obj_backend (S3): erase devId mapping, unique_ptr deletes metadata (no static freed set).
 * unloadMD does not delete (see below).
 * Log: deregisterMem: removing devId=..., key=...
 */
nixl_status_t
nixlMockKVEngine::deregisterMem(nixlBackendMD *meta) {
    nixlMockKVMetadata *mockkv_md = static_cast<nixlMockKVMetadata *>(meta);
    if (mockkv_md) {
        NIXL_INFO << "deregisterMem: removing devId=" << mockkv_md->devId
                  << ", key=" << mockkv_md->key;
        devIdToKey_.erase(mockkv_md->devId);
        std::unique_ptr<nixlMockKVMetadata> ptr(mockkv_md);
    }
    return NIXL_SUCCESS;
}

/**
 * @brief Query if memory descriptors exist
 * 
 * Checks if keys exist in in-memory store.
 * 
 * @param descs Descriptors to query
 * @param resp Output responses
 * @return NIXL_SUCCESS
 */
nixl_status_t
nixlMockKVEngine::queryMem(const nixl_reg_dlist_t &descs, 
                           std::vector<nixl_query_resp_t> &resp) const {
    resp.reserve(descs.descCount());
    
    for (auto &desc : descs) {
        // Check if key exists in store
        std::string key = desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo;
        
        // Same pattern as nixlObjEngine::queryMem: no lock; S3 uses network, we use local map
        // with serialized agent/backend calls (devIdToKey_ is also unprotected).
        const bool exists = kv_store_.find(key) != kv_store_.end();
        
        NIXL_DEBUG << "queryMem: key=" << key << ", exists=" << exists;
        
        // Return empty optional if key doesn't exist, otherwise return empty params
        resp.emplace_back(exists ? nixl_query_resp_t{nixl_b_params_t{}} : std::nullopt);
    }
    
    return NIXL_SUCCESS;
}

/**
 * @brief Prepare transfer operation
 *
 * Step 1: Validate WRITE/READ and DRAM_SEG on both sides.
 * Step 2: Allocate placeholder handle (MOCKKV is synchronous — no async tracking).
 * Log: prepXfer: operation=..., local_count=..., remote_count=...
 */
nixl_status_t
nixlMockKVEngine::prepXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    NIXL_INFO << "prepXfer: operation=" << (operation == NIXL_WRITE ? "WRITE" : "READ")
              << ", local_count=" << local.descCount()
              << ", remote_count=" << remote.descCount();

    // 1. Validate parameters
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        NIXL_ERROR << "prepXfer: parameter validation failed";
        return NIXL_ERR_INVALID_PARAM;
    }

    // 2. Placeholder handle for postXfer / checkXfer / releaseReqH
    auto req_h = std::make_unique<nixlMockKVBackendReqH>();
    handle = req_h.release();
    NIXL_DEBUG << "prepXfer: request handle created, ready for postXfer";
    return NIXL_SUCCESS;
}

/**
 * @brief Post transfer operation (execute PUT/GET)
 *
 * Step 1: In-memory kv_store_ only (no initiator/target split, no filesystem).
 * Step 2: Resolve key from remote metadata or devIdToKey_ per descriptor.
 * Step 3: WRITE copies local buffer to kv_store_[key]; READ copies from kv_store_[key] to local buffer.
 * Logs: postXfer: Starting WRITE/READ ... ; Local operation - using in-memory store ; ...
 */
nixl_status_t
nixlMockKVEngine::postXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    NIXL_INFO << "postXfer: ========================================";
    NIXL_INFO << "postXfer: Starting " << (operation == NIXL_WRITE ? "WRITE (PUT)" : "READ (GET)")
              << " operation with " << local.descCount() << " descriptor(s)";
    (void)remote_agent;
    (void)handle;
    (void)opt_args;

    NIXL_INFO << "postXfer: Local operation - using in-memory store";

    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];

        NIXL_INFO << "postXfer: [Descriptor " << i << "/" << local.descCount() << "]";
        NIXL_INFO << "postXfer:   Local: addr=0x" << std::hex << local_desc.addr << std::dec
                  << ", len=" << local_desc.len;
        NIXL_INFO << "postXfer:   Remote: devId=" << remote_desc.devId;

        // 2. Resolve key from remote metadata or devIdToKey_
        std::string key;
        if (remote_desc.metadataP) {
            nixlMockKVMetadata *mockkv_md =
                dynamic_cast<nixlMockKVMetadata *>(remote_desc.metadataP);
            if (mockkv_md) {
                key = mockkv_md->key;
                NIXL_INFO << "postXfer:   Found key in metadata: " << key;
            } else {
                auto key_search = devIdToKey_.find(remote_desc.devId);
                if (key_search == devIdToKey_.end()) {
                    NIXL_ERROR << "postXfer: Key for devId " << remote_desc.devId << " not found";
                    return NIXL_ERR_INVALID_PARAM;
                }
                key = key_search->second;
                NIXL_DEBUG << "postXfer:   Found key in mapping: " << key;
            }
        } else {
            auto key_search = devIdToKey_.find(remote_desc.devId);
            if (key_search == devIdToKey_.end()) {
                NIXL_ERROR << "postXfer: Key for devId " << remote_desc.devId << " not found";
                return NIXL_ERR_INVALID_PARAM;
            }
            key = key_search->second;
            NIXL_DEBUG << "postXfer:   Found key in mapping (no metadata): " << key;
        }

        uintptr_t data_ptr = local_desc.addr;
        size_t data_len = local_desc.len;

        if (operation == NIXL_WRITE) {
            // ============================================================
            // WRITE Operation: Copy data from local buffer to storage
            // ============================================================
            NIXL_INFO << "postXfer:   WRITE: Copying " << data_len 
                     << " bytes from buffer to key=" << key;
            
            // ========================================================
            // Local WRITE: Write to in-memory store
            // ========================================================
            // Resize vector to hold the data
            kv_store_[key].resize(data_len);
            
            // Copy data from user buffer to map
            std::memcpy(kv_store_[key].data(), 
                       reinterpret_cast<const void *>(data_ptr), 
                       data_len);
            
            NIXL_INFO << "postXfer:   WRITE: Successfully stored " << data_len
                     << " bytes in key=" << key;
            NIXL_INFO << "postXfer:   WRITE: kv_store_ now has "
                      << kv_store_.size() << " keys";
            
        } else {
            // ============================================================
            // READ Operation: Copy data from storage to local buffer
            // ============================================================
            NIXL_INFO << "postXfer:   READ: Retrieving data from key=" << key;
            
            // ========================================================
            // Local READ: Read from in-memory store
            // ========================================================
            // Check if key exists
            auto it = kv_store_.find(key);
            if (it == kv_store_.end()) {
                NIXL_ERROR << "postXfer:   READ: Key not found: " << key;
                return NIXL_ERR_BACKEND;
            }
            
            // Check if data size matches
            size_t stored_len = it->second.size();
            if (stored_len < data_len) {
                NIXL_WARN << "postXfer:   READ: Stored data size (" << stored_len 
                         << ") < requested size (" << data_len << ")";
                data_len = stored_len;  // Read only what's available
            }
            
            // Copy data from map to user buffer
            std::memcpy(reinterpret_cast<void *>(data_ptr),
                       it->second.data(),
                       data_len);
            
            NIXL_INFO << "postXfer:   READ: Successfully retrieved " << data_len 
                     << " bytes from key=" << key;
        }
    }

    NIXL_INFO << "postXfer: ========================================";
    NIXL_INFO << "postXfer: All " << local.descCount()
              << " operations completed successfully";
    NIXL_INFO << "postXfer: Returning NIXL_SUCCESS (synchronous operations)";
    return NIXL_SUCCESS;
}

/**
 * @brief Check transfer operation status
 * MOCKKV is synchronous; SUCCESS immediately (work finished in postXfer).
 */
nixl_status_t
nixlMockKVEngine::checkXfer(nixlBackendReqH *handle) const {
    (void)handle;
    NIXL_DEBUG << "checkXfer: Operations are synchronous, always return SUCCESS";
    return NIXL_SUCCESS;
}

/**
 * @brief Release request handle
 * Frees handle from prepXfer. Log: releaseReqH: Releasing request handle
 */
nixl_status_t
nixlMockKVEngine::releaseReqH(nixlBackendReqH *handle) const {
    NIXL_INFO << "releaseReqH: Releasing request handle";
    delete handle;
    return NIXL_SUCCESS;
}

/**
 * @brief Load local metadata for local transfers
 * Pass-through: output = input.
 */
nixl_status_t
nixlMockKVEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    output = input;
    NIXL_DEBUG << "loadLocalMD: Reusing local metadata object";
    return NIXL_SUCCESS;
}

/**
 * @brief Connect to remote agent
 *
 * local-only backend: no-op.
 */
nixl_status_t
nixlMockKVEngine::connect(const std::string &remote_agent) {
    (void)remote_agent;
    return NIXL_SUCCESS;
}

/**
 * @brief Disconnect from remote agent
 * 
 * local-only backend: no-op.
 * 
 * @param remote_agent Remote agent name
 * @return NIXL_SUCCESS
 */
nixl_status_t
nixlMockKVEngine::disconnect(const std::string &remote_agent) {
    (void)remote_agent;
    return NIXL_SUCCESS;
}

/**
 * @brief Unload metadata (may be called from nixlRemoteSection teardown paths)
 *
 * Same as nixlObjEngine::unloadMD: no free; deregisterMem owns release.
 */
nixl_status_t
nixlMockKVEngine::unloadMD(nixlBackendMD *input) {
    (void)input;
    return NIXL_SUCCESS;
}

