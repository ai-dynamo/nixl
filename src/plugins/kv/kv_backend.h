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
#ifndef NIXL_SRC_PLUGINS_KV_KV_BACKEND_H
#define NIXL_SRC_PLUGINS_KV_KV_BACKEND_H

#include <string>
#include <unordered_map>
#include <vector>

#include "backend/backend_engine.h"
#include "kv_req_handle.h"

/**
 * @brief Abstract base class for NIXL key-value storage backend plugins.
 *
 * nixlKVBackendBase factors out the boilerplate that is identical across all
 * KV-family plugins (REDIS, WQSKV, and future backends):
 *
 *  - Capability flags: supportsRemote/Local/Notif all return fixed values.
 *  - getSupportedMems(): defaults to {DRAM_SEG}; override to add OBJ_SEG etc.
 *  - Key management: metaInfo-or-devId resolution, devIdToKey_ map,
 *    registerMem / deregisterMem, resolveKey helper.
 *  - Async request handle lifecycle: checkXfer (poll pending/first_error) and
 *    releaseReqH (block on CV until done) operating on nixlKVReqH.
 *  - No-op lifecycle stubs: connect, disconnect, unloadMD, loadLocalMD.
 *
 * Concrete backends must implement:
 *  - prepXfer   — allocate a nixlKVReqH (or subclass) and validate inputs
 *  - postXfer   — set pending = n, dispatch vendor async ops, return immediately
 *  - queryMem   — backend-specific memory query (strong vs. local-only semantics)
 *
 * Each backend plugin registers itself via nixlBackendPluginCreator<Derived>.
 *
 * Directory convention:
 *   src/plugins/kv/              ← this shared base
 *   src/plugins/kv/redis/        ← REDIS plugin (links kv_base)
 *   src/plugins/kv/wqskv/        ← WQSKV plugin (links kv_base)
 *   src/plugins/kv/<future>/     ← additional KV backends
 */
class nixlKVBackendBase : public nixlBackendEngine {
public:
    explicit nixlKVBackendBase(const nixlBackendInitParams *init_params)
        : nixlBackendEngine(init_params) {}

    ~nixlKVBackendBase() override = default;

    // ── Capability flags ───────────────────────────────────────────────────
    bool supportsRemote() const override { return false; }
    bool supportsLocal()  const override { return true;  }
    bool supportsNotif()  const override { return false; }

    /**
     * @brief Returns {DRAM_SEG} by default.
     *
     * Backends that also accept OBJ_SEG (e.g., REDIS) override this.
     */
    nixl_mem_list_t getSupportedMems() const override { return {DRAM_SEG}; }

    // ── Key management ─────────────────────────────────────────────────────

    /**
     * @brief Register a memory region and resolve its KV key.
     *
     * Key policy (shared across all KV backends):
     *   key = mem.metaInfo  (if non-empty)
     *        else  std::to_string(mem.devId)
     *
     * Stores the mapping in devIdToKey_ and creates a nixlKVMetadata output.
     * Returns NIXL_ERR_NOT_SUPPORTED if nixl_mem is not in getSupportedMems().
     */
    nixl_status_t registerMem(const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD *&out) override;

    /**
     * @brief Deregister a memory region and remove its key mapping.
     */
    nixl_status_t deregisterMem(nixlBackendMD *meta) override;

    // ── Async request handle lifecycle ─────────────────────────────────────

    /**
     * @brief Non-blocking transfer status poll.
     *
     * Returns NIXL_IN_PROG while pending > 0, NIXL_ERR_BACKEND on first error,
     * NIXL_SUCCESS when all operations complete.
     *
     * Expects handle to be a nixlKVReqH (or subclass).
     */
    nixl_status_t checkXfer(nixlBackendReqH *handle) const override;

    /**
     * @brief Block until all in-flight ops finish, then delete the handle.
     *
     * Waits on nixlKVReqH::cv until pending == 0.
     */
    nixl_status_t releaseReqH(nixlBackendReqH *handle) const override;

    // ── No-op lifecycle stubs ───────────────────────────────────────────────

    nixl_status_t connect(const std::string &) override    { return NIXL_SUCCESS; }
    nixl_status_t disconnect(const std::string &) override { return NIXL_SUCCESS; }
    nixl_status_t unloadMD(nixlBackendMD *) override       { return NIXL_SUCCESS; }

    nixl_status_t loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        if (!input) {
            output = nullptr;
            return NIXL_ERR_INVALID_PARAM;
        }
        output = input;
        return NIXL_SUCCESS;
    }

    // ── Pure virtuals that each backend must provide ────────────────────────

    nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args = nullptr) const override = 0;

    nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args = nullptr) const override = 0;

    nixl_status_t queryMem(const nixl_reg_dlist_t &descs,
                           std::vector<nixl_query_resp_t> &resp) const override = 0;

protected:
    /**
     * @brief Shared metadata type for all KV backends.
     *
     * Holds the resolved KV key and the devId used for map lookups.
     * Backends that need additional fields may subclass this.
     */
    class nixlKVMetadata : public nixlBackendMD {
    public:
        nixlKVMetadata(nixl_mem_t mem_type, uint64_t dev_id, std::string kv_key)
            : nixlBackendMD(/*isPrivate=*/true),
              memType(mem_type),
              devId(dev_id),
              key(std::move(kv_key)) {}

        nixl_mem_t  memType;
        uint64_t    devId;
        std::string key;
    };

    /**
     * @brief Map from devId → resolved KV key, populated by registerMem.
     *
     * Used as a fallback when metadataP cannot be cast to nixlKVMetadata
     * (e.g., for remote descriptors that arrive without local metadata).
     */
    std::unordered_map<uint64_t, std::string> devIdToKey_;

    /**
     * @brief Resolve the KV key for a descriptor.
     *
     * Two-step lookup:
     *   1. Cast desc.metadataP to nixlKVMetadata — use its key if valid.
     *   2. Fall back to devIdToKey_[desc.devId].
     *
     * @return true if a key was found, false if the descriptor is unmapped.
     */
    bool resolveKey(const nixlMetaDesc &desc, std::string &out_key) const;
};

#endif // NIXL_SRC_PLUGINS_KV_KV_BACKEND_H
