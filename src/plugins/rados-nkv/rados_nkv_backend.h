/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
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

#ifndef NIXL_SRC_PLUGINS_RADOS_NKV_RADOS_NKV_BACKEND_H
#define NIXL_SRC_PLUGINS_RADOS_NKV_RADOS_NKV_BACKEND_H

#include <cstdint>
#include <string>
#include <vector>

#include "backend/backend_engine.h"
#include "rados_nkv_key.h" // radosNkvDeriveKey (SPDK-free key derivation)

// Forward declaration of the opaque SPDK KV host shim handle (C ABI).
struct kv_host_shim;

/**
 * nixlRadosNkvEngine maps NIXL transfers onto the NVMe Key-Value command set
 * via the in-process SPDK KV host shim (kv_host_shim.h).
 *
 * Memory-type mapping (MIRRORS the OBJ plugin):
 *   - local  source/destination: DRAM_SEG (host DRAM)
 *   - remote key-addressed blob : OBJ_SEG  (reused as the "remote KV key" space)
 *
 * Operation mapping:
 *   - NIXL_WRITE (local DRAM -> remote) becomes a KV Store
 *   - NIXL_READ  (remote -> local DRAM) becomes a KV Retrieve
 *
 * The remote OBJ_SEG descriptor carries the token sequence in its metaInfo blob
 * (the same channel obj uses for the object key). The engine derives the
 * fixed-length NVMe KV key as a hash of those bytes (radosNkvDeriveKey: a
 * 128-bit FNV-1a truncated to min(16, kvkml-advertised-by-the-namespace)). This
 * mirrors OBJ in keying off metaInfo rather than devId, lets an arbitrary-length
 * token sequence map to a stable key with no over-length restriction, and stores
 * the derived key in the per-descriptor metadata so transfers read it back from
 * the descriptor. An empty metaInfo is rejected (NIXL_ERR_INVALID_PARAM).
 *
 * Custom backend params (nixl_b_params_t):
 *   - "vfu_addr" (or "socket"/"vfio_user_path"/"device"): REQUIRED, the
 *     VFIOUSER transport directory for the SPDK KV target.
 *   - "nsid": OPTIONAL, NVMe namespace id (0 / unset auto-selects first KV ns).
 *   - "init_env": OPTIONAL bool ("true"/"1"/"yes"/"on"), DEFAULT false. When
 *     false (production), the host/agent owns the SPDK env and this engine does
 *     not initialize it, allowing multiple engines per process. When true, the
 *     shim brings up its own (no-hugepage) SPDK env; this is single-instance
 *     per process and is intended for standalone tests that have no host env.
 *
 * Staging / zero-copy approach (DOCUMENTED):
 *   The SPDK Store/Retrieve primitives require the value buffer to be an
 *   SPDK-DMA buffer (kv_host_shim_dma_alloc). User DRAM registered through
 *   registerMem() is ordinary host memory and is generally NOT DMA-capable, so
 *   for this skeleton we STAGE through a per-request shim DMA buffer and copy:
 *     - WRITE: memcpy(user DRAM -> DMA buf) then Store(key, DMA buf)
 *     - READ : Retrieve(key, DMA buf) then memcpy(DMA buf -> user DRAM)
 *   This keeps the skeleton correct and simple. A future optimization can
 *   register user buffers that are already DMA memory and skip the copy.
 */
class nixlRadosNkvEngine : public nixlBackendEngine {
public:
    explicit nixlRadosNkvEngine(const nixlBackendInitParams *init_params);
    ~nixlRadosNkvEngine() override;

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
    getSupportedMems() const override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    // Existence probe: maps a NIXL queryMem onto the NVMe KV Exist command.
    // This is llm-d's lookup path (agent.query_memory() -> backend queryMem),
    // used to build a cache hit/miss mask without transferring any value data.
    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
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
    // Maximum NVMe KV key length (bytes) the device key space can hold.
    static constexpr uint8_t kMaxKeyLen = 16;

    // The SPDK KV host shim handle (owns the controller attach + qpair).
    kv_host_shim *shim_ = nullptr;

    // Length of the derived key: min(kMaxKeyLen, kvkml advertised by the shim).
    uint8_t maxKeyLen_ = kMaxKeyLen;
};

#endif // NIXL_SRC_PLUGINS_RADOS_NKV_RADOS_NKV_BACKEND_H
