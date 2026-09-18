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

#include "kv_backend.h"

#include <algorithm>
#include <memory>
#include <string>

#include "common/nixl_log.h"
#include "nixl_descriptors.h"
#include "nixl_types.h"

// ── registerMem ─────────────────────────────────────────────────────────────

nixl_status_t
nixlKVBackendBase::registerMem(const nixlBlobDesc &mem,
                               const nixl_mem_t &nixl_mem,
                               nixlBackendMD *&out) {
    const auto supported = getSupportedMems();
    if (std::find(supported.begin(), supported.end(), nixl_mem) == supported.end()) {
        NIXL_ERROR << "KV backend: unsupported memory type " << nixl_mem;
        out = nullptr;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Key policy: use metaInfo if non-empty, otherwise fall back to devId string.
    std::string kv_key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;

    auto md = std::make_unique<nixlKVMetadata>(nixl_mem, mem.devId, kv_key);
    devIdToKey_[mem.devId] = kv_key;
    out = md.release();
    return NIXL_SUCCESS;
}

// ── deregisterMem ───────────────────────────────────────────────────────────

nixl_status_t
nixlKVBackendBase::deregisterMem(nixlBackendMD *meta) {
    auto *kv_md = static_cast<nixlKVMetadata *>(meta);
    if (kv_md) {
        std::unique_ptr<nixlKVMetadata> owned(kv_md);
        devIdToKey_.erase(kv_md->devId);
    }
    return NIXL_SUCCESS;
}

// ── resolveKey ──────────────────────────────────────────────────────────────

bool
nixlKVBackendBase::resolveKey(const nixlMetaDesc &desc, std::string &out_key) const {
    // Step 1: try the per-descriptor metadata pointer.
    auto *kv_md = dynamic_cast<nixlKVMetadata *>(desc.metadataP);
    if (kv_md) {
        out_key = kv_md->key;
        return true;
    }

    // Step 2: fall back to the devId → key map populated at registerMem time.
    auto it = devIdToKey_.find(desc.devId);
    if (it != devIdToKey_.end()) {
        out_key = it->second;
        return true;
    }

    return false;
}

// ── checkXfer ───────────────────────────────────────────────────────────────

nixl_status_t
nixlKVBackendBase::checkXfer(nixlBackendReqH *handle) const {
    auto *req = static_cast<nixlKVReqH *>(handle);
    if (!req) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (req->pending.load(std::memory_order_acquire) > 0) {
        return NIXL_ERR_NOT_POSTED; // IN_PROG sentinel used by NIXL
    }

    if (req->first_error.load(std::memory_order_relaxed) != 0) {
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

// ── releaseReqH ─────────────────────────────────────────────────────────────

nixl_status_t
nixlKVBackendBase::releaseReqH(nixlBackendReqH *handle) const {
    auto *req = static_cast<nixlKVReqH *>(handle);
    if (!req) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Block until all in-flight operations have called nixlKVXferCallback.
    {
        std::unique_lock<std::mutex> lk(req->mu);
        req->cv.wait(lk, [req] {
            return req->pending.load(std::memory_order_acquire) == 0;
        });
    }

    delete req;
    return NIXL_SUCCESS;
}
