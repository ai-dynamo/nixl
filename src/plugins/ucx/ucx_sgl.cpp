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

#include "ucx_sgl.h"

#ifdef HAVE_UCX_SGL_API

#include "common/nixl_log.h"

namespace nixl::ucx {

sglXfer::sglXfer(const nixl_meta_dlist_t &local,
                 const nixl_meta_dlist_t &remote,
                 size_t worker_id,
                 size_t start_idx,
                 size_t end_idx) {
    NIXL_ASSERT(end_idx > start_idx);
    size_ = end_idx - start_idx;
    conn_ = static_cast<nixlUcxPublicMetadata *>(remote[start_idx].metadataP)->conn;

    static_assert(sizeof(void *) == sizeof(uint64_t) && sizeof(size_t) == sizeof(uint64_t) &&
                      sizeof(ucp_mem_h) == sizeof(uint64_t) &&
                      sizeof(ucp_rkey_h) == sizeof(uint64_t),
                  "sglXfer assumes all field types are 8 bytes wide");
    constexpr size_t field_bytes = sizeof(uint64_t);
    const size_t array_bytes = size_ * field_bytes;

    storage_ = std::make_unique_for_overwrite<std::byte[]>(num_fields * array_bytes);
    std::byte *const base = storage_.get();
    localAddrs_ = reinterpret_cast<void **>(base + 0 * array_bytes);
    remoteAddrs_ = reinterpret_cast<uint64_t *>(base + 1 * array_bytes);
    lengths_ = reinterpret_cast<size_t *>(base + 2 * array_bytes);
    memhs_ = reinterpret_cast<ucp_mem_h *>(base + 3 * array_bytes);
    rkeys_ = reinterpret_cast<ucp_rkey_h *>(base + 4 * array_bytes);

    for (size_t i = start_idx; i < end_idx; ++i) {
        const size_t out = i - start_idx;
        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        NIXL_ASSERT(local[i].len == remote[i].len);
        NIXL_ASSERT(rmd->conn == conn_);

        localAddrs_[out] = reinterpret_cast<void *>(local[i].addr);
        remoteAddrs_[out] = static_cast<uint64_t>(remote[i].addr);
        lengths_[out] = local[i].len;
        memhs_[out] = lmd->getMem().getMemh();
        rkeys_[out] = rmd->getRkey(worker_id).get();
    }
}

} // namespace nixl::ucx

#endif // HAVE_UCX_SGL_API
