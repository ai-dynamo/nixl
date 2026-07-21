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
    const size_t count = end_idx - start_idx;
    resize(count);
    conn_ = static_cast<nixlUcxPublicMetadata *>(remote[start_idx].metadataP)->conn;

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

void
sglXfer::resize(size_t count) {
    localAddrs_.resize(count);
    remoteAddrs_.resize(count);
    lengths_.resize(count);
    memhs_.resize(count);
    rkeys_.resize(count);
}

} // namespace nixl::ucx

#endif // HAVE_UCX_SGL_API
