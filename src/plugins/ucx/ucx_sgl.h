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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_SGL_H
#define NIXL_SRC_PLUGINS_UCX_UCX_SGL_H

#ifdef HAVE_UCX_SGL_API

#include <cstddef>
#include <cstdint>
#include <memory>

extern "C" {
#include <ucp/api/ucp.h>
}

#include "ucx_backend.h"

namespace nixl::ucx {

class sglXfer {
public:
    sglXfer(const nixl_meta_dlist_t &local,
            const nixl_meta_dlist_t &remote,
            size_t worker_id,
            size_t start_idx,
            size_t end_idx);

    [[nodiscard]] size_t
    size() const noexcept {
        return size_;
    }

    [[nodiscard]] const ucx_connection_ptr_t &
    conn() const noexcept {
        return conn_;
    }

    [[nodiscard]] ucp_dt_local_sgl_t
    localView() const noexcept {
        return {
            .field_mask = UCP_DT_LOCAL_SGL_FIELD_BUFFERS | UCP_DT_LOCAL_SGL_FIELD_LENGTHS |
                UCP_DT_LOCAL_SGL_FIELD_MEMHS,
            .buffers = localAddrs_,
            .lengths = lengths_,
            .memhs = memhs_,
        };
    }

    [[nodiscard]] ucp_dt_remote_sgl_t
    remoteView() const noexcept {
        return {
            .field_mask = UCP_DT_REMOTE_SGL_FIELD_REMOTE_ADDRS | UCP_DT_REMOTE_SGL_FIELD_LENGTHS |
                UCP_DT_REMOTE_SGL_FIELD_RKEYS,
            .remote_addrs = remoteAddrs_,
            .lengths = lengths_,
            .rkeys = rkeys_,
        };
    }

private:
    static constexpr size_t num_fields = 5;

    std::unique_ptr<std::byte[]> storage_;
    void **localAddrs_ = nullptr;
    uint64_t *remoteAddrs_ = nullptr;
    size_t *lengths_ = nullptr;
    ucp_mem_h *memhs_ = nullptr;
    ucp_rkey_h *rkeys_ = nullptr;
    size_t size_ = 0;
    ucx_connection_ptr_t conn_;
};

} // namespace nixl::ucx

#endif // HAVE_UCX_SGL_API

#endif // NIXL_SRC_PLUGINS_UCX_UCX_SGL_H
