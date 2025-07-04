/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rkey.h"

#include "common/nixl_log.h"
#include "ucx_utils.h"

namespace nixl::ucx {
rkey::rkey(const nixlUcxEp &ep, const void *rkey_buffer) noexcept
    : rkey_{unpackUcpRkey(ep, rkey_buffer), &ucp_rkey_destroy} {}

ucp_rkey *
rkey::unpackUcpRkey(const nixlUcxEp &ep, const void *rkey_buffer) noexcept {
    ucp_rkey_h rkey = nullptr;
    const auto status = ucp_ep_rkey_unpack(ep.getEp(), rkey_buffer, &rkey);
    if (status != UCS_OK) {
        NIXL_ERROR << "Failed to unpack UCX RKEY: " << ucs_status_string(status);
        return nullptr;
    }
    return rkey;
}
} // namespace nixl::ucx
