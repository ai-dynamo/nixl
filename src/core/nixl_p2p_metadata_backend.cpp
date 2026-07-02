/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nixl_p2p_metadata_backend.h"

#include "agent_data.h"

#include <tuple>
#include <utility>

nixlP2PMetadataBackend::nixlP2PMetadataBackend(nixlMetadataContext &ctx) noexcept : ctx_(ctx) {}

std::string_view
nixlP2PMetadataBackend::name() const {
    return "P2P";
}

nixl_status_t
nixlP2PMetadataBackend::sendLocal(const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_blob_t my_md;
    const nixl_status_t ret = ctx_.getLocalMD(my_md);
    if (ret < 0) {
        return ret;
    }
    ctx_.enqueueCommWork(
        std::make_tuple(SOCK_SEND, extra_params->ipAddr, extra_params->port, std::move(my_md)));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::sendLocalPartial(const nixl_reg_dlist_t &descs,
                                         const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_blob_t my_md;
    const nixl_status_t ret = ctx_.getLocalPartialMD(descs, my_md, extra_params);
    if (ret < 0) {
        return ret;
    }
    ctx_.enqueueCommWork(
        std::make_tuple(SOCK_SEND, extra_params->ipAddr, extra_params->port, std::move(my_md)));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::fetchRemote(const std::string & /*remote_name*/,
                                    const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // Socket fetch is keyed by address, not name; the reply is loaded into the
    // remote-section cache by the communication thread.
    ctx_.enqueueCommWork(std::make_tuple(SOCK_FETCH, extra_params->ipAddr, extra_params->port, ""));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlP2PMetadataBackend::invalidateLocal(const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->ipAddr.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    ctx_.enqueueCommWork(std::make_tuple(SOCK_INVAL, extra_params->ipAddr, extra_params->port, ""));
    return NIXL_SUCCESS;
}
