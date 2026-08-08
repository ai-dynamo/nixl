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
#if HAVE_ETCD

#include "nixl_etcd_metadata_backend.h"

#include "agent_data.h"
#include "nixl_types.h"
#include "common/nixl_log.h"

#include <string>
#include <tuple>
#include <utility>

nixlEtcdMetadataBackend::nixlEtcdMetadataBackend(nixlMetadataContext &ctx) noexcept : ctx_(ctx) {}

std::string_view
nixlEtcdMetadataBackend::name() const {
    return "ETCD";
}

nixl_status_t
nixlEtcdMetadataBackend::sendLocal(const nixl_opt_args_t * /*extra_params*/) {
    nixl_blob_t my_md;
    const nixl_status_t ret = ctx_.getLocalMD(my_md);
    if (ret < 0) {
        return ret;
    }
    ctx_.enqueueCommWork(std::make_tuple(ETCD_SEND, default_metadata_label, 0, std::move(my_md)));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlEtcdMetadataBackend::sendLocalPartial(const nixl_reg_dlist_t &descs,
                                          const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->metadataLabel.empty()) {
        NIXL_ERROR_FUNC << "metadata label is required for etcd send of local partial metadata";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_blob_t my_md;
    const nixl_status_t ret = ctx_.getLocalPartialMD(descs, my_md, extra_params);
    if (ret < 0) {
        return ret;
    }
    ctx_.enqueueCommWork(
        std::make_tuple(ETCD_SEND, extra_params->metadataLabel, 0, std::move(my_md)));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlEtcdMetadataBackend::fetchRemote(const std::string &remote_name,
                                     const nixl_opt_args_t *extra_params) {
    std::string label = (extra_params && !extra_params->metadataLabel.empty()) ?
        extra_params->metadataLabel :
        default_metadata_label;
    ctx_.enqueueCommWork(std::make_tuple(ETCD_FETCH, std::move(label), 0, remote_name));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlEtcdMetadataBackend::invalidateLocal(const nixl_opt_args_t * /*extra_params*/) {
    ctx_.enqueueCommWork(std::make_tuple(ETCD_INVAL, "", 0, ""));
    return NIXL_SUCCESS;
}

#endif // HAVE_ETCD
