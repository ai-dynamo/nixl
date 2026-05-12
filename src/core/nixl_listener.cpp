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
#include "agent_data.h"
#include "backend/backend_engine.h"
#include "common/nixl_log.h"
#include "nixl.h"
#include "serdes/serdes.h"

nixl_status_t
nixlAgentData::loadConnInfo(const std::string &remote_name,
                            const nixl_backend_t &backend,
                            const nixl_blob_t &conn_info) {
    if (backendEngines_.count(backend) == 0) {
        NIXL_DEBUG << "Agent " << name_ << " does not support a remote backend: " << backend;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // No need to reload same conn info, error if it changed
    const auto r_it = remoteBackends_.find(remote_name);
    if (r_it != remoteBackends_.end()) {
        const auto rb_it = r_it->second.find(backend);
        if (rb_it != r_it->second.end()) {
            if (rb_it->second != conn_info) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            return NIXL_SUCCESS;
        }
    }

    nixlBackendEngine *eng = backendEngines_[backend].get();
    if (!eng->supportsRemote()) {
        NIXL_DEBUG << backend << " does not support remote operations";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const nixl_status_t ret = eng->loadRemoteConnInfo(remote_name, conn_info);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    remoteBackends_[remote_name].emplace(backend, conn_info);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgentData::loadRemoteSections(const std::string &remote_name, nixlSerDes &sd) {
    const auto [it, inserted] = remoteSections_.try_emplace(remote_name, remote_name);
    const nixl_status_t ret = it->second.loadRemoteData(&sd, backendEngines_);
    // TODO: can be more graceful, if just the new MD blob was improper
    if (ret != NIXL_SUCCESS) {
        remoteSections_.erase(it);
        remoteBackends_.erase(remote_name);
        return ret;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgentData::invalidateRemoteData(const std::string &remote_name) {
    if (remote_name == name_) {
        NIXL_ERROR << "Agent " << name_ << " cannot invalidate itself";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t ret = NIXL_ERR_NOT_FOUND;
    if (remoteSections_.erase(remote_name) > 0) {
        ret = NIXL_SUCCESS;
    }

    auto it_backends = remoteBackends_.find(remote_name);
    if (it_backends != remoteBackends_.end()) {
        for (auto &it : it_backends->second) {
            backendEngines_[it.first]->disconnect(remote_name);
        }

        remoteBackends_.erase(it_backends);
        ret = NIXL_SUCCESS;
    }

    return ret;
}
