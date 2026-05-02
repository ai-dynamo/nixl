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

#include "agent.h"

#include <stdexcept>

namespace {
const std::string backend_name{"UCX"};

[[nodiscard]] constexpr nixlAgentConfig
getAgentConfig() noexcept {
    nixlAgentConfig config;
    config.useProgThread = true;
    return config;
}
} // namespace

namespace nixl::gpu {
agent::agent(const std::string &name) : agent_(name, getAgentConfig()) {
    nixlBackendH *backend_handle;
    if (agent_.createBackend(backend_name, {}, backend_handle) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to create backend");
    }
}

void
agent::registerMem(const std::vector<memBuffer> &mbs) {
    nixlDescList<nixlBlobDesc> blob_desc_list{VRAM_SEG};
    for (const auto &mb : mbs) {
        blob_desc_list.addDesc(nixlBlobDesc{mb, mb.size(), 0});
    }
    if (agent_.registerMem(blob_desc_list) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to register memory");
    }
}

nixl_blob_t
agent::getLocalMD() {
    nixl_blob_t local_md;
    if (agent_.getLocalMD(local_md) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get local metadata");
    }
    return local_md;
}

void
agent::loadRemoteMD(const nixl_blob_t &md) {
    std::string remote_agent_name;
    if (agent_.loadRemoteMD(md, remote_agent_name) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to load remote metadata");
    }
    remoteAgentName_ = std::move(remote_agent_name);
}

[[nodiscard]] void *
agent::regAndPrepLocalMemView(const std::vector<memBuffer> &mbs) {
    registerMem(mbs);
    return prepLocalMemView(mbs);
}

void *
agent::prepRemoteMemView(const std::vector<memBuffer> &mbs) {
    if (!remoteAgentName_) {
        throw std::runtime_error("Remote agent name is not set");
    }

    nixlDescList<nixlRemoteDesc> remote_desc_list{VRAM_SEG};
    for (const auto &mb : mbs) {
        remote_desc_list.addDesc(nixlRemoteDesc{mb, mb.size(), 0, *remoteAgentName_});
    }
    nixlMemViewH mvh;
    if (agent_.prepMemView(remote_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare remote memory view");
    }
    addMemView(mvh);
    return mvh;
}

void *
agent::prepLocalMemView(const std::vector<memBuffer> &mbs) {
    nixlDescList<nixlBasicDesc> basic_desc_list{VRAM_SEG};
    for (const auto &mb : mbs) {
        basic_desc_list.addDesc(nixlBasicDesc{mb, mb.size(), 0});
    }
    nixlMemViewH mvh;
    if (agent_.prepMemView(basic_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare local memory view");
    }
    addMemView(mvh);
    return mvh;
}

void
agent::addMemView(void *mvh) {
    memViews_.emplace(std::unique_ptr<void, std::function<void(void *)>>{
        mvh, [this](void *mvh) { this->agent_.releaseMemView(mvh); }});
}
} // namespace nixl::gpu
