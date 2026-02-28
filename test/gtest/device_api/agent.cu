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

#include "agent.cuh"

namespace {
constexpr std::string_view backend_name{"UCX"};
}

namespace nixl::gpu {
agent::agent(const std::string &name) : agent_(name, nixlAgentConfig(true)) {
    nixlBackendH *backend_handle;
    if (agent_.createBackend(std::string(backend_name), {}, backend_handle) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to create backend");
    }
}

void
agent::registerMem(const cudaPtr<uint64_t> &cuda_ptr) {
    nixlDescList<nixlBlobDesc> blob_desc_list(VRAM_SEG);
    blob_desc_list.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(cuda_ptr.get()), sizeof(uint64_t), 0));
    if (agent_.registerMem(blob_desc_list) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to register memory");
    }
    blobDescLists_.emplace(std::unique_ptr<nixlDescList<nixlBlobDesc>,
                                           std::function<void(nixlDescList<nixlBlobDesc> *)>>(
        new nixlDescList<nixlBlobDesc>(blob_desc_list),
        [this](nixlDescList<nixlBlobDesc> *blob_desc_list) {
            this->agent_.deregisterMem(*blob_desc_list);
            delete blob_desc_list;
        }));
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
    remoteAgentNames_.emplace(std::unique_ptr<std::string, std::function<void(std::string *)>>(
        new std::string(remote_agent_name), [this](std::string *remote_agent_name) {
            this->agent_.invalidateRemoteMD(*remote_agent_name);
            delete remote_agent_name;
        }));
}

void *
agent::prepLocalMemView(const cudaPtr<uint64_t> &cuda_ptr) {
    nixlDescList<nixlBasicDesc> basic_desc_list(VRAM_SEG);
    basic_desc_list.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(cuda_ptr.get()), sizeof(uint64_t), 0));
    nixlMemViewH mvh;
    if (agent_.prepMemView(basic_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare local memory view");
    }
    addMemView(mvh);
    return mvh;
}

void *
agent::prepRemoteMemView(const cudaPtr<uint64_t> &cuda_ptr, const std::string &remote_agent_names) {
    nixlDescList<nixlRemoteDesc> remote_desc_list(VRAM_SEG);
    remote_desc_list.addDesc(nixlRemoteDesc(
        reinterpret_cast<uintptr_t>(cuda_ptr.get()), sizeof(uint64_t), 0, remote_agent_names));
    nixlMemViewH mvh;
    if (agent_.prepMemView(remote_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare remote memory view");
    }
    addMemView(mvh);
    return mvh;
}

void *
agent::prepLocalMemView(const std::vector<cudaPtr<uint64_t>> &cuda_ptrs) {
    nixlDescList<nixlBasicDesc> basic_desc_list(VRAM_SEG);
    for (const auto &cuda_ptr : cuda_ptrs) {
        basic_desc_list.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(cuda_ptr.get()), sizeof(uint64_t), 0));
    }
    nixlMemViewH mvh;
    if (agent_.prepMemView(basic_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare local memory view");
    }
    addMemView(mvh);
    return mvh;
}

void *
agent::prepRemoteMemView(const std::vector<cudaPtr<uint64_t>> &cuda_ptrs,
                         const std::vector<std::string> &remote_agent_names) {
    nixlDescList<nixlRemoteDesc> remote_desc_list(VRAM_SEG);
    for (size_t i = 0; i < cuda_ptrs.size(); i++) {
        remote_desc_list.addDesc(nixlRemoteDesc(reinterpret_cast<uintptr_t>(cuda_ptrs[i].get()),
                                                sizeof(uint64_t),
                                                0,
                                                remote_agent_names[i]));
    }
    nixlMemViewH mvh;
    if (agent_.prepMemView(remote_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare remote memory view");
    }
    addMemView(mvh);
    return mvh;
}

void *
agent::prepRemoteMemView(const std::vector<cudaPtr<uint64_t>> &cuda_ptrs,
                         const std::string &remote_agent_name) {
    nixlDescList<nixlRemoteDesc> remote_desc_list(VRAM_SEG);
    for (size_t i = 0; i < cuda_ptrs.size(); i++) {
        remote_desc_list.addDesc(nixlRemoteDesc(reinterpret_cast<uintptr_t>(cuda_ptrs[i].get()),
                                                sizeof(uint64_t),
                                                0,
                                                remote_agent_name));
    }
    nixlMemViewH mvh;
    if (agent_.prepMemView(remote_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare remote memory view");
    }
    addMemView(mvh);
    return mvh;
}

void *
agent::prepCounterMemView(const cudaPtr<uint64_t> &counter, const std::string &remote_agent_name) {
    nixlDescList<nixlRemoteDesc> remote_desc_list(VRAM_SEG);
    remote_desc_list.addDesc(nixlRemoteDesc(
        reinterpret_cast<uintptr_t>(counter.get()), sizeof(uint64_t), 0, remote_agent_name));
    nixlMemViewH mvh;
    if (agent_.prepMemView(remote_desc_list, mvh) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare counter memory view");
    }
    addMemView(mvh);
    return mvh;
}

void
agent::addMemView(void *mvh) {
    memViews_.emplace(std::unique_ptr<void, std::function<void(void *)>>(
        mvh, [this](void *mvh) { this->agent_.releaseMemView(mvh); }));
}
} // namespace nixl::gpu
