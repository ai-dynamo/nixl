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

#include "nixl_descriptors.h"

namespace {
constexpr size_t num_ucx_workers = 1;
constexpr size_t device_id = 0;
constexpr std::string_view notification_message = "notification";

[[nodiscard]] nixlAgentConfig
createConfig() noexcept {
    return nixlAgentConfig(true, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 0, 100000);
}

[[nodiscard]] nixl_b_params_t
createBackendParams(std::optional<unsigned> num_channels) {
    nixl_b_params_t params;
    params["num_workers"] = std::to_string(num_ucx_workers);
    if (num_channels) {
        params["ucx_num_device_channels"] = std::to_string(*num_channels);
    }
    return params;
}

template<typename descType>
[[nodiscard]] nixlDescList<descType>
makeDescList(const std::vector<nixl::device_api::memTypeArray<uint8_t>> &buffers) {
    if (buffers.empty()) {
        throw std::runtime_error("No buffers to create descriptor list");
    }

    const nixl_mem_t mem_type = buffers[0].memType();
    nixlDescList<descType> desc_list(mem_type);
    for (const auto &buffer : buffers) {
        desc_list.addDesc(descType(
            reinterpret_cast<uintptr_t>(buffer.get()), buffer.size(), uint64_t(device_id)));
    }
    return desc_list;
}
} // namespace

namespace nixl::device_api {
agent::agent(const std::string &name, std::optional<unsigned> num_channels)
    : agent_(name, createConfig()),
      backendHandle_(createBackend(num_channels)) {}

void
agent::registerMem(const std::vector<memTypeArray<uint8_t>> &buffers) {
    nixlDescList<nixlBlobDesc> reg_list = makeDescList<nixlBlobDesc>(buffers);
    const nixl_status_t status = agent_.registerMem(reg_list);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to register memory");
    }
}

[[nodiscard]] nixl_blob_t
agent::getLocalMD() const {
    nixl_blob_t md;
    const nixl_status_t status = agent_.getLocalMD(md);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get local metadata");
    }
    return md;
}

void
agent::loadRemoteMD(const nixl_blob_t &remote_md) {
    std::string remote_agent_name;
    const nixl_status_t status = agent_.loadRemoteMD(remote_md, remote_agent_name);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to load remote metadata");
    }
    remoteAgentName_ = std::unique_ptr<std::string, std::function<void(std::string *)>>(
        new std::string(remote_agent_name), [this](std::string *agent_name) {
            this->agent_.invalidateRemoteMD(*agent_name);
            delete agent_name;
        });
}

nixlGpuXferReqH
agent::createGpuXferReq(const std::vector<memTypeArray<uint8_t>> &src_buffers,
                        const std::vector<memTypeArray<uint8_t>> &dst_buffers) {
    nixlXferReqH *xfer_req = createXferReq(src_buffers, dst_buffers);
    xferReq_ = std::unique_ptr<nixlXferReqH, std::function<void(nixlXferReqH *)>>(
        xfer_req, [this](nixlXferReqH *xfer_req) { this->agent_.releaseXferReq(xfer_req); });

    nixlGpuXferReqH gpu_xfer_req = nullptr;
    const nixl_status_t status = agent_.createGpuXferReq(*xfer_req, gpu_xfer_req);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to create GPU transfer request");
    }

    gpuXferReq_ = std::unique_ptr<void, std::function<void(nixlGpuXferReqH)>>(
        gpu_xfer_req,
        [this](nixlGpuXferReqH gpu_xfer_req) { this->agent_.releaseGpuXferReq(gpu_xfer_req); });
    return gpu_xfer_req;
}

[[nodiscard]] size_t
agent::getGpuSignalSize() const {
    nixl_opt_args_t extra_params = {.backends = {backendHandle_}};
    size_t signal_size;
    const nixl_status_t status = agent_.getGpuSignalSize(signal_size, &extra_params);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get GPU signal size");
    }
    return signal_size;
}

void
agent::prepGpuSignal(memTypeArray<uint8_t> &signal_buffer) {
    nixlDescList<nixlBlobDesc> signal_dlist(signal_buffer.memType());
    signal_dlist.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(signal_buffer.get()),
                                      signal_buffer.size(),
                                      uint64_t(device_id)));
    nixl_opt_args_t extra_params = {.backends = {backendHandle_}};
    const nixl_status_t status = agent_.prepGpuSignal(signal_dlist, &extra_params);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to prepare GPU signal");
    }
}

[[nodiscard]] nixlBackendH *
agent::createBackend(std::optional<unsigned> num_channels) {
    nixlBackendH *backendHandle;
    const nixl_status_t status =
        agent_.createBackend("UCX", createBackendParams(num_channels), backendHandle);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to create backend");
    }
    return backendHandle;
}

[[nodiscard]] nixlXferReqH *
agent::createXferReq(const std::vector<memTypeArray<uint8_t>> &src_buffers,
                     const std::vector<memTypeArray<uint8_t>> &dst_buffers) {
    if (!remoteAgentName_) {
        throw std::runtime_error("Remote agent name is not set");
    }

    nixl_opt_args_t extra_params;
    extra_params.hasNotif = true;
    extra_params.notifMsg = std::string(notification_message);
    extra_params.backends = {backendHandle_};
    nixlXferReqH *xfer_req = nullptr;
    const nixl_status_t status = agent_.createXferReq(NIXL_WRITE,
                                                      makeDescList<nixlBasicDesc>(src_buffers),
                                                      makeDescList<nixlBasicDesc>(dst_buffers),
                                                      *remoteAgentName_,
                                                      xfer_req,
                                                      &extra_params);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to create xfer request");
    }
    return xfer_req;
}
} // namespace nixl::device_api
