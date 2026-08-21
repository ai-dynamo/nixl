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
#ifndef NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H
#define NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H

#include <cstdint>
#include <string>

#include "../../../core/device_proxy/backend_adapter.h"

class nixlUcxEngine;

class nixlUcxProxyBackendAdapter : public nixlDeviceProxyBackendAdapter {
public:
    explicit nixlUcxProxyBackendAdapter(nixlUcxEngine *engine = nullptr) noexcept
        : engine_(engine) {}

    ~nixlUcxProxyBackendAdapter() override = default;

    nixl_status_t
    init(uint32_t proxy_worker_count, uint32_t channel_count, uint32_t peer_capacity) override;

    nixl_status_t
    resolveDirectPointers(const nixl_remote_meta_dlist_t &dlist,
                          std::vector<void *> &direct_ptrs) override;

    nixl_status_t
    submit(const nixlBackendProxySubmission &submission, nixlBackendProxyRequest &request) override;

    nixl_status_t
    checkCompletion(const nixlBackendProxyRequest &request) override;

    void
    releaseRequest(const nixlBackendProxyRequest &request) override;

    nixl_status_t
    progress() override;

    nixl_status_t
    progress(uint32_t channel_id, uint32_t peer_index) override;

    nixl_status_t
    shutdown() override;

private:
    size_t
    getSharedWorkerIdForChannelPeer(uint32_t channel_id, uint32_t peer_index) const;

    nixl_status_t
    submitPut(const nixlBackendProxySubmission &submission, nixlBackendProxyRequest &request);

    nixl_status_t
    submitAtomicAdd(const nixlBackendProxySubmission &submission, nixlBackendProxyRequest &request);

    nixlUcxEngine *engine_ = nullptr;
    uint32_t peer_capacity_ = 0;
};

#endif // NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H
