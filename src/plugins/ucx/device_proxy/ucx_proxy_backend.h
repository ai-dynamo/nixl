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
#include <mutex>
#include <string>
#include <unordered_map>

#include "backend/backend_aux.h"
#include "../../../core/device_proxy/backend_adapter.h"

class nixlUcxEngine;

class nixlUcxProxyBackendAdapter : public nixlDeviceProxyBackendAdapter {
public:
    explicit nixlUcxProxyBackendAdapter(nixlUcxEngine *engine = nullptr,
                                        bool progress_thread_enabled = false) noexcept
        : engine_(engine),
          progress_thread_enabled_(progress_thread_enabled) {}

    ~nixlUcxProxyBackendAdapter() override = default;

    nixl_status_t
    submit(const nixlBackendProxySubmission &submission, uint64_t &request_token) override;

    nixl_status_t
    checkCompletion(uint64_t request_token) override;

    nixl_status_t
    progress() override;

    nixl_status_t
    shutdown() override;

private:
    nixl_status_t
    submitPut(const nixlBackendProxySubmission &submission, uint64_t &request_token);

    nixl_status_t
    submitAtomicAdd(const nixlBackendProxySubmission &submission, uint64_t &request_token);

    uint64_t
    trackRequest(nixlBackendReqH *handle);

    nixlUcxEngine *engine_ = nullptr;
    bool progress_thread_enabled_ = false;
    std::mutex request_mutex_;
    std::unordered_map<uint64_t, nixlBackendReqH *> tracked_requests_;
    uint64_t next_request_token_ = 1;
};

#endif // NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H
