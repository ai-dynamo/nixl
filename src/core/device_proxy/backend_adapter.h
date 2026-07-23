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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H
#define NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H

#include <cstddef>
#include <cstdint>
#include <string>

#include <nixl_types.h>
#include "backend_aux.h"
#include "proxy_protocol.h"

struct nixlBackendProxyXferDesc {
    nixl_mem_t mem_type = VRAM_SEG;
    nixlMetaDesc desc{};
};

struct nixlBackendProxySubmission {
    uint64_t op_idx = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint32_t channel_id = 0;
    /** Destination peer row this submission drains; selects the (channel, peer) UCX worker. */
    uint32_t peer_index = 0;
    uint64_t flags = 0;

    nixlBackendProxyXferDesc local{};
    nixlBackendProxyXferDesc remote{};
    std::string remote_agent;

    size_t size = 0;
    uint64_t value = 0;
};

class nixlDeviceProxyBackendAdapter {
public:
    virtual ~nixlDeviceProxyBackendAdapter() = default;

    /**
     * Initialize backend-specific proxy resources.
     *
     * proxy_worker_count is the number of CPU drain threads. channel_count is the
     * number of logical channels; peer_capacity is the number of destination peer
     * rows per channel. The backend provisions one worker/QP per (channel, peer),
     * so it needs channel_count * peer_capacity transport workers. The three
     * dimensions are otherwise independent.
     */
    virtual nixl_status_t
    init(uint32_t proxy_worker_count, uint32_t channel_count, uint32_t peer_capacity) {
        (void)proxy_worker_count;
        (void)channel_count;
        (void)peer_capacity;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    virtual nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    virtual nixl_status_t
    submit(const nixlBackendProxySubmission &submission, uint64_t &request_token) = 0;

    virtual nixl_status_t
    checkCompletion(uint64_t request_token) = 0;

    virtual nixl_status_t
    progress() {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    /** Progress backend work for a single (channel, peer) ring and its UCX worker. */
    virtual nixl_status_t
    progress(uint32_t /*channel_id*/, uint32_t /*peer_index*/) {
        return progress();
    }

    virtual nixl_status_t
    shutdown() {
        return NIXL_ERR_NOT_SUPPORTED;
    }
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H
