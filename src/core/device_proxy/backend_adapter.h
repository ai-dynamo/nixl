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
#include <vector>

#include <nixl_types.h>
#include "backend_aux.h"
#include "proxy_protocol.h"

/** @brief Describes one local or remote endpoint of a proxy transfer. */
struct nixlBackendProxyXferDesc {
    nixl_mem_t memType = VRAM_SEG;
    nixlMetaDesc desc{};
};

/** @brief Describes a proxy operation submitted to a backend adapter. */
struct nixlBackendProxySubmission {
    uint64_t opIdx = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint32_t channelId = 0;
    uint32_t peerIndex = 0;
    uint64_t flags = 0;

    nixlBackendProxyXferDesc local{};
    nixlBackendProxyXferDesc remote{};
    std::string remoteAgent;

    size_t size = 0;
    uint64_t value = 0;
};

/** @brief Identifies a backend operation that is still in progress. */
struct nixlBackendProxyRequest {
    uint64_t token = 0;
    size_t context = 0;

    explicit
    operator bool() const {
        return token != 0;
    }
};

/** @brief Backend contract used by the CPU proxy runtime. */
class nixlDeviceProxyBackendAdapter {
public:
    virtual ~nixlDeviceProxyBackendAdapter() = default;

    /** @brief Initialize backend resources for the requested proxy topology. */
    virtual nixl_status_t
    init(uint32_t proxy_worker_count, uint32_t channel_count, uint32_t max_peers) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    /** @brief Import a remote agent's connection information. */
    virtual nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    /** @brief Resolve directly accessible pointers from remote metadata. */
    virtual nixl_status_t
    resolveDirectPointers(const nixl_remote_meta_dlist_t &, std::vector<void *> &) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    /** @brief Submit one proxy operation to the backend. */
    virtual nixl_status_t
    submit(const nixlBackendProxySubmission &submission, nixlBackendProxyRequest &request) = 0;

    /** @brief Poll a previously submitted backend operation. */
    virtual nixl_status_t
    checkCompletion(const nixlBackendProxyRequest &request) = 0;

    /** @brief Release resources associated with a completed backend request. */
    virtual void
    releaseRequest(const nixlBackendProxyRequest &) {}

    /** @brief Progress backend work without channel-specific context. */
    virtual nixl_status_t
    progress() {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    /** @brief Progress backend work for one peer/channel pair. */
    virtual nixl_status_t
    progress(uint32_t, uint32_t) {
        return progress();
    }

    /** @brief Shut down backend resources. */
    virtual nixl_status_t
    shutdown() {
        return NIXL_ERR_NOT_SUPPORTED;
    }
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H
