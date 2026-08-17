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
/**
 * @file nixl_metadata_context.h
 * @brief Core-internal interface through which metadata backends reach the agent.
 */
#ifndef NIXL_SRC_CORE_NIXL_METADATA_CONTEXT_H
#define NIXL_SRC_CORE_NIXL_METADATA_CONTEXT_H

#include "nixl_descriptors.h"
#include "nixl_types.h"

#include <string>

/**
 * @class nixlMetadataContext
 * @brief Core-internal interface: the agent-side operations a metadata backend
 *        needs.
 *
 * Implemented by nixlAgentData. The agent still owns the manager that holds this
 * reference, so the object graph is unchanged; what the interface buys is a
 * narrow surface (backends reach these operations and nothing else on
 * nixlAgentData, with no friendship) and a header boundary, since a backend
 * includes this file rather than agent_data.h.
 *
 * Settings are not exposed here: the manager and each backend take the
 * nixlMDConfig they need as a constructor parameter.
 */
class nixlMetadataContext {
public:
    virtual ~nixlMetadataContext() = default;

    /** Serialize this agent's full local metadata blob. */
    [[nodiscard]] virtual nixl_status_t
    getLocalMD(nixl_blob_t &blob) = 0;

    /** Serialize a partial local metadata blob for the given descriptors. */
    [[nodiscard]] virtual nixl_status_t
    getLocalPartialMD(const nixl_reg_dlist_t &descs,
                      nixl_blob_t &blob,
                      const nixl_opt_args_t *extra_params) = 0;

    /** This agent's name; used by centralized backends to build their KV keys. */
    [[nodiscard]] virtual const std::string &
    agentName() const noexcept = 0;

    /**
     * Deserialize a received metadata blob into the remote-section cache,
     * returning the embedded remote agent name. Used by every backend to load a
     * fetched/received blob.
     */
    [[nodiscard]] virtual nixl_status_t
    loadRemoteMD(const nixl_blob_t &blob, std::string &out_name) = 0;

    /** Evict a remote agent's cached metadata (used by inbound INVL / watches). */
    virtual nixl_status_t
    invalidateRemoteMD(const std::string &remote_name) = 0;
};

#endif // NIXL_SRC_CORE_NIXL_METADATA_CONTEXT_H
