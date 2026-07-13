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
 * @file nixl_p2p_metadata_backend.h
 * @brief Point-to-point (socket) metadata backend.
 */
#ifndef NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H
#define NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H

#include "nixl_metadata_backend.h"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

class nixlMetadataContext;
class nixlMDStreamListener;

/**
 * @class nixlP2PMetadataBackend
 * @brief Self-contained socket-based metadata backend.
 *
 * Owns its transport state: the open peer connections and (when the agent
 * enables listening) the accept socket. Outbound ops validate and serialize
 * synchronously, then submit the socket send as a task on the manager's worker
 * thread; inbound work (accepting peers, reading LOAD/SEND/INVL replies) happens
 * in serviceEvents(), also on that worker thread, so the connection map is only
 * ever touched by one thread. Depends only on nixlMetadataContext, not nixlAgent.
 */
class nixlP2PMetadataBackend : public nixlMetadataBackend {
public:
    explicit nixlP2PMetadataBackend(nixlMetadataContext &ctx);
    ~nixlP2PMetadataBackend() override;

    [[nodiscard]] std::string_view
    name() const override;

    [[nodiscard]] nixlPreparedOp
    prepareSendLocal(const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixlPreparedOp
    prepareSendLocalPartial(const nixl_reg_dlist_t &descs,
                            const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixlPreparedOp
    prepareFetchRemote(const std::string &remote_name,
                       const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixlPreparedOp
    prepareInvalidateLocal(const nixl_opt_args_t *extra_params) override;

    // The worker is needed for inbound servicing when listening is enabled.
    [[nodiscard]] bool
    needsWorker() const override;

    // Accept new peers (if listening) and read/dispatch incoming messages.
    void
    serviceEvents() override;

private:
    // Connect-on-demand to (ip, port) and send msg; disconnect on error.
    // Runs only on the worker thread (submitted task).
    void
    sendToPeer(const std::string &ip, int port, const std::string &msg);
    void
    acceptPeers();
    void
    readIncoming();

    nixlMetadataContext &ctx_;
    std::map<std::pair<std::string, int>, int> remoteSockets_;
    std::unique_ptr<nixlMDStreamListener> listener_;
};

#endif // NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H
