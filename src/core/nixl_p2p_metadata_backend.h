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

#include "nixl_md_config.h"
#include "nixl_metadata_backend.h"
#include "nixl_metadata_worker.h"

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
 * synchronously, then submit the socket send to its worker; inbound work
 * (accepting peers, reading LOAD/SEND/INVL replies) happens in serviceEvents()
 * on the same thread, so the connection map is only ever touched by one thread.
 * A listener is what there is to service, so the worker runs only then: without
 * one the sends run inline on the caller thread and no thread is started.
 * Depends only on nixlMetadataContext, not nixlAgent.
 */
class nixlP2PMetadataBackend : public nixlMetadataBackend {
public:
    nixlP2PMetadataBackend(nixlMetadataContext &ctx, const nixlMDConfig &config);
    ~nixlP2PMetadataBackend() override;

    [[nodiscard]] std::string_view
    name() const override;

    [[nodiscard]] nixl_status_t
    sendLocal(const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    sendLocalPartial(const nixl_reg_dlist_t &descs, const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    fetchRemote(const std::string &remote_name, const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    invalidateLocal(const nixl_opt_args_t *extra_params) override;

    // A thread runs only when there is a listener to service.
    [[nodiscard]] bool
    usesThread() const override;

    void
    start() override;

    void
    stop() override;

private:
    // Accept new peers and read/dispatch incoming messages. Worker poll.
    void
    serviceEvents();

    // Connect-on-demand to (ip, port) and send msg; disconnect on error. Runs as
    // a worker task, so never concurrently with serviceEvents().
    void
    sendToPeer(const std::string &ip, int port, const std::string &msg);
    void
    acceptPeers();
    void
    readIncoming();

    nixlMetadataContext &ctx_;
    const nixlMDConfig config_;
    std::map<std::pair<std::string, int>, int> remoteSockets_;
    std::unique_ptr<nixlMDStreamListener> listener_;
    // Declared last so it joins before the state its tasks touch is destroyed.
    nixlMetadataWorker worker_;
};

#endif // NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H
