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
 * @file nixl_tcpstore_metadata_backend.h
 * @brief TCPStore (centralized key/value) metadata backend.
 */
#ifndef NIXL_SRC_CORE_NIXL_TCPSTORE_METADATA_BACKEND_H
#define NIXL_SRC_CORE_NIXL_TCPSTORE_METADATA_BACKEND_H

#include "nixl_metadata_backend.h"

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

class nixlMetadataContext;
class nixlTcpStoreClient;

/**
 * @class nixlTcpStoreMetadataBackend
 * @brief Centralized-store metadata backend over the c10d TCPStore protocol.
 *
 * Unlike the P2P/ETCD backends (which enqueue work on the agent's comm thread),
 * this backend owns a nixlTcpStoreClient (nixl_tcpstore_client.h) and does its
 * store I/O synchronously: it reuses nixlMetadataContext for serialization
 * (getLocalMD / getLocalPartialMD) and cache load (loadRemoteMD), and builds
 * its own keys. It links no libtorch; it speaks the wire protocol directly, so
 * it interoperates with a torch.distributed.TCPStore server.
 *
 * There is no native watch: fetchRemote loads synchronously, so a subsequent
 * checkRemoteMD reports readiness. When the peer has not published yet,
 * fetchRemote returns NIXL_ERR_NOT_FOUND and the caller re-initiates.
 * Selected by nixlMDManager when NIXL_TCPSTORE_ENDPOINTS is set.
 */
class nixlTcpStoreMetadataBackend : public nixlMetadataBackend {
public:
    // Health gate: reads NIXL_TCPSTORE_ENDPOINTS (host:port), connects the
    // client, and throws on failure.
    explicit nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx);

    ~nixlTcpStoreMetadataBackend() override;

    [[nodiscard]] std::string_view
    name() const override {
        return "TCPStore";
    }

    // Ops run their store I/O on the manager's worker thread (no background poll).
    [[nodiscard]] bool
    needsWorker() const override {
        return true;
    }

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

private:
    // Publish blob under key, tracking it so invalidateLocal can remove it.
    [[nodiscard]] nixl_status_t
    publishKey(const std::string &key, const nixl_blob_t &blob);

    nixlMetadataContext &ctx_;
    std::unique_ptr<nixlTcpStoreClient> client_;
    std::mutex publishedMutex_;
    // Keys this agent has published; TCPStore has no recursive delete, so
    // invalidateLocal removes exactly these.
    std::unordered_set<std::string> publishedKeys_;
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_METADATA_BACKEND_H
