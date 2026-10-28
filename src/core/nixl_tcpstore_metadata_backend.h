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

#include "nixl_md_config.h"
#include "nixl_metadata_backend.h"

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <asio.hpp>

class nixlMetadataContext;
class nixlTcpStoreClientAsync;

/**
 * @class nixlTcpStoreMetadataBackend
 * @brief Centralized-store metadata backend over the c10d TCPStore protocol.
 *
 * Owns a nixlTcpStoreClientAsync (nixl_tcpstore_client_async.h) and runs its store
 * I/O asynchronously on the shared I/O thread: it reuses nixlMetadataContext for
 * serialization (getLocalMD / getLocalPartialMD) and cache load (loadRemoteMD),
 * and builds its own keys. It links no libtorch; it speaks the wire protocol
 * directly, so it interoperates with a torch.distributed.TCPStore server.
 *
 * There is no native watch. A fetch whose key is not published yet is kept
 * pending and re-probed periodically until its deadline, so the caller
 * can fetch then poll checkRemoteMD as it would with etcd.
 * Selected by nixlMDManager when NIXL_TCPSTORE_ENDPOINT is set.
 *
 * The I/O context and the client can be accessed from any thread. Other members
 * must be accessed from the I/O thread only to avoid explicit synchronization.
 */
class nixlTcpStoreMetadataBackend : public nixlMetadataBackend {
public:
    // Parses NIXL_TCPSTORE_ENDPOINT (host:port) and throws when it is malformed.
    // Does no I/O: start() connects on the shared I/O thread.
    nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx,
                                const nixlMDConfig &config,
                                asio::io_context &io_context);

    ~nixlTcpStoreMetadataBackend() override;

    [[nodiscard]] std::string_view
    name() const override {
        return "TCPStore";
    }

    // The store I/O and the pending-fetch retries both need a thread.
    [[nodiscard]] bool
    usesThread() const override {
        return true;
    }

    void
    start() override;

    void
    stop() override;

    [[nodiscard]] nixl_status_t
    sendLocal(const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    sendLocalPartial(const nixl_reg_dlist_t &descs, const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    fetchRemote(const std::string &remote_name, const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    invalidateLocal(const nixl_opt_args_t *extra_params) override;

private:
    // Forward declaration. The base class for metadata operations
    // that change the published metadata. This includes sendLocal,
    // sendLocalPartial, and invalidateLocal.
    class WriteTask;

    // Publish blob under key, tracking it so invalidateLocal can remove it.
    void
    publishKey(const std::string &key, const nixl_blob_t &blob);

    // Initiate one async fetch on the key.
    void
    initiateFetch(const std::string &remote_name, const std::string &key);

    // Retry pending fetches that have not yet succeeded.
    void
    retryPendingFetches();

    void
    postWriteTask(std::unique_ptr<WriteTask> task);

    void
    writeTaskCompleted(const WriteTask *completed_task);

    // Timeout for fetch operations.
    const std::chrono::milliseconds fetchTimeout_;

    // The metadata context.
    nixlMetadataContext &ctx_;

    // The I/O context.
    asio::io_context &ioContext_;
    // The TCPStore client.
    const std::unique_ptr<nixlTcpStoreClientAsync> client_;

    // Keys this agent has published; TCPStore has no recursive delete, so
    // invalidateLocal removes exactly these.
    std::unordered_set<std::string> publishedKeys_;

    // Keys pending fetch retries. Each key is mapped to its fetch deadline and remote name.
    struct pendingInfo {
        std::chrono::steady_clock::time_point deadline;
        std::string remoteName;
        bool fetchInProgress = false;
    };

    std::unordered_map<std::string, pendingInfo> pendingKeys_;
    // Timer for fetch retries.
    asio::steady_timer fetchRetryTimer_;

    // Queued metadata operations that change the published metadata.
    std::deque<std::unique_ptr<WriteTask>> writeQueue_;
    // The metadata operation that is currently being processed.
    std::unique_ptr<WriteTask> inflightTask_;

    // Whether stop() has been called.
    bool stopCalled_ = false;
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_METADATA_BACKEND_H
