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

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

#include <asio.hpp>

class nixlMetadataContext;
class nixlTcpStoreClientAsync;

/**
 * @class nixlTcpStoreMetadataBackend
 * @brief Centralized-store metadata backend over the c10d TCPStore protocol.
 *
 * Owns a nixlTcpStoreClient (nixl_tcpstore_client.h) and runs its store I/O as
 * tasks on its own worker thread: it reuses nixlMetadataContext for
 * serialization (getLocalMD / getLocalPartialMD) and cache load (loadRemoteMD),
 * and builds its own keys. It links no libtorch; it speaks the wire protocol
 * directly, so it interoperates with a torch.distributed.TCPStore server.
 *
 * There is no native watch. A fetch whose key is not published yet is kept
 * pending and re-probed from serviceEvents() until its deadline, so the caller
 * can fetch then poll checkRemoteMD as it would with etcd.
 * Selected by nixlMDManager when NIXL_TCPSTORE_ENDPOINT is set.
 *
 * Every member here is touched only from that worker thread: the tasks the
 * operations submit and serviceEvents() both run there, and the thread always
 * runs. Nothing is synchronized.
 */
class nixlTcpStoreMetadataBackend : public nixlMetadataBackend {
public:
    // Parses NIXL_TCPSTORE_ENDPOINT (host:port) and throws when it is malformed.
    // Does no I/O: start() connects on the worker thread.
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
    // Publish blob under key, tracking it so invalidateLocal can remove it.
    void
    publishKey(const std::string &key, const nixl_blob_t &blob);

    // One fetch attempt.
    void
    tryFetch(const std::string &remote_name,
             const std::string &key,
             std::chrono::steady_clock::time_point deadline,
             std::chrono::milliseconds retry_delay);

    nixlMetadataContext &ctx_;
    const std::chrono::milliseconds fetchTimeout_;
    asio::io_context &ioContext_;
    const std::unique_ptr<nixlTcpStoreClientAsync> client_;
    // Keys this agent has published; TCPStore has no recursive delete, so
    // invalidateLocal removes exactly these.
    std::mutex publishedKeysMutex_;
    std::unordered_set<std::string> publishedKeys_;

    // Whether stop() has been called.
    std::atomic<bool> stopCalled_ = false;
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_METADATA_BACKEND_H
