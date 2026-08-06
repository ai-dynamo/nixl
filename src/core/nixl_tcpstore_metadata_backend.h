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
#include "nixl_metadata_worker.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

class nixlMetadataContext;
class nixlTcpStoreClient;

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
    nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx, const nixlMDConfig &config);

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

    // Starts the worker and connects on it, so an unreachable store is reported
    // at bring-up without agent construction waiting for it.
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
    // Re-probe the fetches whose key was not published yet. Worker poll.
    void
    serviceEvents();

    // A fetch waiting for its key to appear in the store.
    struct pendingFetch {
        std::string remoteName;
        std::chrono::steady_clock::time_point deadline;
    };

    // Publish blob under key, tracking it so invalidateLocal can remove it.
    [[nodiscard]] nixl_status_t
    publishKey(const std::string &key, const nixl_blob_t &blob);

    // One fetch attempt. False means "not published yet, or the store was
    // unreachable" - i.e. worth retrying; true means the fetch is settled
    // (loaded, or rejected for a reason a retry cannot fix).
    [[nodiscard]] bool
    tryFetch(const std::string &remote_name, const std::string &key);

    nixlMetadataContext &ctx_;
    const std::chrono::milliseconds fetchTimeout_;
    const std::chrono::microseconds workerDelay_;
    const std::unique_ptr<nixlTcpStoreClient> client_;
    // Keys this agent has published; TCPStore has no recursive delete, so
    // invalidateLocal removes exactly these.
    std::unordered_set<std::string> publishedKeys_;
    // Store key -> in-flight fetch. Keyed by the store key, not the agent name:
    // one peer can have a fetch pending per metadata label.
    std::unordered_map<std::string, pendingFetch> pendingFetches_;
    // Declared last so it joins before the state its tasks touch is destroyed.
    nixlMetadataWorker worker_;
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_METADATA_BACKEND_H
