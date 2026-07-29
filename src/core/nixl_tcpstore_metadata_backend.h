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

#include <chrono>
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
 * tasks on the manager's worker thread: it reuses nixlMetadataContext for
 * serialization (getLocalMD / getLocalPartialMD) and cache load (loadRemoteMD),
 * and builds its own keys. It links no libtorch; it speaks the wire protocol
 * directly, so it interoperates with a torch.distributed.TCPStore server.
 *
 * There is no native watch. A fetch whose key is not published yet is kept
 * pending and re-probed from serviceEvents() until its deadline, so the caller
 * can fetch then poll checkRemoteMD as it would with etcd.
 * Selected by nixlMDManager when NIXL_TCPSTORE_ENDPOINT is set.
 *
 * Every member here is touched only from the manager's worker thread: the tasks
 * returned by prepare* and serviceEvents() both run there, and this backend
 * always requires the worker. Nothing is synchronized.
 */
class nixlTcpStoreMetadataBackend : public nixlMetadataBackend {
public:
    // Parses NIXL_TCPSTORE_ENDPOINT (host:port) and throws when it is malformed.
    // Does no I/O: the client connects on its first operation.
    explicit nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx);

    ~nixlTcpStoreMetadataBackend() override;

    [[nodiscard]] std::string_view
    name() const override {
        return "TCPStore";
    }

    // Ops run their store I/O on the manager's worker thread, which also drives
    // the pending-fetch retries.
    [[nodiscard]] bool
    needsWorker() const override {
        return true;
    }

    // Re-probe the fetches whose key was not published yet.
    void
    serviceEvents() override;

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
    const std::unique_ptr<nixlTcpStoreClient> client_;
    // Keys this agent has published; TCPStore has no recursive delete, so
    // invalidateLocal removes exactly these.
    std::unordered_set<std::string> publishedKeys_;
    // Store key -> in-flight fetch. Keyed by the store key, not the agent name:
    // one peer can have a fetch pending per metadata label.
    std::unordered_map<std::string, pendingFetch> pendingFetches_;
};

#endif // NIXL_SRC_CORE_NIXL_TCPSTORE_METADATA_BACKEND_H
