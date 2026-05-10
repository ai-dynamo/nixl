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
 * @file etcd_metadata_backend.h
 * @brief ETCD-backed metadata transport.
 *
 * The backend keys differ depending on identity mode (selected per-call by
 * the manager when it forms keys):
 *   - UUID_BACKED:            `{ns}/agents/[{label}/]{src_uuid}/{dst|null_agent}`
 *   - LEGACY_AGENT_NAME_KEYS: `{ns}/{name}/{label}` for agent-name-key
 *                              compatibility
 */
#ifndef NIXL_SRC_PLUGINS_METADATA_ETCD_METADATA_BACKEND_H
#define NIXL_SRC_PLUGINS_METADATA_ETCD_METADATA_BACKEND_H

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "metadata/metadata_backend.h"
#include "nixl_md.h"
#include "nixl_types.h"

#if HAVE_ETCD

namespace etcd {
class SyncClient;
class Watcher;
} // namespace etcd

class nixlAgent;

/**
 * @class nixlEtcdMetadataBackend
 * @brief ETCD implementation of the southbound metadata contract.
 *
 * Supports agent-name-key compatibility and UUID-backed key formation, plus
 * the existing watcher-on-prefix invalidation behaviour.
 */
class nixlEtcdMetadataBackend final : public nixlMetadataBackend {
public:
    nixlEtcdMetadataBackend(std::string my_agent_name, std::chrono::microseconds watch_timeout);
    ~nixlEtcdMetadataBackend() override;

    nixlEtcdMetadataBackend(nixlEtcdMetadataBackend &&) = delete;
    nixlEtcdMetadataBackend(const nixlEtcdMetadataBackend &) = delete;
    nixlEtcdMetadataBackend &
    operator=(nixlEtcdMetadataBackend &&) = delete;
    nixlEtcdMetadataBackend &
    operator=(const nixlEtcdMetadataBackend &) = delete;

    /*** nixlMetadataBackend ***/

    [[nodiscard]] nixl_status_t
    publish(const std::string &key, const nixl_blob_t &value) override;

    [[nodiscard]] nixl_status_t
    fetch(const std::string &key, nixl_blob_t &value) override;

    [[nodiscard]] nixl_status_t
    remove(const std::string &key) override;

    [[nodiscard]] nixl_status_t
    watch(const std::string &prefix, nixl_md_watch_cb_t cb) override;

    [[nodiscard]] bool
    isHealthy() const noexcept override;

    [[nodiscard]] nixl_status_t
    fetchBatch(const std::vector<std::string> &keys,
               std::vector<nixl_blob_t> &out,
               std::vector<nixl_status_t> &per_key_status) override;

    /*** Helpers driven by the manager tick ***/

    /** @brief Agent-name-key compatibility shape: `{namespace}/{agent_name}/{metadata_type}`. */
    [[nodiscard]] std::string
    legacyKey(const std::string &agent_name, const std::string &metadata_type) const;

    /**
     * @brief UUID-backed key formation:
     *        `{namespace}/agents/[{label}/]{src_uuid}/{dst|null_agent}`.
     */
    [[nodiscard]] std::string
    uuidBackedKey(const std::string &src_uuid,
                  const std::string &dst_name_or_null,
                  const std::string &metadata_label) const;

    /** @brief Fetch with watch fallback (waits up to `watch_timeout`). */
    [[nodiscard]] nixl_status_t
    fetchOrWait(const std::string &agent_name,
                const std::string &metadata_label,
                nixl_blob_t &value);

    /** @brief Subscribe to invalidation events under the given agent's prefix. */
    void
    setupAgentInvalWatcher(const std::string &agent_name);

    /**
     * @brief Drain any pending invalidations recorded by watchers and ask
     *        @p agent to invalidate the matching remote metadata.
     */
    void
    processInvalidatedAgents(nixlAgent &agent);

private:
    struct GenericWatcherState;
    struct InvalidationWatcherState;

    [[nodiscard]] std::string
    namespaceForAgent() const;

    void
    shutdownWatchers() noexcept;

    const std::string my_agent_name_;
    const std::string namespace_prefix_;
    const std::chrono::microseconds watch_timeout_;

    std::shared_ptr<InvalidationWatcherState> invalidation_state_;
    std::vector<std::shared_ptr<GenericWatcherState>> generic_watcher_states_;
    std::unique_ptr<etcd::SyncClient> etcd_;
    std::unordered_map<std::string, std::unique_ptr<etcd::Watcher>> agent_watchers_;
    std::vector<std::unique_ptr<etcd::Watcher>> generic_watchers_;
    bool healthy_ = false;
};

#endif // HAVE_ETCD

#endif // NIXL_SRC_PLUGINS_METADATA_ETCD_METADATA_BACKEND_H
