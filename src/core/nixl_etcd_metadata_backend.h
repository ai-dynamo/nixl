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
 * @file nixl_etcd_metadata_backend.h
 * @brief ETCD (centralized key/value) metadata backend.
 */
#ifndef NIXL_SRC_CORE_NIXL_ETCD_METADATA_BACKEND_H
#define NIXL_SRC_CORE_NIXL_ETCD_METADATA_BACKEND_H

#if HAVE_ETCD

#include "nixl_md_config.h"
#include "nixl_metadata_backend.h"
#include "nixl_metadata_worker.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

class nixlMetadataContext;

/**
 * @class nixlEtcdMetadataBackend
 * @brief Self-contained centralized-store metadata backend (etcd).
 *
 * Owns its own nixlEtcdClient (connection + watchers). Outbound ops reuse the
 * context's serialization (getLocalMD / getLocalPartialMD) and submit the etcd
 * I/O as tasks on its own worker thread; watch-driven invalidations are drained
 * in serviceEvents() on that thread. Depends only on nixlMetadataContext, not
 * nixlAgent. Selected by nixlMDManager when NIXL_ETCD_ENDPOINTS is set.
 */
class nixlEtcdMetadataBackend : public nixlMetadataBackend {
public:
    // Builds the etcd client without contacting the store; start() announces
    // this agent on the worker thread, which is what connects.
    nixlEtcdMetadataBackend(nixlMetadataContext &ctx, const nixlMDConfig &config);
    ~nixlEtcdMetadataBackend() override;

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

    // The etcd I/O and the watch-invalidation drain both need a thread.
    [[nodiscard]] bool
    usesThread() const override {
        return true;
    }

    // Starts the worker and announces this agent on it, so an unreachable store
    // is reported at bring-up without agent construction waiting for it.
    void
    start() override;

    void
    stop() override;

private:
    // etcd connection and watchers; defined in the .cpp, nested so the type is
    // private to this backend.
    class etcdClient;

    // Drain the invalidations the watchers queued. Worker poll.
    void
    serviceEvents();

    nixlMetadataContext &ctx_;
    const std::chrono::microseconds workerDelay_;
    const std::unique_ptr<etcdClient> client_;
    // Declared last so it joins before the state its tasks touch is destroyed.
    nixlMetadataWorker worker_;
};

#endif // HAVE_ETCD

#endif // NIXL_SRC_CORE_NIXL_ETCD_METADATA_BACKEND_H
