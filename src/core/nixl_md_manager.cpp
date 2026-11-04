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
#include "nixl_md_manager.h"

#include "agent_data.h"
#include "nixl_p2p_metadata_backend.h"
#include "nixl_tcpstore_metadata_backend.h"

#if HAVE_ETCD
#include "nixl_etcd_metadata_backend.h"
#endif

#include "common/configuration.h"
#include "common/nixl_log.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

// Definition of the default metadata label (declared in nixl_types.h). It used
// to live in the now-deleted nixl_listener.cpp; the manager is a natural home.
const std::string default_metadata_label = "metadata";

namespace {

// The name-addressed backend for this run, chosen from the environment (null
// when none is configured, i.e. address-only / P2P). Adding a name-addressed
// transport is one class plus a branch here; it is not tied to any storage kind.
// ETCD and TCPStore are mutually exclusive (a run uses exactly one).
[[nodiscard]] std::unique_ptr<nixlMetadataBackend>
makeBackend([[maybe_unused]] nixlMetadataContext &ctx,
            [[maybe_unused]] const nixlMDConfig &config) {
    const bool use_tcpstore = nixl::config::checkExistence("NIXL_TCPSTORE_ENDPOINT");
#if HAVE_ETCD
    if (nixlMDManager::etcdConfigured()) {
        if (use_tcpstore) {
            throw std::runtime_error(
                "NIXL_ETCD_ENDPOINTS and NIXL_TCPSTORE_ENDPOINT are mutually exclusive");
        }
        return std::make_unique<nixlEtcdMetadataBackend>(ctx, config);
    }
#endif
    if (use_tcpstore) {
        return std::make_unique<nixlTcpStoreMetadataBackend>(ctx, config);
    }
    return nullptr;
}

// A call is address-routed (P2P) when it carries a peer address; otherwise it is
// name-addressed and handled by the configured backend.
[[nodiscard]] bool
hasAddress(const nixl_opt_args_t *extra_params) {
    return extra_params && !extra_params->ipAddr.empty();
}

// Error when a call carries no peer address and no name-addressed backend
// (etcd/tcpstore) is configured.
[[nodiscard]] nixl_status_t
noTransport() {
#if HAVE_ETCD
    NIXL_ERROR_FUNC << "no peer address provided and no centralized store configured "
                       "(set NIXL_ETCD_ENDPOINTS or NIXL_TCPSTORE_ENDPOINT)";
    return NIXL_ERR_INVALID_PARAM;
#else
    NIXL_ERROR_FUNC << "no peer address provided and no centralized store configured "
                       "(set NIXL_TCPSTORE_ENDPOINT; this build has no ETCD)";
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

} // namespace

bool
nixlMDManager::etcdConfigured() {
#if HAVE_ETCD
    return nixl::config::checkExistence("NIXL_ETCD_ENDPOINTS");
#else
    return false;
#endif
}

// The manager holds the P2P backend (always) plus an optional name-addressed
// backend, and routes each call by precedence: a peer address selects P2P,
// otherwise the configured backend. This preserves the agent's original per-call
// precedence (address wins over a configured backend).
nixlMDManager::nixlMDManager(nixlMetadataContext &ctx, const nixlMDConfig &config)
    : p2pBackend_(std::make_unique<nixlP2PMetadataBackend>(ctx, config)),
      backend_(makeBackend(ctx, config)) {}

// Each backend joins its own thread as it is destroyed, so nothing to do here.
nixlMDManager::~nixlMDManager() = default;

nixlMetadataBackend *
nixlMDManager::select(const nixl_opt_args_t *extra_params) const noexcept {
    return hasAddress(extra_params) ? p2pBackend_.get() : backend_.get();
}

nixl_status_t
nixlMDManager::sendLocalMD(const nixl_opt_args_t *extra_params) {
    nixlMetadataBackend *b = select(extra_params);
    return b ? b->sendLocal(extra_params) : noTransport();
}

nixl_status_t
nixlMDManager::sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                                  const nixl_opt_args_t *extra_params) {
    nixlMetadataBackend *b = select(extra_params);
    return b ? b->sendLocalPartial(descs, extra_params) : noTransport();
}

nixl_status_t
nixlMDManager::fetchRemoteMD(const std::string &remote_name, const nixl_opt_args_t *extra_params) {
    nixlMetadataBackend *b = select(extra_params);
    return b ? b->fetchRemote(remote_name, extra_params) : noTransport();
}

nixl_status_t
nixlMDManager::invalidateLocalMD(const nixl_opt_args_t *extra_params) {
    nixlMetadataBackend *b = select(extra_params);
    return b ? b->invalidateLocal(extra_params) : noTransport();
}

std::string_view
nixlMDManager::backendName() const noexcept {
    return backend_ ? backend_->name() : p2pBackend_->name();
}

bool
nixlMDManager::usesThread() const noexcept {
    // The backends answer from configuration they already hold.
    return p2pBackend_->usesThread() || (backend_ && backend_->usesThread());
}

void
nixlMDManager::start() {
    p2pBackend_->start();
    if (backend_) {
        backend_->start();
    }
}

void
nixlMDManager::stop() {
    p2pBackend_->stop();
    if (backend_) {
        backend_->stop();
    }
}
