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

#include "nixl_p2p_metadata_backend.h"
#include "nixl_tcpstore_metadata_backend.h"

#if HAVE_ETCD
#include "nixl_etcd_metadata_backend.h"
#endif

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

// Select the single backend for this run. `use_etcd` / `use_tcpstore` mirror the
// agent's cached env checks, so the env is not read again here. A centralized
// store wins when configured; P2P is the default. ETCD and TCPStore are mutually
// exclusive because a run publishes to exactly one store.
[[nodiscard]] std::unique_ptr<nixlMetadataBackend>
makeBackend([[maybe_unused]] bool use_etcd, bool use_tcpstore, nixlMetadataContext &ctx) {
#if HAVE_ETCD
    if (use_etcd && use_tcpstore) {
        throw std::runtime_error(
            "NIXL_ETCD_ENDPOINTS and NIXL_TCPSTORE_ENDPOINTS are mutually exclusive");
    }
    if (use_etcd) {
        return std::make_unique<nixlEtcdMetadataBackend>(ctx);
    }
#endif
    if (use_tcpstore) {
        return std::make_unique<nixlTcpStoreMetadataBackend>(ctx);
    }
    return std::make_unique<nixlP2PMetadataBackend>(ctx);
}

} // namespace

nixlMDManager::nixlMDManager(nixlMetadataContext &ctx, bool use_etcd, bool use_tcpstore)
    : backend_(makeBackend(use_etcd, use_tcpstore, ctx)) {}

nixlMDManager::~nixlMDManager() = default;

nixl_status_t
nixlMDManager::sendLocalMD(const nixl_opt_args_t *extra_params) const {
    return backend_->sendLocal(extra_params);
}

nixl_status_t
nixlMDManager::sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                                  const nixl_opt_args_t *extra_params) const {
    return backend_->sendLocalPartial(descs, extra_params);
}

nixl_status_t
nixlMDManager::fetchRemoteMD(const std::string &remote_name,
                             const nixl_opt_args_t *extra_params) const {
    return backend_->fetchRemote(remote_name, extra_params);
}

nixl_status_t
nixlMDManager::invalidateLocalMD(const nixl_opt_args_t *extra_params) const {
    return backend_->invalidateLocal(extra_params);
}

std::string_view
nixlMDManager::backendName() const noexcept {
    return backend_->name();
}
