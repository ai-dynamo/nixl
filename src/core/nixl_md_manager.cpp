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

#include <memory>
#include <string>
#include <string_view>

nixlMDManager::nixlMDManager(nixlMetadataContext &ctx)
    : backend_(std::make_unique<nixlP2PMetadataBackend>(ctx)) {}

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
