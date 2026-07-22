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
 * @file nixl_p2p_metadata_backend.h
 * @brief Point-to-point (socket) metadata backend.
 */
#ifndef NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H
#define NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H

#include "nixl_metadata_backend.h"

#include <string>
#include <string_view>

class nixlMetadataContext;

/**
 * @class nixlP2PMetadataBackend
 * @brief Socket-based metadata backend.
 *
 * Serializes via nixlMetadataContext (getLocalMD / getLocalPartialMD) and
 * enqueues socket work on the agent's existing comm thread. Depends only on
 * nixlMetadataContext, not nixlAgent. Duplicates the agent's inline socket path,
 * removed in a later PR.
 */
class nixlP2PMetadataBackend : public nixlMetadataBackend {
public:
    explicit nixlP2PMetadataBackend(nixlMetadataContext &ctx) noexcept;

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

private:
    nixlMetadataContext &ctx_;
};

#endif // NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H
