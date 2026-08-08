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

#include "nixl_metadata_backend.h"

#include <string>
#include <string_view>

class nixlMetadataContext;

/**
 * @class nixlEtcdMetadataBackend
 * @brief Centralized-store metadata backend.
 *
 * Like the P2P backend, it implements the exchange by reusing the metadata
 * port's serialization (getLocalMD / getLocalPartialMD) and enqueuing ETCD work
 * on the agent's existing communication thread via the port. It depends only on
 * nixlMetadataContext (not nixlAgent) and does not call the agent's public
 * send/fetch methods. Selected by nixlMDManager when NIXL_ETCD_ENDPOINTS is set.
 */
class nixlEtcdMetadataBackend : public nixlMetadataBackend {
public:
    explicit nixlEtcdMetadataBackend(nixlMetadataContext &ctx) noexcept;

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

#endif // HAVE_ETCD

#endif // NIXL_SRC_CORE_NIXL_ETCD_METADATA_BACKEND_H
