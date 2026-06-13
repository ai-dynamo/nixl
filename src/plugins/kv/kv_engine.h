/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file kv_engine.h
 * @brief Shared thin wrapper base class for KV-style NIXL backend engines.
 *
 * nixlKVEngine implements nixlBackendEngine and forwards all data-plane calls
 * to a nixlKVEngineImpl instance supplied by each plugin.
 *
 * Plugin authors subclass nixlKVEngine (or use it directly with a factory) and
 * register the resulting engine with nixlBackendPluginCreator.
 */

#ifndef NIXL_SRC_PLUGINS_KV_KV_ENGINE_H
#define NIXL_SRC_PLUGINS_KV_KV_ENGINE_H

#include "kv_engine_impl.h"
#include <memory>

/**
 * @class nixlKVEngine
 * @brief Thin nixlBackendEngine wrapper that delegates to nixlKVEngineImpl.
 *
 * Common KV backend capabilities (local-only, no notifications) are declared here
 * so individual plugins only need to supply a concrete nixlKVEngineImpl.
 */
class nixlKVEngine : public nixlBackendEngine {
public:
    /**
     * @brief Construct a KV engine with a plugin-specific implementation.
     * @param init_params NIXL initialization parameters.
     * @param impl Concrete nixlKVEngineImpl (ownership transferred).
     */
    nixlKVEngine(const nixlBackendInitParams *init_params, std::unique_ptr<nixlKVEngineImpl> impl);

    ~nixlKVEngine() override;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    /**
     * @brief Passes local metadata through unchanged.
     *
     * Local-only KV backends do not transform metadata. input must be the
     * nixlBackendMD* returned by registerMem().
     */
    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        if (input == nullptr) {
            output = nullptr;
            return NIXL_ERR_INVALID_PARAM;
        }
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

private:
    std::unique_ptr<nixlKVEngineImpl> impl_;
};

#endif // NIXL_SRC_PLUGINS_KV_KV_ENGINE_H
