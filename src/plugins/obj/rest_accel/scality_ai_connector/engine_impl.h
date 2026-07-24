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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H

#include "obj_backend.h"
#include "rest_accel/scality_ai_connector/rest_client.h"
#include "rest_accel/scality_ai_connector/rdma_token_client.h"
#include <unordered_map>

/**
 * Scality AI Connector RDMA Engine Implementation.
 * Provides RDMA-accelerated object storage operations using Scality's
 * AI Connector HTTP service. Implements nixlObjEngineImpl directly and
 * uses the cuObject API for GPU-direct storage operations.
 *
 * The RDMA descriptor obtained from cuObject is transmitted via the
 * x-scal-rdma custom HTTP header, enabling the connector to perform the data
 * transfer via RDMA instead of the HTTP body.
 */
class ScalityObjEngineImpl : public nixlObjEngineImpl {
public:
    /**
     * Constructor that initializes the Scality AI Connector engine.
     * Creates both the HTTP client and cuObject client for RDMA operations.
     *
     * @param init_params Backend initialization parameters
     */
    explicit ScalityObjEngineImpl(const nixlBackendInitParams *init_params);

    /**
     * Constructor that accepts an injected client (for testing).
     *
     * @param init_params Backend initialization parameters
     * @param connector_client Pre-configured client (can be mock for testing)
     */
    ScalityObjEngineImpl(const nixlBackendInitParams *init_params,
                         const std::shared_ptr<iRestClient> &connector_client);

    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

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
    /// Maps device IDs to object keys
    std::unordered_map<uint64_t, std::string> devIdToObjKey_;
    /// RDMA token client (DC via cuObjClient)
    const std::shared_ptr<iRdmaTokenClient> cuClient_;
    /// Scality AI Connector HTTP client with RDMA support
    const std::shared_ptr<iRestClient> connectorClient_;
    /// Number of CUDA devices, queried once at construction (0 if none).
    /// Used to spread DRAM MRs across GPUs so cuObject fans them across NICs.
    int gpuCount_ = 0;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H
