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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H

#include "obj_backend.h"
#include "rest_accel/scality_ai_connector/client.h"
#include "rest_accel/scality_ai_connector/rdma_descriptor_provider.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

/// DC access key used when neither 'rdma_dc_key' nor cufile.json sets one.
constexpr uint64_t default_dc_key = 0xffeeddccULL;

/// Defaults for 'rdma_sl' and 'rdma_traffic_class': NVIDIA's standard lossless
/// RoCE setup, with RoCE on priority 3 (PFC enabled) and marked DSCP 26. SL 3
/// selects that priority under PCP trust; traffic class 106 is DSCP 26 (104)
/// plus the ECN-capable bit, which selects it under DSCP trust.
constexpr uint8_t default_rdma_sl = 3;
constexpr uint8_t default_rdma_traffic_class = 106;

/**
 * OBJ engine for the Scality AI Connector.
 *
 * Object bytes move by RDMA between the local buffer (DRAM or VRAM) and the
 * server, which performs the one-sided transfer itself. A header-only HTTP
 * request names the object and carries the buffer's RDMA descriptor in the
 * x-scal-rdma header.
 */
class ScalityObjEngineImpl : public nixlObjEngineImpl {
public:
    explicit ScalityObjEngineImpl(const nixlBackendInitParams *init_params);

    /**
     * Constructor that accepts an injected REST client (for testing).
     *
     * @param init_params Backend initialization parameters
     * @param connector_client Pre-configured client (can be mock for testing)
     */
    ScalityObjEngineImpl(const nixlBackendInitParams *init_params,
                         std::shared_ptr<iRestClient> connector_client);

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
    /// Build the DC descriptor provider from the resolved NIC list, once. Deferred
    /// to the first DRAM/VRAM registration so a backend that only queries objects
    /// needs no NICs.
    nixl_status_t
    ensureDescriptorProvider();

    /// Maps device IDs to object keys
    std::unordered_map<uint64_t, std::string> devIdToObjKey_;
    std::shared_ptr<iRdmaDescriptorProvider> descriptorProvider_;
    std::mutex descriptorProviderMu_;
    /// RDMA NIC specifiers (IPv4 or device names), resolved at construction from
    /// customParams 'rdma_nics' or cufile.json.
    std::vector<std::string> rdmaNics_;
    uint64_t dcKey_ = default_dc_key;
    uint8_t rdmaSl_ = default_rdma_sl;
    uint8_t rdmaTrafficClass_ = default_rdma_traffic_class;
    std::shared_ptr<iRestClient> connectorClient_;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H
