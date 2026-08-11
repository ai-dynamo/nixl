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
 * @file nixl_service.h
 * @brief nixlServiceAgent — compression, staging, and encryption services
 *        layered transparently on top of nixlAgent.
 */
#ifndef NIXL_SERVICE_H
#define NIXL_SERVICE_H

#include "nixl.h"
#include "nixl_service_types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class nixlServiceAgentData;
struct nixlServiceXferReqH;

namespace nixlService {
/**
 * @brief Recommend the service memory, in bytes per descriptor, to register for a given
 *        marshal mode.
 *
 * This is how a caller learns how much memory to hand to
 * nixlServiceAgent::registerServiceMem.
 *
 * @note The recommendation is a minimum: registering more is allowed and yields more slots.
 * @note Undersizing maxConcurrentTransfers is especially visible for NIXL_READ: a served
 * READ that finds the peer's slot pool exhausted is rejected outright (RNAK), not
 * queued, so back-to-back READs on a minimal pool can transiently fail. Size with
 * headroom for the actual number of concurrent in-flight transfers, not just 1.
 *
 * @param mode The marshal mode.
 * @param maxConcurrentTransfers The maximum expected number of concurrent transfers.
 * @return The recommended service memory size per descriptor, in bytes.
 * @throws std::invalid_argument for direct mode, which uses no service memory; for a zero
 *         maxConcurrentTransfers; and for compression when the library was built without
 *         nvCOMP.
 */
size_t
recommendServiceMemSize(const nixl_marshal_config_t &mode, uint32_t max_concurrent_transfers = 1);
}; // namespace nixlService

/**
 * @class nixlServiceAgent
 * @brief Extends nixlAgent with transparent marshalling.
 *
 * @note DIRECT mode is identical to the base class.
 */
class nixlServiceAgent : public nixlAgent {
public:
    static constexpr bool trackCompressionRatio = false;
    /**
     * @brief Construct a service agent.
     *
     * @param name  Agent name (same semantics as nixlAgent)
     * @param cfg   Service configuration (defaultMode, optional defaultCompAlg).
     */
    nixlServiceAgent(const std::string &name, nixlServiceAgentConfig cfg);

    /**
     * @brief Destructor.  Virtual via nixlAgent.
     */
    ~nixlServiceAgent() override;

    // Non-copyable, non-movable (same constraint as nixlAgent)
    nixlServiceAgent(const nixlServiceAgent &) = delete;
    nixlServiceAgent &
    operator=(const nixlServiceAgent &) = delete;
    nixlServiceAgent(nixlServiceAgent &&) = delete;
    nixlServiceAgent &
    operator=(nixlServiceAgent &&) = delete;

    /**
     * @brief  Register a staging buffer for the given service mode.
     *
     * Delegates to the marshal backend for transport registration and any
     * library-specific setup.
     *
     * @param  descs        Staging buffer descriptors
     * @param  extra_params Optional backend hints
     * @return nixl_status_t
     */
    nixl_status_t
    registerServiceMem(const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t *extra_params = nullptr);

    /**
     * @brief  Deregister previously registered staging memory.
     *         Must be followed by metadata re-exchange.
     */
    nixl_status_t
    deregisterServiceMem(const nixl_reg_dlist_t &descs,
                         const nixl_opt_args_t *extra_params = nullptr);

    /**
     * @brief  Create a service transfer request from two descriptor lists.
     *
     * @param  operation    NIXL_WRITE or NIXL_READ
     * @param  local_descs  Local descriptor list
     * @param  remote_descs Remote descriptor list
     * @param  remote_agent Remote agent name
     * @param  req_hndl     [out] Service transfer handle
     * @param  extra_params Optional per-transfer mode/algorithm overrides
     * @return nixl_status_t
     */
    nixl_status_t
    createXferReq(const nixl_xfer_op_t &operation,
                  const nixl_xfer_dlist_t &local_descs,
                  const nixl_xfer_dlist_t &remote_descs,
                  const std::string &remote_agent,
                  nixlServiceXferReqH *&req_hndl,
                  const nixl_service_opt_args_t *extra_params = nullptr);

    /**
     * @brief  Create a service transfer request from pre-prepared dlist handles.
     *
     * @param  operation      NIXL_WRITE or NIXL_READ
     * @param  local_side     Pre-prepared local dlist handle
     * @param  local_indices  Descriptor indices to use from local_side
     * @param  remote_side    Pre-prepared remote dlist handle
     * @param  remote_indices Descriptor indices to use from remote_side
     * @param  req_hndl       [out] Service transfer handle
     * @param  extra_params   Optional per-transfer overrides
     * @return nixl_status_t
     */
    nixl_status_t
    makeXferReq(const nixl_xfer_op_t &operation,
                const nixlDlistH *local_side,
                const std::vector<int> &local_indices,
                const nixlDlistH *remote_side,
                const std::vector<int> &remote_indices,
                nixlServiceXferReqH *&req_hndl,
                const nixl_service_opt_args_t *extra_params = nullptr);

    /**
     * @brief  Post a service transfer request, initiating the protocol.
     *
     * @param  req_hndl    Service transfer handle from createXferReq / makeXferReq
     * @param  extra_params Optional per-post overrides
     * @return nixl_status_t  NIXL_IN_PROG or error
     */
    nixl_status_t
    postXferReq(nixlServiceXferReqH *req_hndl,
                const nixl_service_opt_args_t *extra_params = nullptr);

    /**
     * @brief  Query transfer status, driving service progress in the process.
     *
     * @param  req_hndl  Service transfer handle
     * @return nixl_status_t  NIXL_SUCCESS (done), NIXL_IN_PROG, or error
     */
    nixl_status_t
    getXferStatus(nixlServiceXferReqH *req_hndl);

    /**
     * @brief  Release a service transfer handle and free all resources.
     *
     * @note   Releasing a NIXL_READ (or NIXL_WRITE) that is still in progress drains
     *         asynchronously rather than freeing synchronously: the destination buffer must
     *         not be reused until a prior getXferStatus() reported a terminal status, since
     *         in-flight local decodes/writes into it may briefly outlive this call.
     *
     * @param  req_hndl  Service transfer handle to release
     * @return nixl_status_t
     */
    nixl_status_t
    releaseXferReq(nixlServiceXferReqH *req_hndl);

    /**
     * @brief  Collect user-visible notifications, driving service progress first.
     *
     * @param  notifs       [in/out] Appended with non-service notifications
     * @param  extra_params Optional backend filter
     * @return nixl_status_t
     */
    nixl_status_t
    getNotifs(nixl_notifs_t &notifs, const nixl_opt_args_t *extra_params = nullptr);

    nixl_status_t
    createXferReq(const nixl_xfer_op_t &,
                  const nixl_xfer_dlist_t &,
                  const nixl_xfer_dlist_t &,
                  const std::string &,
                  nixlXferReqH *&,
                  const nixl_opt_args_t * = nullptr) const = delete;

    nixl_status_t
    makeXferReq(const nixl_xfer_op_t &,
                const nixlDlistH *,
                const std::vector<int> &,
                const nixlDlistH *,
                const std::vector<int> &,
                nixlXferReqH *&,
                const nixl_opt_args_t * = nullptr) const = delete;

    nixl_status_t
    postXferReq(nixlXferReqH *, const nixl_opt_args_t * = nullptr) const = delete;

    nixl_status_t
    getXferStatus(nixlXferReqH *) const = delete;

    nixl_status_t
    releaseXferReq(nixlXferReqH *) const = delete;

    nixl_status_t
    estimateXferCost(const nixlXferReqH *,
                     std::chrono::microseconds &,
                     std::chrono::microseconds &,
                     nixl_cost_t &,
                     const nixl_opt_args_t * = nullptr) const = delete;

    nixl_status_t
    getXferTelemetry(const nixlXferReqH *, nixl_xfer_telem_t &) const = delete;

private:
    /**
     * @brief  Pre-create nixlServiceAgentData.
     */
    static std::pair<nixlServiceAgentConfig, std::shared_ptr<nixlServiceAgentData>>
    prepare(nixlServiceAgentConfig cfg);

protected:
    /**
     * @brief  Pre-create nixlServiceAgentData with an explicit chunked payload size.
     *
     * @note   Test-only seam. The payload size is baked into the layout fingerprint, so
     *         production agents must go through the public constructor to stay compatible
     *         with their peers.
     */
    static std::pair<nixlServiceAgentConfig, std::shared_ptr<nixlServiceAgentData>>
    prepare(nixlServiceAgentConfig cfg, size_t chunked_payload_size);

    /**
     * @brief  Delegating constructor that receives the pre-built tag.
     */
    nixlServiceAgent(
        const std::string &name,
        std::pair<nixlServiceAgentConfig, std::shared_ptr<nixlServiceAgentData>> &&tag);

private:
    std::shared_ptr<nixlServiceAgentData> data_;
};

#endif // NIXL_SERVICE_H
