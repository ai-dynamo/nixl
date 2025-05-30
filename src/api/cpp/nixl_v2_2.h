/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#ifndef _NIXL_H
#define _NIXL_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nixl {

class Config {
};

class MemDesc {
public:
    enum class MemType { DRAM, VRAM, BLK, OBJ, FILE, PODMEM };

    // Local memory
    MemDesc(uint64_t address, size_t length, MemType type, uint64_t dev_id);

    // Remote memory
    MemDesc(uint64_t address, size_t length, MemType type, uint64_t dev_id,
            std::string_view agent_name);
};

class Agent {
public:
    class XferList;
    class XferRequest;
    class OptionalParams;

    Agent(std::string_view name, const Config &cfg);
    ~Agent();

    ///////////////////////////////////////////////////////////////////
    // Metadata / connection setup

    // Maybe return remote agent name from metadata
    std::string getMetadata();
    void connect(const std::string &remote_metadata);


    // TODO support storage get-closest APIs (Vish)
    // TODO How callbacks/no callbacks will map to Rust API (Tim)

    ///////////////////////////////////////////////////////////////////
    // Prepare transfer list

    // Prepare local and remote buffers for transfer:
    // - Register local buffers
    // - Potentially calculate the communication pattern and breakdown to
    //   collective operations
    // - Potentially send a message to the remote agent(s) asking to resolve the
    //   remote addresses to memory access keys, or use one-sided RDMA_READ
    //   to resolve remote keys.
    // - List sizes and corresponding buffer sizes are the same.
    // We considered add signal list / inline with remote dst, but for now we
    // decided to only keep the separate message API.
    XferList makeXferList(const std::vector<MemDesc> &dst_list,
                          const std::vector<MemDesc> &src_list,
                          const OptionalParams &params);

    // Estimate the time to complete the transfer, in seconds.
    // In case of storage it uses the storage SLA from vendor SDK.
    double estimateXferTime(const XferList &xfer_list);

    ///////////////////////////////////////////////////////////////////
    // Posting list of operations

    // Post all operations in the list
    // There is no ordering guarantee between different transfer requests or
    // between operations in the same transfer request.
    XferRequest postXfer(const XferList &xfer_list);

    // Post operations in the list with specific dst and src indices
    XferRequest postXfer(const XferList &xfer_list,
                         const std::vector<int> &dst_indices,
                         const std::vector<int> &src_indices);

    ///////////////////////////////////////////////////////////////////
    // Completion checking

    // Poll Option1: check if a specific transfer is done
    // TODO: allow returning failures
    // TODO move it to XferRequest class?
    bool isXferDone(const XferRequest &xfer_request);

    // Poll Option2: get all completed transfers
    void getCompletedXfers(std::vector<XferRequest> &xfer_requests);

    ///////////////////////////////////////////////////////////////////
    // Signals / Notifications

    // Post a signal to a remote agent
    // Signal is ordered w.r.t. to any previous transfers to the same remote
    // agent.
    void postMessage(std::string_view agent_name, const std::string &message);

    // Option1: poll for all signals
    void pollMessages(std::multimap<std::string_view, std::string> &signals);

    // Option2: poll for signals from a specific agent
    void pollMessages(std::string_view remote_agent,
                      std::vector<std::string> &signals);
};

}  // namespace nixl

#endif