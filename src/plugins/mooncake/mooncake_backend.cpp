/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mooncake_backend.h"
#include "serdes/serdes.h"

#include <arpa/inet.h>
#include <bits/stdint-uintn.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/socket.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cufile.h>

#endif

std::vector<std::string> findLocalIpAddresses() {
    std::vector<std::string> ips;
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        return ips;
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        if (ifa->ifa_addr->sa_family == AF_INET) {
            if (strcmp(ifa->ifa_name, "lo") == 0) {
                continue;
            }

            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                            NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
                ips.push_back(host);
            }
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

nixlMooncakeEngine::nixlMooncakeEngine (const nixlBackendInitParams* init_params)
: nixlBackendEngine (init_params) {
    local_agent_name_ = init_params->localAgent;
    auto ips = findLocalIpAddresses();
    std::string segment_name = "127.0.0.1";
    if (!ips.empty()) segment_name = ips[0];
    if (getenv("NIXL_MOONCAKE_IP_ADDR"))
        segment_name = std::string(getenv("NIXL_MOONCAKE_IP_ADDR"));
    engine_ = createTransferEngine("P2PHANDSHAKE",
                                   segment_name.c_str(),
                                   "", 0, true);
}

nixl_mem_list_t nixlMooncakeEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

// Through parent destructor the unregister will be called.
nixlMooncakeEngine::~nixlMooncakeEngine () {
    destroyTransferEngine(engine_);
}

nixl_status_t nixlMooncakeEngine::getConnInfo(std::string &str) const {
    const static size_t kBufLen = 64;
    char buf_out[kBufLen];
    getLocalIpAndPort(engine_, buf_out, kBufLen);
    str = buf_out;
    return NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::connect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connected_agents_.count(remote_agent)) {
        connected_agents_[remote_agent] = AgentInfo{};
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::disconnect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (connected_agents_.count(remote_agent))
        connected_agents_.erase(remote_agent);
    return NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                      const std::string &remote_conn_info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto &agent = connected_agents_[remote_agent];
    agent.ip_and_port = remote_conn_info;
    return NIXL_SUCCESS;
}

struct nixlMooncakeBackendMD : public nixlBackendMD {
    nixlMooncakeBackendMD(bool isPrivate) : nixlBackendMD(isPrivate) {}
    virtual ~nixlMooncakeBackendMD(){}
    void *addr;
    size_t length;
};

nixl_status_t nixlMooncakeEngine::registerMem (const nixlBlobDesc &mem,
                                               const nixl_mem_t &nixl_mem,
                                               nixlBackendMD* &out)
{
    int err = registerLocalMemory(engine_, (void *) mem.addr, mem.len, "*", 1);
    if (err) return NIXL_ERR_BACKEND;
    auto priv = new nixlMooncakeBackendMD(true);
    priv->addr = (void *) mem.addr;
    priv->length = mem.len;
    out = priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::deregisterMem (nixlBackendMD* meta)
{
    return NIXL_SUCCESS;
    auto priv = (nixlMooncakeBackendMD *) meta;
    int err = unregisterLocalMemory(engine_, priv->addr);
    delete priv;
    return err == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

nixl_status_t nixlMooncakeEngine::getPublicData (const nixlBackendMD* meta,
                                                 std::string &str) const {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::loadLocalMD (nixlBackendMD* input,
                                 nixlBackendMD* &output)
{
    output = nullptr;
    return NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::loadRemoteMD (const nixlBlobDesc &input,
                                                const nixl_mem_t &nixl_mem,
                                                const std::string &remote_agent,
                                                nixlBackendMD* &output)
{
    output = nullptr;
    return NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::unloadMD (nixlBackendMD* input) {
    return NIXL_SUCCESS;
}

struct nixlMooncakeBackendReqH : public nixlBackendReqH {
    nixlMooncakeBackendReqH() : nixlBackendReqH() {}
    virtual ~nixlMooncakeBackendReqH(){}
    uint64_t batch_id;
    size_t request_count;
};

nixl_status_t nixlMooncakeEngine::prepXfer (const nixl_xfer_op_t &operation,
                                            const nixl_meta_dlist_t &local,
                                            const nixl_meta_dlist_t &remote,
                                            const std::string &remote_agent,
                                            nixlBackendReqH* &handle,
                                            const nixl_opt_b_args_t* opt_args)
{
    return NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::postXfer (const nixl_xfer_op_t &operation,
                                            const nixl_meta_dlist_t &local,
                                            const nixl_meta_dlist_t &remote,
                                            const std::string &remote_agent,
                                            nixlBackendReqH* &handle,
                                            const nixl_opt_b_args_t* opt_args)
{
    std::string remote_ip_and_port;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!connected_agents_.count(remote_agent))
            return NIXL_ERR_INVALID_PARAM;
        remote_ip_and_port = connected_agents_[remote_agent].ip_and_port;
    }
    if (local.descCount() != remote.descCount()) return NIXL_ERR_INVALID_PARAM;
    size_t request_count = local.descCount();
    uint64_t batch_id = allocateBatchID(engine_, request_count);
    if (batch_id == INVALID_BATCH) return NIXL_ERR_BACKEND;
    auto target_id = openSegment(engine_, remote_ip_and_port.c_str());
    if (target_id < 0) return NIXL_ERR_BACKEND;
    transfer_request_t *request = new transfer_request_t[request_count];
    for (size_t index = 0; index < request_count; ++index) {
        if (local[index].len != remote[index].len) return NIXL_ERR_INVALID_PARAM;
        request[index].opcode = (operation == NIXL_READ) ? OPCODE_READ : OPCODE_WRITE;
        request[index].source = (void *)local[index].addr;
        request[index].target_offset = remote[index].addr;
        request[index].length = local[index].len;
        request[index].target_id = target_id;
    }
    int rc = submitTransfer(engine_, batch_id, request, request_count);
    if (rc) return NIXL_ERR_BACKEND;
    auto priv = new nixlMooncakeBackendReqH();
    priv->batch_id = batch_id;
    priv->request_count = request_count;
    handle = priv;
    return NIXL_IN_PROG;
}

nixl_status_t nixlMooncakeEngine::checkXfer (nixlBackendReqH* handle)
{
    auto priv = (nixlMooncakeBackendReqH *) handle;
    bool has_failed = false;
    for (size_t index = 0; index < priv->request_count; ++index) {
        transfer_status_t status;
        int rc = getTransferStatus(engine_, priv->batch_id, index, &status);
        if (rc || status.status == STATUS_FAILED)
            has_failed = true;
        else if (status.status == STATUS_PENDING || status.status == STATUS_WAITING)
            return NIXL_IN_PROG;
    }
    return has_failed ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t nixlMooncakeEngine::releaseReqH(nixlBackendReqH* handle)
{
    auto priv = (nixlMooncakeBackendReqH *) handle;
    freeBatchID(engine_, priv->batch_id);
    delete priv;
    return NIXL_SUCCESS;
}
