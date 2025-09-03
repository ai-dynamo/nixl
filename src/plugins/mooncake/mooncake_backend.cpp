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
#include "common/nixl_log.h"

#include <arpa/inet.h>
#include <bits/stdint-uintn.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

std::vector<std::string>
findLocalIpAddresses() {
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

            // Check if interface is UP and RUNNING
            if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) {
                NIXL_INFO << "Skipping interface " << ifa->ifa_name << " (not UP or not RUNNING)";
                continue;
            }

            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr,
                            sizeof(struct sockaddr_in),
                            host,
                            NI_MAXHOST,
                            nullptr,
                            0,
                            NI_NUMERICHOST) == 0) {
                ips.push_back(host);
            }
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

nixlMooncakeEngine::nixlMooncakeEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    local_agent_name_ = init_params->localAgent;
    auto ips = findLocalIpAddresses();
    std::string segment_name = "127.0.0.1";
    if (!ips.empty()) segment_name = ips[0];
    if (getenv("NIXL_MOONCAKE_IP_ADDR"))
        segment_name = std::string(getenv("NIXL_MOONCAKE_IP_ADDR"));
    engine_ = createTransferEngine("P2PHANDSHAKE", segment_name.c_str(), "", 0, true);
}

nixl_mem_list_t
nixlMooncakeEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

// Through parent destructor the unregister will be called.
nixlMooncakeEngine::~nixlMooncakeEngine() {
    destroyTransferEngine(engine_);
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
//
// Mooncake Transfer Engine exchanges metadata by itself without any explicit interface,
// and it does not need to connect remote agent before transferring data.
// Instead, getConnInfo() obtains the mapping between agent name and connect info
// (segment name in the context of Mooncake Transfer Engine).
// loadRemoteConnInfo() opens the segment, which implicitly retrieves metadata
// (such as QP numbers) of the remote agent.
nixl_status_t
nixlMooncakeEngine::connect(const std::string &remote_agent) {
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::disconnect(const std::string &remote_agent) {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::getConnInfo(std::string &str) const {
    const static size_t kBufLen = 64;
    char buf_out[kBufLen];
    getLocalIpAndPort(engine_, buf_out, kBufLen);
    str = buf_out;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                       const std::string &remote_conn_info) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto segment_id = openSegment(engine_, remote_conn_info.c_str());
    if (segment_id < 0) return NIXL_ERR_BACKEND;
    connected_agents_[remote_agent].segment_id = segment_id;
    return NIXL_SUCCESS;
}

struct nixlMooncakeBackendMD : public nixlBackendMD {
    nixlMooncakeBackendMD(bool isPrivate) : nixlBackendMD(isPrivate) {}

    virtual ~nixlMooncakeBackendMD() {}

    void *addr;
    size_t length;
    int ref_cnt;
};

nixl_status_t
nixlMooncakeEngine::registerMem(const nixlBlobDesc &mem,
                                const nixl_mem_t &nixl_mem,
                                nixlBackendMD *&out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mem_reg_info_.count(mem.addr)) {
        auto priv = mem_reg_info_[mem.addr];
        priv->ref_cnt++;
        out = priv;
        return NIXL_SUCCESS;
    }
    int err = registerLocalMemory(engine_, (void *)mem.addr, mem.len, "*", 1);
    if (err) return NIXL_ERR_BACKEND;
    auto priv = new nixlMooncakeBackendMD(true);
    priv->addr = (void *)mem.addr;
    priv->length = mem.len;
    priv->ref_cnt = 1;
    out = priv;
    mem_reg_info_[mem.addr] = priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::deregisterMem(nixlBackendMD *meta) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto priv = (nixlMooncakeBackendMD *)meta;
    priv->ref_cnt--;
    if (priv->ref_cnt) return NIXL_SUCCESS;
    int err = unregisterLocalMemory(engine_, priv->addr);
    mem_reg_info_.erase((uint64_t)priv->addr);
    delete priv;
    return err == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
//
// Mooncake Transfer Engine exchanges metadata by itself without any explicit interface,
// which is different from NIXL's paradigm.
// Therefore no metadata needs to be exposed to the outside.
nixl_status_t
nixlMooncakeEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    output = nullptr;
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::loadRemoteMD(const nixlBlobDesc &input,
                                 const nixl_mem_t &nixl_mem,
                                 const std::string &remote_agent,
                                 nixlBackendMD *&output) {
    output = nullptr;
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::unloadMD(nixlBackendMD *input) {
    return NIXL_SUCCESS;
}

struct nixlMooncakeBackendReqH : public nixlBackendReqH {
    nixlMooncakeBackendReqH() : nixlBackendReqH() {}

    virtual ~nixlMooncakeBackendReqH() {}

    uint64_t batch_id;
    size_t request_count;
};

nixl_status_t
nixlMooncakeEngine::prepXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    auto priv = new nixlMooncakeBackendReqH();
    priv->batch_id = INVALID_BATCH;
    handle = priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::postXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    auto priv = (nixlMooncakeBackendReqH *)handle;
    int segment_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto agent = connected_agents_.find(remote_agent);
        if (agent == connected_agents_.end()) return NIXL_ERR_INVALID_PARAM;
        segment_id = agent->second.segment_id;
    }
    if (local.descCount() != remote.descCount()) return NIXL_ERR_INVALID_PARAM;

    const static size_t kMaxRequestCount = 1024;
    if (priv->batch_id == INVALID_BATCH) {
        uint64_t batch_id = allocateBatchID(engine_, kMaxRequestCount);
        if (batch_id == INVALID_BATCH) {
            return NIXL_ERR_BACKEND;
        }
        priv->batch_id = batch_id;
        priv->request_count = 0;
    }

    size_t request_count = local.descCount();
    transfer_request_t *request = new transfer_request_t[request_count];
    for (size_t index = 0; index < request_count; ++index) {
        if (local[index].len != remote[index].len) return NIXL_ERR_INVALID_PARAM;
        request[index].opcode = (operation == NIXL_READ) ? OPCODE_READ : OPCODE_WRITE;
        request[index].source = (void *)local[index].addr;
        request[index].target_offset = remote[index].addr;
        request[index].length = local[index].len;
        request[index].target_id = segment_id;
    }
    int rc = 0;
    if (opt_args->hasNotif) {
        notify_msg_t notify_msg;
        notify_msg.name = const_cast<char *>(local_agent_name_.c_str());
        notify_msg.msg = const_cast<char *>(opt_args->notifMsg.c_str());
        rc = submitTransferWithNotify(engine_, priv->batch_id, request, request_count, notify_msg);
    } else {
        rc = submitTransfer(engine_, priv->batch_id, request, request_count);
    }
    delete[] request;
    if (rc) return NIXL_ERR_BACKEND;
    priv->request_count += request_count;
    return NIXL_IN_PROG;
}

nixl_status_t
nixlMooncakeEngine::checkXfer(nixlBackendReqH *handle) const {
    auto priv = (nixlMooncakeBackendReqH *)handle;
    bool has_failed = false;
    for (size_t index = 0; index < priv->request_count; ++index) {
        transfer_status_t status;
        int rc = getTransferStatus(engine_, priv->batch_id, index, &status);
        if (rc || status.status == STATUS_FAILED)
            has_failed = true;
        else if (status.status == STATUS_PENDING || status.status == STATUS_WAITING)
            return NIXL_IN_PROG;
    }
    if (!has_failed) {
        // Each batch_id has the batch size, and cannot process more requests
        // than the batch size. So, free the batch id here to workaround the issue
        // where the same nixlBackendReqH could be used to post multiple transfer.
        freeBatchID(engine_, priv->batch_id);
        priv->batch_id = INVALID_BATCH;
    }
    return has_failed ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto priv = (nixlMooncakeBackendReqH *)handle;
    if (priv->batch_id != INVALID_BATCH) {
        freeBatchID(engine_, priv->batch_id);
    }
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::getNotifs(notif_list_t &notif_list) {
    if (notif_list.size() != 0) return NIXL_ERR_INVALID_PARAM;
    int size = 0;
    notify_msg_t *notify_msgs = getNotifsFromEngine(engine_, &size);
    for (int i = 0; i < size; i++) {
        notif_list.push_back(std::make_pair(notify_msgs[i].name, notify_msgs[i].msg));
    }
    freeNotifsMsgBuf(notify_msgs, size);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    int segment_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto agent = connected_agents_.find(remote_agent);
        if (agent == connected_agents_.end()) return NIXL_ERR_INVALID_PARAM;
        segment_id = agent->second.segment_id;
    }
    notify_msg_t notify_msg;
    notify_msg.name = const_cast<char *>(local_agent_name_.c_str());
    notify_msg.msg = const_cast<char *>(msg.c_str());
    int ret = genNotifyInEngine(engine_, segment_id, notify_msg);
    return nixl_status_t(ret);
}
