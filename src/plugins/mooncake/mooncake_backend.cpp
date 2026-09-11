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
#include "mooncake_backend.h"
#include "serdes/serdes.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include "mooncake_backend_internal.h"

// Both engine headers can coexist: the OPCODE_*/STATUS_* macros they share are
// defined identically by each.
#ifdef HAVE_MOONCAKE_TENT
#include <tent/transfer_engine.h>
#endif

#include <arpa/inet.h>
#include <bits/stdint-uintn.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace {

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

[[nodiscard]] std::string
chooseIpAddress() {
    static const std::string local = "127.0.0.1";
    static const std::vector<std::string> ips = findLocalIpAddresses();
    static const std::string &fallback = ips.empty() ? local : ips[0];
    return nixl::config::getValueDefaulted("NIXL_MOONCAKE_IP_ADDR", fallback);
}

} // namespace

nixlMooncakeEngine::nixlMooncakeEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      local_agent_name_(init_params->localAgent) {
    const std::string segment_name = chooseIpAddress();

    // Engine selection: backend parameter "mooncake_mode", overridable through
    // NIXL_MOONCAKE_MODE. The default is the classic engine, so existing
    // deployments are unaffected by the addition of the TENT mode.
    std::string mode = "classic";
    (void)getInitParam("mooncake_mode", mode);
    mode = nixl::config::getValueDefaulted("NIXL_MOONCAKE_MODE", mode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode.empty()) {
        mode = "classic";
    }

    if (mode == "tent") {
#ifdef HAVE_MOONCAKE_TENT
        mode_ = mode::TENT;
        invalid_batch_ = kTentInvalidBatch;
#else
        NIXL_ERROR << "Mooncake backend was built without TENT support; rebuild Mooncake with "
                      "-DUSE_TENT=ON (and NIXL against it) to use mooncake_mode=tent";
        initErr = true;
        return;
#endif
    } else if (mode != "classic") {
        NIXL_ERROR << "Unknown mooncake_mode '" << mode << "', expected 'classic' or 'tent'";
        initErr = true;
        return;
    }

#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        // Optional user-provided TENT configuration file; the settings below
        // override whatever it contains.
        std::string config_path;
        if (getInitParam("tent_config_path", config_path) == NIXL_SUCCESS && !config_path.empty()) {
            tent_load_config_from_file(config_path.c_str());
        }
        // Equivalent of the classic createTransferEngine("P2PHANDSHAKE", ...):
        // peer-to-peer metadata exchange with no external metadata service.
        // tent_set_config() stages values in thread-local storage consumed by
        // tent_create_engine(), so these calls must stay on the same thread.
        tent_set_config("metadata_type", "p2p");
        tent_set_config("local_segment_name", segment_name.c_str());
        tent_engine_ = tent_create_engine();
        if (!tent_engine_ || !tent_available(tent_engine_)) {
            NIXL_ERROR << "Failed to initialize the Mooncake TENT engine";
            if (tent_engine_) {
                tent_destroy_engine(tent_engine_);
                tent_engine_ = nullptr;
            }
            initErr = true;
            return;
        }
        if (local_agent_name_.size() > kMaxNotifNameLen) {
            NIXL_WARN << "Agent name exceeds " << kMaxNotifNameLen
                      << " bytes; Mooncake notifications from this agent will be rejected";
        }
        return;
    }
#endif

    engine_ = createTransferEngine("P2PHANDSHAKE", segment_name.c_str(), "", 0, true);
    if (!engine_) {
        NIXL_ERROR << "Failed to initialize the Mooncake transfer engine";
        initErr = true;
    }
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
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        if (tent_engine_) {
            reclaimParkedBatches();
            tent_destroy_engine(tent_engine_);
        }
        return;
    }
#endif
    if (engine_) {
        destroyTransferEngine(engine_);
    }
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
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        char addr[64] = {0};
        uint16_t port = 0;
        if (tent_rpc_server_addr_port(tent_engine_, addr, sizeof(addr), &port)) {
            return NIXL_ERR_BACKEND;
        }
        addr[sizeof(addr) - 1] = '\0';
        // In TENT p2p mode the segment name is the "ip:port" of the local RPC
        // server, the same shape the classic engine reports.
        str = std::string(addr) + ":" + std::to_string(port);
        return NIXL_SUCCESS;
    }
#endif
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
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        tent_segment_id_t segment_id = 0;
        if (tent_open_segment(tent_engine_, &segment_id, remote_conn_info.c_str())) {
            return NIXL_ERR_BACKEND;
        }
        connected_agents_[remote_agent].segment_id = segment_id;
        return NIXL_SUCCESS;
    }
#endif
    auto segment_id = openSegment(engine_, remote_conn_info.c_str());
    if (segment_id < 0) return NIXL_ERR_BACKEND;
    connected_agents_[remote_agent].segment_id = static_cast<uint64_t>(segment_id);
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
    int err;
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        tent_memory_options_t opts;
        memset(&opts, 0, sizeof(opts));
        // A zero-initialized permission is PERM_LOCAL_READ_WRITE, which would
        // hide the region from remote peers - the classic call registers with
        // remote_accessible=1, so ask for global access explicitly.
        opts.permission = PERM_GLOBAL_READ_WRITE;
        // UNSPEC registers with every available transport, leaving TENT free
        // to route and to fail over between transports.
        opts.transport_type = TRANSPORT_UNSPEC;
        if (nixl_mem == VRAM_SEG) {
            snprintf(opts.location,
                     sizeof(opts.location),
                     "cuda:%llu",
                     static_cast<unsigned long long>(mem.devId));
        }
        // DRAM: leave the location empty, TENT falls back to the wildcard.
        err = tent_register_memory_ex(tent_engine_, (void *)mem.addr, mem.len, &opts);
    } else
#endif
    {
        err = registerLocalMemory(engine_, (void *)mem.addr, mem.len, "*", 1);
    }
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
    int err;
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        err = tent_unregister_memory(tent_engine_, priv->addr, priv->length);
    } else
#endif
    {
        err = unregisterLocalMemory(engine_, priv->addr);
    }
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

nixl_status_t
nixlMooncakeEngine::prepXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    handle = new nixlMooncakeBackendReqH(invalid_batch_);
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
    uint64_t segment_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto agent = connected_agents_.find(remote_agent);
        if (agent == connected_agents_.end()) return NIXL_ERR_INVALID_PARAM;
        segment_id = agent->second.segment_id;
    }
    if (local.descCount() != remote.descCount()) return NIXL_ERR_INVALID_PARAM;
    if (priv->abort_requested) {
        return NIXL_ERR_INVALID_PARAM;
    }

#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        return postXferTent(operation, local, remote, segment_id, priv, opt_args);
    }
#endif
    return postXferClassic(operation, local, remote, segment_id, priv, opt_args);
}

nixl_status_t
nixlMooncakeEngine::postXferClassic(const nixl_xfer_op_t &operation,
                                    const nixl_meta_dlist_t &local,
                                    const nixl_meta_dlist_t &remote,
                                    uint64_t segment_id,
                                    nixlMooncakeBackendReqH *priv,
                                    const nixl_opt_b_args_t *opt_args) const {
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
        if (local[index].len != remote[index].len) {
            delete[] request;
            return NIXL_ERR_INVALID_PARAM;
        }
        request[index].opcode = (operation == NIXL_READ) ? OPCODE_READ : OPCODE_WRITE;
        request[index].source = (void *)local[index].addr;
        request[index].target_offset = remote[index].addr;
        request[index].length = local[index].len;
        request[index].target_id = static_cast<segment_id_t>(segment_id);
    }
    int rc = 0;
    // opt_args is declared optional (nullptr default) by the SB API.
    if (opt_args && opt_args->hasNotif) {
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
    // The batch is reclaimed once every request completed (see below); a later
    // status query on the same handle is a terminal no-op. Without this guard a
    // stale batch id would be handed back to the engine, which casts it to a
    // descriptor pointer and dereferences it.
    if (priv->batch_id == invalid_batch_) {
        return NIXL_SUCCESS;
    }
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        return checkXferTent(priv);
    }
#endif
    return checkXferClassic(priv);
}

nixl_status_t
nixlMooncakeEngine::checkXferClassic(nixlMooncakeBackendReqH *priv) const {
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
        priv->request_count = 0;
    }
    return has_failed ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto priv = (nixlMooncakeBackendReqH *)handle;
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        return releaseReqHTent(priv);
    }
#endif
    return releaseReqHClassic(priv);
}

nixl_status_t
nixlMooncakeEngine::releaseReqHClassic(nixlMooncakeBackendReqH *priv) const {
    // The classic engine exposes no cancellation primitive: freeBatchID()
    // refuses to release a batch that still has unfinished tasks (BatchBusy)
    // and nothing can stop them, so an in-flight release can only leak the
    // batch. Keep the historical behavior here; mooncake_mode=tent implements
    // the BackendGuide cancellation protocol.
    if (priv->batch_id != INVALID_BATCH) {
        freeBatchID(engine_, priv->batch_id);
    }
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::getNotifs(notif_list_t &notif_list) {
    if (notif_list.size() != 0) return NIXL_ERR_INVALID_PARAM;
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        reclaimParkedBatches();
        tent_notifi_info info;
        info.num_records = 0;
        info.records = nullptr;
        if (tent_recv_notifs(tent_engine_, &info)) {
            return NIXL_ERR_BACKEND;
        }
        for (int i = 0; i < info.num_records; i++) {
            // The C shim copies with strncpy and does not guarantee
            // termination at full length; clamp before constructing strings.
            info.records[i].name[sizeof(info.records[i].name) - 1] = '\0';
            info.records[i].msg[sizeof(info.records[i].msg) - 1] = '\0';
            notif_list.push_back(std::make_pair(std::string(info.records[i].name),
                                                std::string(info.records[i].msg)));
        }
        tent_free_notifs(&info);
        return NIXL_SUCCESS;
    }
#endif
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
    uint64_t segment_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto agent = connected_agents_.find(remote_agent);
        if (agent == connected_agents_.end()) return NIXL_ERR_INVALID_PARAM;
        segment_id = agent->second.segment_id;
    }
#ifdef HAVE_MOONCAKE_TENT
    if (mode_ == mode::TENT) {
        if (msg.size() > kMaxNotifMsgLen || local_agent_name_.size() > kMaxNotifNameLen) {
            return NIXL_ERR_INVALID_PARAM;
        }
        int ret =
            tent_send_notifs(tent_engine_, segment_id, local_agent_name_.c_str(), msg.c_str());
        return ret == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }
#endif
    notify_msg_t notify_msg;
    notify_msg.name = const_cast<char *>(local_agent_name_.c_str());
    notify_msg.msg = const_cast<char *>(msg.c_str());
    int ret = genNotifyInEngine(engine_, static_cast<segment_id_t>(segment_id), notify_msg);
    return nixl_status_t(ret);
}
