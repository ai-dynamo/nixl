/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_BACKEND_H
#define NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_BACKEND_H

#include <cstddef>
#include <cstdint>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "backend/backend_aux.h"
#include "backend/backend_engine.h"
#include "common/nixl_log.h"
#include "nixl_descriptors.h" // for nixlBlobDesc
#include "nixl_types.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/oss/status_macros.h" // for ASSIGN_OR_RETURN, RETURN_IF_ERROR
#else
#include "rxdm_dxs_stub.h"
#endif
#include "control_channel.h"
#include "control_channel.pb.h"
#include "dxs_endpoint.h"
#include "host_connection.h"
#include "mpmc_queue.h"
#include "params.h"
#include "tcpxo_common.h"
#include "tcpxo_nixl_memory_metadata.h"

namespace tcpxo {

// Key: [addr, size]
using CacheKeyType = std::pair<uintptr_t, size_t>;

struct CacheValueType {
    std::unique_ptr<nixlTcpxoLocalMemoryMetadata> mem_md;
    int ref_cnt;
};

using CacheMap = absl::flat_hash_map<CacheKeyType, CacheValueType>;

struct nixlMemDev {
    absl::Mutex reg_mutex;
    CacheMap cache ABSL_GUARDED_BY(reg_mutex);
};

/* Connection metadata for remote agents */
class nixlTcpxoConnection : public nixlBackendConnMD {
public:
    friend class nixlTcpxoEngine;

    nixlTcpxoConnection() = default;

    explicit nixlTcpxoConnection(AgentAddress peer_control_channel_addr)
        : nixlBackendConnMD(),
          peer_control_channel_addr_(peer_control_channel_addr) {}

private:
    AgentAddress peer_control_channel_addr_;
    // In A3-High, we presume each GPU pair will have different ports, but the same IPAddress. So
    // the right number of elements is the number of GPUs, not the number of NICs
    std::array<DxsAddress, kMaxGpuDevices> peer_dxs_addrs_;
};

class nixlTcpxoEngine : public nixlBackendEngine {
public:
    nixlTcpxoEngine(const nixlBackendInitParams *init_params);
    ~nixlTcpxoEngine();

    friend class TcpxoValidationTest;
    friend class TcpxoBackendTest;

    // Feature select subsystem
    bool
    supportsRemote() const override {
        return true;
    }

    bool
    supportsLocal() const override {
        return false;
    }

    bool
    supportsNotif() const override {
        return true;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {
            VRAM_SEG,
        };
    }

    // Connection subsystem
    /*
     * Returns our control channel service address
     */
    nixl_status_t
    getConnInfo(std::string &str) const override;

    /* Load remote agent connection info */
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent,
                       const std::string &remote_conn_info) override;

    nixl_status_t
    connect(const std::string &remote_agent) override;
    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    // Memory subsystem
    /* Serialize memory metadata for remote agents */
    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const override;

    /* Register local memory */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    /* Load memory metadata for local operations. Not needed until/if we support local transfers */
    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    /* Load memory metadata for remote operations */
    nixl_status_t
    loadRemoteMD(const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output) override;

    /* Release local or remote memory metadata resources */
    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    // Xfer subsystem
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

    // Notification subsystem
    /* Retrieve available notifications */
    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;

    /* Send notification to remote agent */
    nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;

    inline const Params &
    params() const {
        return params_;
    }

private:
    struct RemoteAgent {
        nixlTcpxoConnection info;
        std::unique_ptr<HostConnection> conn{nullptr};
    };

    static std::vector<DxsOpParams>
    CoalesceDxsOps(std::vector<DxsOpParams> uncoalesced_ops);

    // This function is run with the host connection mapping mutex locked. It should not be a
    // long running function.
    template<typename Func>
    nixl_status_t
    EnsureHostConnectionAndRun(const std::string &, Func &&) const;

    void
    ProgressThread();

    absl::StatusOr<HostConnection *>
    LookupHostConnectionUnlocked(const std::string &)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(remote_agents_mutex_) const;

    std::optional<std::string>
    ReadRawParam(const std::string &);

    void
    LoadIntegerParam(IntegerParamDef &);

    void
    LoadStringParam(StringParamDef &);

    void
    LoadBoolParam(BoolParamDef &);

    Params
    InitializeParams();

    absl::Status
    InitializeRxDM();

    absl::Status
    InitializeDxs();

    bool
    CheckIfAgentLoaded(const std::string &) const;

    void
    HandlePeerEvent(PeerEvent &&);

    void
    HandleConnectionEvent(PeerHandle,
                          const AgentAddress &socket_addr,
                          const AgentAddress &service_addr,
                          const std::string &agent_name,
                          const DxsAddressExchangeMessage &);

    void
    HandleConnectionEventUnlocked(PeerHandle,
                                  const AgentAddress &socket_addr,
                                  const AgentAddress &service_addr,
                                  const std::string &agent_name,
                                  const DxsAddressExchangeMessage &)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(remote_agents_mutex_);

    nixl_status_t
    LoadRemoteConnInfoUnlocked(const std::string &remote_agent, const AgentAddress &service_addr)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(remote_agents_mutex_);

    void
    HandleDisconnectionEvent(PeerHandle,
                             const AgentAddress &socket_addr,
                             const AgentAddress &service_addr);

    void
    HandleNotificationEvent(PeerHandle,
                            const AgentAddress &service_addr,
                            WorkloadNotificationMessage &&);

    /**
     * @brief This is triggered when the target sends the initiator its DXS endpoints
     * @details
     * We should drain any pending DXS ops we have on the host connection in this callback. We will
     * likely need to establish endpoint connections first. Since this is running on a worker
     * thread, we can keep that thread occupied with the work of establishing the DXS endpoint
     * connections.
     *
     * The chain of calls here is:
     *
     *     Node A (us, local)            |      Node B (peer, remote)
     * =====================================================================
     *        connect   ->               | HandlePeerRead(IdentityMessage)
     *                                   |     HandleConnectionEvent
     *  HandleDxsAddressExchangeEvent <- | ScheduleSend(DxsAddressExchange)
     */
    void
    HandleDxsAddressExchangeEvent(PeerHandle,
                                  const AgentAddress &service_addr,
                                  const DxsAddressExchangeMessage &);

    void
    HandleXferMessageEvent(PeerHandle handle,
                           const AgentAddress &service_addr,
                           const XferMessage &msg);

    void
    HandleHostConnectionTask(HostConnectionTask task);

    void
    HandleDxsConnectionEstablishTask(const std::string &remote_name);

    Worker &
    GetLeastBusyWorker();

    Worker &
    GetWorkerForPeer(const PeerEvent &);

    absl::Status
    ValidateXferDlists(const nixl_xfer_op_t &operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote) const;

    // Helper function for registerMem()
    CacheKeyType
    GetPageOrientedRegion(uintptr_t addr, size_t len) const;

    CacheMap::iterator
    FindRegionOrSubregionInCache(CacheMap &cache,
                                 const CacheKeyType &cache_key,
                                 uintptr_t addr,
                                 size_t len) const;

    absl::StatusOr<tcpdirect::BufferManagerClientInterface * absl_nonnull>
        GetBufferManagerClient(uint8_t);

    absl::StatusOr<std::vector<DxsEndpointListenInfo>>
    PrepareLocalListenHandles(absl::flat_hash_map<EndpointPair, EndpointConnection> &) const;

    // Page size that can only be determined in runtime
    const size_t page_size_;

    // Spawn a progress thread and repeatedly call DXS Test()
    // for both locally and remote initiated pending DXS ops.
    const absl::Duration progress_thread_delay_;
    absl::Notification stop_progress_thread_;
    std::thread progress_thread_;

    Params params_;

    std::unique_ptr<DxsEndpointManager> endpoint_manager_;
    std::vector<std::unique_ptr<Worker>> workers_;
    // The below two are only modified by the owner of our HandlePeerEvent callback, i.e. the
    // ControlChannel Epoll Thread
    absl::flat_hash_map<AgentAddressKey, Worker * absl_nonnull> peer_to_worker_map_;
    size_t next_worker_idx_{0};
    ControlChannel control_channel_;

    mutable absl::Mutex remote_agents_mutex_;
    mutable absl::flat_hash_map<std::string, RemoteAgent>
        remote_agents_ ABSL_GUARDED_BY(remote_agents_mutex_);

    MPMCQueue<std::pair<std::string, std::string>> pending_notifs_;

    // Keep cache and mutex to search for registered memory handles
    mutable std::array<nixlMemDev, kMaxGpuDevices> mem_devices_;
};

template<typename Func>
nixl_status_t
nixlTcpxoEngine::EnsureHostConnectionAndRun(const std::string &remote_agent, Func &&func) const {
    absl::MutexLock lock(remote_agents_mutex_);
    auto it = remote_agents_.find(remote_agent);
    if (it == remote_agents_.end()) {
        NIXL_ERROR << "Failed to find remote agent connection info for " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    if (it->second.conn) {
        return func(*it->second.conn);
    }

    // Store EndpointConnection info in HostConnection
    absl::flat_hash_map<EndpointPair, EndpointConnection> connection_map;
    absl::StatusOr<std::vector<DxsEndpointListenInfo>> endpoint_listen_infos =
        PrepareLocalListenHandles(connection_map);
    if (!endpoint_listen_infos.ok()) {
        NIXL_ERROR << "PrepareLocalListenHandles(connection_map) failed: "
                   << endpoint_listen_infos.status();
        return AbslStatusToNixlStatus(endpoint_listen_infos.status(), __FILE__, __LINE__);
    }

    NIXL_DEBUG << "Connecting to agent: " << remote_agent << " within " << __func__;
    auto &addr = it->second.info.peer_control_channel_addr_;
    auto handle_or_status = const_cast<ControlChannel &>(control_channel_)
                                .Connect(addr, localAgent, *std::move(endpoint_listen_infos));
    ASSIGN_OR_RETURN_NIXL(auto handle, handle_or_status);
    NIXL_DEBUG << "Connected to agent: " << remote_agent << " within " << __func__;
    it->second.conn = std::make_unique<HostConnection>(
        remote_agent,
        handle,
        const_cast<ControlChannel &>(control_channel_),
        std::move(connection_map),
        params_,
        const_cast<MPMCQueue<std::pair<std::string, std::string>> &>(pending_notifs_),
        absl::bind_front(&nixlTcpxoEngine::HandleHostConnectionTask,
                         const_cast<nixlTcpxoEngine *>(this),
                         HostConnectionTask{DxsConnectionEstablishTask{remote_agent}}));

    return func(*it->second.conn);
}

} // namespace tcpxo

#endif // NIXL_SRC_PLUGINS_GPUDIRECT_TCPXO_TCPXO_BACKEND_H
