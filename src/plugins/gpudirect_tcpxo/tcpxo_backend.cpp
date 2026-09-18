/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#include "tcpxo_backend.h"

#include <algorithm>
#include <bits/local_lim.h>
#include <cstdint>
#include <cstdlib>
#include <ctype.h>
#include <unistd.h>

#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "backend/backend_aux.h"
#include "backend/backend_engine.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h" // for ASSIGN_OR_RETURN, RETURN_IF_ERROR
#else
#include "rxdm_dxs_stub.h"
#endif
#include "dxs_endpoint.h"
#include "host_connection.h"
#include "nixl_cuda/cuda_common.h"
#include "params.h"
#include "tcpxo_common.h"
#include "tcpxo_nixl_memory_metadata.h"

namespace tcpxo {

static constexpr char kAddrKey[] = "addr";
static constexpr char kPortKey[] = "port";

static constexpr uint16_t kMaxPort = std::numeric_limits<uint16_t>::max();

namespace {

    // Removes leading and trailing whitespaces from a string.
    void
    TrimString(std::string &str) {
        if (str.empty()) {
            return;
        }
        str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    }

    void
    EnsureNonEmptyStringParam(StringParamDef &param) {
        // Safely convert to std::string (defaults to "" if missing)
        std::string str = "";
        if (!param.value.empty()) {
            str = std::string(param.value);
        }
        TrimString(str);
        if (str.empty()) {
            param.value = param.default_value;
            NIXL_DEBUG << "Parameter " << param.env_var_name
                       << " set to an empty string. Using default value: " << param.default_value;
        }
    }

    bool
    CanCoalesce(const DxsOpParams &current_op, const DxsOpParams &next_op) {
        return current_op.endpoint_pair == next_op.endpoint_pair &&
            current_op.local_reg_handle == next_op.local_reg_handle &&
            current_op.remote_reg_handle == next_op.remote_reg_handle &&
            current_op.local_mr.addr + current_op.local_mr.len == next_op.local_mr.addr &&
            current_op.remote_mr.addr + current_op.remote_mr.len == next_op.remote_mr.addr;
    }

} // namespace

// nixlTcpxoEngine
nixlTcpxoEngine::nixlTcpxoEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE))),
      progress_thread_delay_(absl::Microseconds(init_params->pthrDelay)),
      params_(InitializeParams()),
      control_channel_(params_.fastrak_ctrl_dev.value,
                       absl::Milliseconds(params_.fastrak_heartbeat_send_period_ms.value),
                       absl::Milliseconds(params_.fastrak_heartbeat_timeout_ms.value),
                       absl::bind_front(&nixlTcpxoEngine::HandlePeerEvent, this)) {
    NIXL_DEBUG << "Initializing GPUDirect TCPXO Backend";

    // Explicitly initialize CUDA before anything else
    if (const auto status = InitCuda(); !status.ok()) {
        NIXL_ERROR << "Failed to initialize CUDA driver: " << status.message();
#ifdef HAVE_CUDA
        initErr = true;
        return;
#endif
    }

    if (const auto status = InitializeRxDM(); !status.ok()) {
        initErr = true;
        NIXL_ERROR << "Initializing RxDM failed!";
        return;
    }

    if (const auto status = InitializeDxs(); !status.ok()) {
        initErr = true;
        NIXL_ERROR << "Initializing DXS failed: " << status.message();
        return;
    }

    if (const auto status = CheckDeviceCount(kMaxGpuDevices); !status.ok()) {
        initErr = true;
        NIXL_ERROR << "CheckDeviceCount failed: " << status.message();
        return;
    }

    for (auto i = 0u; i < params_.fastrak_num_control_channel_workers.value; ++i) {
        workers_.push_back(std::make_unique<Worker>(
            absl::bind_front(&nixlTcpxoEngine::HandleConnectionEvent, this),
            absl::bind_front(&nixlTcpxoEngine::HandleDisconnectionEvent, this),
            absl::bind_front(&nixlTcpxoEngine::HandleNotificationEvent, this),
            absl::bind_front(&nixlTcpxoEngine::HandleDxsAddressExchangeEvent, this),
            absl::bind_front(&nixlTcpxoEngine::HandleXferMessageEvent, this),
            absl::bind_front(&nixlTcpxoEngine::HandleDxsConnectionEstablishTask, this)));
        workers_.back()->Start();
    }

    if (const auto status = control_channel_.Listen(); !status.ok()) {
        initErr = true;
        NIXL_ERROR << "ControlChannel Listen failed: " << status.status().message();
        return;
    }

    progress_thread_ = std::thread(&nixlTcpxoEngine::ProgressThread, this);

    NIXL_DEBUG << "GPUDirect TCPXO initialized successfully";
}

nixlTcpxoEngine::~nixlTcpxoEngine() {
    control_channel_.Stop();
    for (auto &worker : workers_) {
        worker->Stop();
    }

    stop_progress_thread_.Notify();
    if (progress_thread_.joinable()) {
        progress_thread_.join();
    }

    // Manually cleanup cache if deregisterMem() is not called
    for (uint8_t fastrak_idx = 0; fastrak_idx < kMaxGpuDevices; ++fastrak_idx) {
        auto &mem_device = mem_devices_[fastrak_idx];
        absl::MutexLock lock(&mem_device.reg_mutex);
        for (auto &[key, val] : mem_device.cache) {
            MemoryHandle mem_handle = val.mem_md->GetMemHandle();
            // Make sure not to double-deregister the memory handle
            if (mem_handle.reg_handle == dxs::kInvalidRegistration) {
                continue;
            }
            if (mem_handle.dmabuf_fd > 0) {
                close(mem_handle.dmabuf_fd);
            }
            absl::StatusOr<tcpdirect::BufferManagerClientInterface *> buf_mgr =
                GetBufferManagerClient(fastrak_idx);
            if (!buf_mgr.ok()) {
                NIXL_ERROR << absl::StrFormat(
                    "Cannot get valid buffer manager client with FasTrak index: %d", fastrak_idx);
                return;
            }
            absl::Status ret = (*buf_mgr)->DeregBuf(mem_handle.reg_handle);
            if (!ret.ok()) {
                NIXL_ERROR << absl::StrFormat(
                    "Deregister memory with buffer manager failed at data: %p size: %d",
                    mem_handle.start_addr,
                    mem_handle.size);
                return;
            }
        }
        mem_device.cache.clear();
    }
}

nixl_status_t
nixlTcpxoEngine::getConnInfo(std::string &str) const {
    const AgentAddress &addr = control_channel_.GetServiceAddress();

    nixlSerDes ser_des;
    ser_des.addStr(kAddrKey, addr.ip());
    ser_des.addStr(kPortKey, absl::StrCat(addr.port()));
    str = ser_des.exportStr();

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxoEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                    const std::string &remote_conn_info) {
    nixlSerDes ser_des;
    if (ser_des.importStr(remote_conn_info) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to parse remote connection information";
        return NIXL_ERR_BACKEND;
    }

    std::string addr_str = ser_des.getStr(kAddrKey);
    if (addr_str.empty()) {
        NIXL_ERROR << "Missing 'addr' in remote connection information";
        return NIXL_ERR_BACKEND;
    }

    std::string port_str = ser_des.getStr(kPortKey);
    if (port_str.empty()) {
        NIXL_ERROR << "Missing 'port' in remote connection information";
        return NIXL_ERR_BACKEND;
    }

    uint32_t port;
    if (!absl::SimpleAtoi(port_str, &port) || port > kMaxPort) {
        NIXL_ERROR << "Invalid 'port' value: " << port_str;
        return NIXL_ERR_BACKEND;
    }

    IPAddress ip_addr;
    ip_addr.set_addr(addr_str);
    AgentAddress service_addr(ip_addr, port);

    absl::MutexLock lock(&remote_agents_mutex_);
    return LoadRemoteConnInfoUnlocked(remote_agent, service_addr);
}

nixl_status_t
nixlTcpxoEngine::LoadRemoteConnInfoUnlocked(const std::string &remote_agent,
                                            const AgentAddress &service_addr) {
    nixlTcpxoConnection conn_info(service_addr);
    remote_agents_[remote_agent].info = conn_info;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxoEngine::connect(const std::string &remote_agent) {
    return EnsureHostConnectionAndRun(remote_agent,
                                      [](HostConnection &conn) { return NIXL_SUCCESS; });
}

nixl_status_t
nixlTcpxoEngine::disconnect(const std::string &remote_agent) {
    absl::MutexLock lock(&remote_agents_mutex_);

    auto it = remote_agents_.find(remote_agent);
    if (it == remote_agents_.end() || !it->second.conn) {
        return NIXL_SUCCESS;
    }

    if (const auto status = control_channel_.Disconnect(it->second.conn->remote_handle());
        !status.ok()) {
        NIXL_WARN << "Failed to disconnect from peer: "
                  << it->second.info.peer_control_channel_addr_ << ". Cause: " << status;
    }

    it->second.conn.reset();

    return NIXL_SUCCESS;
}

void
nixlTcpxoEngine::ProgressThread() {
    while (!stop_progress_thread_.WaitForNotificationWithTimeout(progress_thread_delay_)) {
        // Loop over ops
        absl::MutexLock lock(&remote_agents_mutex_);
        for (const auto &[name, agent] : remote_agents_) {
            if (agent.conn) {
                agent.conn->TestProgress();
            }
        }
    }
}

nixl_status_t
nixlTcpxoEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    const nixlTcpxoLocalMemoryMetadata *raw_mem_md =
        static_cast<const nixlTcpxoLocalMemoryMetadata *>(meta);
    auto status = ::tcpxo::SerializeMemoryMetadata(raw_mem_md, str);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    return NIXL_SUCCESS;
}

CacheKeyType
nixlTcpxoEngine::GetPageOrientedRegion(uintptr_t addr, size_t len) const {
    /**
     * Let's say page_size_ is 4096 (0x1000) and addr is 0x7FFF1234.
     * addr:            0x7FFF1234
     * -4096:           0xFFFFF000
     * page_start_addr  0x7FFF1000
     * This is an efficient way to calculate the starting address of the page containing the given
     * addr
     */
    uintptr_t page_start_addr = addr & -page_size_;
    // Ceiling of number of pages needed for the given data.
    size_t pages = (addr + len - page_start_addr + page_size_ - 1) / page_size_;
    return std::make_pair(page_start_addr, pages * page_size_);
}

CacheMap::iterator
nixlTcpxoEngine::FindRegionOrSubregionInCache(CacheMap &cache,
                                              const CacheKeyType &cache_key,
                                              uintptr_t addr,
                                              size_t len) const {
    auto it = cache.find(cache_key);
    if (it != cache.end()) {
        return it;
    }
    for (auto sub_it = cache.begin(); sub_it != cache.end(); ++sub_it) {
        const uintptr_t reg_addr = sub_it->first.first;
        const size_t reg_len = sub_it->first.second;
        if (addr >= reg_addr && (addr + len) <= (reg_addr + reg_len)) {
            return sub_it;
        }
    }
    return cache.end();
}

nixl_status_t
nixlTcpxoEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    // We only support memory type VRAM for CUDA.
    if (VRAM_SEG != nixl_mem) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Register memory regions in a page-oriented way
    const CacheKeyType &region_info = GetPageOrientedRegion(mem.addr, mem.len);
    ASSIGN_OR_RETURN_NIXL(const auto fastrak_idx,
                          endpoint_manager_->GetFastrakIdxFromDevId(mem.devId));
    nixlMemDev &mem_device = mem_devices_[fastrak_idx];
    absl::MutexLock lock(&mem_device.reg_mutex);
    // Found region_info in the cache, increment the ref_cnt and return the raw
    // pointer to the local mem_md
    if (auto it = mem_device.cache.find(region_info); it != mem_device.cache.end()) {
        ++it->second.ref_cnt;
        MemoryHandle mem_handle = it->second.mem_md->GetMemHandle();
        NIXL_DEBUG << absl::StrFormat(
            "Register memory with existing registration handle found for data: %p size: %d",
            mem_handle.start_addr,
            mem_handle.size);
        out = absl::implicit_cast<nixlBackendMD *>(it->second.mem_md.get());
    } else {
        const auto &[addr, size] = region_info;
        ASSIGN_OR_RETURN_NIXL(int dmabuf_fd,
                              GetDmabufFd(mem.devId, reinterpret_cast<void *>(addr), size));
        absl::Cleanup fd_cleanup = [&] { close(dmabuf_fd); };

        ASSIGN_OR_RETURN_NIXL(auto base_info,
                              GetDmabufBase(mem.devId, reinterpret_cast<void *>(addr)));
        void *base_addr = base_info.first;
        // Still need to evaluate these approaches with NIXLBench
        NIXL_DEBUG << "GetDmabufBase: [" << std::hex << reinterpret_cast<uintptr_t>(base_addr)
                   << std::dec << ", " << base_info.second << "] vs. GetPageOrientedRegion: ["
                   << std::hex << reinterpret_cast<uintptr_t>(addr) << std::dec << ", " << size
                   << "]";

        ASSIGN_OR_RETURN_NIXL(tcpdirect::BufferManagerClientInterface * buf_mgr,
                              GetBufferManagerClient(fastrak_idx));
        ASSIGN_OR_RETURN_NIXL(dxs::Reg reg_handle, buf_mgr->RegBuf(dmabuf_fd, size));
        auto [inserted_itr, inserted] = mem_device.cache.insert(
            {region_info,
             CacheValueType{std::make_unique<nixlTcpxoLocalMemoryMetadata>(
                                reg_handle, base_addr, size, dmabuf_fd, fastrak_idx),
                            1}});
        if (!inserted) {
            NIXL_ERROR << absl::StrFormat(
                "Insert memory metadata into cache failed at data: %p size: %d", addr, size);
            return NIXL_ERR_BACKEND;
        }
        out = absl::implicit_cast<nixlBackendMD *>(inserted_itr->second.mem_md.get());
        std::move(fd_cleanup).Cancel();
        NIXL_INFO << absl::StrFormat(
            "Register memory with new registration handle at data: %p size: %d", addr, size);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxoEngine::deregisterMem(nixlBackendMD *meta) {
    auto *raw_mem_md = reinterpret_cast<nixlTcpxoLocalMemoryMetadata *>(meta);
    MemoryHandle mem_handle = raw_mem_md->GetMemHandle();

    // Make sure not to double-deregister the memory handle
    if (dxs::kInvalidRegistration != mem_handle.reg_handle) {
        NIXL_INFO << absl::StrFormat(
            "Deregister memory at data: %p size: %d", mem_handle.start_addr, mem_handle.size);
        const CacheKeyType &region_info =
            std::make_pair(reinterpret_cast<uintptr_t>(mem_handle.start_addr), mem_handle.size);
        uint8_t fastrak_idx = raw_mem_md->GetFastrakIdx();
        nixlMemDev &mem_device = mem_devices_[fastrak_idx];
        absl::MutexLock lock(&mem_device.reg_mutex);
        if (auto it = mem_device.cache.find(region_info); it != mem_device.cache.end()) {
            if (mem_handle.dmabuf_fd > 0) {
                close(mem_handle.dmabuf_fd);
            }
            ASSIGN_OR_RETURN_NIXL(tcpdirect::BufferManagerClientInterface * buf_mgr,
                                  GetBufferManagerClient(fastrak_idx));
            RETURN_IF_ERROR_NIXL(buf_mgr->DeregBuf(mem_handle.reg_handle));
            if (--it->second.ref_cnt == 0) {
                mem_device.cache.erase(it);
            }
        } else {
            NIXL_ERROR << absl::StrFormat(
                "Could not find registration handle in cache for data: %p size: %d",
                mem_handle.start_addr,
                mem_handle.size);
            return NIXL_ERR_NOT_FOUND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxoEngine::loadRemoteMD(const nixlBlobDesc &input,
                              const nixl_mem_t &nixl_mem,
                              const std::string &remote_agent,
                              nixlBackendMD *&output) {
    nixlTcpxoLocalMemoryMetadata *local = nullptr;
    auto status = ::tcpxo::DeserializeMemoryMetadata(input.metaInfo, local);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    output = static_cast<nixlBackendMD *>(local);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxoEngine::unloadMD(nixlBackendMD *input) {
    delete static_cast<nixlTcpxoLocalMemoryMetadata *>(input);
    return NIXL_SUCCESS;
}

bool
nixlTcpxoEngine::CheckIfAgentLoaded(const std::string &remote_agent) const {
    throw std::runtime_error("not implemented");
}

nixl_status_t
nixlTcpxoEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    RETURN_IF_ERROR_NIXL(ValidateXferDlists(operation, local, remote));

    // If there are no descriptors, there is nothing to do
    if (local.descCount() == 0) {
        return NIXL_SUCCESS;
    }

    std::vector<DxsOpParams> uncoalesced_ops;
    uncoalesced_ops.reserve(local.descCount());

    for (int idx = 0; idx < local.descCount(); ++idx) {
        const auto &local_desc = local[idx];
        const auto &remote_desc = remote[idx];

        ASSIGN_OR_RETURN_NIXL(const auto local_fastrak_idx,
                              endpoint_manager_->GetFastrakIdxFromDevId(local_desc.devId));

        auto *remote_mem_md = static_cast<nixlTcpxoLocalMemoryMetadata *>(remote_desc.metadataP);
        const auto remote_fastrak_idx = remote_mem_md->GetFastrakIdx();
        const auto remote_mem_handle = remote_mem_md->GetMemHandle();

        auto *local_mem_md = static_cast<nixlTcpxoLocalMemoryMetadata *>(local_desc.metadataP);
        auto local_mem_handle = local_mem_md->GetMemHandle();
        const auto page_start_addr = reinterpret_cast<uintptr_t>(local_mem_handle.start_addr);
        ptrdiff_t offset = local_desc.addr - page_start_addr;

        uncoalesced_ops.push_back({{local_fastrak_idx, remote_fastrak_idx},
                                   {local_desc.addr, local_desc.len},
                                   {remote_desc.addr, remote_desc.len},
                                   local_mem_handle.reg_handle,
                                   remote_mem_handle.reg_handle,
                                   offset});
    }

    std::vector<DxsOpParams> dxs_op_params_list = CoalesceDxsOps(std::move(uncoalesced_ops));

    // Lookup the host connection and prepare transfer request for this host
    // connection.
    std::optional<std::string> notification = std::nullopt;
    if (opt_args && opt_args->hasNotif) {
        notification = opt_args->notifMsg;
    }

    nixlTcpxoBackendReqH *request_handle = nullptr;
    nixl_status_t connection_status = EnsureHostConnectionAndRun(
        remote_agent,
        [operation, &dxs_op_params_list, &request_handle, &notification](
            HostConnection &connection) {
            auto status = connection.PrepXfer(
                operation, std::move(dxs_op_params_list), request_handle, notification);
            return status.ok() ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
        });

    if (connection_status != NIXL_SUCCESS) {
        return connection_status;
    }

    handle = static_cast<nixlBackendReqH *>(request_handle);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxoEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        NIXL_ERROR << "Transfer handle is null. Call prepXfer before postXfer";
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *request_handle = static_cast<nixlTcpxoBackendReqH *>(handle);
    if (opt_args && opt_args->hasNotif) {
        request_handle->set_notification(opt_args->notifMsg);
    }
    absl::MutexLock lock(&remote_agents_mutex_);
    ASSIGN_OR_RETURN_NIXL(HostConnection * connection, LookupHostConnectionUnlocked(remote_agent));
    RETURN_IF_ERROR_NIXL(connection->PostXfer(*request_handle));
    return NIXL_IN_PROG;
}

nixl_status_t
nixlTcpxoEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *request_handle = static_cast<nixlTcpxoBackendReqH *>(handle);
    HostConnection &connection_cache = request_handle->connection();
    absl::MutexLock lock(&remote_agents_mutex_);
    ASSIGN_OR_RETURN_NIXL(HostConnection * connection,
                          LookupHostConnectionUnlocked(connection_cache.remote_name()));

    return connection->CheckXfer(*request_handle);
}

nixl_status_t
nixlTcpxoEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *request_handle = static_cast<nixlTcpxoBackendReqH *>(handle);
    HostConnection &connection_cache = request_handle->connection();
    absl::MutexLock lock(&remote_agents_mutex_);
    ASSIGN_OR_RETURN_NIXL(HostConnection * connection,
                          LookupHostConnectionUnlocked(connection_cache.remote_name()));

    return connection->ReleaseReqH(*request_handle);
}

nixl_status_t
nixlTcpxoEngine::getNotifs(notif_list_t &notif_list) {
    while (true) {
        auto notif_or_status = pending_notifs_.TryDequeue();
        if (!notif_or_status.ok()) {
            break;
        }
        notif_list.push_back(std::move(*notif_or_status));
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxoEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    NIXL_DEBUG << "Sending notification to: " << remote_agent;

    return EnsureHostConnectionAndRun(remote_agent, [&msg](HostConnection &conn) {
        RETURN_IF_ERROR_NIXL(conn.SendNotification(msg));
        return NIXL_SUCCESS;
    });
}

void
nixlTcpxoEngine::HandlePeerEvent(PeerEvent &&event) {
    auto &worker = GetWorkerForPeer(event);
    worker.EnqueueEvent(std::move(event));
}

void
nixlTcpxoEngine::HandleConnectionEvent(PeerHandle handle,
                                       const AgentAddress &socket_addr,
                                       const AgentAddress &service_addr,
                                       const std::string &agent_name,
                                       const DxsAddressExchangeMessage &dxs_addr_exchange_msg) {
    absl::MutexLock lock(&remote_agents_mutex_);
    HandleConnectionEventUnlocked(
        handle, socket_addr, service_addr, agent_name, dxs_addr_exchange_msg);
}

void
nixlTcpxoEngine::HandleConnectionEventUnlocked(
    PeerHandle handle,
    const AgentAddress &socket_addr,
    const AgentAddress &service_addr,
    const std::string &agent_name,
    const DxsAddressExchangeMessage &dxs_addr_exchange_msg) {
    std::string remote_agent;
    bool already_connected = false;

    for (auto &[name, state] : remote_agents_) {
        if (state.info.peer_control_channel_addr_ != service_addr) {
            continue;
        }
        remote_agent = name;
        if (state.conn) {
            already_connected = true;
            break;
        }

        // Store EndpointConnection info in HostConnection
        absl::flat_hash_map<EndpointPair, EndpointConnection> connection_map;
        absl::StatusOr<std::vector<DxsEndpointListenInfo>> endpoint_listen_infos =
            PrepareLocalListenHandles(connection_map);
        if (!endpoint_listen_infos.ok()) {
            NIXL_WARN << "Get all listen handles failed: " << endpoint_listen_infos.status();
            break;
        }

        if (const auto send_status =
                control_channel_.ExchangeDxsAddress(handle, *std::move(endpoint_listen_infos));
            !send_status.ok()) {
            NIXL_WARN << "Failed to send DXS listen addresses to initiator: " << remote_agent
                      << ". Cause: " << send_status.status();
            break;
        }
        NIXL_DEBUG << "Sent DXS address exchange to agent: " << remote_agent << " within "
                   << __func__;

        state.conn = std::make_unique<HostConnection>(
            remote_agent,
            handle,
            control_channel_,
            std::move(connection_map),
            params_,
            pending_notifs_,
            absl::bind_front(&nixlTcpxoEngine::HandleHostConnectionTask,
                             this,
                             HostConnectionTask{DxsConnectionEstablishTask{remote_agent}}));

        const auto status = state.conn->ParseRemoteListenMap(dxs_addr_exchange_msg);
        if (!status.ok()) {
            NIXL_WARN << "Failed to parse remote listen map: " << status.message();
            break;
        }

        NIXL_INFO << "Created HostConnection for newly connected peer: " << remote_agent;
        return;
    }

    if (remote_agent.empty()) {
        remote_agent = agent_name;
        LOG_EVERY_N_SEC(INFO, 1) << "Connection attempt from unknown peer: " << socket_addr
                                 << " with name " << remote_agent
                                 << ". Making host data structures and accepting.";
        LoadRemoteConnInfoUnlocked(remote_agent, service_addr);
        return HandleConnectionEventUnlocked(
            handle, socket_addr, service_addr, agent_name, dxs_addr_exchange_msg);
    } else if (already_connected) {
        NIXL_WARN << "Received connection from already connected agent! Rejecting duplicate "
                     "connection from peer: "
                  << remote_agent;
        if (const auto status = control_channel_.Disconnect(handle); !status.ok()) {
            NIXL_WARN << "Unable to disconnect duplicate peer " << remote_agent
                      << ". The old one has probably just now disconnected";
        }
    }
}

void
nixlTcpxoEngine::HandleDisconnectionEvent(PeerHandle handle,
                                          const AgentAddress &socket_addr,
                                          const AgentAddress &service_addr) {
    {
        absl::MutexLock lock(&remote_agents_mutex_);
        for (auto &[name, state] : remote_agents_) {
            if (state.conn && state.conn->remote_handle() == handle) {
                NIXL_DEBUG << "Remote " << service_addr << " with handle " << handle
                           << " disconnected";
                state.conn.reset();
                return;
            }
        }
    }

    // If we reach here, we did not find a HostConnection to teardown
    NIXL_WARN << "Received disconnection from unknown peer: " << service_addr
              << ". We are probably mid-connection teardown.";
    return;
}

void
nixlTcpxoEngine::HandleNotificationEvent(PeerHandle handle,
                                         const AgentAddress &service_addr,
                                         WorkloadNotificationMessage &&msg) {
    std::string remote_agent;
    {
        absl::MutexLock lock(&remote_agents_mutex_);
        for (const auto &[name, state] : remote_agents_) {
            if (state.conn && state.conn->remote_handle() == handle) {
                remote_agent = name;
                break;
            }
        }
    }
    if (remote_agent.empty()) {
        NIXL_WARN << "Received notification from unknown peer: " << service_addr
                  << ". We are probably mid-connection teardown.";
        return;
    }
    pending_notifs_.Enqueue(std::make_pair(remote_agent, std::string(msg.message())));
}

void
nixlTcpxoEngine::HandleDxsAddressExchangeEvent(PeerHandle handle,
                                               const AgentAddress &socket_addr,
                                               const DxsAddressExchangeMessage &msg) {
    NIXL_DEBUG << "HandleDxsAddressExchangeEvent called for handle " << handle;
    absl::MutexLock lock(&remote_agents_mutex_);

    HostConnection *conn{nullptr};
    for (const auto &[name, state] : remote_agents_) {
        if (state.conn && state.conn->remote_handle() == handle) {
            conn = state.conn.get();
            break;
        }
    }

    if (!conn) {
        NIXL_ERROR << "Did not find HostConnection for peer with handle " << handle
                   << ". We are probably mid-connection teardown.";
        return;
    }

    const auto status = conn->ParseRemoteListenMap(msg);
    if (!status.ok()) {
        NIXL_ERROR << "Failed to parse remote listen map: " << status.message();
        return;
    }

    if (auto status = conn->EstablishEndpointConnectionsAndDrainPendingOps(); !status.ok()) {
        NIXL_ERROR << "Failed to drain pending ops: " << status;
    }
}

void
nixlTcpxoEngine::HandleXferMessageEvent(PeerHandle handle,
                                        const AgentAddress &service_addr,
                                        const XferMessage &msg) {
    absl::MutexLock lock(&remote_agents_mutex_);

    HostConnection *conn{nullptr};
    for (auto &[name, state] : remote_agents_) {
        if (state.conn && state.conn->remote_handle() == handle) {
            conn = state.conn.get();
            break;
        }
    }

    if (!conn) {
        NIXL_ERROR << "Did not find HostConnection for peer with handle " << handle
                   << ". We are probably mid-connection teardown.";
        return;
    }

    std::vector<nixlTcpxoDxsOp> remote_ops;
    nixl_xfer_op_t op_type = (msg.type() == XFER_MESSAGE_TYPE_SEND) ? NIXL_WRITE : NIXL_READ;

    for (const auto &xfer_op_details : msg.ops()) {
        // Swap the initiator/target index to match the local/remote view
        EndpointPair endpoint_pair{xfer_op_details.target_fastrak_idx(),
                                   xfer_op_details.initiator_fastrak_idx()};

        auto cache_key =
            GetPageOrientedRegion(xfer_op_details.memory_address(), xfer_op_details.len());
        auto &mem_device = mem_devices_[xfer_op_details.target_fastrak_idx()];
        absl::MutexLock lock(&mem_device.reg_mutex);
        auto cache_it = FindRegionOrSubregionInCache(
            mem_device.cache, cache_key, xfer_op_details.memory_address(), xfer_op_details.len());
        if (cache_it == mem_device.cache.end()) {
            NIXL_ERROR << "Memory not registered on target for addr: "
                       << xfer_op_details.memory_address();
            continue;
        }

        auto reg_handle = cache_it->second.mem_md->GetMemHandle().reg_handle;
        ptrdiff_t offset = xfer_op_details.memory_address() -
            reinterpret_cast<uintptr_t>(cache_it->second.mem_md->GetMemHandle().start_addr);

        DxsOpParams params;
        params.endpoint_pair = endpoint_pair;
        params.local_mr.addr = xfer_op_details.memory_address();
        params.local_mr.len = xfer_op_details.len();
        params.local_reg_handle = reg_handle;
        params.local_page_offset = offset;

        remote_ops.emplace_back(op_type, nullptr, params, xfer_op_details.flow_idx());
    }

    if (auto status = conn->HandleRemoteXferOps(std::move(remote_ops)); !status.ok()) {
        NIXL_ERROR << "Failed to handle remote ops: " << status;
    }
}

void
nixlTcpxoEngine::HandleHostConnectionTask(HostConnectionTask task) {
    GetLeastBusyWorker().EnqueueHostConnectionTask(std::move(task));
}

void
nixlTcpxoEngine::HandleDxsConnectionEstablishTask(const std::string &remote_name) {
    absl::MutexLock lock(&remote_agents_mutex_);
    auto it = remote_agents_.find(remote_name);
    if (it != remote_agents_.end() && it->second.conn) {
        it->second.conn->ProgressDxsConnectionState(
            absl::Milliseconds(params_.fastrak_plugin_connect_timeout_ms.value),
            absl::Milliseconds(params_.fastrak_plugin_accept_timeout_ms.value));
    }
}

Worker &
nixlTcpxoEngine::GetLeastBusyWorker() {
    auto least_busy_worker_idx = 0u;
    auto least_busy_worker_pending_events = workers_[0]->pending_task_count();

    if (least_busy_worker_pending_events == 0) {
        return *workers_[0];
    }

    for (auto i = 1u; i < workers_.size(); ++i) {
        const auto curr_worker_pending_events = workers_[i]->pending_task_count();
        if (curr_worker_pending_events == 0) {
            return *workers_[i];
        }
        if (curr_worker_pending_events < least_busy_worker_pending_events) {
            least_busy_worker_idx = i;
            least_busy_worker_pending_events = curr_worker_pending_events;
        }
    }
    return *workers_[least_busy_worker_idx];
}

Worker &
nixlTcpxoEngine::GetWorkerForPeer(const PeerEvent &event) {
    Worker *assigned_worker{nullptr};
    AgentAddressKey socket_key;
    std::visit([&socket_key](const auto &e) { socket_key = e.socket_addr.MakeKey(); }, event);

    auto it = peer_to_worker_map_.find(socket_key);
    if (it != peer_to_worker_map_.end()) {
        assigned_worker = it->second;
    } else {
        assigned_worker = workers_[next_worker_idx_].get();
        peer_to_worker_map_[socket_key] = assigned_worker;
        next_worker_idx_ = (next_worker_idx_ + 1) % workers_.size();
    }

    if (std::holds_alternative<DisconnectedEvent>(event)) {
        // Erase here so we don't have to introduce a lock between this function and
        // HandleDisconnectionEvent
        peer_to_worker_map_.erase(socket_key);
    }

    return *assigned_worker;
}

std::vector<DxsOpParams>
nixlTcpxoEngine::CoalesceDxsOps(std::vector<DxsOpParams> uncoalesced_ops) {
    if (uncoalesced_ops.empty()) {
        return uncoalesced_ops;
    }

    std::sort(uncoalesced_ops.begin(),
              uncoalesced_ops.end(),
              [](const DxsOpParams &a, const DxsOpParams &b) {
                  return a.local_mr.addr < b.local_mr.addr;
              });

    std::vector<DxsOpParams> dxs_op_params_list;
    dxs_op_params_list.reserve(uncoalesced_ops.size());
    DxsOpParams current_op = uncoalesced_ops[0];
    for (size_t i = 1; i < uncoalesced_ops.size(); ++i) {
        const auto &next_op = uncoalesced_ops[i];

        if (next_op.local_mr.addr < current_op.local_mr.addr + current_op.local_mr.len) {
            NIXL_WARN << "Overlapping memory regions detected in dlist. "
                      << "Next op's local addr: " << reinterpret_cast<void *>(next_op.local_mr.addr)
                      << ", overlaps with previous op ending at: "
                      << reinterpret_cast<void *>(current_op.local_mr.addr +
                                                  current_op.local_mr.len);
            dxs_op_params_list.push_back(current_op);
            current_op = next_op;
            continue;
        }

        if (CanCoalesce(current_op, next_op)) {
            current_op.local_mr.len += next_op.local_mr.len;
            current_op.remote_mr.len += next_op.remote_mr.len;
        } else {
            dxs_op_params_list.push_back(current_op);
            current_op = next_op;
        }
    }
    dxs_op_params_list.push_back(current_op);

    NIXL_DEBUG << "Coalesced " << uncoalesced_ops.size() << " operation(s) to "
               << dxs_op_params_list.size() << " operation(s)";

    return dxs_op_params_list;
}

absl::Status
nixlTcpxoEngine::ValidateXferDlists(const nixl_xfer_op_t &operation,
                                    const nixl_meta_dlist_t &local,
                                    const nixl_meta_dlist_t &remote) const {
    if (local.descCount() != remote.descCount()) {
        return absl::InvalidArgumentError(
            "Local and remote dlists must have the same number of elements");
    }

    // Make sure local dlists are registered and buffer sizes are valid
    for (int idx = 0; idx < local.descCount(); ++idx) {
        const auto &local_desc = local[idx];
        const auto &remote_desc = remote[idx];

        const auto target_buf_len = operation == NIXL_WRITE ? remote_desc.len : local_desc.len;
        const auto source_buf_len = operation == NIXL_WRITE ? local_desc.len : remote_desc.len;
        if (target_buf_len < source_buf_len) {
            return absl::InvalidArgumentError(absl::StrCat("Target buffer length (",
                                                           target_buf_len,
                                                           ") must be >= source buffer length(",
                                                           source_buf_len,
                                                           ")"));
        }

        ASSIGN_OR_RETURN(uint8_t fastrak_idx,
                         endpoint_manager_->GetFastrakIdxFromDevId(local_desc.devId));

        nixlMemDev &mem_device = mem_devices_[fastrak_idx];
        absl::MutexLock lock(&mem_device.reg_mutex);

        const CacheKeyType region_info = GetPageOrientedRegion(local_desc.addr, local_desc.len);
        auto it = FindRegionOrSubregionInCache(
            mem_device.cache, region_info, local_desc.addr, local_desc.len);
        if (it == mem_device.cache.end()) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Local memory region not registered in cache for addr: %p size: %d",
                                reinterpret_cast<void *>(local_desc.addr),
                                local_desc.len));
        }

        auto *local_mem_md = static_cast<nixlTcpxoLocalMemoryMetadata *>(local_desc.metadataP);
        const auto &local_mem_handle = local_mem_md->GetMemHandle();
        const auto &local_addr_specified_mem_handle = it->second.mem_md->GetMemHandle();
        if (local_mem_handle != local_addr_specified_mem_handle) {
            return absl::InvalidArgumentError(
                absl::StrCat("Address (0x",
                             local_desc.addr,
                             ") with length ",
                             local_desc.len,
                             "in local dlist lies in a different page than its metadata indicates. "
                             "Page according to its address: ",
                             local_addr_specified_mem_handle,
                             ". Page in local dlsit: ",
                             local_mem_handle,
                             ". Did you frankenstein together this dlist?"));
        }
    }

    return absl::OkStatus();
}

std::optional<std::string>
nixlTcpxoEngine::ReadRawParam(const std::string &param_name) {
    std::string param_str;
    if (getInitParam(param_name, param_str) == NIXL_SUCCESS) {
        NIXL_DEBUG << "Parsed " << param_name
                   << " from engine initialization parameters. Value: " << param_str;
        return param_str;
        // Allow environment variables, in addition to engine parameters
    } else if (const char *env_str = std::getenv(param_name.c_str())) {
        NIXL_DEBUG << "Parsed " << param_name << " from environment variables. Value " << env_str;
        return env_str;
    }
    return std::nullopt;
}

void
nixlTcpxoEngine::LoadIntegerParam(IntegerParamDef &param) {
    const auto param_name = std::string(param.env_var_name);
    const auto maybe_value_str = ReadRawParam(param_name);

    if (!maybe_value_str.has_value()) {
        NIXL_DEBUG << param.env_var_name
                   << " is unset. Using default value: " << param.default_value;
        param.value = static_cast<uint64_t>(param.default_value);
        return;
    }

    int64_t value;
    if (!absl::SimpleAtoi(*maybe_value_str, &value)) {
        NIXL_WARN << "Invalid value for " << param.env_var_name << ": " << *maybe_value_str
                  << ". Using default: " << param.default_value;
        param.value = static_cast<uint64_t>(param.default_value);
        return;
    }

    if (value < param.min_value) {
        NIXL_WARN << "Too small value for " << param.env_var_name << ": " << value
                  << ". Using minimum: " << param.min_value;
        param.value = static_cast<uint64_t>(param.min_value);
    } else if (value > param.max_value) {
        NIXL_WARN << "Too large value for " << param.env_var_name << ": " << value
                  << ". Using maximum: " << param.max_value;
        param.value = static_cast<uint64_t>(param.max_value);
    } else {
        NIXL_DEBUG << "Parameter " << param.env_var_name << " set to " << value;
        param.value = static_cast<uint64_t>(value);
    }
}

void
nixlTcpxoEngine::LoadStringParam(StringParamDef &param) {
    const auto param_name = std::string(param.env_var_name);
    auto maybe_value_str = ReadRawParam(param_name);

    if (!maybe_value_str.has_value()) {
        NIXL_DEBUG << param.env_var_name
                   << " is unset. Using default value: " << param.default_value;
        param.value = param.default_value;
    } else {
        NIXL_DEBUG << "Parameter " << param.env_var_name << " set to " << *maybe_value_str;
        param.value = std::move(*maybe_value_str);
    }
}

void
nixlTcpxoEngine::LoadBoolParam(BoolParamDef &param) {
    const auto param_name = std::string(param.env_var_name);
    auto maybe_value_str = ReadRawParam(param_name);

    if (!maybe_value_str.has_value()) {
        NIXL_DEBUG << param.env_var_name
                   << " is unset. Using default value: " << param.default_value;
        param.value = param.default_value;
        return;
    }

    bool value;
    if (!absl::SimpleAtob(*maybe_value_str, &value)) {
        NIXL_WARN << "Invalid value for " << param.env_var_name << ": " << *maybe_value_str
                  << ". Using default: " << param.default_value;
        param.value = param.default_value;
    } else {
        NIXL_DEBUG << "Parameter " << param.env_var_name << " set to " << value;
        param.value = value;
    }
}

Params
nixlTcpxoEngine::InitializeParams() {
    auto params = GetUnsetParams();

    LoadIntegerParam(params.fastrak_data_transfer_timeout_ms);
    LoadIntegerParam(params.fastrak_data_transfer_slowness_ms);
    LoadIntegerParam(params.fastrak_dxs_listen_timeout_ms);
    LoadIntegerParam(params.fastrak_plugin_connect_timeout_ms);
    LoadIntegerParam(params.fastrak_plugin_accept_timeout_ms);
    LoadIntegerParam(params.fastrak_num_flows_per_dxs_connection);
    LoadIntegerParam(params.fastrak_num_control_channel_workers);
    LoadIntegerParam(params.fastrak_heartbeat_send_period_ms);
    LoadIntegerParam(params.fastrak_heartbeat_timeout_ms);

    LoadStringParam(params.fastrak_ctrl_dev);
    LoadIntegerParam(params.fastrak_rxdm_init_timeout_sec);

    LoadBoolParam(params.fastrak_loopback_only);
    LoadStringParam(params.fastrak_ifname);
    LoadStringParam(params.fastrak_socket_family);
    LoadStringParam(params.fastrak_socket_ifname);
    LoadStringParam(params.fastrak_comm_id);
    LoadBoolParam(params.fastrak_use_llcm);
    LoadBoolParam(params.fastrak_close_send_on_done);

    LoadStringParam(params.fastrak_llcm_device_directory);
    EnsureNonEmptyStringParam(params.fastrak_llcm_device_directory);

    LoadIntegerParam(params.fastrak_num_gpus_per_node);

    return params;
}

absl::StatusOr<tcpdirect::BufferManagerClientInterface * absl_nonnull>
nixlTcpxoEngine::GetBufferManagerClient(uint8_t dev_id) {
    ASSIGN_OR_RETURN(DxsEndpoint * endpoint, endpoint_manager_->GetEndpoint(dev_id));
    return endpoint->GetBufferManagerClient();
}

absl::Status
nixlTcpxoEngine::InitializeRxDM() {
    uint64_t rxdm_init_timeout = params_.fastrak_rxdm_init_timeout_sec.value;
    if (rxdm_init_timeout == 0) {
        rxdm_init_timeout = params_.fastrak_rxdm_init_timeout_sec.max_value;
    }

    auto deadline = absl::Now() + absl::Seconds(rxdm_init_timeout);
    absl::Status status;
    while (!(status = tcpdirect::rxdm_running()).ok() && absl::Now() < deadline) {
        NIXL_DEBUG << absl::StrFormat(
            "RxDM not ready after %d seconds (status: %s), retrying... (timeout "
            "%d s)",
            absl::ToInt64Seconds(absl::Now() - (deadline - absl::Seconds(rxdm_init_timeout))),
            status.ToString(),
            rxdm_init_timeout);
        absl::SleepFor(absl::Seconds(1));
    }
    if (status.ok()) {
        NIXL_INFO << "RxDM ready";
        return absl::OkStatus();
    }

    std::array<char, HOST_NAME_MAX + 1> hostname;
    if (::gethostname(hostname.data(), hostname.size()) != 0) {
        absl::SNPrintF(hostname.data(), hostname.size(), "<unknown>");
    }
    hostname[HOST_NAME_MAX] = '\0';
    NIXL_ERROR << absl::StrFormat("Timeout: RxDM not ready on %s after %d seconds: %s",
                                  hostname.data(),
                                  rxdm_init_timeout,
                                  status.ToString());
    return status;
}

absl::Status
nixlTcpxoEngine::InitializeDxs() {
    ASSIGN_OR_RETURN(
        std::unique_ptr<DxsEndpointManager> manager,
        DxsEndpointManager::InitializeNetIfs(params_.fastrak_loopback_only.value,
                                             params_.fastrak_ifname.value,
                                             params_.fastrak_socket_family.value,
                                             params_.fastrak_socket_ifname.value,
                                             params_.fastrak_comm_id.value,
                                             params_.fastrak_ctrl_dev.value,
                                             params_.fastrak_use_llcm.value,
                                             params_.fastrak_close_send_on_done.value,
                                             params_.fastrak_llcm_device_directory.value));
    endpoint_manager_ = std::move(manager);
    return absl::OkStatus();
}

absl::StatusOr<HostConnection *>
nixlTcpxoEngine::LookupHostConnectionUnlocked(const std::string &remote_agent)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(remote_agents_mutex_) const {
    auto it = remote_agents_.find(remote_agent);
    if (it == remote_agents_.end() || !it->second.conn) {
        return absl::NotFoundError(
            absl::StrFormat("Host Connection for Remote Agent %s not found", remote_agent));
    }
    return static_cast<HostConnection *>(it->second.conn.get());
}

absl::StatusOr<std::vector<DxsEndpointListenInfo>>
nixlTcpxoEngine::PrepareLocalListenHandles(
    absl::flat_hash_map<EndpointPair, EndpointConnection> &connection_map) const {
    // Get all managing local endpoints to iterate through
    const std::vector<std::unique_ptr<DxsEndpoint>> &local_endpoints =
        endpoint_manager_->endpoints();
    const auto highest_fastrak_idx_for_workload = params_.fastrak_num_gpus_per_node.value;

    // Build DxsEndpointListenInfo messages and send them directly to `ControlChannel::Connect()`
    // Each represent a pair of (local, remote) endpoints1
    std::vector<DxsEndpointListenInfo> endpoint_listen_infos;
    endpoint_listen_infos.reserve(local_endpoints.size() * highest_fastrak_idx_for_workload);

    for (auto &endpoint : local_endpoints) {
        uint8_t local_fastrak_idx = endpoint->fastrak_idx();
        for (size_t remote_fastrak_idx = 0; remote_fastrak_idx < highest_fastrak_idx_for_workload;
             ++remote_fastrak_idx) {
            ASSIGN_OR_RETURN(
                DxsConnection conn,
                endpoint->Listen(params_.fastrak_num_flows_per_dxs_connection.value,
                                 absl::Milliseconds(params_.fastrak_dxs_listen_timeout_ms.value)));

            DxsEndpointListenInfo endpoint_listen_info;
            endpoint_listen_info.set_local_fastrak_idx(local_fastrak_idx);
            endpoint_listen_info.set_remote_fastrak_idx(remote_fastrak_idx);
            for (auto &flow : conn.flows()) {
                DxsAddress *listen_handle = endpoint_listen_info.add_listen_handles();
                listen_handle->mutable_addr()->set_addr(flow.listen_socket->Address());
                listen_handle->set_port(flow.listen_socket->Port());
            }

            auto [unused_it, inserted] =
                connection_map.insert({EndpointPair{
                                           .local_fastrak_idx = local_fastrak_idx,
                                           .remote_fastrak_idx = remote_fastrak_idx,
                                       },
                                       EndpointConnection{
                                           .endpoint_ = *endpoint,
                                           .conn_ = std::move(conn),
                                       }});
            if (!inserted) {
                auto error_msg = absl::StrCat("Store EndpointConnection error within ", __func__);
                NIXL_ERROR << error_msg;
                return absl::InternalError(error_msg);
            }
            endpoint_listen_infos.push_back(std::move(endpoint_listen_info));
        }
    }
    return endpoint_listen_infos;
}

} // namespace tcpxo
