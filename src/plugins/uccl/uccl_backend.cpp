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
#include "uccl_backend.h"
#include "serdes/serdes.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>

// Parse connection string in format: ip_addr:port?gpu_index
bool
parseConnectionString(const std::string &conn_str,
                      std::unique_ptr<char[]> &ip_addr,
                      int &port,
                      std::string &gpu_bdf) {
    // Format: ip:port?bdf  (e.g. "10.0.0.1:41861?0000:18:00.0")
    size_t colon_pos = conn_str.find(':');
    if (colon_pos == std::string::npos) {
        NIXL_ERROR << "Invalid connection string format: missing colon separator";
        return false;
    }
    size_t question_pos = conn_str.find('?', colon_pos);
    if (question_pos == std::string::npos) {
        NIXL_ERROR << "Invalid connection string format: missing question mark separator";
        return false;
    }

    std::string ip_str = conn_str.substr(0, colon_pos);
    ip_addr = std::make_unique<char[]>(ip_str.length() + 1);
    strcpy(ip_addr.get(), ip_str.c_str());

    std::string port_str = conn_str.substr(colon_pos + 1, question_pos - colon_pos - 1);
    try {
        port = std::stoi(port_str);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Invalid port number: " << port_str;
        return false;
    }

    gpu_bdf = conn_str.substr(question_pos + 1);

    return true;
}

int
getNixlParam(const nixl_b_params_t *custom_params, const std::string &key, int default_value) {
    if (!custom_params) {
        return default_value;
    }

    auto it = custom_params->find(key);
    if (it == custom_params->end()) {
        return default_value;
    }

    try {
        return std::stoi(it->second);
    }
    catch (const std::exception &) {
        return default_value;
    }
}

nixlUcclEngine::nixlUcclEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      stop_listener_(false) {

    local_agent_name_ = init_params->localAgent;
    nixl_b_params_t *custom_params = init_params->customParams;

    size_t num_cpus = getNixlParam(custom_params, "num_cpus", 4);
    int in_python = getNixlParam(custom_params, "in_python", 1);
    engine_ = uccl_engine_create(num_cpus, (in_python == 1));
    NIXL_DEBUG << "UCCL engine created";

    listener_thread_ = std::thread(&nixlUcclEngine::startListener, this);
}

nixlUcclEngine::~nixlUcclEngine() {
    stop_listener_ = true;

    if (engine_) {
        uccl_engine_stop_accept(engine_);
    }

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(mem_mutex_);
        for (auto &[addr, priv] : mem_reg_info_) {
            if (priv && priv->mr_id != 0) {
                uccl_engine_mr_destroy(engine_, priv->mr_id);
            }
            delete priv;
        }
        mem_reg_info_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        std::set<std::string> destroyed_agents;
        for (auto &[agent_name, conn_id] : connected_agents_) {
            if (destroyed_agents.find(agent_name) == destroyed_agents.end()) {
                uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_id);
                if (conn) {
                    uccl_engine_conn_destroy(conn);
                    destroyed_agents.insert(agent_name);
                }
            }
        }
        connected_agents_.clear();
    }

    if (engine_) {
        uccl_engine_destroy(engine_);
        engine_ = nullptr;
    }
}

void
nixlUcclEngine::startListener() {
    // The listener waits for connections from remote agents
    NIXL_DEBUG << "UCCL accepting connections";
    while (!stop_listener_) {

        char ip_buf[256];
        int remote_gpu_idx;
        uccl_conn_t *conn = uccl_engine_accept(engine_, ip_buf, sizeof(ip_buf), &remote_gpu_idx);
        if (!conn) {
            // Check if we should stop
            if (stop_listener_) {
                NIXL_DEBUG << "Listener thread stopping";
                break;
            }
            NIXL_ERROR << "Failed to accept connection from remote agent";
            continue;
        }
        // Start the listener thread to send/get notifications from the remote agent
        uccl_engine_start_listener(conn);
        NIXL_DEBUG << "Connected to remote agent: " << ip_buf;
        {
            std::lock_guard<std::mutex> lock(conn_mutex_);
            connected_agents_[ip_buf] = reinterpret_cast<uint64_t>(conn);
        }
    }
}

nixl_mem_list_t
nixlUcclEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);

    return mems;
}

nixl_status_t
nixlUcclEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    nixlUcclBackendMD *priv = (nixlUcclBackendMD *)meta;

    // Export fifo_item as hex string (FIFO_SIZE * 2 hex chars).
    str.clear();
    str.reserve(FIFO_SIZE * 2 + 2 + IPC_INFO_SIZE * 2);
    for (int i = 0; i < FIFO_SIZE; i++) {
        char hex[3];
        snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned char>(priv->fifo_item[i]));
        str += hex;
    }

    // Append IPC info: "01" + hex(ipc_info) if has_ipc, else "00".
    // This enables cross-process local (IPC) transfers.
    str += (priv->has_ipc ? "01" : "00");
    if (priv->has_ipc) {
        for (int i = 0; i < IPC_INFO_SIZE; i++) {
            char hex[3];
            snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned char>(priv->ipc_info[i]));
            str += hex;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::getConnInfo(std::string &str) const {
    if (!engine_) {
        return NIXL_ERR_BACKEND;
    }

    char *metadata = nullptr;
    int result = uccl_engine_get_metadata(engine_, &metadata);
    if (result != 0 || !metadata) {
        return NIXL_ERR_BACKEND;
    }

    str = std::string(metadata);
    delete[] metadata;
    NIXL_DEBUG << "UCCL engine metadata: " << str;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                   const std::string &remote_conn_info) {
    // Parse remote_conn_info and establish connection using UCCL engine
    NIXL_DEBUG << "UCCL engine remote_agent: " << remote_agent
               << " loadRemoteConnInfo: " << remote_conn_info;
    std::lock_guard<std::mutex> lock(conn_mutex_);

    std::unique_ptr<char[]> ip_addr;
    int port = 0;
    std::string gpu_bdf;

    if (!parseConnectionString(remote_conn_info, ip_addr, port, gpu_bdf)) {
        return NIXL_ERR_BACKEND;
    }

    uccl_conn_t *conn = nullptr;

    NIXL_DEBUG << "Connecting to " << ip_addr.get() << ":" << port << "?gpu=" << gpu_bdf;
    conn = uccl_engine_connect(engine_, ip_addr.get(), gpu_bdf.c_str(), port, false);
    if (!conn) {
        NIXL_ERROR << "Failed to connect to remote agent " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Successfully connected to remote agent " << remote_agent;
    // Start the listener thread for notifications
    uccl_engine_start_listener(conn);

    connected_agents_[remote_agent] = reinterpret_cast<uint64_t>(conn);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::connect(const std::string &remote_agent) {
    // We cannot establish the local (IPC) connection now because the UCCL engine
    // is lazily initialized on first register_memory (which sets local_gpu_idx and
    // creates the shm inbox rings).  Record the agent name; the actual connection
    // is established after memory registration
    std::lock_guard<std::mutex> lock(conn_mutex_);
    pending_local_agent_ = remote_agent;
    NIXL_DEBUG << "Deferred local connect for agent " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::prepareLocalConn() const {
    // Must be called after register_memory so the UCCL engine is initialized.
    std::string agent_name;
    {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        if (pending_local_agent_.empty()) {
            return NIXL_SUCCESS;
        }
        if (connected_agents_.count(pending_local_agent_)) {
            pending_local_agent_.clear();
            return NIXL_SUCCESS;
        }
        agent_name = pending_local_agent_;
    }

    std::string conn_info;
    if (getConnInfo(conn_info) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get own conn info for local connect";
        return NIXL_ERR_BACKEND;
    }

    std::unique_ptr<char[]> ip_addr;
    int port = 0;
    std::string gpu_bdf;
    if (!parseConnectionString(conn_info, ip_addr, port, gpu_bdf)) {
        return NIXL_ERR_BACKEND;
    }

    uccl_conn_t *conn = uccl_engine_connect(engine_,
                                            ip_addr.get(),
                                            gpu_bdf.c_str(),
                                            port,
                                            /*same_process=*/true);
    if (!conn) {
        NIXL_ERROR << "Failed to establish local connection for agent " << agent_name;
        return NIXL_ERR_BACKEND;
    }

    {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        NIXL_DEBUG << "Same-process local connection established for agent " << agent_name;
        connected_agents_[agent_name] = reinterpret_cast<uint64_t>(conn);
        pending_local_agent_.clear();
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::disconnect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    auto conn_iter = connected_agents_.find(remote_agent);
    if (conn_iter == connected_agents_.end()) {
        NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }
    uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
    if (!conn) {
        NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    if (conn) {
        NIXL_DEBUG << "Disconnecting from agent: " << remote_agent;
        uccl_engine_conn_destroy(conn);
        connected_agents_.erase(remote_agent);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    std::lock_guard<std::mutex> lock(mem_mutex_);

    if (mem_reg_info_.count(mem.addr)) {
        auto priv = mem_reg_info_[mem.addr];
        NIXL_DEBUG << "Registering memory: " << std::hex << mem.addr << ", len: " << std::dec
                   << mem.len;
        priv->ref_cnt++;
        out = priv;
        return NIXL_SUCCESS;
    }

    // Register memory with UCCL engine
    uccl_mr_t mr;
    int result = uccl_engine_reg(engine_, mem.addr, mem.len, mr);
    if (result != 0) {
        NIXL_ERROR << "Failed to register memory with UCCL engine";
        return NIXL_ERR_BACKEND;
    }

    auto priv = new nixlUcclBackendMD(true);
    priv->addr = (void *)mem.addr;
    priv->length = mem.len;
    priv->ref_cnt = 1;
    priv->mr_id = mr;

    // Pre-compute fifo_item for one-sided RDMA operations
    result = uccl_engine_prepare_fifo(engine_, mr, (void *)mem.addr, mem.len, priv->fifo_item);
    if (result != 0) {
        NIXL_ERROR << "Failed to prepare fifo_item for memory region";
        uccl_engine_mr_destroy(engine_, mr);
        delete priv;
        return NIXL_ERR_BACKEND;
    }

    // Retrieve pre-computed IPC info for GPU buffers (used by cross-process local transfers)
    bool has_ipc = false;
    uccl_engine_get_ipc_info(engine_, mem.addr, priv->ipc_info, &has_ipc);
    priv->has_ipc = has_ipc;

    out = priv;
    mem_reg_info_[mem.addr] = priv;
    NIXL_DEBUG << "Registering memory: " << std::hex << mem.addr << " Device: " << mem.devId
               << " ref_cnt: " << priv->ref_cnt << " mr_id: " << priv->mr_id;

    // Prepare the deferred local connection if there are pending agents.
    nixl_status_t conn_status = prepareLocalConn();
    if (conn_status != NIXL_SUCCESS) {
        NIXL_WARN << "Deferred local connection failed";
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::deregisterMem(nixlBackendMD *meta) {
    std::lock_guard<std::mutex> lock(mem_mutex_);
    auto priv = static_cast<nixlUcclBackendMD *>(meta);
    priv->ref_cnt--;
    if (priv->ref_cnt > 0) return NIXL_SUCCESS;

    // Deregister memory from UCCL engine
    uccl_engine_mr_destroy(engine_, priv->mr_id);
    NIXL_DEBUG << "Deregistered memory: " << std::hex << priv->addr << " mr_id: " << priv->mr_id;

    mem_reg_info_.erase((uint64_t)priv->addr);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    nixlUcclBackendMD *input_md = (nixlUcclBackendMD *)input;
    NIXL_DEBUG << "UCCL Load Local MD: " << std::hex << input_md->addr
               << "Meta Info:" << input_md->mr_id;

    output = new nixlUcclBackendMD(true);
    nixlUcclBackendMD *output_md = static_cast<nixlUcclBackendMD *>(output);
    output_md->addr = input_md->addr;
    output_md->length = input_md->length;
    output_md->ref_cnt = 1;
    output_md->mr_id = input_md->mr_id;
    memcpy(output_md->fifo_item, input_md->fifo_item, FIFO_SIZE);
    memcpy(output_md->ipc_info, input_md->ipc_info, IPC_INFO_SIZE);
    output_md->has_ipc = input_md->has_ipc;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::loadRemoteMD(const nixlBlobDesc &input,
                             const nixl_mem_t &nixl_mem,
                             const std::string &remote_agent,
                             nixlBackendMD *&output) {
    NIXL_DEBUG << "UCCL Load Remote MD: " << std::hex << input.addr
               << " Meta Info:" << input.metaInfo << " remote_agent: " << remote_agent;

    output = new nixlUcclBackendMD(true);
    nixlUcclBackendMD *output_md = static_cast<nixlUcclBackendMD *>(output);
    output_md->addr = (void *)input.addr;
    output_md->length = input.len;
    output_md->ref_cnt = 1;

    // Decode fifo_item and optional IPC info from hex string.
    const std::string &hex_str = input.metaInfo;
    size_t min_len = FIFO_SIZE * 2; // 128 chars for fifo_item

    if (hex_str.length() < min_len) {
        NIXL_ERROR << "Invalid metaInfo hex string length: " << hex_str.length()
                   << " (expected at least " << min_len << ")";
        delete output_md;
        output = nullptr;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Decode fifo_item (first FIFO_SIZE*2 hex chars)
    for (int i = 0; i < FIFO_SIZE; i++) {
        std::string byte_str = hex_str.substr(i * 2, 2);
        output_md->fifo_item[i] = static_cast<char>(strtoul(byte_str.c_str(), NULL, 16));
    }

    // Decode IPC info (appended after fifo_item)
    size_t pos = min_len;
    if (hex_str.length() >= pos + 2) {
        std::string ipc_flag = hex_str.substr(pos, 2);
        pos += 2;
        if (ipc_flag == "01") {
            if (hex_str.length() < pos + IPC_INFO_SIZE * 2) {
                NIXL_ERROR << "IPC flag set but payload truncated: got " << (hex_str.length() - pos)
                           << " hex chars, expected " << (IPC_INFO_SIZE * 2);
                delete output_md;
                output = nullptr;
                return NIXL_ERR_INVALID_PARAM;
            }
            for (int i = 0; i < IPC_INFO_SIZE; i++) {
                std::string byte_str = hex_str.substr(pos + i * 2, 2);
                output_md->ipc_info[i] = static_cast<char>(strtoul(byte_str.c_str(), NULL, 16));
            }
            output_md->has_ipc = true;
        } else if (ipc_flag != "00") {
            NIXL_ERROR << "Unknown IPC flag in metaInfo: " << ipc_flag;
            delete output_md;
            output = nullptr;
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::unloadMD(nixlBackendMD *input) {
    nixlUcclBackendMD *md = (nixlUcclBackendMD *)input;
    delete md;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    nixlUcclBackendMD *lmd;
    nixlUcclBackendMD *rmd;
    handle = nullptr;

    NIXL_DEBUG << "UCCL PrepXfer: " << operation << " remote_agent: " << remote_agent;

    uccl_conn_t *conn = nullptr;
    {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        // Get the connection for this remote agent
        auto conn_iter = connected_agents_.find(remote_agent);
        if (conn_iter == connected_agents_.end()) {
            NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
            return NIXL_ERR_BACKEND;
        }
        conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
        if (!conn) {
            NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
            return NIXL_ERR_BACKEND;
        }
    }

    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local and remote descriptor counts don't match: " << lcnt << " != " << rcnt;
        return NIXL_ERR_INVALID_PARAM;
    }

    handle = new nixlUcclReqH(conn);
    nixlUcclReqH *uccl_handle = static_cast<nixlUcclReqH *>(handle);

    uccl_handle->fifo_items.resize(lcnt);

    // Check if this is a local connection (same-process or cross-process on same node)
    bool is_local_conn = uccl_engine_conn_is_local(conn);
    bool is_same_process = (remote_agent == local_agent_name_);

    if (is_local_conn) {
        uccl_handle->ipc_infos.resize(lcnt);
    }

    std::lock_guard<std::mutex> lock(mem_mutex_);
    for (size_t i = 0; i < lcnt; i++) {
        lmd = (nixlUcclBackendMD *)local[i].metadataP;
        rmd = (nixlUcclBackendMD *)remote[i].metadataP;
        size_t rsize = remote[i].len;
        uintptr_t remote_addr = remote[i].addr;

        // Validate the local address is registered
        auto local_mem_iter = mem_reg_info_.find((uint64_t)lmd->addr);
        if (local_mem_iter == mem_reg_info_.end()) {
            NIXL_ERROR << "Local memory not registered for address:" << std::hex << lmd->addr;
            return NIXL_ERR_BACKEND;
        }

        // Deserialize fifo_item from char[] into FifoItem struct
        deserialize_fifo_item(rmd->fifo_item, &uccl_handle->fifo_items[i]);

        uccl_engine_update_fifo(uccl_handle->fifo_items[i], remote_addr, rsize);

        // For local connections: prepare IPC info from remote MD
        // - has_ipc (GPU memory): works for both same-process and cross-process
        // - !has_ipc (DRAM): only works for same-process (direct_addr)
        //   Cross-process DRAM falls back to RDMA
        if (is_local_conn) {
            if (rmd->has_ipc) {
                uccl_handle->ipc_infos[i].assign(rmd->ipc_info, rmd->ipc_info + IPC_INFO_SIZE);
                uccl_engine_update_ipc_info(
                    uccl_handle->ipc_infos[i].data(), remote_addr, (uintptr_t)rmd->addr, rsize);
                uccl_handle->use_ipc = true;
            } else if (is_same_process) {
                uccl_handle->ipc_infos[i].assign(IPC_INFO_SIZE, 0);
                uccl_handle->use_ipc = true;
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    nixlUcclReqH *uccl_handle;
    nixlUcclBackendMD *lmd;

    NIXL_DEBUG << "UCCL PostXfer: " << operation << " remote_agent: " << remote_agent;

    uccl_conn_t *conn = nullptr;
    {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        // Get the connection for this remote agent
        auto conn_iter = connected_agents_.find(remote_agent);
        if (conn_iter == connected_agents_.end()) {
            NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
            return NIXL_ERR_BACKEND;
        }

        conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
        if (!conn) {
            NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
            return NIXL_ERR_BACKEND;
        }
    }

    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local and remote descriptor counts don't match: " << lcnt << " != " << rcnt;
        return NIXL_ERR_INVALID_PARAM;
    }

    std::vector<uccl_mr_t> mr_ids;
    std::vector<void *> addr_v;
    std::vector<size_t> size_v;

    std::lock_guard<std::mutex> lock(mem_mutex_);
    for (size_t i = 0; i < lcnt; i++) {
        lmd = (nixlUcclBackendMD *)local[i].metadataP;
        size_t lsize = local[i].len;
        size_t rsize = remote[i].len;
        // Use local[i].addr for the actual iovec address, not lmd->addr (which is base address)
        uintptr_t local_addr = local[i].addr;

        if (lsize != rsize) {
            NIXL_ERROR << "Local and remote sizes don't match: " << lsize << " != " << rsize;
            return NIXL_ERR_INVALID_PARAM;
        }

        // Validate the local address is registered
        auto local_mem_iter = mem_reg_info_.find((uint64_t)lmd->addr);
        if (local_mem_iter == mem_reg_info_.end()) {
            NIXL_ERROR << "Local memory not registered for base address: " << std::hex << lmd->addr;
            return NIXL_ERR_BACKEND;
        }

        auto local_priv = local_mem_iter->second;

        mr_ids.push_back(local_priv->mr_id);
        addr_v.push_back((void *)local_addr);
        size_v.push_back(lsize);
    }

    // Perform a vector read/write operation
    int result = 0;
    uint64_t transfer_id = 0;
    uccl_handle = static_cast<nixlUcclReqH *>(handle);

    // Build optional IPC info pointers for cross-process local transfers
    std::vector<char *> ipc_ptrs;
    if (uccl_handle->use_ipc) {
        ipc_ptrs.resize(lcnt);
        for (size_t i = 0; i < lcnt; i++) {
            ipc_ptrs[i] = uccl_handle->ipc_infos[i].data();
        }
        NIXL_DEBUG << "Using IPC for transfer";
    }

    switch (operation) {
    case NIXL_READ: {
        result = uccl_engine_read_vector(
            conn, mr_ids, addr_v, size_v, uccl_handle->fifo_items, lcnt, &transfer_id, ipc_ptrs);
        break;
    }
    case NIXL_WRITE: {
        result = uccl_engine_write_vector(
            conn, mr_ids, addr_v, size_v, uccl_handle->fifo_items, lcnt, &transfer_id, ipc_ptrs);
        break;
    }
    default:
        NIXL_ERROR << "Unsupported operation type: " << operation;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (result != 0) {
        NIXL_ERROR << "UCCL operation failed with result: " << result;
        return NIXL_ERR_BACKEND;
    }

    if (!handle) {
        handle = new nixlUcclReqH(conn);
    }
    uccl_handle->transfer_id = transfer_id;

    NIXL_DEBUG << "Successfully posted vector " << (operation == NIXL_READ ? "READ" : "WRITE")
               << " operation with " << lcnt << " iovecs, transfer_id: " << transfer_id;

    if (opt_args && opt_args->hasNotif) {
        uccl_handle->notif_msg = opt_args->notifMsg;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlUcclEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "Invalid handle provided to checkXfer";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlUcclReqH *uccl_handle = dynamic_cast<nixlUcclReqH *>(handle);
    if (!uccl_handle) {
        NIXL_ERROR << "Invalid handle type for UCCL backend";
        return NIXL_ERR_INVALID_PARAM;
    }

    uccl_conn_t *conn = uccl_handle->conn;
    if (!conn) {
        NIXL_ERROR << "No connection found in handle";
        return NIXL_ERR_BACKEND;
    }

    bool is_done = uccl_engine_xfer_status(conn, uccl_handle->transfer_id);
    if (is_done) {
        nixlSerDes ser_des;
        ser_des.addStr("msg", uccl_handle->notif_msg);
        std::string serialized = ser_des.exportStr();

        if (serialized.size() > sizeof(notify_msg_t::msg)) {
            NIXL_ERROR << "Notification message too large: " << serialized.size()
                       << " bytes, max: " << sizeof(notify_msg_t::msg) << " bytes";
        } else {
            notify_msg_t notify_msg = {};
            strncpy(notify_msg.name, local_agent_name_.c_str(), sizeof(notify_msg.name) - 1);
            memcpy(notify_msg.msg, serialized.c_str(), serialized.size());

            int result = uccl_engine_send_notif(conn, &notify_msg);
            if (result < 0) {
                NIXL_ERROR << "Failed to send notify message";
                return NIXL_ERR_BACKEND;
            }
            NIXL_DEBUG << "Transfer complete, sent notification: " << uccl_handle->notif_msg;
        }
        return NIXL_SUCCESS;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlUcclEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_SUCCESS;
    }

    nixlUcclReqH *uccl_handle = dynamic_cast<nixlUcclReqH *>(handle);
    if (uccl_handle) {
        delete uccl_handle;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::getNotifs(notif_list_t &notif_list) {
    if (notif_list.size() != 0) return NIXL_ERR_INVALID_PARAM;

    std::vector<notify_msg_t> notify_msgs = uccl_engine_get_notifs();
    for (size_t i = 0; i < notify_msgs.size(); i++) {
        size_t msg_len = sizeof(notify_msgs[i].msg);
        std::string serialized_str(notify_msgs[i].msg, msg_len);
        nixlSerDes ser_des;
        nixl_status_t ret = ser_des.importStr(serialized_str);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to deserialize notification message";
            continue;
        }
        std::string remote_name(notify_msgs[i].name);
        std::string msg = ser_des.getStr("msg");

        notif_list.push_back(std::make_pair(remote_name, msg));
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    auto conn_iter = connected_agents_.find(remote_agent);
    if (conn_iter == connected_agents_.end()) {
        NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
    if (!conn) {
        NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    nixlSerDes ser_des;
    ser_des.addStr("msg", msg);
    std::string serialized = ser_des.exportStr();

    if (serialized.size() > sizeof(notify_msg_t::msg)) {
        NIXL_ERROR << "Notification message too large: " << serialized.size()
                   << " bytes, max: " << sizeof(notify_msg_t::msg) << " bytes";
        return NIXL_ERR_INVALID_PARAM;
    }

    notify_msg_t notify_msg;
    memset(&notify_msg, 0, sizeof(notify_msg));
    strncpy(notify_msg.name, local_agent_name_.c_str(), sizeof(notify_msg.name) - 1);
    memcpy(notify_msg.msg, serialized.c_str(), serialized.size());

    int result = uccl_engine_send_notif(conn, &notify_msg);
    if (result < 0) {
        NIXL_ERROR << "Failed to send notify message";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}
