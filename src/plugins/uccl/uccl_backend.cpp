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
                      int &gpu_index) {
    // Exit with error if neither : or ? is found in conn_str
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

    std::string gpu_str = conn_str.substr(question_pos + 1);
    try {
        gpu_index = std::stoi(gpu_str);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Invalid GPU index: " << gpu_str;
        return false;
    }

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
    : nixlBackendEngine(init_params) {

    local_agent_name_ = init_params->localAgent;
    nixl_b_params_t *custom_params = init_params->customParams;

    size_t num_cpus = getNixlParam(custom_params, "num_cpus", 4);
    int in_python = getNixlParam(custom_params, "in_python", 1);
    engine_ = uccl_engine_create(num_cpus, (in_python == 1));
    NIXL_DEBUG << "UCCL engine created";

    listener_thread_ = std::thread(&nixlUcclEngine::startListener, this);
}

nixlUcclEngine::~nixlUcclEngine() {
    {
        std::lock_guard<std::mutex> lock(mem_mutex_);
        for (auto &[addr, priv] : mem_reg_info_) {
            if (priv && priv->mr_id != 0) {
                uccl_mr_t *mr = reinterpret_cast<uccl_mr_t *>(priv->mr_id);
                if (mr) {
                    uccl_engine_mr_destroy(mr);
                }
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

    if (listener_thread_.joinable()) {
        listener_thread_.detach();
    }

    if (engine_) {
        // Add a small delay to allow UCCL internal cleanup to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        uccl_engine_destroy(engine_);
        engine_ = nullptr;
    }
}

void
nixlUcclEngine::startListener() {
    // The listener waits for connections from remote agents
    NIXL_DEBUG << "UCCL accepting connections";
    while (true) {
        char ip_buf[256];
        int remote_gpu_idx;
        uccl_conn_t *conn = uccl_engine_accept(engine_, ip_buf, sizeof(ip_buf), &remote_gpu_idx);
        if (!conn) {
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
    str = std::to_string(priv->mr_id);

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
    int gpu_index = 0;

    if (!parseConnectionString(remote_conn_info, ip_addr, port, gpu_index)) {
        return NIXL_ERR_BACKEND;
    }

    uccl_conn_t *conn = nullptr;

    NIXL_DEBUG << "Connecting to " << ip_addr.get() << ":" << port << "?gpu=" << gpu_index
               << std::endl;
    conn = uccl_engine_connect(engine_, ip_addr.get(), gpu_index, port);
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
    // Unused
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
    uccl_mr_t *mr = uccl_engine_reg(engine_, mem.addr, mem.len);
    if (!mr) {
        NIXL_ERROR << "Failed to register memory with UCCL engine";
        return NIXL_ERR_BACKEND;
    }

    auto priv = new nixlUcclBackendMD(true);
    priv->addr = (void *)mem.addr;
    priv->length = mem.len;
    priv->ref_cnt = 1;
    priv->mr_id = reinterpret_cast<uint64_t>(mr); // Store the memory region handle
    out = priv;
    mem_reg_info_[mem.addr] = priv;
    NIXL_DEBUG << "Registering memory: " << mem.addr << "Device: " << mem.devId
               << " ref_cnt: " << priv->ref_cnt << " mr_id: " << priv->mr_id;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::deregisterMem(nixlBackendMD *meta) {
    std::lock_guard<std::mutex> lock(mem_mutex_);
    auto priv = static_cast<nixlUcclBackendMD *>(meta);
    priv->ref_cnt--;
    if (priv->ref_cnt > 0) return NIXL_SUCCESS;

    // Deregister memory from UCCL engine
    if (priv->mr_id != 0) {
        uccl_mr_t *mr = reinterpret_cast<uccl_mr_t *>(priv->mr_id);
        if (mr) {
            uccl_engine_mr_destroy(mr);
            NIXL_DEBUG << "Deregistered memory: " << priv->addr << " mr_id: " << priv->mr_id;
        }
        priv->mr_id = 0;
    }

    mem_reg_info_.erase((uint64_t)priv->addr);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    nixlUcclBackendMD *input_md = (nixlUcclBackendMD *)input;
    NIXL_DEBUG << "UCCL Load Local MD: " << input_md->addr << "Meta Info:" << input_md->mr_id;

    nixlUcclBackendMD *output_md = (nixlUcclBackendMD *)output;
    output_md->addr = (void *)input_md->addr;
    output_md->length = input_md->length;
    output_md->ref_cnt = 1;
    output_md->mr_id = reinterpret_cast<uint64_t>(input_md->mr_id);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcclEngine::loadRemoteMD(const nixlBlobDesc &input,
                             const nixl_mem_t &nixl_mem,
                             const std::string &remote_agent,
                             nixlBackendMD *&output) {
    NIXL_DEBUG << "UCCL Load Remote MD: " << input.addr << "Meta Info:" << input.metaInfo
               << " remote_agent: " << remote_agent;

    output = new nixlUcclBackendMD(true);
    nixlUcclBackendMD *output_md = static_cast<nixlUcclBackendMD *>(output);
    output_md->addr = (void *)input.addr;
    output_md->length = input.len;
    output_md->ref_cnt = 1;
    output_md->mr_id = strtoul(input.metaInfo.c_str(), NULL, 10);

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
    int result = 0;
    nixlUcclBackendMD *lmd;
    nixlUcclBackendMD *rmd;
    bool rcmode = false;
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

    const char *uccl_rcmode = std::getenv("UCCL_RCMODE");
    rcmode = (uccl_rcmode && std::strcmp(uccl_rcmode, "1") == 0);

    if (operation == NIXL_READ) {
        if (!rcmode) {
            NIXL_ERROR
                << "UCCL_RCMODE environment variable must be set to 1 for NIXL_READ operations";
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    // Collect all tx_data into vectors for batch sending
    std::vector<md_t> md_vector;
    std::vector<nixlUcclBackendMD *> local_priv_vector;

    std::lock_guard<std::mutex> lock(mem_mutex_);
    for (size_t i = 0; i < lcnt; i++) {
        lmd = (nixlUcclBackendMD *)local[i].metadataP;
        rmd = (nixlUcclBackendMD *)remote[i].metadataP;
        size_t rsize = remote[i].len;
        uintptr_t local_addr = local[i].addr;
        uintptr_t remote_addr = remote[i].addr;

        NIXL_DEBUG << "prepXfer iovec[" << i << "]: local[i].addr=" << std::hex << local_addr
                   << ", lmd->addr=" << std::hex << lmd->addr << ", lmd->mr_id=" << std::dec
                   << lmd->mr_id << ", remote[i].addr=" << std::hex << remote_addr
                   << ", rmd->addr=" << std::hex << rmd->addr << ", rmd->mr_id=" << std::dec
                   << rmd->mr_id;

        // Validate the local address is registered
        auto local_mem_iter = mem_reg_info_.find((uint64_t)lmd->addr);
        if (local_mem_iter == mem_reg_info_.end()) {
            NIXL_ERROR << "Local memory not registered for address:" << std::hex << lmd->addr;
            return NIXL_ERR_BACKEND;
        }

        auto local_priv = local_mem_iter->second;
        if (local_priv->mr_id == 0) {
            NIXL_ERROR << "Local memory region not properly registered";
            return NIXL_ERR_BACKEND;
        }

        // Prepare the memory region metadata for batch sending
        md_t md;
        tx_msg_t tx_data;
        tx_data.data_ptr = remote_addr;
        tx_data.data_size = rsize;

        // RC mode is supported for both READ/WRITE operations
        // UC mode is supported only for WRITE operations
        md.op = rcmode ? UCCL_RW_RC : UCCL_WRITE;
        md.data.tx_data = tx_data;

        // Add to vectors for batch processing
        md_vector.push_back(md);
        local_priv_vector.push_back(local_priv);
    }

    // Send all tx_data as a vector
    result = uccl_engine_send_tx_md_vector(conn, md_vector.data(), md_vector.size());
    if (result < 0) {
        NIXL_ERROR << "Failed to send transfer metadata vector";
        return NIXL_ERR_BACKEND;
    }

    if (rcmode) {
        if (!handle) {
            handle = new nixlUcclReqH(conn);
        }
        nixlUcclReqH *uccl_handle = static_cast<nixlUcclReqH *>(handle);

        uccl_handle->fifo_items.clear();
        uccl_handle->fifo_items.resize(local_priv_vector.size());

        for (size_t i = 0; i < local_priv_vector.size(); i++) {
            char fifo_item[FIFO_ITEM_SIZE];
            int retry_count = 0;
            const int max_retries = 50;
            do {
                result = uccl_engine_get_fifo_item(conn, i, &fifo_item);
                if (result == 0) {
                    memcpy(uccl_handle->fifo_items[i].data(), fifo_item, FIFO_ITEM_SIZE);
                    break;
                }
                retry_count++;
                if (retry_count < max_retries) {
                    NIXL_DEBUG << "Failed to get FIFO item, retry " << retry_count << "/"
                               << max_retries << " for item " << i;
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            } while (retry_count < max_retries);

            if (result != 0) {
                NIXL_ERROR << "Failed to get FIFO item after " << max_retries
                           << " retries for item " << i;
                return NIXL_ERR_BACKEND;
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
    nixlUcclBackendMD *rmd;
    bool rcmode = false;

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

    const char *uccl_rcmode = std::getenv("UCCL_RCMODE");
    rcmode = (uccl_rcmode && std::strcmp(uccl_rcmode, "1") == 0);

    if (operation == NIXL_READ) {
        if (!rcmode) {
            NIXL_ERROR
                << "UCCL_RCMODE environment variable must be set to 1 for NIXL_READ operations";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    // Process each descriptor pair
    // TODO: Use a vector send async API to send all the transfers at once
    std::lock_guard<std::mutex> lock(mem_mutex_); // Lock once for the entire loop
    for (size_t i = 0; i < lcnt; i++) {
        lmd = (nixlUcclBackendMD *)local[i].metadataP;
        rmd = (nixlUcclBackendMD *)remote[i].metadataP;
        size_t lsize = local[i].len;
        size_t rsize = remote[i].len;
        // Use local[i].addr for the actual iovec address, not lmd->addr (which is base address)
        uintptr_t local_addr = local[i].addr;
        uintptr_t remote_addr = remote[i].addr;

        NIXL_DEBUG << "postXfer iovec[" << i << "]: local[i].addr=" << std::hex << local_addr
                   << ", lsize=" << std::dec << lsize << ", remote[i].addr=" << std::hex
                   << remote_addr << ", rsize=" << std::dec << rsize << ", lmd->addr=" << std::hex
                   << lmd->addr << ", rmd->addr=" << std::hex << rmd->addr;

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
        if (local_priv->mr_id == 0) {
            NIXL_ERROR << "Local memory region not properly registered";
            return NIXL_ERR_BACKEND;
        }

        uccl_mr_t *local_mr = reinterpret_cast<uccl_mr_t *>(local_priv->mr_id);

        int result = 0;
        uint64_t transfer_id = 0;

        char *fifo_item_data = nullptr;
        if (rcmode && handle) {
            nixlUcclReqH *uccl_handle = static_cast<nixlUcclReqH *>(handle);
            if (i < uccl_handle->fifo_items.size()) {
                fifo_item_data = uccl_handle->fifo_items[i].data();
            } else {
                NIXL_ERROR << "No FIFO item found for item: " << i
                           << ", fifo_items.size()=" << uccl_handle->fifo_items.size();
                return NIXL_ERR_BACKEND;
            }
        }

        switch (operation) {
        case NIXL_READ: {
            result = uccl_engine_read(
                conn, local_mr, (void *)local_addr, lsize, fifo_item_data, &transfer_id);
            break;
        }
        case NIXL_WRITE:
            if (rcmode) {
                result = uccl_engine_write_rc(
                    conn, local_mr, (void *)local_addr, lsize, fifo_item_data, &transfer_id);
            } else {
                result = uccl_engine_write(conn, local_mr, (void *)local_addr, lsize, &transfer_id);
            }
            break;
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
        uccl_handle = static_cast<nixlUcclReqH *>(handle);
        uccl_handle->pending_transfer_ids.insert(transfer_id);

        NIXL_DEBUG << "Successfully posted " << (operation == NIXL_READ ? "READ" : "WRITE")
                   << " operation: " << lsize << " bytes with transfer_id: " << transfer_id;
    }
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

    auto it = uccl_handle->pending_transfer_ids.begin();
    while (it != uccl_handle->pending_transfer_ids.end()) {
        uint64_t transfer_id = *it;
        int is_done = uccl_engine_xfer_status(conn, transfer_id);
        if (is_done) {
            it = uccl_handle->pending_transfer_ids.erase(it);
        } else {
            ++it;
        }
    }
    bool all_done = uccl_handle->pending_transfer_ids.empty();
    if (all_done && !uccl_handle->notif_msg.empty()) {
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
            NIXL_DEBUG << "All transfers in handle completed, sent notification: "
                       << uccl_handle->notif_msg;
        }
    }

    return (all_done) ? NIXL_SUCCESS : NIXL_IN_PROG;
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
