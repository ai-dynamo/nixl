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
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>

#include "nixl.h"
#include "test_utils.h"


// Change these values to match your etcd setup
const std::string ETCD_ENDPOINT = "http://localhost:2379";
const std::string AGENT1_NAME = "EtcdAgent1";
const std::string AGENT2_NAME = "EtcdAgent2";
const std::string PARTIAL_LABEL_1 = "conn_info_1";
const std::string PARTIAL_LABEL_2 = "conn_info_2";

void printStatus(const std::string& operation, nixl_status_t status) {
    std::cout << operation << ": " << nixlEnumStrings::statusStr(status) << std::endl;
    if (status != NIXL_SUCCESS) {
        std::cerr << "Error: " << nixlEnumStrings::statusStr(status) << std::endl;
    }
}

// Initialize an agent with etcd enabled
nixlAgent* createAgent(const std::string& name) {
    // Create agent configuration with etcd enabled

    if (getenv("NIXL_ETCD_ENDPOINTS")) {
        std::cout << "NIXL_ETCD_ENDPOINTS is set" << std::endl;
    } else {
        std::cout << "NIXL_ETCD_ENDPOINTS is not set, setting to " << ETCD_ENDPOINT << std::endl;
        setenv("NIXL_ETCD_ENDPOINTS", ETCD_ENDPOINT.c_str(), 1);
    }

    nixlAgentConfig cfg(true);

    // Create the agent with the configuration
    nixlAgent* agent = new nixlAgent(name, cfg);

    return agent;
}

void printParams(const nixl_b_params_t& params, const nixl_mem_list_t& mems) {
    if (params.empty()) {
        std::cout << "Parameters: (empty)" << std::endl;
        return;
    }

    std::cout << "Parameters:" << std::endl;
    for (const auto& pair : params) {
        std::cout << "  " << pair.first << " = " << pair.second << std::endl;
    }

    if (mems.empty()) {
        std::cout << "Mems: (empty)" << std::endl;
        return;
    }

    std::cout << "Mems:" << std::endl;
    for (const auto& elm : mems) {
        std::cout << "  " << nixlEnumStrings::memTypeStr(elm) << std::endl;
    }
}

// Register a memory buffer with the agent
nixl_status_t registerMemory(void** addr, nixlAgent* agent, nixl_reg_dlist_t* dlist, nixl_opt_args_t* extra_params, nixlBackendH* backend, uint8_t pattern) {
    // Create an optional parameters structure
    extra_params->backends.push_back(backend);

    // Allocate and initialize a buffer
    size_t buffer_size = 1024;
    *addr = malloc(buffer_size);

    memset(*addr, pattern, buffer_size);

    // Create a descriptor for the buffer
    nixlBlobDesc desc;
    desc.addr = (uintptr_t)(*addr);
    desc.len = buffer_size;
    desc.devId = 0;

    // Add the descriptor to the list
    dlist->addDesc(desc);

    // Register the memory with the agent
    nixl_status_t status = agent->registerMem(*dlist, extra_params);

    std::cout << "Registered memory " << *addr << " with agent "
              << agent << std::endl;

    return status;
}

int main() {
    void* addr1 = nullptr;
    void* addr2 = nullptr;
    nixl_status_t ret1, ret2;
    nixl_status_t status;

    // Create two agents (normally these would be in separate processes or machines)
    nixlAgentConfig cfg(true);
    nixl_b_params_t init1, init2;
    nixl_mem_list_t mems1, mems2;
    nixl_reg_dlist_t dlist1(DRAM_SEG), dlist2(DRAM_SEG), empty_dlist(DRAM_SEG);

    nixl_opt_args_t extra_params1, extra_params2;

    std::cout << "NIXL Etcd Metadata Example\n";
    std::cout << "==========================\n";

    // populate required/desired inits
    nixlAgent A1(AGENT1_NAME, cfg);
    nixlAgent A2(AGENT2_NAME, cfg);

    std::vector<nixl_backend_t> plugins;

    ret1 = A1.getAvailPlugins(plugins);
    nixl_exit_on_failure(ret1, "Failed to get available plugins", AGENT1_NAME);

    std::cout << "Available plugins:\n";

    for (nixl_backend_t b: plugins)
        std::cout << b << "\n";

    ret1 = A1.getPluginParams("UCX", mems1, init1);
    ret2 = A2.getPluginParams("UCX", mems2, init2);

    nixl_exit_on_failure(ret1, "Failed to get plugin params for UCX", AGENT1_NAME);
    nixl_exit_on_failure(ret2, "Failed to get plugin params for UCX", AGENT2_NAME);

    std::cout << "Params before init:\n";
    printParams(init1, mems1);
    printParams(init2, mems2);

    // Create backends
    nixlBackendH* ucx1, *ucx2;
    ret1 = A1.createBackend("UCX", init1, ucx1);
    ret2 = A2.createBackend("UCX", init2, ucx2);
    nixl_exit_on_failure(ret1, "Failed to create UCX backend", AGENT1_NAME);
    nixl_exit_on_failure(ret2, "Failed to create UCX backend", AGENT2_NAME);

    ret1 = A1.getBackendParams(ucx1, mems1, init1);
    ret2 = A2.getBackendParams(ucx2, mems2, init2);

    nixl_exit_on_failure(ret1, "Failed to get UCX backend params", AGENT1_NAME);
    nixl_exit_on_failure(ret2, "Failed to get UCX backend params", AGENT2_NAME);


    std::cout << "Params after init:\n";
    printParams(init1, mems1);
    printParams(init2, mems2);

    // Register memory with both agents
    status = registerMemory(&addr1, &A1, &dlist1, &extra_params1, ucx1, 0xaa);
    nixl_exit_on_failure(status, "Failed to register memory", AGENT1_NAME);
    status = registerMemory(&addr2, &A2, &dlist2, &extra_params2, ucx2, 0xbb);
    nixl_exit_on_failure(status, "Failed to register memory", AGENT2_NAME);

    std::cout << "\nEtcd Metadata Exchange Demo\n";
    std::cout << "==========================\n";

    // 1. Send Local Metadata to etcd
    std::cout << "\n1. Sending local metadata to etcd...\n";

    // Both agents send their metadata to etcd
    status = A1.sendLocalMD();
    nixl_exit_on_failure(status, "Failed to send local MD", AGENT1_NAME);

    status = A2.sendLocalMD();
    nixl_exit_on_failure(status, "Failed to send local MD", AGENT2_NAME);

    // Give etcd time to process
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 2. Fetch Remote Metadata from etcd
    std::cout << "\n2. Fetching remote metadata from etcd...\n";

    // Agent1 fetches metadata for Agent2
    status = A1.fetchRemoteMD(AGENT2_NAME);
    nixl_exit_on_failure(status, "Failed to fetch remote MD", AGENT1_NAME);

    // Agent2 fetches metadata for Agent1
    status = A2.fetchRemoteMD(AGENT1_NAME);
    nixl_exit_on_failure(status, "Failed to fetch remote MD", AGENT2_NAME);

    // Do transfer from Agent 1 to Agent 2
    size_t req_size = 8;
    size_t dst_offset = 8;

    std::cout << "Agent1's address: " << addr1 << std::endl;
    std::cout << "Agent2's address: " << addr2 << std::endl;

    nixl_xfer_dlist_t req_src_descs (DRAM_SEG);
    nixlBasicDesc req_src;
    req_src.addr     = (uintptr_t) (((char*) addr1) + 16); //random offset
    req_src.len      = req_size;
    req_src.devId    = 0;
    req_src_descs.addDesc(req_src);

    nixl_xfer_dlist_t req_dst_descs (DRAM_SEG);
    nixlBasicDesc req_dst;
    req_dst.addr   = (uintptr_t) ((char*) addr2) + dst_offset; //random offset
    req_dst.len    = req_size;
    req_dst.devId  = 0;
    req_dst_descs.addDesc(req_dst);

    std::cout << "Transfer request from " << std::hex << (void*)req_src.addr
              << " to " << std::hex << (void*)req_dst.addr << std::endl;
    nixlXferReqH *req_handle;

    std::this_thread::sleep_for(std::chrono::seconds(5));

    extra_params1.notifMsg = "notification";
    extra_params1.hasNotif = true;
    ret1 = A1.createXferReq(NIXL_WRITE, req_src_descs, req_dst_descs, AGENT2_NAME, req_handle, &extra_params1);
    std::cout << "Xfer request created, status: " << nixlEnumStrings::statusStr(ret1) << std::endl;
    nixl_exit_on_failure(ret1, "Failed to create Xfer Req", AGENT1_NAME);

    status = A1.postXferReq(req_handle);
    nixl_exit_on_failure((status >= NIXL_SUCCESS), "Failed to post Xfer Req", AGENT1_NAME);

    std::cout << "Transfer was posted\n";

    nixl_notifs_t notif_map;
    int n_notifs = 0;

    while (status != NIXL_SUCCESS || n_notifs == 0) {
        if (status != NIXL_SUCCESS) status = A1.getXferStatus(req_handle);
        if (n_notifs == 0) ret2 = A2.getNotifs(notif_map);
        nixl_exit_on_failure((status >= NIXL_SUCCESS), "Failed to get Xfer status", AGENT1_NAME);
        nixl_exit_on_failure(ret2, "Failed to get notifs", AGENT2_NAME);
        n_notifs = notif_map.size();
    }

    std::cout << "Transfer verified\n";

    ret1 = A1.releaseXferReq(req_handle);
    nixl_exit_on_failure(ret1, "Failed to release Xfer Req", AGENT1_NAME);

    ret1 = A1.deregisterMem(dlist1, &extra_params1);
    ret2 = A2.deregisterMem(dlist2, &extra_params2);
    nixl_exit_on_failure(ret1, "Failed to deregister memory", AGENT1_NAME);
    nixl_exit_on_failure(ret2, "Failed to deregister memory", AGENT2_NAME);

    // 3. Partial Metadata Exchange
    std::cout << "\n3. Sending partial metadata to etcd...\n";

    // Create empty descriptor lists
    nixl_reg_dlist_t empty_dlist1(DRAM_SEG);
    nixl_reg_dlist_t empty_dlist2(DRAM_SEG);

    // Create optional parameters with includeConnInfo set to true
    nixl_opt_args_t conn_params1, conn_params2;
    conn_params1.includeConnInfo = true;
    conn_params1.backends.push_back(ucx1);
    conn_params1.metadataLabel = PARTIAL_LABEL_1;

    conn_params2.includeConnInfo = true;
    conn_params2.backends.push_back(ucx2);
    conn_params2.metadataLabel = PARTIAL_LABEL_1;

    // Send partial metadata
    status = A1.sendLocalPartialMD(empty_dlist1, &conn_params1);
    nixl_exit_on_failure(status, "Failed to send local partial MD", AGENT1_NAME);

    status = A2.sendLocalPartialMD(empty_dlist2, &conn_params2);
    nixl_exit_on_failure(status, "Failed to send local partial MD", AGENT2_NAME);

    // Send once partial with different label
    conn_params1.metadataLabel = PARTIAL_LABEL_2;
    status = A1.sendLocalPartialMD(empty_dlist1, &conn_params1);
    nixl_exit_on_failure(status, "Failed to send local partial MD", AGENT1_NAME);

    conn_params2.metadataLabel = PARTIAL_LABEL_2;
    status = A2.sendLocalPartialMD(empty_dlist2, &conn_params2);
    nixl_exit_on_failure(status, "Failed to send local partial MD", AGENT2_NAME);

    nixl_opt_args_t fetch_params;
    fetch_params.metadataLabel = PARTIAL_LABEL_1;
    status = A1.fetchRemoteMD(AGENT2_NAME, &fetch_params);
    nixl_exit_on_failure(status, "Failed to fetch remote MD", AGENT1_NAME);

    status = A2.fetchRemoteMD(AGENT1_NAME, &fetch_params);
    nixl_exit_on_failure(status, "Failed to fetch remote MD", AGENT2_NAME);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 4. Invalidate Metadata
    std::cout << "\n4. Invalidating metadata in etcd...\n";

    // Invalidate AGENT1_NAME's metadata
    status = A1.invalidateLocalMD();
    nixl_exit_on_failure(status, "Failed to invalidate local MD", AGENT1_NAME);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Try fetching the invalidated metadata
    std::cout << "\nTrying to fetch invalidated metadata for Agent1...\n";
    status = A2.fetchRemoteMD(AGENT1_NAME, &extra_params2);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Try invalidating again, this should log a debug message
    std::cout << "Trying to invalidate again...\n";
    status = A1.invalidateLocalMD();
    nixl_exit_on_failure(status, "Failed to invalidate local MD", AGENT1_NAME);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 5. Fetch metadata with invalid label. This should not block forever and print error message.
    std::cout << "\n5. Fetching metadata with invalid label...\n";
    status = A2.fetchRemoteMD("INVALID_AGENT", &fetch_params);
    nixl_exit_on_failure(status, "Failed to fetch remote MD", AGENT2_NAME);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    free(addr1);
    free(addr2);

    std::cout << "\nExample completed.\n";
    return 0;
}
