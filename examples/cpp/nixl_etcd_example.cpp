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
#include <cassert>
#include <thread>
#include <chrono>
#include <cstring>

#include "nixl.h"

// Change these values to match your etcd setup
const std::string ETCD_ENDPOINT = "http://localhost:2379";
const std::string AGENT1_NAME = "EtcdAgent1";
const std::string AGENT2_NAME = "EtcdAgent2";

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

// Register a memory buffer with the agent
nixl_status_t registerMemory(nixlAgent* agent, nixlBackendH* backend) {
    // Create an optional parameters structure
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);

    // Allocate and initialize a buffer
    size_t buffer_size = 1024;
    void* buffer = malloc(buffer_size);
    memset(buffer, 0xaa, buffer_size); // Initialize with pattern

    // Create a descriptor list
    nixl_reg_dlist_t dlist(DRAM_SEG);

    // Create a descriptor for the buffer
    nixlBlobDesc desc;
    desc.addr = (uintptr_t)buffer;
    desc.len = buffer_size;
    desc.devId = 0;

    // Add the descriptor to the list
    dlist.addDesc(desc);

    // Register the memory with the agent
    nixl_status_t status = agent->registerMem(dlist, &extra_params);

    return status;
}

int main() {
    std::cout << "NIXL Etcd Metadata Example\n";
    std::cout << "==========================\n";

    // Create two agents (normally these would be in separate processes or machines)
    nixlAgent* agent1 = createAgent(AGENT1_NAME);
    nixlAgent* agent2 = createAgent(AGENT2_NAME);

    // Check for UCX backend
    nixl_backend_t backend_type = "UCX";
    nixl_b_params_t params1, params2;
    nixl_mem_list_t mems1, mems2;

    // Check if UCX is available
    std::vector<nixl_backend_t> plugins;
    nixl_status_t status = agent1->getAvailPlugins(plugins);
    printStatus("Getting available plugins", status);

    std::cout << "Available plugins:" << std::endl;
    bool ucx_available = false;
    for (const auto& plugin : plugins) {
        std::cout << "  - " << plugin << std::endl;
        if (plugin == backend_type) {
            ucx_available = true;
        }
    }

    if (!ucx_available) {
        std::cerr << "UCX plugin not available, exiting" << std::endl;
        delete agent1;
        delete agent2;
        return 1;
    }

    // Get UCX plugin parameters for DRAM memory type
    printStatus("Getting plugin parameters for agent1", status);
    assert(status == NIXL_SUCCESS);

    status = agent2->getPluginParams(backend_type, mems2, params2);
    assert(status == NIXL_SUCCESS);

    // Create backends
    nixlBackendH *backend1, *backend2;
    status = agent1->createBackend(backend_type, params1, backend1);
    assert(status == NIXL_SUCCESS);

    status = agent2->createBackend(backend_type, params2, backend2);
    assert(status == NIXL_SUCCESS);

    // Register memory with both agents
    status = registerMemory(agent1, backend1);
    assert(status == NIXL_SUCCESS);

    status = registerMemory(agent2, backend2);
    assert(status == NIXL_SUCCESS);

    std::cout << "\nEtcd Metadata Exchange Demo\n";
    std::cout << "==========================\n";

    // 1. Send Local Metadata to etcd
    std::cout << "\n1. Sending local metadata to etcd...\n";

    // Both agents send their metadata to etcd
    status = agent1->sendLocalMD();
    assert(status == NIXL_SUCCESS);

    status = agent2->sendLocalMD();
    assert(status == NIXL_SUCCESS);

    // Give etcd time to process
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 2. Fetch Remote Metadata from etcd
    std::cout << "\n2. Fetching remote metadata from etcd...\n";

    // Agent1 fetches metadata for Agent2
    status = agent1->fetchRemoteMD(AGENT2_NAME);
    assert(status == NIXL_SUCCESS);

    // Agent2 fetches metadata for Agent1
    status = agent2->fetchRemoteMD(AGENT1_NAME);
    assert(status == NIXL_SUCCESS);

    // 3. Partial Metadata Exchange
    std::cout << "\n3. Sending partial metadata to etcd...\n";

    // Create empty descriptor lists
    nixl_reg_dlist_t empty_dlist1(DRAM_SEG);
    nixl_reg_dlist_t empty_dlist2(DRAM_SEG);

    // Create optional parameters with includeConnInfo set to true
    nixl_opt_args_t conn_params1, conn_params2;
    conn_params1.includeConnInfo = true;
    conn_params1.backends.push_back(backend1);

    conn_params2.includeConnInfo = true;
    conn_params2.backends.push_back(backend2);

    // Send partial metadata
    status = agent1->sendLocalPartialMD(empty_dlist1, &conn_params1);
    assert(status == NIXL_SUCCESS);

    status = agent2->sendLocalPartialMD(empty_dlist2, &conn_params2);
    assert(status == NIXL_SUCCESS);

    // 4. Invalidate Metadata
    std::cout << "\n4. Invalidating metadata in etcd...\n";

    // Invalidate agent1's metadata
    status = agent1->invalidateLocalMD();
    assert(status == NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Try fetching the invalidated metadata
    std::cout << "\nTrying to fetch invalidated metadata for Agent1...\n";
    status = agent2->fetchRemoteMD(AGENT1_NAME);

    // Clean up
    delete agent1;
    delete agent2;

    std::cout << "\nExample completed.\n";
    return 0;
}
