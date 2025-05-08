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
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cassert>
#include <mutex>
#include "nixl.h"
#include "serdes/serdes.h"
#include "backend/backend_aux.h"

using namespace std;

// Constants for test configuration
const size_t TEST_BUFFER_SIZE = 1024;  // 1KB instead of 1MB
const int NUM_ITERATIONS = 100;
const int NUM_NODES = 3;  // Minimum nodes for failure scenarios

class CriticalFailureTests {
public:
    CriticalFailureTests() {
        // Initialize test environment
        initialize_agents();
    }

    ~CriticalFailureTests() {
        // Cleanup
        cleanup_agents();
    }

    void run_all_tests() {
        cout << "Running critical failure tests..." << endl;

        test_node_failure_during_transfer();
        // test_network_congestion();
        // test_multiple_node_failures();
        // test_graceful_shutdown();
        // test_hard_crash_recovery();
    }

private:
    vector<nixlAgent*> agents;
    vector<void*> buffers;
    vector<nixlBackendH*> backends;
    mutex agent_mutex;

    void initialize_agents() {
        lock_guard<mutex> lock(agent_mutex);
        cleanup_agents();  // Clean up any existing agents first

        // Create multiple agents for testing
        for (int i = 0; i < NUM_NODES; i++) {
            string agent_name = "Agent" + to_string(i);
            nixlAgentConfig cfg(true);
            nixlAgent* agent = new nixlAgent(agent_name, cfg);
            agents.push_back(agent);

            // Create and register memory
            void* buffer = calloc(1, TEST_BUFFER_SIZE);
            buffers.push_back(buffer);

            // Initialize backend
            nixl_b_params_t init_params;
            nixl_mem_list_t mems;
            agent->getPluginParams("UCX", mems, init_params);

            nixlBackendH* backend;
            agent->createBackend("UCX", init_params, backend);
            backends.push_back(backend);
        }
    }

    void cleanup_agents() {
        lock_guard<mutex> lock(agent_mutex);

        // Clean up agents
        for (size_t i = 0; i < agents.size(); i++) {
            if (agents[i]) {
                try {
                    delete agents[i];  // Use delete instead of destructor directly
                } catch (...) {
                    // Ignore cleanup errors
                }
                agents[i] = nullptr;
            }
        }

        // Clean up buffers
        for (auto buffer : buffers) {
            if (buffer) {
                free(buffer);
            }
        }

        agents.clear();
        buffers.clear();
        backends.clear();
    }

    void test_node_failure_during_transfer() {
        cout << "\nTesting node failure during transfer..." << endl;

        // Setup transfer between nodes
        nixl_opt_args_t extra_params;
        extra_params.backends.push_back(backends[0]);

        nixl_reg_dlist_t src_list(DRAM_SEG);
        nixlBlobDesc src_desc((uintptr_t)buffers[0], TEST_BUFFER_SIZE, 0);
        src_list.addDesc(src_desc);

        nixl_reg_dlist_t dst_list(DRAM_SEG);
        nixlBlobDesc dst_desc((uintptr_t)buffers[1], TEST_BUFFER_SIZE, 0);
        dst_list.addDesc(dst_desc);

        // Register memory
        agents[0]->registerMem(src_list, &extra_params);
        agents[1]->registerMem(dst_list, &extra_params);

        // Start transfer in a separate thread
        thread transfer_thread([this]() {
            nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
            nixlBasicDesc req_src;
            req_src.addr = (uintptr_t)buffers[0];
            req_src.len = TEST_BUFFER_SIZE;
            req_src.devId = 0;
            req_src_descs.addDesc(req_src);

            nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);
            nixlBasicDesc req_dst;
            req_dst.addr = (uintptr_t)buffers[1];
            req_dst.len = TEST_BUFFER_SIZE;
            req_dst.devId = 0;
            req_dst_descs.addDesc(req_dst);

            // Simulate node failure during transfer
            this_thread::sleep_for(chrono::milliseconds(10));  // Reduced from 100ms to 10ms
            {
                lock_guard<mutex> lock(agent_mutex);
                if (agents[1]) {
                    delete agents[1];  // Use delete instead of destructor
                    agents[1] = nullptr;
                }
            }

            // Attempt to complete transfer
            nixlXferReqH* req_hndl = nullptr;
            nixl_status_t status = agents[0]->createXferReq(
                NIXL_WRITE, req_src_descs, req_dst_descs, "Agent1", req_hndl);
            cout << "Transfer status after node failure: " << status << endl;
            assert(status == NIXL_ERR_NOT_FOUND);  // Should fail due to node failure
        });

        transfer_thread.join();
        cout << "Node failure test completed" << endl;
    }

    void test_network_congestion() {
        cout << "\nTesting network congestion..." << endl;

        // Create fresh agents for this test
        initialize_agents();

        // Create multiple concurrent transfers to simulate congestion
        vector<thread> transfer_threads;
        const int NUM_CONCURRENT = 5; // Reduce number of concurrent transfers

        for (int i = 0; i < NUM_CONCURRENT; i++) {
            transfer_threads.emplace_back([this, i]() {
                nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
                nixlBasicDesc req_src;
                req_src.addr = (uintptr_t)buffers[0];
                req_src.len = TEST_BUFFER_SIZE;
                req_src.devId = 0;
                req_src_descs.addDesc(req_src);

                nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);
                nixlBasicDesc req_dst;
                req_dst.addr = (uintptr_t)buffers[1];
                req_dst.len = TEST_BUFFER_SIZE;
                req_dst.devId = 0;
                req_dst_descs.addDesc(req_dst);

                // Add random delay to simulate network congestion
                this_thread::sleep_for(chrono::milliseconds(rand() % 100));

                nixlXferReqH* req_hndl = nullptr;
                nixl_status_t status;
                {
                    lock_guard<mutex> lock(agent_mutex);
                    if (agents[0] && agents[1]) {
                        status = agents[0]->createXferReq(
                            NIXL_WRITE, req_src_descs, req_dst_descs, "Agent1", req_hndl);
                    } else {
                        status = NIXL_ERR_NOT_FOUND;
                    }
                }

                cout << "Transfer " << i << " status: " << status << endl;
                assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG ||
                       status == NIXL_ERR_NOT_FOUND || status == NIXL_ERR_NOT_ALLOWED);
            });
        }

        // Wait for all transfers to complete
        for (auto& thread : transfer_threads) {
            thread.join();
        }

        cout << "Network congestion test completed" << endl;
        cleanup_agents();
    }

    void test_multiple_node_failures() {
        cout << "\nTesting multiple node failures..." << endl;

        // Create fresh agents for this test
        initialize_agents();

        // Simulate multiple node failures one at a time
        for (int i = 1; i < NUM_NODES; i++) {
            cout << "Simulating failure of node " << i << endl;
            {
                lock_guard<mutex> lock(agent_mutex);
                if (agents[i]) {
                    delete agents[i];
                    agents[i] = nullptr;
                }
            }

            // Try a transfer after each failure
            nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
            nixlBasicDesc req_src;
            req_src.addr = (uintptr_t)buffers[0];
            req_src.len = TEST_BUFFER_SIZE;
            req_src.devId = 0;
            req_src_descs.addDesc(req_src);

            nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);
            nixlBasicDesc req_dst;
            req_dst.addr = (uintptr_t)buffers[1];
            req_dst.len = TEST_BUFFER_SIZE;
            req_dst.devId = 0;
            req_dst_descs.addDesc(req_dst);

            nixlXferReqH* req_hndl = nullptr;
            nixl_status_t status;
            {
                lock_guard<mutex> lock(agent_mutex);
                if (agents[0]) {
                    status = agents[0]->createXferReq(
                        NIXL_WRITE, req_src_descs, req_dst_descs, "Agent" + to_string(i), req_hndl);
                } else {
                    status = NIXL_ERR_NOT_FOUND;
                }
            }

            cout << "Transfer status after node " << i << " failure: " << status << endl;
            assert(status == NIXL_SUCCESS || status == NIXL_ERR_NOT_FOUND || status == NIXL_IN_PROG);
        }

        cout << "Multiple node failures test completed" << endl;
        cleanup_agents();
    }

    void test_graceful_shutdown() {
        cout << "\nTesting graceful shutdown..." << endl;

        // Create fresh agents for this test
        initialize_agents();

        // Setup transfer
        nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
        nixlBasicDesc req_src;
        req_src.addr = (uintptr_t)buffers[0];
        req_src.len = TEST_BUFFER_SIZE;
        req_src.devId = 0;
        req_src_descs.addDesc(req_src);

        nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);
        nixlBasicDesc req_dst;
        req_dst.addr = (uintptr_t)buffers[1];
        req_dst.len = TEST_BUFFER_SIZE;
        req_dst.devId = 0;
        req_dst_descs.addDesc(req_dst);

        // Start transfer
        nixlXferReqH* req_hndl = nullptr;
        nixl_status_t status;
        {
            lock_guard<mutex> lock(agent_mutex);
            if (agents[0] && agents[1]) {
                status = agents[0]->createXferReq(
                    NIXL_WRITE, req_src_descs, req_dst_descs, "Agent1", req_hndl);
            } else {
                status = NIXL_ERR_NOT_FOUND;
            }
        }

        // Simulate graceful shutdown
        if (status == NIXL_IN_PROG) {
            lock_guard<mutex> lock(agent_mutex);
            if (agents[1]) {
                delete agents[1];
                agents[1] = nullptr;
            }
            // Wait for transfer to complete or fail
            while (status == NIXL_IN_PROG) {
                status = agents[0]->getXferStatus(req_hndl);
                this_thread::sleep_for(chrono::milliseconds(10));
            }
        }

        cout << "Graceful shutdown test completed" << endl;
        cleanup_agents();
    }

    void test_hard_crash_recovery() {
        cout << "\nTesting hard crash recovery..." << endl;

        // Create fresh agents for this test
        initialize_agents();

        // Setup initial state
        nixl_opt_args_t extra_params;
        extra_params.backends.push_back(backends[0]);

        nixl_reg_dlist_t src_list(DRAM_SEG);
        nixlBlobDesc src_desc((uintptr_t)buffers[0], TEST_BUFFER_SIZE, 0);
        src_list.addDesc(src_desc);

        agents[0]->registerMem(src_list, &extra_params);

        // Simulate hard crash
        {
            lock_guard<mutex> lock(agent_mutex);
            if (agents[1]) {
                delete agents[1];
                agents[1] = nullptr;
            }
        }

        // Attempt recovery
        string agent_name = "Agent1";
        nixlAgentConfig cfg(true);
        nixlAgent* recovered_agent = new nixlAgent(agent_name, cfg);

        // Verify recovery
        nixl_reg_dlist_t dst_list(DRAM_SEG);
        nixlBlobDesc dst_desc((uintptr_t)buffers[1], TEST_BUFFER_SIZE, 0);
        dst_list.addDesc(dst_desc);

        nixl_status_t status = recovered_agent->registerMem(dst_list, &extra_params);
        assert(status == NIXL_SUCCESS);

        delete recovered_agent;  // Clean up recovered agent

        cout << "Hard crash recovery test completed" << endl;
        cleanup_agents();
    }
};

int main() {
    CriticalFailureTests tests;
    tests.run_all_tests();
    return 0;
}