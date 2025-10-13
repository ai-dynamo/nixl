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
#include <algorithm>
#include <thread>
#include <chrono>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include "test_utils.h"
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"
#include <mutex>
#include <vector>

#define NUM_TRANSFERS 2
#define NUM_THREADS 4
#define SIZE 1024
#define MEM_VAL 0xBB

/**
 * This test does p2p from using PUT.
 * intitator -> target so the metadata and
 * desc list needs to move from
 * target to initiator
 */

struct SharedNotificationState {
    std::mutex mtx;
    std::vector<nixlSerDes> remote_serdes;
};

static const std::string target("target");
static const std::string initiator("initiator");

static std::vector<std::unique_ptr<uint8_t[]>> initMem(nixlAgent &agent,
                                                       nixl_reg_dlist_t &dram,
                                                       nixl_opt_args_t *extra_params,
                                                       uint8_t val) {
    std::vector<std::unique_ptr<uint8_t[]>> addrs;

    for (int i = 0; i < NUM_TRANSFERS; i++) {
        auto addr = std::make_unique<uint8_t[]>(SIZE);

        std::fill_n(addr.get(), SIZE, val);
        std::cout << "Allocating : " << (void *)addr.get() << ", "
                  << "Setting to 0x" << std::hex << (unsigned)val << std::dec << std::endl;
        dram.addDesc(nixlBlobDesc((uintptr_t)(addr.get()), SIZE, 0, ""));

        addrs.push_back(std::move(addr));
    }
    agent.registerMem(dram, extra_params);

    return addrs;
}

static void targetThread(nixlAgent &agent, nixl_opt_args_t *extra_params, int thread_id) {
    nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
    auto addrs = initMem(agent, dram_for_ucx, extra_params, 0);

    nixl_blob_t tgt_metadata;
    agent.getLocalMD(tgt_metadata);

    std::cout << "Thread " << thread_id << " Start Control Path metadata exchanges\n";

    std::cout << "Thread " << thread_id << " Desc List from Target to Initiator\n";
    dram_for_ucx.print();

    /** Only send desc list */
    nixlSerDes serdes;
    nixl_status_t st = dram_for_ucx.trim().serialize(&serdes);
    nixl_exit_on_failure(st, "Failed to serialize registry dlist");

    std::cout << "Thread " << thread_id << " Wait for initiator and then send xfer descs\n";
    std::string message = serdes.exportStr();
    while (agent.genNotif(initiator, message, extra_params) != NIXL_SUCCESS);
    std::cout << "Thread " << thread_id << " End Control Path metadata exchanges\n";

    std::cout << "Thread " << thread_id << " Start Data Path Exchanges\n";
    std::cout << "Thread " << thread_id << " Waiting to receive Data from Initiator\n";

    bool rc = false;
    for (int n_tries = 0; !rc && n_tries < 100; n_tries++) {
        //Only works with progress thread now, as backend is protected
        /** Sanity Check */
        rc = std::all_of(addrs.begin(), addrs.end(), [](auto &addr) {
            return std::all_of(addr.get(), addr.get() + SIZE, [](int x) {
                return x == MEM_VAL;
            });
        });
        if (!rc)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!rc)
        std::cerr << "Thread " << thread_id << " UCX Transfer failed, buffers are different\n";
    else
        std::cout << "Thread " << thread_id << " Transfer completed and Buffers match with Initiator\n"
                  << "Thread " << thread_id << " UCX Transfer Success!!!\n";

    std::cout << "Thread " << thread_id << " Cleanup..\n";
    agent.deregisterMem(dram_for_ucx, extra_params);
}

static void initiatorThread(nixlAgent &agent, nixl_opt_args_t *extra_params,
                          const std::string &target_ip, int target_port, int thread_id,
                          SharedNotificationState &shared_state) {
    nixl_status_t st;
    nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
    auto addrs = initMem(agent, dram_for_ucx, extra_params, MEM_VAL);

    std::cout << "Thread " << thread_id << " Start Control Path metadata exchanges\n";
    std::cout << "Thread " << thread_id << " Exchange metadata with Target\n";

    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = target_ip;
    md_extra_params.port = target_port;

    st = agent.fetchRemoteMD(target, &md_extra_params);
    nixl_exit_on_failure(st, "Failed to fetch remote MD");

    st = agent.sendLocalMD(&md_extra_params);
    nixl_exit_on_failure(st, "Failed to send local MD");

    // Wait for notifications and populate shared state
    while (true) {
        {
            std::lock_guard<std::mutex> lock(shared_state.mtx);
            if (shared_state.remote_serdes.size() >= NUM_THREADS) {
                break;
            }
        }

        nixl_notifs_t notifs;
        nixl_status_t ret = agent.getNotifs(notifs, extra_params);
        nixl_exit_on_failure(ret, "Failed to get notifs");

        if (notifs.size() > 0) {
            std::lock_guard<std::mutex> lock(shared_state.mtx);
            for (const auto &notif : notifs[target]) {
                nixlSerDes serdes;
                serdes.importStr(notif);
                shared_state.remote_serdes.push_back(serdes);
            }
        }
    }

    // Get our thread's serdes instance
    nixlSerDes remote_serdes;
    {
        std::lock_guard<std::mutex> lock(shared_state.mtx);
        remote_serdes = shared_state.remote_serdes[thread_id];
    }

    std::cout << "Thread " << thread_id << " Verify Deserialized Target's Desc List at Initiator\n";
    nixl_xfer_dlist_t dram_target_ucx(&remote_serdes);
    nixl_xfer_dlist_t dram_initiator_ucx = dram_for_ucx.trim();
    dram_target_ucx.print();

    std::cout << "Thread " << thread_id << " End Control Path metadata exchanges\n";
    std::cout << "Thread " << thread_id << " Start Data Path Exchanges\n\n";
    std::cout << "Thread " << thread_id << " Create transfer request with UCX backend\n";

    // Need to do this in a loop with NIXL_ERR_NOT_FOUND
    // UCX AM with desc list is faster than listener thread can recv/load MD with sockets
    // Will be deprecated with ETCD or callbacks
    nixlXferReqH *treq;
    nixl_status_t ret = NIXL_SUCCESS;
    do {
        ret = agent.createXferReq(NIXL_WRITE, dram_initiator_ucx, dram_target_ucx,
                                  target, treq, extra_params);
    } while (ret == NIXL_ERR_NOT_FOUND);

    if (ret != NIXL_SUCCESS) {
        std::cerr << "Thread " << thread_id << " Error creating transfer request " << ret << "\n";
        exit(-1);
    }

    std::cout << "Thread " << thread_id << " Post the request with UCX backend\n";
    ret = agent.postXferReq(treq);
    std::cout << "Thread " << thread_id << " Initiator posted Data Path transfer\n";
    std::cout << "Thread " << thread_id << " Waiting for completion\n";

    while (ret != NIXL_SUCCESS) {
        ret = agent.getXferStatus(treq);
        nixl_exit_on_failure((ret >= NIXL_SUCCESS), "Failed to get transfer status");
    }

    std::cout << "Thread " << thread_id << " Completed Sending Data using UCX backend\n";
    agent.releaseXferReq(treq);
    agent.invalidateLocalMD(&md_extra_params);

    std::cout << "Thread " << thread_id << " Cleanup..\n";
    agent.deregisterMem(dram_for_ucx, extra_params);
}

static void runTarget(const std::string &ip, int port, nixl_thread_sync_t sync_mode) {
    nixlAgentConfig cfg(true, true, port, sync_mode, 1, 0, 100000, false);

    std::cout << "Starting Agent for target\n";
    nixlAgent agent(target, cfg);

    nixl_b_params_t params = {
        { "num_workers", "4" },
    };
    nixlBackendH *ucx;
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(targetThread, std::ref(agent), &extra_params, i);

    for (auto &thread : threads)
        thread.join();
}

static void runInitiator(const std::string &target_ip, int target_port, nixl_thread_sync_t sync_mode) {
    nixlAgentConfig cfg(true, true, 0, sync_mode, 1, 0, 100000, false);

    std::cout << "Starting Agent for initiator\n";
    nixlAgent agent(initiator, cfg);

    nixl_b_params_t params = {
        { "num_workers", "4" },
    };
    nixlBackendH *ucx;
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    SharedNotificationState shared_state;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(initiatorThread, std::ref(agent), &extra_params,
                             target_ip, target_port, i, std::ref(shared_state));

    for (auto &thread : threads)
        thread.join();
}

int main(int argc, char *argv[]) {
    /** Argument Parsing */
    if (argc < 4) {
        std::cout <<"Enter the required arguments\n" << std::endl;
        std::cout <<"<Role> " <<"<Target IP> <Target Port>"
                  << std::endl;
        exit(-1);
    }

    std::string role = std::string(argv[1]);
    const char  *target_ip   = argv[2];
    int         target_port = std::stoi(argv[3]);

    std::transform(role.begin(), role.end(), role.begin(), ::tolower);

    if (!role.compare(initiator) && !role.compare(target)) {
            std::cerr << "Invalid role. Use 'initiator' or 'target'."
                      << "Currently "<< role <<std::endl;
            return 1;
    }

    auto sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    if (argc == 5) {
        std::string sync_mode_str{argv[4]};
        std::transform(sync_mode_str.begin(), sync_mode_str.end(), sync_mode_str.begin(), ::tolower);
        if (sync_mode_str == "rw") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
            std::cout << "Using RW sync mode" << std::endl;
        } else if (sync_mode_str == "strict") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            std::cout << "Using Strict sync mode" << std::endl;
        } else {
            std::cerr << "Invalid sync mode. Use 'rw' or 'strict'." << std::endl;
            return 1;
        }
    }

    /*** End - Argument Parsing */

    if (role == target)
        runTarget(target_ip, target_port, sync_mode);
    else
        runInitiator(target_ip, target_port, sync_mode);

    return 0;
}
