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
#include <iostream>
#include <cassert>
#include <cstring>

#include <sys/time.h>

#include "nixl.h"
#include "test_utils.h"


std::string agent1("Agent001");
std::string agent2("Agent002");

void check_buf(void* buf, size_t len) {

    // Do some checks on the data.
    for(size_t i = 0; i<len; i++){
        nixl_exit_on_failure((((uint8_t *)buf)[i] == 0xbb), "Data mismatch!", agent1);
    }
}

bool equal_buf (void* buf1, void* buf2, size_t len) {

    // Do some checks on the data.
    for (size_t i = 0; i<len; i++)
        if (((uint8_t*) buf1)[i] != ((uint8_t*) buf2)[i])
            return false;
    return true;
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

int
main(int argc, char **argv) {
    nixl_status_t ret1, ret2;
    std::string ret_s1, ret_s2;

    // Backend name can be provided as the first CLI argument; default to "UCX"
    std::string backend = "UCX";
    if (argc > 1) {
        backend = argv[1];
    }

    // Example: assuming two agents running on the same machine,
    // with separate memory regions in DRAM

    nixlAgentConfig cfg(true);
    nixl_b_params_t init1, init2;
    nixl_mem_list_t mems1, mems2;

    // populate required/desired inits
    nixlAgent A1(agent1, cfg);
    nixlAgent A2(agent2, cfg);

    std::vector<nixl_backend_t> plugins;

    ret1 = A1.getAvailPlugins(plugins);
    nixl_exit_on_failure(ret1, "Failed to get available plugins", agent1);

    std::cout << "Available plugins:\n";

    for (nixl_backend_t b: plugins)
        std::cout << b << "\n";

    std::cout << "Using backend: " << backend << "\n";
    ret1 = A1.getPluginParams(backend, mems1, init1);
    ret2 = A2.getPluginParams(backend, mems2, init2);

    nixl_exit_on_failure(ret1, "Failed to get plugin params", agent1);
    nixl_exit_on_failure(ret2, "Failed to get plugin params", agent2);

    std::cout << "Params before init:\n";
    printParams(init1, mems1);
    printParams(init2, mems2);

    nixlBackendH *bknd1, *bknd2;
    ret1 = A1.createBackend(backend, init1, bknd1);
    ret2 = A2.createBackend(backend, init2, bknd2);

    nixl_opt_args_t extra_params1, extra_params2;
    extra_params1.backends.push_back(bknd1);
    extra_params2.backends.push_back(bknd2);

    nixl_exit_on_failure(ret1, "Failed to create " + backend + " backend", agent1);
    nixl_exit_on_failure(ret2, "Failed to create " + backend + " backend", agent2);

    ret1 = A1.getBackendParams(bknd1, mems1, init1);
    ret2 = A2.getBackendParams(bknd2, mems2, init2);

    nixl_exit_on_failure(ret1, "Failed to get " + backend + " backend params", agent1);
    nixl_exit_on_failure(ret2, "Failed to get " + backend + " backend params", agent2);

    std::cout << "Params after init:\n";
    printParams(init1, mems1);
    printParams(init2, mems2);

    // // One side gets to listen, one side to initiate. Same string is passed as the last 2 steps
    // ret1 = A1->makeConnection(agent2, 0);
    // ret2 = A2->makeConnection(agent1, 1);

    // assert (ret1 == NIXL_SUCCESS);
    // assert (ret2 == NIXL_SUCCESS);

    // User allocates memories, and passes the corresponding address
    // and length to register with the backend
    nixlBlobDesc buff1, buff2, buff3;
    nixl_reg_dlist_t dlist1(DRAM_SEG), dlist2(DRAM_SEG);
    size_t len = 256;
    void* addr1 = calloc(1, len);
    void* addr2 = calloc(1, len);

    memset(addr1, 0xbb, len);
    memset(addr2, 0, len);

    buff1.addr   = (uintptr_t) addr1;
    buff1.len    = len;
    buff1.devId = 0;
    dlist1.addDesc(buff1);

    buff2.addr   = (uintptr_t) addr2;
    buff2.len    = len;
    buff2.devId = 0;
    dlist2.addDesc(buff2);

    // dlist1.print();
    // dlist2.print();

    ret1 = A1.registerMem(dlist1, &extra_params1);
    ret2 = A2.registerMem(dlist2, &extra_params2);
    nixl_exit_on_failure(ret1, "Failed to register memory", agent1);
    nixl_exit_on_failure(ret2, "Failed to register memory", agent2);

    std::string meta1;
    ret1 = A1.getLocalMD(meta1);
    std::string meta2;
    ret2 = A2.getLocalMD(meta2);
    nixl_exit_on_failure(ret1, "Failed to get local MD", agent1);
    nixl_exit_on_failure(ret2, "Failed to get local MD", agent2);

    ret1 = A1.loadRemoteMD (meta2, ret_s1);

    nixl_exit_on_failure(ret1, "Failed to load remote MD", agent1);

    size_t req_size = 8;
    size_t dst_offset = 8;

    nixl_xfer_dlist_t req_src_descs (DRAM_SEG);
    nixlBasicDesc req_src;
    req_src.addr     = (uintptr_t) (((char*) addr1) + 16); //random offset
    req_src.len      = req_size;
    req_src.devId   = 0;
    req_src_descs.addDesc(req_src);

    nixl_xfer_dlist_t req_dst_descs (DRAM_SEG);
    nixlBasicDesc req_dst;
    req_dst.addr   = (uintptr_t) ((char*) addr2) + dst_offset; //random offset
    req_dst.len    = req_size;
    req_dst.devId = 0;
    req_dst_descs.addDesc(req_dst);

    std::cout << "Transfer request from " << addr1 << " to " << addr2 << "\n";
    nixlXferReqH *req_handle;

    extra_params1.notifMsg = "notification";
    extra_params1.hasNotif = true;
    ret1 = A1.createXferReq(NIXL_WRITE, req_src_descs, req_dst_descs, agent2, req_handle, &extra_params1);
    nixl_exit_on_failure(ret1, "Failed to create Xfer Req", agent1);

    nixl_status_t status = A1.postXferReq(req_handle);
    nixl_exit_on_failure((status >= NIXL_SUCCESS), "Failed to post Xfer Req", agent1);

    std::cout << "Transfer was posted\n";

    nixl_notifs_t notif_map;
    int n_notifs = 0;

    while (status != NIXL_SUCCESS || n_notifs == 0) {
        if (status != NIXL_SUCCESS) status = A1.getXferStatus(req_handle);
        if (n_notifs == 0) ret2 = A2.getNotifs(notif_map);
        nixl_exit_on_failure((status >= NIXL_SUCCESS), "Failed to post Xfer Req", agent1);
        nixl_exit_on_failure(ret2, "Failed to get notifs", agent2);
        n_notifs = notif_map.size();
    }

    std::vector<std::string> agent1_notifs = notif_map[agent1];
    nixl_exit_on_failure((agent1_notifs.size() == 1), "Incorrect notif size", agent1);
    nixl_exit_on_failure(
        (agent1_notifs.front() == "notification"), "Incorrect notification", agent1);

    notif_map[agent1].clear(); // Redundant, for testing
    notif_map.clear();
    n_notifs = 0;

    std::cout << "Transfer verified\n";

    ret1 = A1.releaseXferReq(req_handle);
    nixl_exit_on_failure(ret1, "Failed to release Xfer Req", agent1);

    ret1 = A1.deregisterMem(dlist1, &extra_params1);
    ret2 = A2.deregisterMem(dlist2, &extra_params2);
    nixl_exit_on_failure(ret1, "Failed to deregister memory", agent1);
    nixl_exit_on_failure(ret2, "Failed to deregister memory", agent2);

    //only initiator should call invalidate
    ret1 = A1.invalidateRemoteMD(agent2);
    nixl_exit_on_failure(ret1, "Failed to invalidate remote MD", agent1);

    free(addr1);
    free(addr2);

    std::cout << "Test done\n";
}
