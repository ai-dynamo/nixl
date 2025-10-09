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
#include <cstring>

#include <sys/time.h>

#include "nixl.h"

std::string agent1("Agent001");
std::string agent2("Agent002");

void check_buf(void* buf, size_t len) {

    // Do some checks on the data.
    for(size_t i = 0; i<len; i++){
        assert (((uint8_t*) buf)[i] == 0xbb);
    }
}

bool equal_buf (void* buf1, void* buf2, size_t len) {

    // Do some checks on the data.
    for (size_t i = 0; i<len; i++)
        if (((uint8_t*) buf1)[i] != ((uint8_t*) buf2)[i])
            return false;
    return true;
}

void test_side_perf(nixlAgent* A1, nixlAgent* A2, nixlBackendH* backend, nixlBackendH* backend2) {

    int n_mems = 32;
    int descs_per_mem = 64*1024;
    int n_iters = 10;
    nixl_reg_dlist_t mem_list1(DRAM_SEG), mem_list2(DRAM_SEG);
    nixl_xfer_dlist_t src_list(DRAM_SEG), dst_list(DRAM_SEG);
    nixl_status_t status;

    nixl_opt_args_t extra_params1, extra_params2;
    extra_params1.backends.push_back(backend);
    extra_params2.backends.push_back(backend2);

    struct timeval start_time, end_time, diff_time;

    nixlDlistH *src_side[n_iters];
    nixlDlistH *dst_side[n_iters];

    void* src_buf = malloc(n_mems*descs_per_mem*8);
    void* dst_buf = malloc(n_mems*descs_per_mem*8);

    for(int i = 0; i<n_mems; i++) {
        nixlBlobDesc src_desc((uintptr_t) src_buf + i*descs_per_mem*8, descs_per_mem*8, 0);
        nixlBlobDesc dst_desc((uintptr_t) dst_buf + i*descs_per_mem*8, descs_per_mem*8, 0);

        mem_list1.addDesc(src_desc);
        mem_list2.addDesc(dst_desc);

        //std::cout << "mem region " << i << " working \n";

        for(int j = 0; j<descs_per_mem; j++){
            nixlBasicDesc src_desc2((uintptr_t) src_buf + i*descs_per_mem*8 + 8*j, 8, 0);
            nixlBasicDesc dst_desc2((uintptr_t) dst_buf + i*descs_per_mem*8 + 8*j, 8, 0);

            src_list.addDesc(src_desc2);
            dst_list.addDesc(dst_desc2);
        }
    }

    assert (mem_list1.descCount() == n_mems);
    assert (mem_list2.descCount() == n_mems);

    assert (src_list.descCount() == n_mems*descs_per_mem);
    assert (dst_list.descCount() == n_mems*descs_per_mem);

    status = A1->registerMem(mem_list1, &extra_params1);
    assert (status == NIXL_SUCCESS);

    status = A2->registerMem(mem_list2, &extra_params2);
    assert (status == NIXL_SUCCESS);

    std::string meta2;
    status = A2->getLocalMD(meta2);
    assert (status == NIXL_SUCCESS);
    assert (meta2.size() > 0);

    std::string remote_name;
    status = A1->loadRemoteMD(meta2, remote_name);
    assert (status == NIXL_SUCCESS);
    assert (remote_name == agent2);

    std::cout << "perf setup done\n";

    gettimeofday(&start_time, NULL);

    for(int i = 0; i<n_iters; i++) {
        status = A1->prepXferDlist(agent2, dst_list, dst_side[i], &extra_params1);
        assert (status == NIXL_SUCCESS);

        status = A1->prepXferDlist(NIXL_INIT_AGENT, src_list, src_side[i], &extra_params1);
        assert (status == NIXL_SUCCESS);
    }

    gettimeofday(&end_time, NULL);

    timersub(&end_time, &start_time, &diff_time);
    std::cout << "prepXferDlist, total time for " << n_iters << " iters: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    float time_per_iter = ((diff_time.tv_sec * 1000000) + diff_time.tv_usec);
    time_per_iter /=  (n_iters) ;
    std::cout << "time per 2 preps " << time_per_iter << "us\n";

    //test makeXfer optimization

    std::vector<int> indices;
    nixlXferReqH* reqh1, *reqh2;

    for(int i = 0; i<(n_mems*descs_per_mem); i++)
        indices.push_back(i);

    //should print n_mems number of final descriptors
    extra_params1.notifMsg = "test";
    extra_params1.hasNotif = true;
    status = A1->makeXferReq(NIXL_WRITE, src_side[0], indices, dst_side[0], indices, reqh1, &extra_params1);
    assert (status == NIXL_SUCCESS);

    indices.clear();
    for(int i = 0; i<(n_mems*descs_per_mem); i+=2)
        indices.push_back(i);

    //should print (n_mems*descs_per_mem/2) number of final descriptors
    status = A1->makeXferReq(NIXL_WRITE, src_side[0], indices, dst_side[0], indices, reqh2, &extra_params1);
    assert (status == NIXL_SUCCESS);

    status = A1->releaseXferReq(reqh1);
    assert (status == NIXL_SUCCESS);
    status = A1->releaseXferReq(reqh2);
    assert (status == NIXL_SUCCESS);

    // Commented out to test auto deregistration
    // status = A1->deregisterMem(mem_list1, &extra_params1);
    // assert (status == NIXL_SUCCESS);
    status = A2->deregisterMem(mem_list2, &extra_params2);
    assert (status == NIXL_SUCCESS);

    for(int i = 0; i<n_iters; i++){
        status = A1->releasedDlistH(src_side[i]);
        assert (status == NIXL_SUCCESS);
        status = A1->releasedDlistH(dst_side[i]);
        assert (status == NIXL_SUCCESS);
    }

    free(src_buf);
    free(dst_buf);
}

nixl_status_t partialMdTest(nixlAgent* A1, nixlAgent* A2, nixlBackendH* backend1, nixlBackendH* backend2) {
    std::cout << "Starting partialMdTest\n";

    nixl_status_t status;
    nixl_opt_args_t extra_params1, extra_params2;
    extra_params1.backends.push_back(backend1);
    extra_params2.backends.push_back(backend2);

    const int NUM_BUFFERS = 4;
    const int NUM_UPDATES = 3;
    const size_t BUF_SIZE = 1024;

    // Allocate memory for the test
    void* src_bufs[NUM_UPDATES][NUM_BUFFERS];
    void* dst_bufs[NUM_UPDATES][NUM_BUFFERS];

    // Create mem_lists for updates - using std::vector instead of C-style arrays
    std::vector<nixl_reg_dlist_t> src_mem_lists(NUM_UPDATES, nixl_reg_dlist_t(DRAM_SEG));
    std::vector<nixl_reg_dlist_t> dst_mem_lists(NUM_UPDATES, nixl_reg_dlist_t(DRAM_SEG));

    // Allocate buffers and create memory descriptors
    for (int update_idx = 0; update_idx < NUM_UPDATES; update_idx++) {
        for (int buf_idx = 0; buf_idx < NUM_BUFFERS; buf_idx++) {
            src_bufs[update_idx][buf_idx] = calloc(1, BUF_SIZE);
            dst_bufs[update_idx][buf_idx] = calloc(1, BUF_SIZE);

            nixlBlobDesc src_desc((uintptr_t)src_bufs[update_idx][buf_idx], BUF_SIZE, 0);
            nixlBlobDesc dst_desc((uintptr_t)dst_bufs[update_idx][buf_idx], BUF_SIZE, 0);

            src_mem_lists[update_idx].addDesc(src_desc);
            dst_mem_lists[update_idx].addDesc(dst_desc);

            // Fill source buffers with test pattern
            memset(src_bufs[update_idx][buf_idx], 0xbb, BUF_SIZE);
        }
    }

    // Register memory for each update
    for (int update = 0; update < NUM_UPDATES; update++) {
        status = A1->registerMem(src_mem_lists[update], &extra_params1);
        assert(status == NIXL_SUCCESS);

        status = A2->registerMem(dst_mem_lists[update], &extra_params2);
        assert(status == NIXL_SUCCESS);
    }

    // Test metadata update with only backends and empty descriptor list
    std::cout << "Metadata update - backends only\n";

    // Agent2 might have already been previously loaded.
    // Invalidate it just in case but don't care either way.
    A1->invalidateRemoteMD(agent2);

    nixl_reg_dlist_t empty_dlist(DRAM_SEG);
    std::string partial_meta;
    status = A2->getLocalPartialMD(empty_dlist, partial_meta, NULL);
    assert(status == NIXL_SUCCESS);
    assert(partial_meta.size() > 0);

    std::string remote_name;
    status = A1->loadRemoteMD(partial_meta, remote_name);
    assert(status == NIXL_SUCCESS);
    assert(remote_name == agent2);

    // Make sure unregistered descriptors are not updated
    for (int update = 0; update < NUM_UPDATES; update++) {
        nixlDlistH *dst_side;
        status = A1->prepXferDlist(agent2, dst_mem_lists[update].trim(), dst_side, &extra_params1);
        assert(status != NIXL_SUCCESS);
        assert(dst_side == nullptr);
    }

    // Invalidate remote agent metadata to make sure we received connection info
    status = A1->invalidateRemoteMD(agent2);
    assert(status == NIXL_SUCCESS);
    std::cout << "Metadata update - backends only completed\n";

    // Main test loop - update metadata multiple times
    // and verify those that are not updatedare invalid on remote side.
    extra_params2.includeConnInfo = false;
    for (int update = 0; update < NUM_UPDATES; update++) {
        // Toggle includeConnInfo to test that it doesn't affect metadata update
        extra_params2.includeConnInfo = !extra_params2.includeConnInfo;

        std::cout << "Metadata update #" << update << "\n";
        // Get partial metadata from A2
        status = A2->getLocalPartialMD(dst_mem_lists[update], partial_meta, &extra_params2);
        assert(status == NIXL_SUCCESS);
        assert(partial_meta.size() > 0);

        // Load the partial metadata into A1
        std::string remote_name;
        status = A1->loadRemoteMD(partial_meta, remote_name);
        assert(status == NIXL_SUCCESS);
        assert(remote_name == agent2);

        // Make sure loaded descriptors are updated
        nixlDlistH *dst_side;
        status = A1->prepXferDlist(agent2, dst_mem_lists[update].trim(), dst_side, &extra_params1);
        assert(status == NIXL_SUCCESS);
        assert(dst_side != nullptr);

        // Make sure not-loaded descriptors are not updated
        for (int invalid_idx = update + 1; invalid_idx < NUM_UPDATES; invalid_idx++) {
            status = A1->prepXferDlist(agent2, dst_mem_lists[invalid_idx].trim(), dst_side, &extra_params1);
            assert(status != NIXL_SUCCESS);
            assert(dst_side == nullptr);
        }
        std::cout << "Metadata update #" << update << " completed\n";
    }

    // Prepare transfer dlists of all descriptors and buffers
    nixl_xfer_dlist_t src_xfer_list(DRAM_SEG), dst_xfer_list(DRAM_SEG);

    for (int update_idx = 0; update_idx < NUM_UPDATES; update_idx++) {
        nixl_xfer_dlist_t tmp_src_list = src_mem_lists[update_idx].trim();
        nixl_xfer_dlist_t tmp_dst_list = dst_mem_lists[update_idx].trim();

        for (int buf_idx = 0; buf_idx < NUM_BUFFERS; buf_idx++) {
            src_xfer_list.addDesc(tmp_src_list[buf_idx]);
            dst_xfer_list.addDesc(tmp_dst_list[buf_idx]);
        }
    }

    // Prepare for transfers of all descriptors and buffers
    nixlDlistH *src_side, *dst_side;

    status = A1->prepXferDlist(NIXL_INIT_AGENT, src_xfer_list, src_side, &extra_params1);
    assert(status == NIXL_SUCCESS);

    status = A1->prepXferDlist(agent2, dst_xfer_list, dst_side, &extra_params1);
    assert(status == NIXL_SUCCESS);

    std::cout << "Transfer preparation completed\n";

    // Perform a single transfer for all descriptors and buffers
    std::vector<int> indices;
    for (int i = 0; i < NUM_UPDATES * NUM_BUFFERS; i++) {
        indices.push_back(i);
    }

    nixlXferReqH *req;
    extra_params1.notifMsg = "partialMdTest_notification";
    extra_params1.hasNotif = true;

    // Create and post the transfer request
    status = A1->makeXferReq(NIXL_WRITE, src_side, indices, dst_side, indices, req, &extra_params1);
    assert(status == NIXL_SUCCESS);

    nixl_status_t xfer_status = A1->postXferReq(req);

    // Wait for transfer completion
    while (xfer_status != NIXL_SUCCESS) {
        if (xfer_status != NIXL_SUCCESS) xfer_status = A1->getXferStatus(req);
        assert (xfer_status >= 0);
    }

    // Verify transfer results
    for (int update_idx = 0; update_idx < NUM_UPDATES; update_idx++) {
        for (int buf_idx = 0; buf_idx < NUM_BUFFERS; buf_idx++) {
            check_buf(dst_bufs[update_idx][buf_idx], BUF_SIZE);
        }
    }

    std::cout << "Transfer verification completed\n";

    // Cleanup
    status = A1->releaseXferReq(req);
    assert(status == NIXL_SUCCESS);

    status = A1->releasedDlistH(src_side);
    assert(status == NIXL_SUCCESS);

    status = A1->releasedDlistH(dst_side);
    assert(status == NIXL_SUCCESS);

    // Deregister memory
    for (int update = 0; update < NUM_UPDATES; update++) {
        status = A1->deregisterMem(src_mem_lists[update], &extra_params1);
        assert(status == NIXL_SUCCESS);

        status = A2->deregisterMem(dst_mem_lists[update], &extra_params2);
        assert(status == NIXL_SUCCESS);
    }

    // Free allocated memory
    for (int update_idx = 0; update_idx < NUM_UPDATES; update_idx++) {
        for (int buf_idx = 0; buf_idx < NUM_BUFFERS; buf_idx++) {
            free(src_bufs[update_idx][buf_idx]);
            free(dst_bufs[update_idx][buf_idx]);
        }
    }

    std::cout << "partialMdTest completed successfully\n";
    return NIXL_SUCCESS;
}

nixl_status_t sideXferTest(nixlAgent* A1, nixlAgent* A2, nixlXferReqH* src_handle, nixlBackendH* dst_backend) {
    std::cout << "Starting sideXferTest\n";

    nixlBackendH* src_backend;
    nixl_status_t status = A1->queryXferBackend(src_handle, src_backend);

    nixl_opt_args_t extra_params1, extra_params2;
    extra_params1.backends.push_back(src_backend);
    extra_params2.backends.push_back(dst_backend);

    assert (status == NIXL_SUCCESS);
    assert (src_backend);

    std::cout << "Got backend\n";

    test_side_perf(A1, A2, src_backend, dst_backend);

    int n_bufs = 32; // must be even
    size_t len = 1024;
    void* src_bufs[n_bufs], *dst_bufs[n_bufs];

    nixl_reg_dlist_t mem_list1(DRAM_SEG), mem_list2(DRAM_SEG);
    nixl_xfer_dlist_t src_list(DRAM_SEG), dst_list(DRAM_SEG);
    nixlBlobDesc src_desc[n_bufs], dst_desc[n_bufs];
    for(int i = 0; i<n_bufs; i++) {

        src_bufs[i] = calloc(1, len);
        std::cout << " src " << i << " " << src_bufs[i] << "\n";
        dst_bufs[i] = calloc(1, len);
        std::cout << " dst " << i << " " << dst_bufs[i] << "\n";

        src_desc[i].len = len;
        src_desc[i].devId = 0;
        src_desc[i].addr = (uintptr_t) src_bufs[i];
        dst_desc[i].len = len;
        dst_desc[i].devId = 0;
        dst_desc[i].addr = (uintptr_t) dst_bufs[i];

        mem_list1.addDesc(src_desc[i]);
        mem_list2.addDesc(dst_desc[i]);
    }

    src_list = mem_list1.trim();
    dst_list = mem_list2.trim();

    status = A1->registerMem(mem_list1, &extra_params1);
    assert (status == NIXL_SUCCESS);

    status = A2->registerMem(mem_list2, &extra_params2);
    assert (status == NIXL_SUCCESS);

    std::string meta2;
    status = A2->getLocalMD(meta2);
    assert (status == NIXL_SUCCESS);
    assert (meta2.size() > 0);

    std::string remote_name;
    status = A1->loadRemoteMD(meta2, remote_name);
    assert (status == NIXL_SUCCESS);
    assert (remote_name == agent2);

    std::cout << "Ready to prepare side\n";

    nixlDlistH *src_side, *dst_side;

    status = A1->prepXferDlist(NIXL_INIT_AGENT, src_list, src_side, &extra_params1);
    assert (status == NIXL_SUCCESS);

    status = A1->prepXferDlist(remote_name, dst_list, dst_side, &extra_params1);
    assert (status == NIXL_SUCCESS);

    std::cout << "prep done, starting transfers\n";

    std::vector<int> indices1, indices2;

    for(int i = 0; i<(n_bufs/2); i++) {
        //initial bufs
        memset(src_bufs[i], 0xbb, len);
        indices1.push_back(i);
    }
    for(int i = (n_bufs/2); i<n_bufs; i++)
        indices2.push_back(i);

    nixlXferReqH *req1, *req2, *req3;

    //write first half of src_bufs to dst_bufs
    status = A1->makeXferReq(NIXL_WRITE, src_side, indices1, dst_side, indices1, req1, &extra_params1);
    assert (status == NIXL_SUCCESS);

    nixl_status_t xfer_status = A1->postXferReq(req1);

    while (xfer_status != NIXL_SUCCESS) {
        if (xfer_status != NIXL_SUCCESS) xfer_status = A1->getXferStatus(req1);
        assert (xfer_status >= 0);
    }

    for(int i = 0; i<(n_bufs/2); i++)
        check_buf(dst_bufs[i], len);

    std::cout << "transfer 1 done\n";

    //read first half of dst_bufs back to second half of src_bufs
    status = A1->makeXferReq(NIXL_READ, src_side, indices2, dst_side, indices1, req2, &extra_params1);
    assert (status == NIXL_SUCCESS);

    xfer_status = A1->postXferReq(req2);

    while (xfer_status != NIXL_SUCCESS) {
        if (xfer_status != NIXL_SUCCESS) xfer_status = A1->getXferStatus(req2);
        assert (xfer_status >= 0);
    }

    for(int i = (n_bufs/2); i<n_bufs; i++)
        check_buf(src_bufs[i], len);

    std::cout << "transfer 2 done\n";

    //write second half of src_bufs to dst_bufs
    status = A1->makeXferReq(NIXL_WRITE, src_side, indices2, dst_side, indices2, req3, &extra_params1);
    assert (status == NIXL_SUCCESS);

    xfer_status = A1->postXferReq(req3);

    while (xfer_status != NIXL_SUCCESS) {
        if (xfer_status != NIXL_SUCCESS) xfer_status = A1->getXferStatus(req3);
        assert (xfer_status >= 0);
    }

    for(int i = (n_bufs/2); i<n_bufs; i++)
        check_buf(dst_bufs[i], len);

    std::cout << "transfer 3 done\n";

    status = A1->releaseXferReq(req1);
    assert (status == NIXL_SUCCESS);
    status = A1->releaseXferReq(req2);
    assert (status == NIXL_SUCCESS);
    status = A1->releaseXferReq(req3);
    assert (status == NIXL_SUCCESS);

    // Commented out to test auto deregistration
    // status = A1->deregisterMem(mem_list1, &extra_params1);
    // assert (status == NIXL_SUCCESS);
    // status = A2->deregisterMem(mem_list2, &extra_params2);
    // assert (status == NIXL_SUCCESS);

    status = A1->releasedDlistH(src_side);
    assert (status == NIXL_SUCCESS);
    status = A1->releasedDlistH(dst_side);
    assert (status == NIXL_SUCCESS);

    for(int i = 0; i<n_bufs; i++) {
        free(src_bufs[i]);
        free(dst_bufs[i]);
    }

    return NIXL_SUCCESS;
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
    assert (ret1 == NIXL_SUCCESS);

    std::cout << "Available plugins:\n";

    for (nixl_backend_t b: plugins)
        std::cout << b << "\n";

    std::cout << "Using backend: " << backend << "\n";
    ret1 = A1.getPluginParams(backend, mems1, init1);
    ret2 = A2.getPluginParams(backend, mems2, init2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

    std::cout << "Params before init:\n";
    printParams(init1, mems1);
    printParams(init2, mems2);

    nixlBackendH *bknd1, *bknd2;
    ret1 = A1.createBackend(backend, init1, bknd1);
    ret2 = A2.createBackend(backend, init2, bknd2);

    nixl_opt_args_t extra_params1, extra_params2;
    extra_params1.backends.push_back(bknd1);
    extra_params2.backends.push_back(bknd2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

    ret1 = A1.getBackendParams(bknd1, mems1, init1);
    ret2 = A2.getBackendParams(bknd2, mems2, init2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

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
    void* addr3 = calloc(1, len);

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

    buff3.addr   = (uintptr_t) addr3;
    buff3.len    = len;
    buff3.devId = 0;
    dlist1.addDesc(buff3);

    // dlist1.print();
    // dlist2.print();

    ret1 = A1.registerMem(dlist1, &extra_params1);
    ret2 = A2.registerMem(dlist2, &extra_params2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

    std::string meta1;
    ret1 = A1.getLocalMD(meta1);
    std::string meta2;
    ret2 = A2.getLocalMD(meta2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

    std::cout << "Agent1's Metadata: " << meta1 << "\n";
    std::cout << "Agent2's Metadata: " << meta2 << "\n";

    ret1 = A1.loadRemoteMD (meta2, ret_s1);
    ret2 = A2.loadRemoteMD (meta1, ret_s2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

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

    nixl_xfer_dlist_t req_ldst_descs (DRAM_SEG);
    nixlBasicDesc req_ldst;
    req_ldst.addr   = (uintptr_t) ((char*) addr3) + dst_offset; //random offset
    req_ldst.len    = req_size;
    req_ldst.devId = 0;
    req_ldst_descs.addDesc(req_ldst);

    std::cout << "Transfer request from " << addr1 << " to " << addr2 << "\n";
    nixlXferReqH *req_handle, *req_handle2;

    extra_params1.notifMsg = "notification";
    extra_params1.hasNotif = true;
    ret1 = A1.createXferReq(NIXL_WRITE, req_src_descs, req_dst_descs, agent2, req_handle, &extra_params1);
    assert (ret1 == NIXL_SUCCESS);

    nixl_status_t status = A1.postXferReq(req_handle);

    std::cout << "Transfer was posted\n";

    nixl_notifs_t notif_map;
    int n_notifs = 0;

    while (status != NIXL_SUCCESS || n_notifs == 0) {
        if (status != NIXL_SUCCESS) status = A1.getXferStatus(req_handle);
        if (n_notifs == 0) ret2 = A2.getNotifs(notif_map);
        assert (status >= 0);
        assert (ret2 == NIXL_SUCCESS);
        n_notifs = notif_map.size();
    }

    std::vector<std::string> agent1_notifs = notif_map[agent1];
    assert (agent1_notifs.size() == 1);
    assert (agent1_notifs.front() == "notification");
    notif_map[agent1].clear(); // Redundant, for testing
    notif_map.clear();
    n_notifs = 0;

    std::cout << "Transfer verified\n";

    std::cout << "performing partialMdTest with backends " << bknd1 << " " << bknd2 << "\n";
    ret1 = partialMdTest(&A1, &A2, bknd1, bknd2);
    assert (ret1 == NIXL_SUCCESS);

    std::cout << "performing sideXferTest with backends " << bknd1 << " " << bknd2 << "\n";
    ret1 = sideXferTest(&A1, &A2, req_handle, bknd2);
    assert (ret1 == NIXL_SUCCESS);

    std::cout << "Performing local test\n";
    extra_params1.notifMsg = "local_notif";
    extra_params1.hasNotif = true;
    ret2 = A1.createXferReq(NIXL_WRITE, req_src_descs, req_ldst_descs, agent1, req_handle2, &extra_params1);
    assert (ret2 == NIXL_SUCCESS);

    status = A1.postXferReq(req_handle2);
    std::cout << "Local transfer was posted\n";

    while (status != NIXL_SUCCESS || n_notifs == 0) {
        if (status != NIXL_SUCCESS) status = A1.getXferStatus(req_handle2);
        if (n_notifs == 0) ret2 = A1.getNotifs(notif_map);
        assert (status >= 0);
        assert (ret2 == NIXL_SUCCESS);
        n_notifs = notif_map.size();
    }

    agent1_notifs = notif_map[agent1];
    assert (agent1_notifs.size() == 1);
    assert (agent1_notifs.front() == "local_notif");
    assert (equal_buf((void*) req_src.addr, (void*) req_ldst.addr, req_size) == true);

    ret1 = A1.releaseXferReq(req_handle);
    ret2 = A1.releaseXferReq(req_handle2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

    ret1 = A1.deregisterMem(dlist1, &extra_params1);
    ret2 = A2.deregisterMem(dlist2, &extra_params2);

    assert (ret1 == NIXL_SUCCESS);
    assert (ret2 == NIXL_SUCCESS);

    //only initiator should call invalidate
    ret1 = A1.invalidateRemoteMD(agent2);
    //A2.invalidateRemoteMD(agent1);

    assert (ret1 == NIXL_SUCCESS);

    free(addr1);
    free(addr2);
    free(addr3);

    std::cout << "Test done\n";
}
