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
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ucx/ucx_utils.h"

using namespace std;

struct sample_header {
    uint32_t test;
};

ucs_status_t
check_buffer(
        void* arg, const void* header, size_t header_length, void* data, size_t length,
        const ucp_am_recv_param_t* param)
{
    struct sample_header* hdr = (struct sample_header*)header;
    // TODO: is data 8 byte aligned?
    uint64_t recv_data = *((uint64_t*)data);

    if (hdr->test != 0xcee)
        return UCS_ERR_INVALID_PARAM;

    assert(length == 8);
    assert(recv_data == 0xdeaddeaddeadbeef);

    std::cout << "check_buffer passed\n";

    return UCS_OK;
}

ucs_status_t
rndv_test(
        void* arg, const void* header, size_t header_length, void* data, size_t length,
        const ucp_am_recv_param_t* param)
{
    struct sample_header* hdr = (struct sample_header*)header;
    void* recv_buffer = calloc(1, length);
    nixlUcxWorker* am_worker = (nixlUcxWorker*)arg;
    ucp_request_param_t recv_param = {0};
    uint64_t check_data;
    nixlUcxReq req;
    int ret = 0;

    if (hdr->test != 0xcee)
        return UCS_ERR_INVALID_PARAM;

    std::cout << "rndv_test started\n";
    assert(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV);
    ret = am_worker->getRndvData(data, recv_buffer, length, &recv_param, req);
    assert(ret == 0);

    while (ret == 0) {
        ret = am_worker->test(req);
    }

    check_data = ((uint64_t*)recv_buffer)[0];
    assert(check_data == 0xdeaddeaddeadbeef);
    free(recv_buffer);

    std::cout << "rndv_test passed\n";
    return UCS_OK;
}

int
main()
{
    vector<string> devs;
    devs.push_back("mlx5_0");

    nixlUcxContext c[2] = {
            nixlUcxContext(devs, 0, NULL, NULL, NIXL_UCX_MT_SINGLE),
            nixlUcxContext(devs, 0, NULL, NULL, NIXL_UCX_MT_SINGLE)};

    nixlUcxWorker w[2] = {nixlUcxWorker(&c[0]), nixlUcxWorker(&c[1])};
    nixlUcxEp ep[2];
    nixlUcxReq req;
    uint64_t buffer;
    int ret, i;

    unsigned check_cb_id = 1, rndv_cb_id = 2;

    void* big_buffer = calloc(1, 8192);
    struct sample_header hdr = {0};

    buffer = 0xdeaddeaddeadbeef;
    ((uint64_t*)big_buffer)[0] = 0xdeaddeaddeadbeef;
    hdr.test = 0xcee;

    /* Test control path */
    for (i = 0; i < 2; i++) {
        uint64_t addr;
        size_t size;
        assert(0 == w[i].epAddr(addr, size));
        assert(0 == w[!i].connect((void*)addr, size, ep[!i]));

        // no need for mem_reg with active messages
        // assert (0 == w[i].mem_reg(buffer[i], 128, mem[i]));
        // assert (0 == w[i].mem_addr(mem[i], addr, size));
        // assert (0 == w[!i].rkey_import(ep[!i], (void*) addr, size, rkey[!i]));
        free((void*)addr);
    }

    /* Register active message callbacks */
    ret = w[0].regAmCallback(check_cb_id, check_buffer, NULL);
    assert(ret == 0);

    w[0].progress();
    w[1].progress();

    ret = w[0].regAmCallback(rndv_cb_id, rndv_test, &(w[0]));
    assert(ret == 0);

    w[0].progress();

    /* Test first callback */
    ret = w[1].sendAm(ep[1], check_cb_id, &hdr, sizeof(struct sample_header), (void*)&buffer, sizeof(buffer), 0, req);
    assert(ret == 0);

    while (ret == 0) {
        ret = w[1].test(req);
        w[0].progress();
    }

    std::cout << "first active message complete\n";

    /* Test second callback */
    uint32_t flags = 0;
    flags |= UCP_AM_SEND_FLAG_RNDV;

    ret = w[1].sendAm(ep[1], rndv_cb_id, &hdr, sizeof(struct sample_header), big_buffer, 8192, flags, req);
    assert(ret == 0);

    while (ret == 0) {
        ret = w[1].test(req);
        w[0].progress();
    }

    std::cout << "second active message complete\n";

    // make sure callbacks are complete
    w[0].progress();

    /* Test shutdown */
    for (i = 0; i < 2; i++) {
        assert(0 == w[i].disconnect(ep[i]));
    }

    free(big_buffer);
}
