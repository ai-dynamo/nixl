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
#include <vector>
#include <string>
#include <cassert>
#include <cstring>
#include <iostream>

#include "ucx/ucx_utils.h"

using namespace std;

struct sample_header {
    uint32_t test;
};

ucs_status_t check_buffer (void *arg, const void *header,
   		                   size_t header_length, void *data,
				           size_t length,
				           const ucp_am_recv_param_t *param)
{
    struct sample_header* hdr = (struct sample_header*) header;
    //TODO: is data 8 byte aligned?
    uint64_t recv_data = *((uint64_t*) data);

    if (hdr->test != 0xcee)
	    return UCS_ERR_INVALID_PARAM;

    assert (length == 8);
    assert (recv_data == 0xdeaddeaddeadbeef);

    std::cout << "check_buffer passed\n";

    return UCS_OK;
}

int main()
{
    vector<string> devs;
    devs.push_back("mlx5_0");

    nixlUcxContext c[2] = {{devs, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE},
                           {devs, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE}};

    nixlUcxWorker w[2] = {nixlUcxWorker(c[0]), nixlUcxWorker(c[1])};
    std::unique_ptr<nixlUcxEp> ep[2];
    nixlUcxReq req;
    uint64_t buffer;
    int ret, i;

    unsigned check_cb_id = 1, rndv_cb_id = 2;

    void* big_buffer = calloc(1, 8192);
    struct sample_header hdr = {0};

    buffer = 0xdeaddeaddeadbeef;
    ((uint64_t*) big_buffer)[0] = 0xdeaddeaddeadbeef;
    hdr.test = 0xcee;

    /* Test control path */
    for (i = 0; i < 2; i++) {
        const std::string addr = w[i].epAddr();
        assert(!addr.empty());
        auto result = w[!i].connect((void*)addr.data(), addr.size());
        assert(result.ok());
        ep[!i] = std::move(*result);
    }

    /* Register active message callbacks */
    ret = w[0].regAmCallback(check_cb_id, check_buffer, NULL);
    assert (ret == 0);

    w[0].progress();
    w[1].progress();
    w[0].progress();

    /* Test first callback */
    ret = ep[1]->sendAm(check_cb_id, &hdr, sizeof(hdr), (void *)&buffer, sizeof(buffer), 0, &req);
    assert(ret >= 0);

    while (ret == NIXL_IN_PROG) {
        ret = w[1].test(req);
        w[0].progress();
    }

    std::cout << "first active message complete\n";

    /* Test second callback */
    bool buffer_freed = false;
    auto deleter = [&buffer_freed](void *request, void *buffer) {
        buffer_freed = true;
        free(buffer);
        if (request != nullptr) {
            ucp_request_free(request);
        }
    };
    ret = ep[1]->sendAm(rndv_cb_id, &hdr, sizeof(hdr), big_buffer, 8192, 0, &req, deleter);
    assert(ret >= 0);

    while (ret == NIXL_IN_PROG) {
        ret = w[1].test(req);
        w[0].progress();
    }

    assert(buffer_freed);
    std::cout << "second active message complete\n";

    //make sure callbacks are complete
    while (w[0].progress())
        ;
}
