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
#ifndef __UCX_UTILS_H
#define __UCX_UTILS_H

extern "C" {
#include <ucp/api/ucp.h>
}

#include "nixl.h"

typedef enum { NIXL_UCX_MT_SINGLE, NIXL_UCX_MT_CTX, NIXL_UCX_MT_WORKER, NIXL_UCX_MT_MAX } nixl_ucx_mt_t;
class nixlUcxEp {
private:
    ucp_ep_h eph;

public:
    friend class nixlUcxWorker;
};

class nixlUcxMem {
private:
    void* base;
    size_t size;
    ucp_mem_h memh;

public:
    friend class nixlUcxWorker;
};

class nixlUcxRkey {
private:
    ucp_rkey_h rkeyh;

public:
    friend class nixlUcxWorker;
};

typedef void* nixlUcxReq;

class nixlUcxContext {
private:
    /* Local UCX stuff */
    ucp_context_h ctx;
    nixl_ucx_mt_t mt_type;

public:
    typedef void req_cb_t(void* request);
    nixlUcxContext(
            std::vector<std::string> devices, size_t req_size, req_cb_t init_cb, req_cb_t fini_cb,
            nixl_ucx_mt_t mt_type);
    ~nixlUcxContext();

    static bool mtLevelIsSupproted(nixl_ucx_mt_t mt_type);

    friend class nixlUcxWorker;
};

class nixlUcxWorker {
private:
    /* Local UCX stuff */
    nixlUcxContext* ctx;
    ucp_worker_h worker;

public:
    nixlUcxWorker(nixlUcxContext* ctx);
    ~nixlUcxWorker();

    /* Connection */
    int epAddr(uint64_t& addr, size_t& size);
    int connect(void* addr, size_t size, nixlUcxEp& ep);
    int disconnect(nixlUcxEp& ep);
    int disconnect_nb(nixlUcxEp& ep);

    /* Memory management */
    int memReg(void* addr, size_t size, nixlUcxMem& mem);
    size_t packRkey(nixlUcxMem& mem, uint64_t& addr, size_t& size);
    void memDereg(nixlUcxMem& mem);

    /* Rkey */
    int rkeyImport(nixlUcxEp& ep, void* addr, size_t size, nixlUcxRkey& rkey);
    void rkeyDestroy(nixlUcxRkey& rkey);

    /* Active message handling */
    int regAmCallback(unsigned msg_id, ucp_am_recv_callback_t cb, void* arg);
    nixl_status_t sendAm(
            nixlUcxEp& ep, unsigned msg_id, void* hdr, size_t hdr_len, void* buffer, size_t len, uint32_t flags,
            nixlUcxReq& req);
    int getRndvData(void* data_desc, void* buffer, size_t len, const ucp_request_param_t* param, nixlUcxReq& req);

    /* Data access */
    int progress();
    nixl_status_t flushEp(nixlUcxEp& ep, nixlUcxReq& req);
    nixl_status_t read(
            nixlUcxEp& ep, uint64_t raddr, nixlUcxRkey& rk, void* laddr, nixlUcxMem& mem, size_t size, nixlUcxReq& req);
    nixl_status_t write(
            nixlUcxEp& ep, void* laddr, nixlUcxMem& mem, uint64_t raddr, nixlUcxRkey& rk, size_t size, nixlUcxReq& req);
    nixl_status_t test(nixlUcxReq req);

    void reqRelease(nixlUcxReq req);
    void reqCancel(nixlUcxReq req);
};

#endif
