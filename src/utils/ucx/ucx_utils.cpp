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
#include "ucx_utils.h"

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

using namespace std;


bool
nixlUcxContext::mtLevelIsSupproted(nixl_ucx_mt_t mt_type)
{
    ucp_lib_attr_t attr;
    attr.field_mask = UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL;
    ucp_lib_query(&attr);

    switch (mt_type) {
        case NIXL_UCX_MT_SINGLE:
            return (attr.max_thread_level >= UCS_THREAD_MODE_SERIALIZED);
        case NIXL_UCX_MT_CTX:
        case NIXL_UCX_MT_WORKER:
            return (attr.max_thread_level >= UCS_THREAD_MODE_MULTI);
        default:
            assert(mt_type < NIXL_UCX_MT_MAX);
            abort();
    }
    return false;
}

nixlUcxContext::nixlUcxContext(
        std::vector<std::string> devs, size_t req_size, nixlUcxContext::req_cb_t init_cb,
        nixlUcxContext::req_cb_t fini_cb, nixl_ucx_mt_t __mt_type)
{
    ucp_params_t ucp_params;
    ucp_config_t* ucp_config;
    ucs_status_t status = UCS_OK;

    mt_type = __mt_type;

    ucp_params.field_mask =
            UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    ucp_params.features = UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64 | UCP_FEATURE_AM;
    switch (mt_type) {
        case NIXL_UCX_MT_SINGLE:
        case NIXL_UCX_MT_WORKER:
            ucp_params.mt_workers_shared = 0;
            break;
        case NIXL_UCX_MT_CTX:
            ucp_params.mt_workers_shared = 1;
            break;
        default:
            assert(mt_type < NIXL_UCX_MT_MAX);
            abort();
    }
    ucp_params.estimated_num_eps = 3;

    if (req_size) {
        ucp_params.request_size = req_size;
        ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_SIZE;
    }

    if (init_cb) {
        ucp_params.request_init = init_cb;
        ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_INIT;
    }

    if (fini_cb) {
        ucp_params.request_cleanup = fini_cb;
        ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_CLEANUP;
    }

    ucp_config_read(NULL, NULL, &ucp_config);

    /* If requested, restrict the set of network devices */
    if (devs.size()) {
        /* TODO: check if this is the best way */
        string dev_str = "";
        unsigned int i;
        for (i = 0; i < devs.size() - 1; i++) {
            dev_str = dev_str + devs[i] + ":1,";
        }
        dev_str = dev_str + devs[i] + ":1";
        ucp_config_modify(ucp_config, "NET_DEVICES", dev_str.c_str());
    }

    status = ucp_init(&ucp_params, ucp_config, &ctx);
    if (status != UCS_OK) {
        /* TODO: proper cleanup */
        // TODO: MSW_NET_ERROR(priv->net, "failed to ucp_init(%s)\n", ucs_status_string(status));
        return;
    }
    ucp_config_release(ucp_config);
}

nixlUcxContext::~nixlUcxContext()
{
    ucp_cleanup(ctx);
}


nixlUcxWorker::nixlUcxWorker(nixlUcxContext* _ctx)
{
    ucp_worker_params_t worker_params;
    ucs_status_t status = UCS_OK;

    ctx = _ctx;

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;

    switch (ctx->mt_type) {
        case NIXL_UCX_MT_CTX:
            worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
            break;
        case NIXL_UCX_MT_SINGLE:
            worker_params.thread_mode = UCS_THREAD_MODE_SERIALIZED;
            break;
        case NIXL_UCX_MT_WORKER:
            worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
            break;
        default:
            assert(ctx->mt_type < NIXL_UCX_MT_MAX);
            abort();
    }

    status = ucp_worker_create(ctx->ctx, &worker_params, &worker);
    if (status != UCS_OK) {
        // TODO: MSW_NET_ERROR(priv->net, "failed to create ucp_worker (%s)\n", ucs_status_string(status));
        return;
    }
}

nixlUcxWorker::~nixlUcxWorker()
{
    ucp_worker_destroy(worker);
}

int
nixlUcxWorker::epAddr(uint64_t& addr, size_t& size)
{
    ucp_worker_attr_t wattr;
    ucs_status_t status;
    void* new_addr;

    wattr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS | UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    wattr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
    status = ucp_worker_query(worker, &wattr);
    if (UCS_OK != status) {
        // TODO: printf
        return -1;
    }

    new_addr = calloc(wattr.address_length, sizeof(char));
    memcpy(new_addr, wattr.address, wattr.address_length);
    ucp_worker_release_address(worker, wattr.address);

    addr = (uint64_t)new_addr;
    size = wattr.address_length;
    return 0;
}

/* ===========================================
 * EP management
 * =========================================== */

static void
err_cb(void* arg, ucp_ep_h ep, ucs_status_t status)
{
    printf("error handling callback was invoked with status %d (%s)\n", status, ucs_status_string(status));
}


int
nixlUcxWorker::connect(void* addr, size_t size, nixlUcxEp& ep)
{
    ucp_ep_params_t ep_params;
    ucs_status_t status;

    // ep.uw = this;

    ep_params.field_mask =
            UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLER | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    // TODO: Need to be UCP_ERR_HANDLING_MODE_PEER to properly handle disconnect
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_NONE;

    ep_params.err_handler.cb = err_cb;
    ep_params.address = (ucp_address_t*)addr;

    status = ucp_ep_create(worker, &ep_params, &ep.eph);
    if (status != UCS_OK) {
        /* TODO: proper cleanup */
        /* TODO:  MSW_NET_ERROR(priv->net, "!!! failed to create endpoint to remote %d (%s)\n",
                      status, ucs_status_string(status)); */
        return -1;
    }

    return 0;
}

int
nixlUcxWorker::disconnect(nixlUcxEp& ep)
{
    ucs_status_ptr_t request = ucp_ep_close_nb(ep.eph, UCP_EP_CLOSE_MODE_FLUSH);

    if (UCS_PTR_IS_ERR(request)) {
        // TODO: proper cleanup
        // if (UCS_PTR_IS_ERR(request)) {
        //     MSW_NET_ERROR(priv->net, "ucp_disconnect_nb() failed: %s",
        //                  ucs_status_string(UCS_PTR_STATUS(request)));
        //     return -1;
        // }
        return -1;
    }

    if (request) {
        while (ucp_request_check_status(request) == UCS_INPROGRESS) {
            ucp_worker_progress(worker);
        }
        ucp_request_free(request);
    }

    return 0;
}

int
nixlUcxWorker::disconnect_nb(nixlUcxEp& ep)
{
    ucs_status_ptr_t request = ucp_ep_close_nb(ep.eph, UCP_EP_CLOSE_MODE_FLUSH);

    if (UCS_PTR_IS_ERR(request)) {
        // TODO: proper cleanup
        // if (UCS_PTR_IS_ERR(request)) {
        //     MSW_NET_ERROR(priv->net, "ucp_disconnect_nb() failed: %s",
        //                  ucs_status_string(UCS_PTR_STATUS(request)));
        //     return -1;
        // }
        return -1;
    }

    if (request) {
        // don't care
        ucp_request_free(request);
    }

    return 0;
}

/* ===========================================
 * Memory management
 * =========================================== */


int
nixlUcxWorker::memReg(void* addr, size_t size, nixlUcxMem& mem)
{
    ucs_status_t status;

    // mem.uw = this;
    mem.base = addr;
    mem.size = size;

    ucp_mem_map_params_t mem_params = {
            .field_mask =
                    UCP_MEM_MAP_PARAM_FIELD_FLAGS | UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_ADDRESS,
            .address = mem.base,
            .length = mem.size,
    };

    status = ucp_mem_map(ctx->ctx, &mem_params, &mem.memh);
    if (status != UCS_OK) {
        /* TODOL: MSW_NET_ERROR(priv->net, "failed to ucp_mem_map (%s)\n", ucs_status_string(status)); */
        return -1;
    }

    return 0;
}


size_t
nixlUcxWorker::packRkey(nixlUcxMem& mem, uint64_t& addr, size_t& size)
{
    ucs_status_t status;
    void* rkey_buf;

    status = ucp_rkey_pack(ctx->ctx, mem.memh, &rkey_buf, &size);
    if (status != UCS_OK) {
        /* TODO: MSW_NET_ERROR(priv->net, "failed to ucp_rkey_pack (%s)\n", ucs_status_string(status)); */
        return -1;
    }

    /* Allocate the buffer */
    addr = (uint64_t)calloc(size, sizeof(char));
    if (!addr) {
        /* TODO: proper cleanup */
        /* TODO: MSW_NET_ERROR(priv->net, "failed to allocate memory key buffer\n"); */
        return -1;
    }
    memcpy((void*)addr, rkey_buf, size);
    ucp_rkey_buffer_release(rkey_buf);

    return 0;
}

void
nixlUcxWorker::memDereg(nixlUcxMem& mem)
{
    ucp_mem_unmap(ctx->ctx, mem.memh);
}

/* ===========================================
 * RKey management
 * =========================================== */

int
nixlUcxWorker::rkeyImport(nixlUcxEp& ep, void* addr, size_t size, nixlUcxRkey& rkey)
{
    ucs_status_t status;

    status = ucp_ep_rkey_unpack(ep.eph, addr, &rkey.rkeyh);
    if (status != UCS_OK) {
        /* TODO: MSW_NET_ERROR(priv->net, "unable to unpack key!\n"); */
        return -1;
    }

    return 0;
}

void
nixlUcxWorker::rkeyDestroy(nixlUcxRkey& rkey)
{
    ucp_rkey_destroy(rkey.rkeyh);
}

/* ===========================================
 * Active message handling
 * =========================================== */

int
nixlUcxWorker::regAmCallback(unsigned msg_id, ucp_am_recv_callback_t cb, void* arg)
{
    ucs_status_t status;
    ucp_am_handler_param_t params = {0};

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;

    params.id = msg_id;
    params.cb = cb;
    params.arg = arg;

    status = ucp_worker_set_am_recv_handler(worker, &params);

    if (status != UCS_OK) {
        // TODO: error handling
        return -1;
    }
    return 0;
}

nixl_status_t
nixlUcxWorker::sendAm(
        nixlUcxEp& ep, unsigned msg_id, void* hdr, size_t hdr_len, void* buffer, size_t len, uint32_t flags,
        nixlUcxReq& req)
{
    ucs_status_ptr_t request;
    ucp_request_param_t param = {0};

    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags = flags;

    request = ucp_am_send_nbx(ep.eph, msg_id, hdr, hdr_len, buffer, len, &param);

    if (request == NULL) {
        return NIXL_SUCCESS;
    } else if (UCS_PTR_IS_ERR(request)) {
        /* TODO: MSW_NET_ERROR(priv->net, "unable to complete UCX request (%s)\n",
                         ucs_status_string(UCS_PTR_STATUS(request))); */
        return NIXL_ERR_BACKEND;
    }

    req = (void*)request;
    return NIXL_IN_PROG;
}

int
nixlUcxWorker::getRndvData(void* data_desc, void* buffer, size_t len, const ucp_request_param_t* param, nixlUcxReq& req)
{
    ucs_status_ptr_t status;

    status = ucp_am_recv_data_nbx(worker, data_desc, buffer, len, param);
    if (UCS_PTR_IS_ERR(status)) {
        // TODO: error handling
        return -1;
    }
    req = (void*)status;

    return 0;
}

/* ===========================================
 * Data transfer
 * =========================================== */

int
nixlUcxWorker::progress()
{
    return ucp_worker_progress(worker);
}

nixl_status_t
nixlUcxWorker::read(
        nixlUcxEp& ep, uint64_t raddr, nixlUcxRkey& rk, void* laddr, nixlUcxMem& mem, size_t size, nixlUcxReq& req)
{
    ucs_status_ptr_t request;

    ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_MEMH,
            .memh = mem.memh,
    };

    request = ucp_get_nbx(ep.eph, laddr, size, raddr, rk.rkeyh, &param);
    if (request == NULL) {
        return NIXL_SUCCESS;
    } else if (UCS_PTR_IS_ERR(request)) {
        /* TODO: MSW_NET_ERROR(priv->net, "unable to complete UCX request (%s)\n",
                         ucs_status_string(UCS_PTR_STATUS(request))); */
        return NIXL_ERR_BACKEND;
    }

    req = (void*)request;
    return NIXL_IN_PROG;
}

nixl_status_t
nixlUcxWorker::write(
        nixlUcxEp& ep, void* laddr, nixlUcxMem& mem, uint64_t raddr, nixlUcxRkey& rk, size_t size, nixlUcxReq& req)
{
    ucs_status_ptr_t request;

    ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_MEMH,
            .memh = mem.memh,
    };

    request = ucp_put_nbx(ep.eph, laddr, size, raddr, rk.rkeyh, &param);
    if (request == NULL) {
        return NIXL_SUCCESS;
    } else if (UCS_PTR_IS_ERR(request)) {
        /* TODO: MSW_NET_ERROR(priv->net, "unable to complete UCX request (%s)\n",
                         ucs_status_string(UCS_PTR_STATUS(request))); */
        return NIXL_ERR_BACKEND;
    }

    req = (void*)request;
    return NIXL_IN_PROG;
}

nixl_status_t
nixlUcxWorker::test(nixlUcxReq req)
{
    ucs_status_t status;

    if (req == NULL) {
        return NIXL_SUCCESS;
    }

    ucp_worker_progress(worker);
    status = ucp_request_check_status(req);
    if (status == UCS_INPROGRESS) {
        return NIXL_IN_PROG;
    }

    if (status == UCS_OK) {
        return NIXL_SUCCESS;
    } else {
        // TODO: error
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxWorker::flushEp(nixlUcxEp& ep, nixlUcxReq& req)
{
    ucp_request_param_t param;
    ucs_status_ptr_t request;

    param.op_attr_mask = 0;
    request = ucp_ep_flush_nbx(ep.eph, &param);

    if (request == NULL) {
        return NIXL_SUCCESS;
    } else if (UCS_PTR_IS_ERR(request)) {
        /* TODO: MSW_NET_ERROR(priv->net, "unable to complete UCX request (%s)\n",
                         ucs_status_string(UCS_PTR_STATUS(request))); */
        return NIXL_ERR_BACKEND;
    }

    req = (void*)request;
    return NIXL_IN_PROG;
}

void
nixlUcxWorker::reqRelease(nixlUcxReq req)
{
    ucp_request_free((void*)req);
}

void
nixlUcxWorker::reqCancel(nixlUcxReq req)
{
    ucp_request_cancel(worker, (void*)req);
}
