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
#ifndef NIXL_SRC_UTILS_UCX_UCX_UTILS_H
#define NIXL_SRC_UTILS_UCX_UCX_UTILS_H

#include <memory>
#include <type_traits>

extern "C"
{
#include <ucp/api/ucp.h>
}

#include <nixl_types.h>

#include "absl/status/statusor.h"

enum class nixl_ucx_mt_t {
    SINGLE,
    CTX,
    WORKER
};

template<typename Enum>
[[nodiscard]] constexpr auto enumToInteger(const Enum e) noexcept
{
    static_assert(std::is_enum_v<Enum>);
    return std::underlying_type_t<Enum>(e);
}

[[nodiscard]] std::string_view constexpr to_string_view(const nixl_ucx_mt_t t) noexcept
{
    switch(t) {
        case nixl_ucx_mt_t::SINGLE:
            return "SINGLE";
        case nixl_ucx_mt_t::CTX:
            return "CTX";
        case nixl_ucx_mt_t::WORKER:
            return "WORKER";
    }
    return "INVALID";  // It is not a to_string function's job to validate.
}

using nixlUcxReq = void*;

class nixlUcxRkey;
class nixlUcxMem;

class nixlUcxEp {
    enum nixl_ucx_ep_state_t {
        NIXL_UCX_EP_STATE_NULL,
        NIXL_UCX_EP_STATE_CONNECTED,
        NIXL_UCX_EP_STATE_FAILED,
        NIXL_UCX_EP_STATE_DISCONNECTED
    };
private:
    ucp_ep_h            eph{nullptr};
    nixl_ucx_ep_state_t state{NIXL_UCX_EP_STATE_NULL};

    void setState(nixl_ucx_ep_state_t new_state);
    nixl_status_t closeImpl(ucp_ep_close_flags_t flags);

    /* Connection */
    nixl_status_t disconnect_nb();
public:
    void err_cb(ucp_ep_h ucp_ep, ucs_status_t status);

    nixl_status_t checkTxState() const {
        switch (state) {
        case NIXL_UCX_EP_STATE_CONNECTED:
            return NIXL_SUCCESS;
        case NIXL_UCX_EP_STATE_FAILED:
            return NIXL_ERR_REMOTE_DISCONNECT;
        case NIXL_UCX_EP_STATE_NULL:
        case NIXL_UCX_EP_STATE_DISCONNECTED:
        default:
            return NIXL_ERR_BACKEND;
        }
    }

    nixlUcxEp(ucp_worker_h worker, void* addr, ucp_err_handling_mode_t err_handling_mode);
    ~nixlUcxEp();
    nixlUcxEp(const nixlUcxEp&) = delete;
    nixlUcxEp& operator=(const nixlUcxEp&) = delete;

    /* Rkey */
    int rkeyImport(void* addr, size_t size, nixlUcxRkey &rkey);
    void rkeyDestroy(nixlUcxRkey &rkey);

    /* Active message handling */
    nixl_status_t sendAm(unsigned msg_id,
                         void* hdr, size_t hdr_len,
                         void* buffer, size_t len,
                         uint32_t flags, nixlUcxReq &req);

    /* Data access */
    nixl_status_t read(uint64_t raddr, nixlUcxRkey &rk,
                       void *laddr, nixlUcxMem &mem,
                       size_t size, nixlUcxReq &req);
    nixl_status_t write(void *laddr, nixlUcxMem &mem,
                        uint64_t raddr, nixlUcxRkey &rk,
                        size_t size, nixlUcxReq &req);
    nixl_status_t estimateCost(size_t size,
                               std::chrono::microseconds &duration,
                               std::chrono::microseconds &err_margin,
                               nixl_cost_t &method);
    nixl_status_t flushEp(nixlUcxReq &req);
};

class nixlUcxMem {
private:
    void *base;
    size_t size;
    ucp_mem_h memh;
public:
    friend class nixlUcxWorker;
    friend class nixlUcxContext;
    friend class nixlUcxEp;
};

class nixlUcxRkey {
private:
    ucp_rkey_h rkeyh;

public:

    friend class nixlUcxWorker;
    friend class nixlUcxEp;
};

class nixlUcxContext {
private:
    /* Local UCX stuff */
    ucp_context_h ctx;
    nixl_ucx_mt_t mt_type;
    ucp_err_handling_mode_t err_handling_mode = UCP_ERR_HANDLING_MODE_NONE;
public:

    using req_cb_t = void(void *request);
    nixlUcxContext(std::vector<std::string> devices,
                   size_t req_size, req_cb_t init_cb, req_cb_t fini_cb,
                   bool prog_thread, ucp_err_handling_mode_t err_handling_mode,
                   unsigned long num_workers, nixl_thread_sync_t sync_mode);
    ~nixlUcxContext();

    /* Memory management */
    int memReg(void *addr, size_t size, nixlUcxMem &mem);
    [[nodiscard]] std::string packRkey(nixlUcxMem &mem);
    void memDereg(nixlUcxMem &mem);

    friend class nixlUcxWorker;
};

[[nodiscard]] bool nixlUcxMtLevelIsSupported(const nixl_ucx_mt_t) noexcept;

class nixlUcxWorker {
private:
    /* Local UCX stuff */
    const std::shared_ptr<nixlUcxContext> ctx;
    const std::unique_ptr<ucp_worker, void(*)(ucp_worker*)> worker;

    [[nodiscard]] static ucp_worker* createUcpWorker(nixlUcxContext&);

  public:
    explicit nixlUcxWorker(const std::shared_ptr<nixlUcxContext> &_ctx);

    nixlUcxWorker( nixlUcxWorker&& ) = delete;
    nixlUcxWorker( const nixlUcxWorker& ) = delete;
    void operator=( nixlUcxWorker&& ) = delete;
    void operator=( const nixlUcxWorker& ) = delete;

    /* Connection */
    [[nodiscard]] std::string epAddr();
    absl::StatusOr<std::unique_ptr<nixlUcxEp>> connect(void* addr, size_t size);

    /* Active message handling */
    int regAmCallback(unsigned msg_id, ucp_am_recv_callback_t cb, void* arg);

    /* Data access */
    int progress();
    [[nodiscard]] nixl_status_t test(nixlUcxReq req);

    void reqRelease(nixlUcxReq req);
    void reqCancel(nixlUcxReq req);

    /* Worker access */
    [[nodiscard]] ucp_worker_h getWorker() const noexcept { return worker.get(); }
};

[[nodiscard]] static inline nixl_b_params_t get_ucx_backend_common_options() {
    return {
        { "ucx_devices", "" },
        { "ucx_error_handling_mode", "none" }, // or "peer"
        { "num_workers", "1" }
    };
}

nixl_status_t ucx_status_to_nixl(ucs_status_t status);

#endif
