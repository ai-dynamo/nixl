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
#ifndef NIXL_SRC_UTILS_UCX_UCX_UTILS_H
#define NIXL_SRC_UTILS_UCX_UCX_UTILS_H

#include <memory>
#include <type_traits>

extern "C" {
#include <ucp/api/ucp.h>
}

#include <nixl_types.h>

#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"


enum class nixl_ucx_mt_t { SINGLE, CTX, WORKER };

inline constexpr std::string_view nixl_ucx_err_handling_param_name = "ucx_error_handling_mode";

template<typename Enum>
[[nodiscard]] constexpr auto
enumToInteger(const Enum e) noexcept {
    static_assert(std::is_enum_v<Enum>);
    return std::underlying_type_t<Enum>(e);
}

[[nodiscard]] std::string_view constexpr to_string_view(const nixl_ucx_mt_t t) noexcept {
    switch (t) {
    case nixl_ucx_mt_t::SINGLE:
        return "SINGLE";
    case nixl_ucx_mt_t::CTX:
        return "CTX";
    case nixl_ucx_mt_t::WORKER:
        return "WORKER";
    }
    return "INVALID"; // It is not a to_string function's job to validate.
}

template<typename T>
[[nodiscard]] T
nixl_b_params_get(const nixl_b_params_t *custom_params, const std::string &key, T default_value) {
    if (!custom_params) {
        return default_value;
    }

    auto it = custom_params->find(key);
    if (it == custom_params->end()) {
        return default_value;
    }

    if constexpr (std::is_same_v<T, int>) {
        T result;
        return absl::SimpleAtoi(it->second, &result) ? result : default_value;
    }
}

using nixlUcxReq = void *;

namespace nixl::ucx {
class rkey;
}
class nixlUcxMem;

class nixlUcxEp {
    enum nixl_ucx_ep_state_t {
        NIXL_UCX_EP_STATE_NULL,
        NIXL_UCX_EP_STATE_CONNECTED,
        NIXL_UCX_EP_STATE_FAILED,
        NIXL_UCX_EP_STATE_DISCONNECTED
    };

private:
    ucp_ep_h eph{nullptr};
    nixl_ucx_ep_state_t state{NIXL_UCX_EP_STATE_NULL};

    void
    setState(nixl_ucx_ep_state_t new_state);
    nixl_status_t
    closeImpl(ucp_ep_close_flags_t flags);

    /* Connection */
    nixl_status_t
    disconnect_nb();

    static void
    sendAmCallback(void *request, ucs_status_t status, void *user_data);

public:
    void
    err_cb(ucp_ep_h ucp_ep, ucs_status_t status);

    nixl_status_t
    checkTxState() const {
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

    nixlUcxEp(ucp_worker_h worker, void *addr, ucp_err_handling_mode_t err_handling_mode);
    ~nixlUcxEp();
    nixlUcxEp(const nixlUcxEp &) = delete;
    nixlUcxEp &
    operator=(const nixlUcxEp &) = delete;

    using am_deleter_t = std::function<void(void *request, void *buffer)>;

    /* Active message handling */
    nixl_status_t
    sendAm(unsigned msg_id,
           void *hdr,
           size_t hdr_len,
           void *buffer,
           size_t len,
           uint32_t flags,
           nixlUcxReq *req = nullptr,
           const am_deleter_t &deleter = nullptr);

    /* Data access */
    [[nodiscard]] nixl_status_t
    read(uint64_t raddr,
         const nixl::ucx::rkey &rkey,
         void *laddr,
         nixlUcxMem &mem,
         size_t size,
         nixlUcxReq &req);
    [[nodiscard]] nixl_status_t
    write(void *laddr,
          nixlUcxMem &mem,
          uint64_t raddr,
          const nixl::ucx::rkey &rkey,
          size_t size,
          nixlUcxReq &req);
    nixl_status_t
    estimateCost(size_t size,
                 std::chrono::microseconds &duration,
                 std::chrono::microseconds &err_margin,
                 nixl_cost_t &method);
    nixl_status_t
    flushEp(nixlUcxReq &req);

    [[nodiscard]] ucp_ep_h
    getEp() const noexcept {
        return eph;
    }
};

class nixlUcxMem {
private:
    void *base;
    size_t size;
    ucp_mem_h memh;

public:
    [[nodiscard]] ucp_mem_h
    getMemh() const noexcept {
        return memh;
    }

    [[nodiscard]] void *
    getBase() const noexcept {
        return base;
    }

    [[nodiscard]] size_t
    getSize() const noexcept {
        return size;
    }

    friend class nixlUcxWorker;
    friend class nixlUcxContext;
    friend class nixlUcxEp;
};

class nixlUcxContext {
private:
    /* Local UCX stuff */
    ucp_context_h ctx;
    const nixl_ucx_mt_t mtType_;
    const unsigned ucpVersion_;

public:
    nixlUcxContext(const std::vector<std::string> &devs,
                   bool prog_thread,
                   unsigned long num_workers,
                   nixl_thread_sync_t sync_mode,
                   size_t num_device_channels,
                   const std::string &engine_conf = "");
    ~nixlUcxContext();

    nixlUcxContext(nixlUcxContext &&) = delete;
    nixlUcxContext(const nixlUcxContext &) = delete;

    void
    operator=(nixlUcxContext &&) = delete;
    void
    operator=(const nixlUcxContext &) = delete;

    /* Memory management */
    int
    memReg(void *addr, size_t size, nixlUcxMem &mem, nixl_mem_t nixl_mem_type);
    [[nodiscard]] std::string
    packRkey(nixlUcxMem &mem);
    void
    memDereg(nixlUcxMem &mem);

    void
    warnAboutHardwareSupportMismatch() const;

    friend class nixlUcxWorker;
};

[[nodiscard]] bool
nixlUcxMtLevelIsSupported(const nixl_ucx_mt_t) noexcept;

class nixlUcxWorker {
public:
    explicit nixlUcxWorker(
        const nixlUcxContext &,
        ucp_err_handling_mode_t ucp_err_handling_mode = UCP_ERR_HANDLING_MODE_NONE);

    nixlUcxWorker(nixlUcxWorker &&) = delete;
    nixlUcxWorker(const nixlUcxWorker &) = delete;
    void
    operator=(nixlUcxWorker &&) = delete;
    void
    operator=(const nixlUcxWorker &) = delete;

    /* Connection */
    [[nodiscard]] std::string
    epAddr();
    absl::StatusOr<std::unique_ptr<nixlUcxEp>>
    connect(void *addr, size_t size);

    /* Active message handling */
    int
    regAmCallback(unsigned msg_id, ucp_am_recv_callback_t cb, void *arg);

    /* Data access */
    int
    progress();
    [[nodiscard]] nixl_status_t
    test(nixlUcxReq req);

    void
    reqRelease(nixlUcxReq req);
    void
    reqCancel(nixlUcxReq req);

    [[nodiscard]] nixl_status_t
    arm() const noexcept;

    [[nodiscard]] int
    getEfd() const;

    /* GPU signal management */
    void
    prepGpuSignal(const nixlUcxMem &mem, void *signal) const;

    [[nodiscard]] ucp_worker_h
    get() const noexcept {
        return worker.get();
    }

private:
    [[nodiscard]] static ucp_worker *
    createUcpWorker(const nixlUcxContext &);

    const std::unique_ptr<ucp_worker, void (*)(ucp_worker *)> worker;
    ucp_err_handling_mode_t err_handling_mode_;
};

[[nodiscard]] nixl_b_params_t
get_ucx_backend_common_options();

[[nodiscard]] nixl_status_t
ucx_status_to_nixl(ucs_status_t status);

[[nodiscard]] std::string_view
ucx_err_mode_to_string(ucp_err_handling_mode_t t);

[[nodiscard]] ucp_err_handling_mode_t
ucx_err_mode_from_string(std::string_view s);

#endif
