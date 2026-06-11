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

#include "rkey.h"
#include "ucx_enums.h"

#include "absl/strings/numbers.h"

inline constexpr std::string_view nixl_ucx_err_handling_param_name = "ucx_error_handling_mode";
inline constexpr std::string_view nixl_ucx_vram_staging_param_name = "vram_staging";
inline constexpr std::string_view nixl_ucx_staging_chunk_size_param_name = "staging_chunk_size";
inline constexpr std::string_view nixl_ucx_staging_slots_param_name = "staging_slots_per_gpu";
inline constexpr std::string_view nixl_ucx_staging_force_progress_param_name =
    "staging_force_progress_thread";
inline constexpr std::string_view nixl_ucx_staging_cuda_streams_param_name =
    "staging_cuda_copy_streams";
inline constexpr std::string_view nixl_ucx_staging_slot_window_param_name =
    "staging_slot_request_window";
inline constexpr std::string_view nixl_ucx_staging_batch_flush_param_name =
    "staging_batch_flush";
inline constexpr std::string_view nixl_ucx_staging_target_h2d_worker_param_name =
    "staging_target_h2d_worker";
inline constexpr std::string_view nixl_ucx_staging_source_d2h_prefetch_param_name =
    "staging_source_d2h_prefetch";
inline constexpr std::string_view nixl_ucx_staging_lease_timeout_param_name =
    "staging_lease_timeout_ms";
inline constexpr std::string_view nixl_ucx_vram_local_staging_param_name = "vram_local_staging";
inline constexpr std::string_view nixl_ucx_local_staging_shm_dir_param_name =
    "local_staging_shm_dir";
inline constexpr std::string_view nixl_ucx_local_staging_fallback_param_name =
    "local_staging_fallback";
inline constexpr std::string_view nixl_ucx_vram_staging_env_name = "NIXL_UCX_VRAM_STAGING";
inline constexpr std::string_view nixl_ucx_staging_chunk_size_env_name =
    "NIXL_UCX_STAGING_CHUNK_SIZE";
inline constexpr std::string_view nixl_ucx_staging_slots_env_name = "NIXL_UCX_STAGING_SLOTS";
inline constexpr std::string_view nixl_ucx_staging_force_progress_env_name =
    "NIXL_UCX_STAGING_FORCE_PROGRESS_THREAD";
inline constexpr std::string_view nixl_ucx_staging_cuda_streams_env_name =
    "NIXL_UCX_STAGING_CUDA_COPY_STREAMS";
inline constexpr std::string_view nixl_ucx_staging_slot_window_env_name =
    "NIXL_UCX_STAGING_SLOT_REQUEST_WINDOW";
inline constexpr std::string_view nixl_ucx_staging_batch_flush_env_name =
    "NIXL_UCX_STAGING_BATCH_FLUSH";
inline constexpr std::string_view nixl_ucx_staging_target_h2d_worker_env_name =
    "NIXL_UCX_STAGING_TARGET_H2D_WORKER";
inline constexpr std::string_view nixl_ucx_staging_source_d2h_prefetch_env_name =
    "NIXL_UCX_STAGING_SOURCE_D2H_PREFETCH";
inline constexpr std::string_view nixl_ucx_staging_lease_timeout_env_name =
    "NIXL_UCX_STAGING_LEASE_TIMEOUT_MS";
inline constexpr std::string_view nixl_ucx_vram_local_staging_env_name =
    "NIXL_UCX_VRAM_LOCAL_STAGING";
inline constexpr std::string_view nixl_ucx_local_staging_shm_dir_env_name =
    "NIXL_UCX_LOCAL_STAGING_SHM_DIR";
inline constexpr std::string_view nixl_ucx_local_staging_fallback_env_name =
    "NIXL_UCX_LOCAL_STAGING_FALLBACK";
inline constexpr std::string_view nixl_ucx_local_staging_force_attach_fail_env_name =
    "NIXL_UCX_LOCAL_STAGING_FORCE_ATTACH_FAIL";

// The API `ucp_context_query(ctx, &attr)` sets `UCS_MEMORY_TYPE_RDMA` in `attr.memory_types`
// field only from UCX 1.22
inline constexpr unsigned ucp_version_mem_type_rdma = UCP_VERSION(1, 22);

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

[[nodiscard]] bool
nixl_b_params_get_bool(const nixl_b_params_t *custom_params,
                       std::string_view key,
                       bool default_value);

[[nodiscard]] size_t
nixl_b_params_get_size(const nixl_b_params_t *custom_params,
                       std::string_view key,
                       size_t default_value);

[[nodiscard]] bool
nixl_env_get_bool(std::string_view name, bool default_value);

[[nodiscard]] size_t
nixl_env_get_size(std::string_view name, size_t default_value);

using nixlUcxReq = void *;

class nixlUcxMem;

class nixlUcxEp {
private:
    ucp_ep_h eph{nullptr};
    nixl::ucx::ep_state_t state = nixl::ucx::ep_state_t::UNINITIALIZED;

    void
    setState(nixl::ucx::ep_state_t new_state);
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

    [[nodiscard]] nixl_status_t
    checkTxState() const noexcept {
        return nixl::ucx::toNixlStatus(state);
    }

    nixlUcxEp(ucp_worker_h worker, void *addr, ucp_err_handling_mode_t err_handling_mode);
    ~nixlUcxEp();
    nixlUcxEp(const nixlUcxEp &) = delete;
    nixlUcxEp &
    operator=(const nixlUcxEp &) = delete;

    using am_deleter_t = std::function<void(void *request, void *buffer)>;

    /* Active message handling */
    nixl_status_t
    sendAm(nixl::ucx::am_cb_op_t msg_id,
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
    const nixl::ucx::mt_mode_t mtType_;
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
nixlUcxMtLevelIsSupported(const nixl::ucx::mt_mode_t) noexcept;

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
    [[nodiscard]] std::unique_ptr<nixlUcxEp>
    connect(void *addr, size_t size);

    /* Active message handling */
    int
    regAmCallback(nixl::ucx::am_cb_op_t msg_id, ucp_am_recv_callback_t cb, void *arg);

    /* Data access */
    unsigned
    progress();

    void
    progressLoop();

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

[[nodiscard]] std::string_view
ucx_err_mode_to_string(ucp_err_handling_mode_t t);

[[nodiscard]] ucp_err_handling_mode_t
ucx_err_mode_from_string(std::string_view s);

#endif
