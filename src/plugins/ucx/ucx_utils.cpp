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

#include "ucx_utils.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <nixl_types.h>

extern "C" {
#ifdef HAVE_UCX_GPU_DEVICE_API
#include <ucp/api/device/ucp_host.h>
#endif
}

#include "common/hw_info.h"
#include "common/nixl_log.h"
#include "config.h"
#include "serdes/serdes.h"
#include "rkey.h"

using namespace std;

[[nodiscard]] nixl_b_params_t
get_ucx_backend_common_options() {
    nixl_b_params_t params = {{"ucx_devices", ""}, {"num_workers", "1"}};

    params.emplace(nixl_ucx_err_handling_param_name,
                   ucx_err_mode_to_string(UCP_ERR_HANDLING_MODE_PEER));
    return params;
}

nixl_status_t
ucx_status_to_nixl(ucs_status_t status) {
    if (status == UCS_OK) {
        return NIXL_SUCCESS;
    }

    switch (status) {
    case UCS_INPROGRESS:
    case UCS_ERR_BUSY:
        return NIXL_IN_PROG;
    case UCS_ERR_NOT_CONNECTED:
    case UCS_ERR_CONNECTION_RESET:
    case UCS_ERR_ENDPOINT_TIMEOUT:
        return NIXL_ERR_REMOTE_DISCONNECT;
    case UCS_ERR_INVALID_PARAM:
        return NIXL_ERR_INVALID_PARAM;
    case UCS_ERR_CANCELED:
        return NIXL_ERR_CANCELED;
    default:
        NIXL_WARN << "Unexpected UCX error: " << ucs_status_string(status);
        return NIXL_ERR_BACKEND;
    }
}

[[nodiscard]] std::string_view
ucx_err_mode_to_string(ucp_err_handling_mode_t t) {
    switch (t) {
    case UCP_ERR_HANDLING_MODE_NONE:
        return "none";
    case UCP_ERR_HANDLING_MODE_PEER:
        return "peer";
    default:
        throw std::invalid_argument(std::to_string(t));
    }
}

[[nodiscard]] ucp_err_handling_mode_t
ucx_err_mode_from_string(std::string_view s) {
    constexpr std::array<ucp_err_handling_mode_t, 2> nixl_ucx_err_handling_modes = {
        UCP_ERR_HANDLING_MODE_NONE,
        UCP_ERR_HANDLING_MODE_PEER,
    };

    for (const auto mode : nixl_ucx_err_handling_modes) {
        if (ucx_err_mode_to_string(mode) == s) {
            return mode;
        }
    }

    std::stringstream err_msg;
    err_msg << "Invalid error handling mode: " << s << ". Valid values are: <";
    for (size_t i = 0; i < nixl_ucx_err_handling_modes.size(); ++i) {
        err_msg << ucx_err_mode_to_string(nixl_ucx_err_handling_modes[i]);
        if (i < nixl_ucx_err_handling_modes.size() - 1) {
            err_msg << "|";
        }
    }

    err_msg << ">";
    throw std::invalid_argument(err_msg.str());
}

static void
err_cb_wrapper(void *arg, ucp_ep_h ucp_ep, ucs_status_t status) {
    auto ep = static_cast<nixlUcxEp *>(arg);
    ep->err_cb(ucp_ep, status);
}

void
nixlUcxEp::err_cb(ucp_ep_h ucp_ep, ucs_status_t status) {
    ucs_status_ptr_t request;

    NIXL_DEBUG << "ep " << eph << ": state " << state
               << ", UCX error handling callback was invoked with status " << status << " ("
               << ucs_status_string(status) << ")";

    NIXL_ASSERT(eph == ucp_ep);

    switch (state) {
    case NIXL_UCX_EP_STATE_NULL:
    case NIXL_UCX_EP_STATE_FAILED:
        // The error was already handled, nothing to do
    case NIXL_UCX_EP_STATE_DISCONNECTED:
        // The EP has been disconnected, nothing to do
        return;
    case NIXL_UCX_EP_STATE_CONNECTED:
        setState(NIXL_UCX_EP_STATE_FAILED);
        request = ucp_ep_close_nb(ucp_ep, UCP_EP_CLOSE_MODE_FORCE);
        if (UCS_PTR_IS_PTR(request)) {
            ucp_request_free(request);
        }
        return;
    }
    NIXL_FATAL << "Invalid endpoint state: " << state;
    std::terminate();
}

void
nixlUcxEp::setState(nixl_ucx_ep_state_t new_state) {
    NIXL_ASSERT(new_state != state);
    NIXL_DEBUG << "ep " << eph << ": state " << state << " -> " << new_state;
    state = new_state;
}

nixl_status_t
nixlUcxEp::closeImpl(ucp_ep_close_flags_t flags) {
    ucs_status_ptr_t request = nullptr;
    ucp_request_param_t req_param = {.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS, .flags = flags};

    switch (state) {
    case NIXL_UCX_EP_STATE_NULL:
    case NIXL_UCX_EP_STATE_DISCONNECTED:
        // The EP has not been connected, or already disconnected.
        // Nothing to do.
        NIXL_ASSERT(eph == nullptr);
        return NIXL_SUCCESS;
    case NIXL_UCX_EP_STATE_FAILED:
        // The EP was closed in error callback, just return error.
        eph = nullptr;
        return NIXL_ERR_REMOTE_DISCONNECT;
    case NIXL_UCX_EP_STATE_CONNECTED:
        request = ucp_ep_close_nbx(eph, &req_param);
        if (request == nullptr) {
            eph = nullptr;
            return NIXL_SUCCESS;
        }

        if (UCS_PTR_IS_ERR(request)) {
            eph = nullptr;
            return ucx_status_to_nixl(UCS_PTR_STATUS(request));
        }

        ucp_request_free(request);
        eph = nullptr;
        return NIXL_SUCCESS;
    }
    NIXL_FATAL << "Invalid endpoint state: " << state;
    std::terminate();
}

nixlUcxEp::nixlUcxEp(ucp_worker_h worker, void *addr, ucp_err_handling_mode_t err_handling_mode) {
    ucp_ep_params_t ep_params;
    nixl_status_t status;

    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLER |
        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode = err_handling_mode;
    ep_params.err_handler.cb = err_cb_wrapper;
    ep_params.err_handler.arg = static_cast<void *>(this);
    ep_params.address = reinterpret_cast<ucp_address_t *>(addr);

    status = ucx_status_to_nixl(ucp_ep_create(worker, &ep_params, &eph));
    if (status == NIXL_SUCCESS)
        setState(NIXL_UCX_EP_STATE_CONNECTED);
    else
        throw std::runtime_error("failed to create ep");
}

nixlUcxEp::~nixlUcxEp() {
    nixl_status_t status = disconnect_nb();
    if (status) NIXL_ERROR << "Failed to disconnect ep with status " << status;
}

/* ===========================================
 * EP management
 * =========================================== */

nixl_status_t
nixlUcxEp::disconnect_nb() {
    nixl_status_t status = closeImpl(ucp_ep_close_flags_t(0));

    // At step of disconnect we can ignore the remote disconnect error.
    return (status == NIXL_ERR_REMOTE_DISCONNECT) ? NIXL_SUCCESS : status;
}

/* ===========================================
 * Active message handling
 * =========================================== */

using nixl_ucx_am_cb_ctx_t = std::pair<void *, nixlUcxEp::am_deleter_t>;
using nixl_ucx_am_cb_ctx_ptr_t = std::unique_ptr<nixl_ucx_am_cb_ctx_t>;

void
nixlUcxEp::sendAmCallback(void *request, ucs_status_t status, void *user_data) {
    auto ctx = static_cast<nixl_ucx_am_cb_ctx_t *>(user_data);
    ctx->second(request, ctx->first);
    delete ctx;
}

nixl_status_t
nixlUcxEp::sendAm(unsigned msg_id,
                  void *hdr,
                  size_t hdr_len,
                  void *buffer,
                  size_t len,
                  uint32_t flags,
                  nixlUcxReq *req,
                  const am_deleter_t &deleter) {
    const nixl_status_t status = checkTxState();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ucp_request_param_t param = {0};

    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags = flags;

    nixl_ucx_am_cb_ctx_ptr_t ctx;
    if (deleter) {
        ctx = std::make_unique<nixl_ucx_am_cb_ctx_t>(buffer, deleter);
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        param.cb.send = sendAmCallback;
        param.user_data = ctx.get();
    }

    const ucs_status_ptr_t request =
        ucp_am_send_nbx(eph, msg_id, hdr, hdr_len, buffer, len, &param);
    if (UCS_PTR_IS_PTR(request)) {
        ctx.release();
        if (req != nullptr) {
            *req = static_cast<nixlUcxReq>(request);
        }
        return NIXL_IN_PROG;
    } else if (deleter) {
        deleter(nullptr, buffer);
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

/* ===========================================
 * Data transfer
 * =========================================== */

nixl_status_t
nixlUcxEp::read(uint64_t raddr,
                const nixl::ucx::rkey &rkey,
                void *laddr,
                nixlUcxMem &mem,
                size_t size,
                nixlUcxReq &req) {
    nixl_status_t status = checkTxState();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMH | UCP_OP_ATTR_FLAG_MULTI_SEND,
        .memh = mem.memh,
    };

    ucs_status_ptr_t request = ucp_get_nbx(eph, laddr, size, raddr, rkey.get(), &param);
    if (UCS_PTR_IS_PTR(request)) {
        req = static_cast<nixlUcxReq>(request);
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t
nixlUcxEp::write(void *laddr,
                 nixlUcxMem &mem,
                 uint64_t raddr,
                 const nixl::ucx::rkey &rkey,
                 size_t size,
                 nixlUcxReq &req) {
    nixl_status_t status = checkTxState();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMH | UCP_OP_ATTR_FLAG_MULTI_SEND,
        .memh = mem.memh,
    };

    ucs_status_ptr_t request = ucp_put_nbx(eph, laddr, size, raddr, rkey.get(), &param);
    if (UCS_PTR_IS_PTR(request)) {
        req = static_cast<nixlUcxReq>(request);
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t
nixlUcxEp::estimateCost(size_t size,
                        std::chrono::microseconds &duration,
                        std::chrono::microseconds &err_margin,
                        nixl_cost_t &method) {
    ucp_ep_evaluate_perf_param_t params = {
        .field_mask = UCP_EP_PERF_PARAM_FIELD_MESSAGE_SIZE,
        .message_size = size,
    };

    ucp_ep_evaluate_perf_attr_t cost_result = {
        .field_mask = UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME,
    };

    ucs_status_t status = ucp_ep_evaluate_perf(this->eph, &params, &cost_result);
    if (status != UCS_OK) {
        NIXL_ERROR << "ucp_ep_evaluate_perf failed: " << ucs_status_string(status);
        return NIXL_ERR_BACKEND;
    }

    duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::duration<double>(cost_result.estimated_time));
    method = nixl_cost_t::ANALYTICAL_BACKEND;
    // Currently, we do not have a way to estimate the error margin
    err_margin = std::chrono::microseconds(0);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEp::flushEp(nixlUcxReq &req) {
    ucp_request_param_t param;
    ucs_status_ptr_t request;

    param.op_attr_mask = 0;
    request = ucp_ep_flush_nbx(eph, &param);

    if (UCS_PTR_IS_PTR(request)) {
        req = static_cast<nixlUcxReq>(request);
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

bool
nixlUcxMtLevelIsSupported(const nixl_ucx_mt_t mt_type) noexcept {
    ucp_lib_attr_t attr;
    attr.field_mask = UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL;
    ucp_lib_query(&attr);

    switch (mt_type) {
    case nixl_ucx_mt_t::SINGLE:
        return attr.max_thread_level >= UCS_THREAD_MODE_SERIALIZED;
    case nixl_ucx_mt_t::CTX:
    case nixl_ucx_mt_t::WORKER:
        return attr.max_thread_level >= UCS_THREAD_MODE_MULTI;
    }
    NIXL_FATAL << "invalid mt type: " << enumToInteger(mt_type);
    std::terminate();
}

namespace {

[[nodiscard]] unsigned
makeUcpVersion() noexcept {
    unsigned major_version, minor_version, release_number;
    ucp_get_version(&major_version, &minor_version, &release_number);
    return UCP_VERSION(major_version, minor_version);
}

[[nodiscard]] nixl_ucx_mt_t
makeMtType(const bool prog_thread, const nixl_thread_sync_t sync_mode) noexcept {
    // With strict synchronization model nixlAgent serializes access to backends, with more
    // permissive models backends need to account for concurrent access and ensure their internal
    // state is properly protected. Progress thread creates internal concurrency in UCX backend
    // irrespective of nixlAgent synchronization model.
    return (sync_mode == nixl_thread_sync_t::NIXL_THREAD_SYNC_RW || prog_thread) ?
        nixl_ucx_mt_t::WORKER :
        nixl_ucx_mt_t::SINGLE;
}

} // namespace

nixlUcxContext::nixlUcxContext(const std::vector<std::string> &devs,
                               bool prog_thread,
                               unsigned long num_workers,
                               nixl_thread_sync_t sync_mode,
                               size_t num_device_channels,
                               const std::string &engine_config)
    : mtType_(makeMtType(prog_thread, sync_mode)),
      ucpVersion_(makeUcpVersion()) {

    ucp_params_t ucp_params;
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    ucp_params.features = UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64 | UCP_FEATURE_AM;
#ifdef HAVE_UCX_GPU_DEVICE_API
    ucp_params.features |= UCP_FEATURE_DEVICE;
#endif

    if (prog_thread) ucp_params.features |= UCP_FEATURE_WAKEUP;
    ucp_params.mt_workers_shared = num_workers > 1 ? 1 : 0;

    nixl::ucx::config config;

    /* If requested, restrict the set of network devices */
    if (devs.size()) {
        /* TODO: check if this is the best way */
        std::string devs_str;
        for (const auto &dev : devs) {
            devs_str += dev + ":1,";
        }
        devs_str.pop_back(); // to remove odd comma after the last device
        config.modifyAlways("NET_DEVICES", devs_str.c_str());
    }

    config.modify("ADDRESS_VERSION", "v2");
    config.modify("RNDV_THRESH", "inf");
    config.modify("MAX_RMA_RAILS", "2");
    config.modify("IB_PCI_RELAXED_ORDERING", "try");
    config.modify("RCACHE_MAX_UNRELEASED", "1024");
    config.modify("RC_GDA_NUM_CHANNELS", std::to_string(num_device_channels));

    if (ucpVersion_ >= UCP_VERSION(1, 19)) {
        config.modify("MAX_COMPONENT_MDS", "32");
    } else {
        NIXL_WARN << "UCX version is less than 1.19, CUDA support is limited, "
                  << "including the lack of support for multi-GPU within a single process.";
    }

    std::string elem;
    std::stringstream stream(engine_config);

    while (std::getline(stream, elem, ',')) {
        std::string_view elem_view = elem;
        size_t pos = elem_view.find('=');

        if (pos != std::string::npos) {
            config.modify(elem_view.substr(0, pos), elem_view.substr(pos + 1));
        }
    }

    const auto status = ucp_init(&ucp_params, config.getUcpConfig(), &ctx);
    if (status != UCS_OK) {
        throw std::runtime_error("Failed to create UCX context: " +
                                 std::string(ucs_status_string(status)));
    }
}

nixlUcxContext::~nixlUcxContext() {
    ucp_cleanup(ctx);
}

namespace {
[[nodiscard]] ucs_thread_mode_t
toUcsThreadModeChecked(const nixl_ucx_mt_t t) {
    switch (t) {
    case nixl_ucx_mt_t::CTX:
        return UCS_THREAD_MODE_SINGLE;
    case nixl_ucx_mt_t::SINGLE:
        return UCS_THREAD_MODE_SERIALIZED;
    case nixl_ucx_mt_t::WORKER:
        return UCS_THREAD_MODE_MULTI;
    }
    NIXL_FATAL << "Invalid UCX worker type: "
               << static_cast<std::underlying_type_t<nixl_ucx_mt_t>>(t);
    std::terminate();
}

struct nixlUcpWorkerParams : ucp_worker_params_t {
    explicit nixlUcpWorkerParams(const nixl_ucx_mt_t t) {
        field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        thread_mode = toUcsThreadModeChecked(t);
    }
};

static_assert(sizeof(nixlUcpWorkerParams) == sizeof(ucp_worker_params_t));

} // namespace

ucp_worker *
nixlUcxWorker::createUcpWorker(const nixlUcxContext &ctx) {
    ucp_worker *worker = nullptr;
    const nixlUcpWorkerParams params(ctx.mtType_);
    const ucs_status_t status = ucp_worker_create(ctx.ctx, &params, &worker);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to create UCX worker: ") +
                                 ucs_status_string(status));
    }

    return worker;
}

nixlUcxWorker::nixlUcxWorker(const nixlUcxContext &ctx, ucp_err_handling_mode_t err_handling_mode)
    : worker(createUcpWorker(ctx), &ucp_worker_destroy),
      err_handling_mode_(err_handling_mode) {}

std::string
nixlUcxWorker::epAddr() {
    ucp_worker_attr_t wattr;

    wattr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
    const ucs_status_t status = ucp_worker_query(worker.get(), &wattr);
    if (UCS_OK != status) {
        throw std::runtime_error(std::string("Unable to query UCX worker address: ") +
                                 ucs_status_string(status));
    }

    const std::string result = nixlSerDes::_bytesToString(wattr.address, wattr.address_length);
    ucp_worker_release_address(worker.get(), wattr.address);
    return result;
}

absl::StatusOr<std::unique_ptr<nixlUcxEp>>
nixlUcxWorker::connect(void *addr, std::size_t size) {
    try {
        return std::make_unique<nixlUcxEp>(worker.get(), addr, err_handling_mode_);
    }
    catch (const std::exception &e) {
        return absl::UnavailableError(e.what());
    }
}

/* ===========================================
 * Memory management
 * =========================================== */


int
nixlUcxContext::memReg(void *addr, size_t size, nixlUcxMem &mem, nixl_mem_t nixl_mem_type) {
    mem.base = addr;
    mem.size = size;

    ucp_mem_map_params_t mem_params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS,
        .address = mem.base,
        .length = mem.size,
    };

    ucs_status_t status = ucp_mem_map(ctx, &mem_params, &mem.memh);
    if (status != UCS_OK) {
        NIXL_ERROR << "Failed to ucp_mem_map: " << ucs_status_string(status);
        return -1;
    }

    if (nixl_mem_type == nixl_mem_t::VRAM_SEG) {
        ucp_mem_attr_t attr;
        attr.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
        status = ucp_mem_query(mem.memh, &attr);
        if (status != UCS_OK) {
            NIXL_ERROR << "Failed to ucp_mem_query: " << ucs_status_string(status);
            ucp_mem_unmap(ctx, mem.memh);
            return -1;
        }

        if (attr.mem_type == UCS_MEMORY_TYPE_HOST) {
            NIXL_WARN << "memory is detected as host, check that UCX is configured"
                         " with CUDA support";
        }
    }

    return 0;
}

std::string
nixlUcxContext::packRkey(nixlUcxMem &mem) {
    void *rkey_buf;
    std::size_t size;

    const ucs_status_t status = ucp_rkey_pack(ctx, mem.memh, &rkey_buf, &size);
    if (status != UCS_OK) {
        NIXL_ERROR << "Failed to ucp_rkey_pack: " << ucs_status_string(status);
        return {};
    }
    const std::string result = nixlSerDes::_bytesToString(rkey_buf, size);
    ucp_rkey_buffer_release(rkey_buf);
    return result;
}

void
nixlUcxContext::memDereg(nixlUcxMem &mem) {
    ucp_mem_unmap(ctx, mem.memh);
}

#ifndef HAVE_UCX_GPU_DEVICE_API
namespace {
constexpr std::string_view ucxGpuDeviceApiUnsupported{
    "UCX was not compiled with GPU device API support"};
}
#endif

size_t
nixlUcxContext::getGpuSignalSize() const {
#ifdef HAVE_UCX_GPU_DEVICE_API
    ucp_context_attr_t attr;
    attr.field_mask = UCP_ATTR_FIELD_DEVICE_COUNTER_SIZE;
    ucs_status_t query_status = ucp_context_query(ctx, &attr);

    if (query_status != UCS_OK) {
        throw std::runtime_error(
            std::string("Failed to query UCX context for device counter size: ") +
            ucs_status_string(query_status));
    }

    return attr.device_counter_size;
#else
    throw std::runtime_error(std::string(ucxGpuDeviceApiUnsupported));
#endif
}

void
nixlUcxContext::warnAboutHardwareSupportMismatch() const {
    ucp_context_attr_t attr = {
        .field_mask = UCP_ATTR_FIELD_MEMORY_TYPES,
    };
    const auto status = ucp_context_query(ctx, &attr);
    if (status != UCS_OK) {
        NIXL_WARN << "Failed to query UCX context: " << ucs_status_string(status) << ", "
                  << "hardware support mismatch check will be skipped";
        return;
    }

    const nixl::hwInfo hw_info;

    NIXL_DEBUG << "hwInfo { "
               << "numNvidiaGpus=" << hw_info.numNvidiaGpus << ", "
               << "numIbDevices=" << hw_info.numIbDevices << " }";

    if (hw_info.numNvidiaGpus > 0 && !UCS_BIT_GET(attr.memory_types, UCS_MEMORY_TYPE_CUDA)) {
        NIXL_WARN << hw_info.numNvidiaGpus
                  << " NVIDIA GPU(s) were detected, but UCX CUDA support was not found! "
                  << "GPU memory is not supported.";
    }

    if (ucpVersion_ >= UCP_VERSION(1, 21)) {
        // `UCS_MEMORY_TYPE_RDMA` is included in `memory_types` only from UCX 1.21
        if (hw_info.numIbDevices > 0 && !UCS_BIT_GET(attr.memory_types, UCS_MEMORY_TYPE_RDMA)) {
            NIXL_WARN << hw_info.numIbDevices
                      << " IB device(s) were detected, but accelerated IB support was not found! "
                         "Performance may be degraded.";
        }
    }
}

/* ===========================================
 * Active message handling
 * =========================================== */

int
nixlUcxWorker::regAmCallback(unsigned msg_id, ucp_am_recv_callback_t cb, void *arg) {
    ucp_am_handler_param_t params = {0};

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB |
        UCP_AM_HANDLER_PARAM_FIELD_ARG;

    params.id = msg_id;
    params.cb = cb;
    params.arg = arg;

    const ucs_status_t status = ucp_worker_set_am_recv_handler(worker.get(), &params);

    if (status != UCS_OK) {
        // TODO: error handling
        return -1;
    }
    return 0;
}

/* ===========================================
 * Data transfer
 * =========================================== */

int
nixlUcxWorker::progress() {
    return ucp_worker_progress(worker.get());
}

nixl_status_t
nixlUcxWorker::test(nixlUcxReq req) {
    if (req == nullptr) {
        return NIXL_SUCCESS;
    }
    ucp_worker_progress(worker.get());
    return ucx_status_to_nixl(ucp_request_check_status(req));
}

void
nixlUcxWorker::reqRelease(nixlUcxReq req) {
    ucp_request_free(req);
}

void
nixlUcxWorker::reqCancel(nixlUcxReq req) {
    ucp_request_cancel(worker.get(), req);
}

nixl_status_t
nixlUcxWorker::arm() const noexcept {
    return ucx_status_to_nixl(ucp_worker_arm(worker.get()));
}

int
nixlUcxWorker::getEfd() const {
    int fd;
    const auto status = ucp_worker_get_efd(worker.get(), &fd);
    if (status != UCS_OK) {
        const auto err_str =
            std::string("Couldn't obtain fd for a worker: ") + ucs_status_string(status);
        NIXL_ERROR << err_str;
        throw std::runtime_error(err_str);
    }
    return fd;
}

void
nixlUcxWorker::prepGpuSignal([[maybe_unused]] const nixlUcxMem &mem,
                             [[maybe_unused]] void *signal) const {
#ifdef HAVE_UCX_GPU_DEVICE_API
    if (!signal) {
        throw std::invalid_argument("Signal pointer cannot be null");
    }

    ucp_device_counter_params_t params;
    params.field_mask = UCP_DEVICE_COUNTER_PARAMS_FIELD_MEMH;
    params.memh = mem.memh;

    ucs_status_t status = ucp_device_counter_init(worker.get(), &params, signal);

    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to initialize GPU signal: ") +
                                 ucs_status_string(status));
    }
#else
    throw std::runtime_error(std::string(ucxGpuDeviceApiUnsupported));
#endif
}
