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

#include "gpu_xfer_req_h.h"
#include "common/nixl_log.h"
#include "ucx_utils.h"
#include "rkey.h"
#include "config.h"

#include <chrono>
#include <cstdlib>
#include <string_view>
#include <thread>

extern "C" {
#ifdef HAVE_UCX_GPU_DEVICE_API
#include <ucp/api/device/ucp_host.h>
#endif
}

namespace nixl::ucx {

#ifdef HAVE_UCX_GPU_DEVICE_API

namespace {

    [[nodiscard]] std::chrono::milliseconds
    get_gpu_xfer_timeout() noexcept {
        constexpr int default_timeout_ms = 5000;
        constexpr std::string_view timeout_env_name = "NIXL_UCX_GPU_XFER_TIMEOUT_MS";

        const char *timeout_env = std::getenv(timeout_env_name.data());
        if (!timeout_env) {
            return std::chrono::milliseconds(default_timeout_ms);
        }

        const int timeout_ms = std::atoi(timeout_env);
        if (timeout_ms <= 0) {
            NIXL_WARN << "Invalid " << timeout_env_name << " value: " << timeout_env
                      << ", using default " << default_timeout_ms << " ms";
            return std::chrono::milliseconds(default_timeout_ms);
        }

        return std::chrono::milliseconds(timeout_ms);
    }

} // namespace

nixlGpuXferReqH
createGpuXferReq(const nixlUcxEp &ep,
                 nixlUcxWorker &worker,
                 const std::vector<nixlUcxMem> &local_mems,
                 const std::vector<const nixl::ucx::rkey *> &remote_rkeys,
                 const std::vector<uint64_t> &remote_addrs) {
    nixl_status_t status = ep.checkTxState();
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Endpoint not in valid state for creating memory list");
    }

    if (local_mems.empty() || remote_rkeys.empty() || remote_addrs.empty()) {
        throw std::invalid_argument("Empty memory, rkey, or address lists provided");
    }

    if (local_mems.size() != remote_rkeys.size() || local_mems.size() != remote_addrs.size()) {
        throw std::invalid_argument(
            "Local memory, remote rkey, and remote address lists must have same size");
    }

    std::vector<ucp_device_mem_list_elem_t> ucp_elements;
    ucp_elements.reserve(local_mems.size());

    for (size_t i = 0; i < local_mems.size(); i++) {
        ucp_device_mem_list_elem_t ucp_elem;
        ucp_elem.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
            UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY | UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR |
            UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR | UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;
        ucp_elem.memh = local_mems[i].getMemh();
        ucp_elem.rkey = remote_rkeys[i]->get();
        ucp_elem.local_addr = local_mems[i].getBase();
        ucp_elem.remote_addr = remote_addrs[i];
        ucp_elem.length = local_mems[i].getSize();
        ucp_elements.push_back(ucp_elem);
    }

    ucp_device_mem_list_params_t params;
    params.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
    params.elements = ucp_elements.data();
    params.element_size = sizeof(ucp_device_mem_list_elem_t);
    params.num_elements = ucp_elements.size();

    const auto timeout = get_gpu_xfer_timeout();

    ucp_device_mem_list_handle_h ucx_handle;
    ucs_status_t ucs_status;
    const auto start = std::chrono::steady_clock::now();
    bool timeout_warned = false;
    while ((ucs_status = ucp_device_mem_list_create(ep.getEp(), &params, &ucx_handle)) ==
           UCS_ERR_NOT_CONNECTED) {
        if (!timeout_warned && std::chrono::steady_clock::now() - start > timeout) {
            NIXL_WARN << "Timeout on creating device memory list has been exceeded (timeout="
                      << timeout.count() << " ms)";
            timeout_warned = true;
        }

        if (worker.progress() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (ucs_status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to create device memory list: ") +
                                 ucs_status_string(ucs_status));
    }

    NIXL_DEBUG << "Created device memory list: ep=" << ep.getEp() << " handle=" << ucx_handle
               << " num_elements=" << local_mems.size() << " worker=" << &worker;
    return reinterpret_cast<nixlGpuXferReqH>(ucx_handle);
}

void
releaseGpuXferReq(nixlGpuXferReqH gpu_req) noexcept {
    auto ucx_handle = reinterpret_cast<ucp_device_mem_list_handle_h>(gpu_req);
    ucp_device_mem_list_release(ucx_handle);
}

#else

nixlGpuXferReqH
createGpuXferReq(const nixlUcxEp &ep,
                 nixlUcxWorker &worker,
                 const std::vector<nixlUcxMem> &local_mems,
                 const std::vector<const nixl::ucx::rkey *> &remote_rkeys,
                 const std::vector<uint64_t> &remote_addrs) {
    NIXL_ERROR << "UCX GPU device API not supported";
    throw std::runtime_error("UCX GPU device API not available");
}

void
releaseGpuXferReq(nixlGpuXferReqH gpu_req) noexcept {
    NIXL_WARN << "UCX GPU device API not supported - cannot release GPU transfer request handle";
}

#endif

} // namespace nixl::ucx
