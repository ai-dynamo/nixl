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

extern "C" {
#ifdef HAVE_UCX_GPU_DEVICE_API
#include <ucp/api/device/ucp_host.h>
#endif
}

namespace nixl::ucx {

#ifdef HAVE_UCX_GPU_DEVICE_API

nixlGpuXferReqH
createGpuXferReq(const nixlUcxEp &ep,
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

    ucp_device_mem_list_handle_h ucx_handle;
    ucs_status_t ucs_status = ucp_device_mem_list_create(ep.getEp(), &params, &ucx_handle);
    if (ucs_status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to create device memory list: ") +
                                 ucs_status_string(ucs_status));
    }

    NIXL_DEBUG << "Created device memory list: ep=" << ep.getEp() << " handle=" << ucx_handle
               << " num_elements=" << local_mems.size();
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
