/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mem_list.h"

#include <stdexcept>

#ifdef HAVE_UCX_GPU_DEVICE_API_V2
extern "C" {
#include <ucp/api/device/ucp_host.h>
}

#include "rkey.h"
#include "ucx_utils.h"

namespace {
using remoteMem = nixl::ucx::remoteMem;

ucp_device_mem_list_elem_t
createElement(const std::unique_ptr<remoteMem> &mem) {
    ucp_device_mem_list_elem_t element;
    if (mem) {
        element.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
            UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR | UCP_DEVICE_MEM_LIST_ELEM_FIELD_EP;
        element.rkey = mem->rkey_.get();
        element.remote_addr = mem->addr_;
        element.ep = mem->ep_.getEp();
    } else {
        element.field_mask = 0;
    }
    return element;
}

ucp_device_mem_list_elem_t
createElement(const nixlUcxMem &mem) {
    ucp_device_mem_list_elem_t element;
    element.field_mask =
        UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH | UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR;
    element.memh = mem.getMemh();
    element.local_addr = mem.getBase();
    return element;
}

template<typename T>
std::vector<ucp_device_mem_list_elem_t>
createElements(const std::vector<T> &mems) {
    std::vector<ucp_device_mem_list_elem_t> elements;
    for (const auto &mem : mems) {
        elements.emplace_back(createElement(mem));
    }
    return elements;
}

ucp_device_mem_list_params_t
addElements(const std::vector<ucp_device_mem_list_elem_t> &elements) {
    ucp_device_mem_list_params_t params;
    params.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
    params.elements = elements.data();
    params.element_size = sizeof(ucp_device_mem_list_elem_t);
    params.num_elements = elements.size();
    return params;
}
} // namespace

namespace nixl::ucx {
void *
createMemList(const std::vector<std::unique_ptr<remoteMem>> &mems, nixlUcxWorker &worker) {
    const auto elements = createElements(mems);
    const auto params = addElements(elements);

    ucp_device_remote_mem_list_h handle{nullptr};
    ucs_status_t status;
    while ((status = ucp_device_remote_mem_list_create(&params, &handle)) ==
           UCS_ERR_NOT_CONNECTED) {
        worker.progress();
    }

    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to create device remote memory list: ") +
                                 ucs_status_string(status));
    }

    return handle;
}

void *
createMemList(const std::vector<nixlUcxMem> &mems, const nixlUcxWorker &worker) {
    const auto elements = createElements(mems);
    auto params = addElements(elements);
    params.field_mask |= UCP_DEVICE_MEM_LIST_PARAMS_FIELD_WORKER;
    params.worker = worker.get();

    ucp_device_local_mem_list_h handle{nullptr};
    const auto status = ucp_device_local_mem_list_create(&params, &handle);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to create device local memory list: ") +
                                 ucs_status_string(status));
    }

    return handle;
}

void
releaseMemList(void *mvh) noexcept {
    ucp_device_mem_list_release(mvh);
}
} // namespace nixl::ucx
#else
namespace nixl::ucx {
void *
createMemList(const std::vector<std::unique_ptr<remoteMem>> &mems, nixlUcxWorker &worker) {
    throw std::runtime_error("UCX GPU device API is not supported");
}

void *
createMemList(const std::vector<nixlUcxMem> &mems, const nixlUcxWorker &worker) {
    throw std::runtime_error("UCX GPU device API is not supported");
}

void
releaseMemList(void *mvh) noexcept {}
} // namespace nixl::ucx
#endif
