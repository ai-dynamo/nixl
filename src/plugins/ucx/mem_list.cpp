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

#ifdef HAVE_UCX_GPU_DEVICE_API_V2
#include "rkey.h"
#include "ucx_backend.h"
#include "ucx_utils.h"

extern "C" {
#include <ucp/api/device/ucp_host.h>
}
#endif

#include <stdexcept>

#ifdef HAVE_UCX_GPU_DEVICE_API_V2
namespace nixl::ucx {
using device_mem_vector_t = std::vector<ucp_device_mem_list_elem_t>;

class memListElement {
public:
    template<typename T>
    [[nodiscard]] static ucp_device_mem_list_elem_t
    get(const T &desc, size_t worker_id) {
        return memListElement(desc, worker_id).element_;
    }

private:
    memListElement(const nixlRemoteMetaDesc &desc, size_t worker_id)
        : element_(create(desc, worker_id)) {}

    memListElement(const nixlMetaDesc &desc, size_t) : element_(create(desc)) {}

    [[nodiscard]] static ucp_device_mem_list_elem_t
    create(const nixlRemoteMetaDesc &, size_t);

    [[nodiscard]] static ucp_device_mem_list_elem_t
    create(const nixlMetaDesc &);

    const ucp_device_mem_list_elem_t element_;
};

class memListParams {
public:
    explicit memListParams(const device_mem_vector_t &elements,
                           const nixlUcxWorker *worker = nullptr) noexcept;
    memListParams(const device_mem_vector_t &&, const nixlUcxWorker *worker = nullptr) = delete;

    [[nodiscard]] const ucp_device_mem_list_params_t *
    get() const noexcept {
        return &params_;
    }

private:
    ucp_device_mem_list_params_t params_;
};

ucp_device_mem_list_elem_t
memListElement::create(const nixlRemoteMetaDesc &desc, size_t worker_id) {
    ucp_device_mem_list_elem_t element;
    if (desc.remoteAgent == nixl_invalid_agent) {
        element.field_mask = 0;
        return element;
    }

    const auto md = static_cast<const nixlUcxPublicMetadata *>(desc.metadataP);
    if (!md) {
        throw std::runtime_error("No public metadata found in remote descriptor");
    }

    if (!md->conn) {
        throw std::runtime_error("No connection found in public metadata");
    }

    element.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
        UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR | UCP_DEVICE_MEM_LIST_ELEM_FIELD_EP;
    element.rkey = md->getRkey(worker_id).get();
    element.remote_addr = static_cast<uint64_t>(desc.addr);
    element.ep = md->conn->getEp(worker_id)->getEp();
    return element;
}

ucp_device_mem_list_elem_t
memListElement::create(const nixlMetaDesc &desc) {
    const auto md = static_cast<const nixlUcxPrivateMetadata *>(desc.metadataP);
    if (!md) {
        throw std::runtime_error("No private metadata found in local descriptor");
    }

    ucp_device_mem_list_elem_t element;
    element.field_mask =
        UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH | UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR;
    element.memh = md->getMem().getMemh();
    element.local_addr = md->getMem().getBase();
    return element;
}

memListParams::memListParams(const device_mem_vector_t &elements,
                             const nixlUcxWorker *worker) noexcept {
    params_.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
    params_.elements = elements.data();
    params_.element_size = sizeof(ucp_device_mem_list_elem_t);
    params_.num_elements = elements.size();
    if (worker) {
        params_.field_mask |= UCP_DEVICE_MEM_LIST_PARAMS_FIELD_WORKER;
        params_.worker = worker->get();
    }
}

template<typename T>
[[nodiscard]] device_mem_vector_t
createElements(const T &dlist, size_t worker_id = 0) {
    device_mem_vector_t elements;
    elements.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        elements.emplace_back(memListElement::get(desc, worker_id));
    }
    return elements;
}

void *
createMemList(const nixl_remote_meta_dlist_t &dlist, size_t worker_id, nixlUcxWorker &worker) {
    const device_mem_vector_t elements = createElements(dlist, worker_id);
    const memListParams params{elements};

    ucp_device_remote_mem_list_h handle{nullptr};
    ucs_status_t status;
    while ((status = ucp_device_remote_mem_list_create(params.get(), &handle)) ==
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
createMemList(const nixl_meta_dlist_t &dlist, const nixlUcxWorker &worker) {
    const device_mem_vector_t elements = createElements(dlist);
    const memListParams params{elements, &worker};

    ucp_device_local_mem_list_h handle{nullptr};
    const auto status = ucp_device_local_mem_list_create(params.get(), &handle);
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
namespace {
const std::string error_message{"UCX GPU device API V2 is not supported"};
}

namespace nixl::ucx {
void *
createMemList(const nixl_remote_meta_dlist_t &, size_t, nixlUcxWorker &) {
    throw std::runtime_error(error_message);
}

void *
createMemList(const nixl_meta_dlist_t &, const nixlUcxWorker &) {
    throw std::runtime_error(error_message);
}

void
releaseMemList(void *) noexcept {
    std::terminate();
}
} // namespace nixl::ucx
#endif
