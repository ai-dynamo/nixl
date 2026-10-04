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
#include "proxy_memview_manager.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "nixl_log.h"
#include "nixl_types.h"

namespace nixl {

proxyMemViewManager::proxyMemViewManager(deviceOps &allocator,
                                         const nixlProxyDeviceContextData *device_context)
    : allocator_(allocator),
      device_context_(device_context) {}

template<typename DlistT>
nixl_status_t
proxyMemViewManager::createView(const DlistT &dlist,
                                const std::vector<void *> &direct_ptrs,
                                proxyHostView *&out) {
    out = nullptr;

    // CPU: the host view, and the device memview's contents, which name the host view by
    // its address.
    auto host_view = std::make_unique<proxyHostView>();
    host_view->remote = std::is_same_v<DlistT, nixl_remote_meta_dlist_t>;
    fillDescs(dlist, host_view->descs);
    const nixlProxyDeviceMemView memview_contents{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host_view.get())),
        device_context_,
        static_cast<uint32_t>(direct_ptrs.size())};

    // GPU: allocate the device memview and copy the contents and direct pointers into it.
    deviceMem device_memview_mem;
    if (allocator_.allocDeviceMem(nixlProxyDeviceMemViewBytes(direct_ptrs.size()),
                                  device_memview_mem) != NIXL_SUCCESS) {
        NIXL_ERROR << "proxyMemViewManager: failed to allocate device memview";
        return NIXL_ERR_BACKEND;
    }
    auto *device_memview = static_cast<nixlProxyDeviceMemView *>(device_memview_mem.get());
    nixl_status_t copy_status = allocator_.copy(device_memview,
                                                &memview_contents,
                                                sizeof(memview_contents),
                                                deviceOps::copyDirection::HostToDevice);
    if (copy_status == NIXL_SUCCESS && !direct_ptrs.empty()) {
        copy_status = allocator_.copy(nixlProxyDeviceMemViewDirectPtrs(device_memview),
                                      direct_ptrs.data(),
                                      direct_ptrs.size() * sizeof(void *),
                                      deviceOps::copyDirection::HostToDevice);
    }
    if (copy_status != NIXL_SUCCESS) {
        NIXL_ERROR << "proxyMemViewManager: failed to initialize device memview";
        return NIXL_ERR_BACKEND;
    }

    // Link: the host view owns its device memview, and the map owns the host view.
    host_view->proxy_memview = device_memview;
    host_view->proxy_memview_mem = std::move(device_memview_mem);
    out = host_view.get();
    views_.emplace(device_memview, std::move(host_view));
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewManager::prepLocal(const nixl_meta_dlist_t &dlist, proxy_view_handle_t &out) {
    proxyHostView *host_view = nullptr;
    const nixl_status_t status = createView(dlist, {}, host_view);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = host_view->proxy_memview;
    NIXL_DEBUG << "proxyMemViewManager::prepLocal: host_view=" << host_view
               << " descs=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewManager::prepRemote(const nixl_remote_meta_dlist_t &dlist,
                                const std::vector<void *> &direct_ptrs,
                                proxy_view_handle_t &out) {
    if (dlist.getType() != VRAM_SEG) {
        NIXL_ERROR << "proxyMemViewManager::prepRemote: unsupported mem type " << dlist.getType();
        return NIXL_ERR_INVALID_PARAM;
    }

    proxyHostView *host_view = nullptr;
    const nixl_status_t status = createView(dlist, direct_ptrs, host_view);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = host_view->proxy_memview;
    NIXL_DEBUG << "proxyMemViewManager::prepRemote: host_view=" << host_view
               << " descs=" << dlist.descCount() << " direct_ptrs=" << direct_ptrs.size();
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewManager::release(proxy_view_handle_t proxy_memview) {
    const auto it = views_.find(proxy_memview);
    if (it == views_.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    views_.erase(it);
    return NIXL_SUCCESS;
}

template<typename DlistT>
void
proxyMemViewManager::fillDescs(const DlistT &dlist, std::vector<proxyHostView::storedDesc> &out) {
    out.clear();
    out.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        proxyHostView::storedDesc stored{desc};
        if constexpr (std::is_same_v<DlistT, nixl_remote_meta_dlist_t>) {
            // A hole (the null agent's descriptor) has no metadata, and nothing without metadata
            // can be posted, so the metadata alone decides.
            stored.usable = desc.metadataP != nullptr;
        }
        out.push_back(std::move(stored));
    }
}

} // namespace nixl
