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
#include "proxy_registry.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "nixl_log.h"
#include "nixl_types.h"

namespace nixl {

proxyMemViewRegistry::proxyMemViewRegistry(deviceOps &allocator,
                                           const nixlProxyDeviceContextData *device_context)
    : allocator_(allocator),
      device_context_(device_context) {}

template<typename DlistT>
nixl_status_t
proxyMemViewRegistry::createEntry(const DlistT &dlist,
                                  const std::vector<void *> &direct_ptrs,
                                  registryEntry *&out) {
    out = nullptr;

    // CPU: the host-side entry, and the device memview's contents, which name the entry by
    // its address.
    auto entry = std::make_unique<registryEntry>();
    entry->remote = std::is_same_v<DlistT, nixl_remote_meta_dlist_t>;
    fillDescs(dlist, entry->descs);
    const nixlProxyDeviceMemView host_memview{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(entry.get())),
        device_context_,
        static_cast<uint32_t>(direct_ptrs.size())};

    // GPU: allocate the device memview and copy the contents and direct pointers into it.
    deviceMem device_memview_mem;
    if (allocator_.allocDeviceMem(nixlProxyDeviceMemViewBytes(direct_ptrs.size()),
                                  device_memview_mem) != NIXL_SUCCESS) {
        NIXL_ERROR << "proxyMemViewRegistry: failed to allocate device memview";
        return NIXL_ERR_BACKEND;
    }
    auto *device_memview = static_cast<nixlProxyDeviceMemView *>(device_memview_mem.get());
    nixl_status_t copy_status = allocator_.copy(device_memview,
                                                &host_memview,
                                                sizeof(host_memview),
                                                deviceOps::copyDirection::HostToDevice);
    if (copy_status == NIXL_SUCCESS && !direct_ptrs.empty()) {
        copy_status = allocator_.copy(nixlProxyDeviceMemViewDirectPtrs(device_memview),
                                      direct_ptrs.data(),
                                      direct_ptrs.size() * sizeof(void *),
                                      deviceOps::copyDirection::HostToDevice);
    }
    if (copy_status != NIXL_SUCCESS) {
        NIXL_ERROR << "proxyMemViewRegistry: failed to initialize device memview";
        return NIXL_ERR_BACKEND;
    }

    // Link: the entry owns its device memview, and the map owns the entry.
    entry->proxy_memview = device_memview;
    entry->proxy_memview_mem = std::move(device_memview_mem);
    out = entry.get();
    views_.emplace(device_memview, std::move(entry));
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewRegistry::prepLocal(const nixl_meta_dlist_t &dlist, proxyViewHandle &out) {
    registryEntry *entry = nullptr;
    const nixl_status_t status = createEntry(dlist, {}, entry);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = entry->proxy_memview;
    NIXL_DEBUG << "proxyMemViewRegistry::prepLocal: host_view=" << entry
               << " descs=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewRegistry::prepRemote(const nixl_remote_meta_dlist_t &dlist,
                                 const std::vector<void *> &direct_ptrs,
                                 proxyViewHandle &out) {
    if (dlist.getType() != VRAM_SEG) {
        NIXL_ERROR << "proxyMemViewRegistry::prepRemote: unsupported mem type " << dlist.getType();
        return NIXL_ERR_INVALID_PARAM;
    }

    registryEntry *entry = nullptr;
    const nixl_status_t status = createEntry(dlist, direct_ptrs, entry);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = entry->proxy_memview;
    NIXL_DEBUG << "proxyMemViewRegistry::prepRemote: host_view=" << entry
               << " descs=" << dlist.descCount() << " direct_ptrs=" << direct_ptrs.size();
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewRegistry::unregister(proxyViewHandle proxy_memview) {
    const auto it = views_.find(proxy_memview);
    if (it == views_.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    views_.erase(it);
    return NIXL_SUCCESS;
}

template<typename DlistT>
void
proxyMemViewRegistry::fillDescs(const DlistT &dlist, std::vector<registryEntry::storedDesc> &out) {
    out.clear();
    out.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        registryEntry::storedDesc stored{desc};
        if constexpr (std::is_same_v<DlistT, nixl_remote_meta_dlist_t>) {
            // A hole is the null agent's descriptor. One without metadata cannot be posted
            // either, so it is unusable too (the direct path rejects it outright).
            stored.usable = desc.remoteAgent != nixl_null_agent && desc.metadataP != nullptr;
        }
        out.push_back(std::move(stored));
    }
}

} // namespace nixl
