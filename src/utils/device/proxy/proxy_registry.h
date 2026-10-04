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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "backend_aux.h"
#include "device/device_ops.h"
#include "proxy_protocol.h"
#include "proxy_registry_entry.h"

namespace nixl {

/**
 * Creates, owns and releases the proxy's prepared memory views. A view is a host-side entry
 * (its descriptors) plus a device memview that the GPU reads and that carries the entry's
 * address as its token. Entries never move, so a token stays valid until its view is
 * released. Not thread-safe: the caller serializes every call.
 */
class proxyMemViewRegistry {
public:
    proxyMemViewRegistry(deviceOps &allocator, const nixlProxyDeviceContextData *device_context);

    proxyMemViewRegistry(const proxyMemViewRegistry &) = delete;
    proxyMemViewRegistry &
    operator=(const proxyMemViewRegistry &) = delete;

    [[nodiscard]] nixl_status_t
    prepLocal(const nixl_meta_dlist_t &dlist, proxyViewHandle &out);

    [[nodiscard]] nixl_status_t
    prepRemote(const nixl_remote_meta_dlist_t &dlist,
               const std::vector<void *> &direct_ptrs,
               proxyViewHandle &out);

    /** Releases the view after all GPU/CPU operations using it have completed. */
    [[nodiscard]] nixl_status_t
    unregister(proxyViewHandle proxy_memview);

private:
    template<typename DlistT>
    [[nodiscard]] nixl_status_t
    createEntry(const DlistT &dlist, const std::vector<void *> &direct_ptrs, registryEntry *&out);

    template<typename DlistT>
    static void
    fillDescs(const DlistT &dlist, std::vector<registryEntry::storedDesc> &out);

    deviceOps &allocator_;
    const nixlProxyDeviceContextData *device_context_;
    /** Workers never access this ownership map. */
    std::unordered_map<proxyViewHandle, std::unique_ptr<registryEntry>> views_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_H
