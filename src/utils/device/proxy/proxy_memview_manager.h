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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_MEMVIEW_MANAGER_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_MEMVIEW_MANAGER_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "backend_aux.h"
#include "device/device_ops.h"
#include "proxy_protocol.h"
#include "proxy_host_view.h"

namespace nixl {

/**
 * Creates, owns and releases the proxy's prepared memory views. A view is a host view
 * (its descriptors) plus a device memview that the GPU reads and that carries the host
 * view's address as its token. Host views never move, so a token stays valid until its
 * view is released. Not thread-safe: the caller serializes every call.
 */
class proxyMemViewManager {
public:
    proxyMemViewManager(deviceOps &allocator, const nixlProxyDeviceContextData *device_context);

    proxyMemViewManager(const proxyMemViewManager &) = delete;
    proxyMemViewManager &
    operator=(const proxyMemViewManager &) = delete;

    [[nodiscard]] nixl_status_t
    prepLocal(const nixl_meta_dlist_t &dlist, proxy_view_handle_t &out);

    [[nodiscard]] nixl_status_t
    prepRemote(const nixl_remote_meta_dlist_t &dlist,
               const std::vector<void *> &direct_ptrs,
               proxy_view_handle_t &out);

    /** Releases the view after all GPU/CPU operations using it have completed. */
    [[nodiscard]] nixl_status_t
    release(proxy_view_handle_t proxy_memview);

private:
    template<typename DlistT>
    [[nodiscard]] nixl_status_t
    createView(const DlistT &dlist, const std::vector<void *> &direct_ptrs, proxyHostView *&out);

    template<typename DlistT>
    static void
    fillDescs(const DlistT &dlist, std::vector<proxyHostView::storedDesc> &out);

    deviceOps &allocator_;
    const nixlProxyDeviceContextData *device_context_;
    /** Workers never access this ownership map. */
    std::unordered_map<proxy_view_handle_t, std::unique_ptr<proxyHostView>> views_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_MEMVIEW_MANAGER_H
