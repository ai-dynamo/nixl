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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "backend_aux.h"
#include "device/device_allocator.h"
#include "proxy_backend_ops.h"
#include "proxy_protocol.h"

/** Owns views until retirement; workers borrow live tokens without accessing the map. */
class nixlProxyMemViewRegistry {
public:
    nixlProxyMemViewRegistry(nixlDeviceAllocator &allocator,
                             const nixlProxyDeviceContextData *device_context);

    nixlProxyMemViewRegistry(const nixlProxyMemViewRegistry &) = delete;
    nixlProxyMemViewRegistry &
    operator=(const nixlProxyMemViewRegistry &) = delete;

    [[nodiscard]] nixl_status_t
    prepLocal(const nixl_meta_dlist_t &dlist, nixlMemViewH &out);

    [[nodiscard]] nixl_status_t
    prepRemote(const nixl_remote_meta_dlist_t &dlist,
               const std::vector<void *> &direct_ptrs,
               nixlMemViewH &out);

    /** Releases the view after all GPU/CPU operations using it have completed. */
    [[nodiscard]] nixl_status_t
    unregister(nixlMemViewH proxy_memview);

    [[nodiscard]] nixl_status_t
    prepareSubmission(const nixlProxySubmission &submission,
                      nixlBackendProxySubmission &prepared_submission) const;

private:
    struct StoredDesc {
        /** metadataP is borrowed and must outlive submissions. */
        nixlMetaDesc desc;
        bool usable = true;
    };

    struct RegistryEntry {
        nixlMemViewH proxy_memview = nullptr;
        nixlDeviceMem proxy_memview_mem;
        bool remote = false;
        nixl_mem_t mem_type = DRAM_SEG;
        std::vector<StoredDesc> descs;
    };

    template<typename DlistT>
    [[nodiscard]] nixl_status_t
    createEntryLocked(const DlistT &dlist,
                      const std::vector<void *> &direct_ptrs,
                      RegistryEntry *&out);

    [[nodiscard]] nixl_status_t
    lookupDesc(uint64_t host_view,
               size_t index,
               size_t offset,
               size_t size,
               bool want_remote,
               const RegistryEntry *&entry_out,
               const StoredDesc *&desc_out) const;

    static bool
    rangeFits(const nixlMetaDesc &desc, size_t offset, size_t size);

    template<typename DlistT>
    static void
    fillDescs(const DlistT &dlist, std::vector<StoredDesc> &out);

    nixlDeviceAllocator &allocator_;
    const nixlProxyDeviceContextData *device_context_;
    mutable std::mutex ctrl_mutex_;
    /** Workers never access this ownership map. */
    std::unordered_map<nixlMemViewH, std::unique_ptr<RegistryEntry>> views_;
};

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_H
