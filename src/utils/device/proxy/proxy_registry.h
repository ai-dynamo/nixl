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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend_aux.h"
#include "device/device_allocator.h"
#include "proxy_backend_ops.h"
#include "proxy_protocol.h"

/**
 * Maps the memview ids the GPU puts in ring records to the transport metadata
 * the backend needs, and owns the device-resident nixlProxyDeviceMemView
 * behind every proxy memview handle.
 *
 * The id indirection exists because nixlProxySubmission is frozen at 64 bytes
 * and can only carry a uint32 per side - the CPU worker gets an id, never a
 * pointer. That is the only reason this class exists; everything else follows
 * the same shape as the direct path's nixl::ucx::createMemList/releaseMemList.
 *
 * Two paths with different rules:
 *  - control path (prepLocal/prepRemote/unregister/resolve), called by
 *    application threads and serialized by ctrl_mutex_;
 *  - data path (prepareSubmission), called by a proxy worker for every single
 *    ring record. It takes no lock and performs no atomic read-modify-write:
 *    an entry is published by release-storing its pointer into by_id_ and read
 *    with one acquire load, and entries themselves are never freed early.
 */
class nixlProxyMemViewRegistry {
    public:
        /** device_context is stamped into every memview handed to the GPU. */
        nixlProxyMemViewRegistry(nixlDeviceAllocator &allocator,
                                 const nixlProxyDeviceContextData *device_context);

        nixlProxyMemViewRegistry(const nixlProxyMemViewRegistry &) = delete;
        nixlProxyMemViewRegistry &
        operator=(const nixlProxyMemViewRegistry &) = delete;

        /** Register a memview over local descriptors and publish it. */
        [[nodiscard]] nixl_status_t
        prepLocal(const nixl_meta_dlist_t &dlist, nixlMemViewH &out);

        /**
         * Register a memview over remote descriptors and publish it.
         * direct_ptrs, when non-empty, is copied into the device memview for
         * the GPU's direct-access fast path.
         */
        [[nodiscard]] nixl_status_t
        prepRemote(const nixl_remote_meta_dlist_t &dlist,
                   const std::vector<void *> &direct_ptrs,
                   nixlMemViewH &out);

        /**
         * Retire a memview and free its device allocation, matching what
         * ucp_device_mem_list_release() does on the direct path: the caller is
         * expected to have quiesced the GPU first. The host-side entry stays
         * alive - a worker may still hold it - but no new submission can reach
         * it, and the worker never touches the device allocation.
         */
        [[nodiscard]] nixl_status_t
        unregister(nixlMemViewH proxy_memview);

        [[nodiscard]] bool
        resolve(nixlMemViewH proxy_memview, nixlMemViewH &backend_out) const;

        /** Data path: resolve a ring record into a transport-ready submission. */
        [[nodiscard]] nixl_status_t
        prepareSubmission(const nixlProxySubmission &submission,
                          nixlBackendProxySubmission &prepared_submission) const;

    private:
        /** One descriptor of a registered dlist; remote_agent is empty if local. */
        struct StoredDesc {
            uintptr_t base_addr = 0;
            size_t len = 0;
            uint64_t dev_id = 0;
            nixlBackendMD *metadata = nullptr;
            std::string remote_agent;
        };

        struct RegistryEntry {
            uint32_t id = 0;
            nixlMemViewH proxy_memview = nullptr;
            /** Owns the device-resident nixlProxyDeviceMemView; freed by unregister(). */
            nixlDeviceMem proxy_memview_mem;
            nixlMemViewH backend_memview = nullptr;
            /** Remote entries are transfer destinations, local ones sources. */
            bool remote = false;
            nixl_mem_t mem_type = DRAM_SEG;
            std::vector<StoredDesc> descs;
        };

        /**
         * Ids are never reused, so this bounds the registrations one runtime
         * can make over its lifetime, not the number live at once. Real
         * workloads register a handful per generation.
         */
        static constexpr uint32_t kMaxProxyMemViews = 4096;

        [[nodiscard]] const RegistryEntry *
        entryForId(uint64_t proxy_memview_id) const noexcept;

        [[nodiscard]] nixl_status_t
        createEntryLocked(const std::vector<void *> &direct_ptrs, RegistryEntry *&out);

        /** Resolve one side of a submission; want_remote selects the role. */
        [[nodiscard]] nixl_status_t
        lookupDesc(uint64_t proxy_memview_id,
                   size_t index,
                   size_t offset,
                   size_t size,
                   bool want_remote,
                   const RegistryEntry *&entry_out,
                   const StoredDesc *&desc_out) const;

        static bool
        rangeFits(const StoredDesc &desc, size_t offset, size_t size);

        template<typename DlistT>
        static void
        fillDescs(const DlistT &dlist, std::vector<StoredDesc> &out);

        nixlDeviceAllocator &allocator_;
        const nixlProxyDeviceContextData *device_context_;
        /**
         * Data-path view: id-1 -> the live entry, or null once retired. Sized
         * once here and never resized, which is what makes an unlocked read of
         * it sound (a growing container rewrites its own bookkeeping - that is
         * repo docs/issues/006).
         */
        std::vector<std::atomic<RegistryEntry *>> by_id_;

        mutable std::mutex ctrl_mutex_;
        /** Every entry ever created, in id order; its size is the id counter. */
        std::vector<std::unique_ptr<RegistryEntry>> owned_;
        /** Live handles only; unregister() erases, so a stale handle misses. */
        std::unordered_map<nixlMemViewH, uint32_t> handle_to_id_;
};

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_REGISTRY_H
