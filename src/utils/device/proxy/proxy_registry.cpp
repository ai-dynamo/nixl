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
proxyMemViewRegistry::createEntryLocked(const DlistT &dlist,
                                        const std::vector<void *> &direct_ptrs,
                                        RegistryEntry *&out) {
    out = nullptr;

    auto entry = std::make_unique<RegistryEntry>();
    entry->remote = std::is_same_v<DlistT, nixl_remote_meta_dlist_t>;
    entry->mem_type = dlist.getType();
    fillDescs(dlist, entry->descs);

    const size_t direct_ptr_bytes = direct_ptrs.size() * sizeof(void *);
    const size_t allocation_size = nixlProxyDeviceMemViewBytes(direct_ptrs.size());

    deviceMem device_memview_mem;
    if (allocator_.allocDeviceMem(allocation_size, device_memview_mem) != NIXL_SUCCESS) {
        NIXL_ERROR << "proxyMemViewRegistry: failed to allocate device memview";
        return NIXL_ERR_BACKEND;
    }
    auto *device_memview = static_cast<nixlProxyDeviceMemView *>(device_memview_mem.get());

    const nixlProxyDeviceMemView host_memview{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(entry.get())),
        device_context_,
        static_cast<uint32_t>(direct_ptrs.size())};
    nixl_status_t copy_status = allocator_.copy(device_memview,
                                                &host_memview,
                                                sizeof(host_memview),
                                                deviceOps::copyDirection::HostToDevice);
    if (copy_status == NIXL_SUCCESS && !direct_ptrs.empty()) {
        copy_status = allocator_.copy(nixlProxyDeviceMemViewDirectPtrs(device_memview),
                                      direct_ptrs.data(),
                                      direct_ptr_bytes,
                                      deviceOps::copyDirection::HostToDevice);
    }
    if (copy_status != NIXL_SUCCESS) {
        NIXL_ERROR << "proxyMemViewRegistry: failed to initialize device memview";
        return NIXL_ERR_BACKEND;
    }

    entry->proxy_memview = device_memview;
    entry->proxy_memview_mem = std::move(device_memview_mem);
    out = entry.get();
    views_.emplace(device_memview, std::move(entry));
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewRegistry::prepLocal(const nixl_meta_dlist_t &dlist, nixlMemViewH &out) {
    const std::lock_guard<std::mutex> lock(ctrl_mutex_);

    RegistryEntry *entry = nullptr;
    const nixl_status_t status = createEntryLocked(dlist, {}, entry);
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
                                 nixlMemViewH &out) {
    if (dlist.getType() != VRAM_SEG) {
        NIXL_ERROR << "proxyMemViewRegistry::prepRemote: unsupported mem type " << dlist.getType();
        return NIXL_ERR_INVALID_PARAM;
    }

    const std::lock_guard<std::mutex> lock(ctrl_mutex_);

    RegistryEntry *entry = nullptr;
    const nixl_status_t status = createEntryLocked(dlist, direct_ptrs, entry);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = entry->proxy_memview;
    NIXL_DEBUG << "proxyMemViewRegistry::prepRemote: host_view=" << entry
               << " descs=" << dlist.descCount() << " direct_ptrs=" << direct_ptrs.size();
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewRegistry::unregister(nixlMemViewH proxy_memview) {
    const std::lock_guard<std::mutex> lock(ctrl_mutex_);

    const auto it = views_.find(proxy_memview);
    if (it == views_.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    views_.erase(it);
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewRegistry::prepareSubmission(const nixlProxySubmission &submission,
                                        proxyBackendSubmission &prepared_submission) const {
    bool needs_source = false;
    size_t transfer_size = 0;
    switch (submission.opcode) {
    case nixl_proxy_opcode_t::PUT:
        needs_source = true;
        transfer_size = submission.size;
        break;
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        transfer_size = sizeof(uint64_t);
        break;
    default:
        NIXL_ERROR << "proxyMemViewRegistry::prepareSubmission: unsupported opcode: "
                   << static_cast<uint32_t>(submission.opcode);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const RegistryEntry *dst_entry = nullptr;
    const StoredDesc *dst_desc = nullptr;
    nixl_status_t status = lookupDesc(submission.dst_view,
                                      submission.dst_index,
                                      submission.dst_offset,
                                      transfer_size,
                                      /*want_remote=*/true,
                                      dst_entry,
                                      dst_desc);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    proxyBackendSubmission prepared{};
    prepared.op_idx = submission.op_idx;
    prepared.opcode = submission.opcode;
    prepared.channel_id = submission.channel_id;
    prepared.flags = submission.flags;
    prepared.size = transfer_size;
    prepared.value = needs_source ? 0 : submission.operand;
    prepared.remote.mem_type = dst_entry->mem_type;
    prepared.remote.desc = dst_desc->desc;
    prepared.remote.desc.addr += submission.dst_offset;
    prepared.remote.desc.len = transfer_size;

    if (needs_source) {
        const RegistryEntry *src_entry = nullptr;
        const StoredDesc *src_desc = nullptr;
        status = lookupDesc(submission.src_view,
                            submission.src_index,
                            submission.operand,
                            transfer_size,
                            /*want_remote=*/false,
                            src_entry,
                            src_desc);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        prepared.local.mem_type = src_entry->mem_type;
        prepared.local.desc = src_desc->desc;
        prepared.local.desc.addr += submission.operand;
        prepared.local.desc.len = transfer_size;
    }

    prepared_submission = prepared;
    return NIXL_SUCCESS;
}

nixl_status_t
proxyMemViewRegistry::lookupDesc(uint64_t host_view,
                                 size_t index,
                                 size_t offset,
                                 size_t size,
                                 bool want_remote,
                                 const RegistryEntry *&entry_out,
                                 const StoredDesc *&desc_out) const {
    entry_out = nullptr;
    desc_out = nullptr;

    const char *const role = want_remote ? "dst" : "src";
    const auto *entry = reinterpret_cast<const RegistryEntry *>(static_cast<uintptr_t>(host_view));
    if (entry == nullptr) {
        NIXL_DEBUG << "proxyMemViewRegistry::prepareSubmission: " << role
                   << " not ready, host_view=" << host_view;
        return NIXL_ERR_NOT_FOUND;
    }
    if (entry->remote != want_remote) {
        NIXL_DEBUG << "proxyMemViewRegistry::prepareSubmission: " << role
                   << " has the wrong role, host_view=" << host_view;
        return NIXL_ERR_INVALID_PARAM;
    }
    if (index >= entry->descs.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const StoredDesc &desc = entry->descs[index];
    if (!desc.usable || !rangeFits(desc.desc, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    entry_out = entry;
    desc_out = &desc;
    return NIXL_SUCCESS;
}

bool
proxyMemViewRegistry::rangeFits(const nixlMetaDesc &desc, size_t offset, size_t size) {
    return offset <= desc.len && size <= desc.len - offset;
}

template<typename DlistT>
void
proxyMemViewRegistry::fillDescs(const DlistT &dlist, std::vector<StoredDesc> &out) {
    out.clear();
    out.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        StoredDesc stored{desc};
        if constexpr (std::is_same_v<DlistT, nixl_remote_meta_dlist_t>) {
            stored.usable = !desc.remoteAgent.empty() && desc.remoteAgent != nixl_null_agent;
        }
        out.push_back(std::move(stored));
    }
}

} // namespace nixl
