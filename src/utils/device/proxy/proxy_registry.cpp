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

nixlProxyMemViewRegistry::nixlProxyMemViewRegistry(nixlDeviceAllocator &allocator,
                                                   const nixlProxyDeviceContextData *device_context)
    : allocator_(allocator),
      device_context_(device_context),
      by_id_(kMaxProxyMemViews) {}

const nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::entryForId(uint64_t proxy_memview_id) const noexcept {
    if (proxy_memview_id == 0 || proxy_memview_id > by_id_.size()) {
        return nullptr;
    }
    return by_id_[proxy_memview_id - 1].load(std::memory_order_acquire);
}

nixl_status_t
nixlProxyMemViewRegistry::createEntryLocked(const std::vector<void *> &direct_ptrs,
                                            RegistryEntry *&out) {
    out = nullptr;

    if (owned_.size() >= by_id_.size()) {
        NIXL_ERROR << "nixlProxyMemViewRegistry: memview capacity exhausted after " << by_id_.size()
                   << " registrations";
        return NIXL_ERR_BACKEND;
    }

    auto entry = std::make_unique<RegistryEntry>();
    entry->id = static_cast<uint32_t>(owned_.size() + 1);

    const size_t direct_ptr_bytes = direct_ptrs.size() * sizeof(void *);
    const size_t allocation_size = nixlProxyDeviceMemViewBytes(direct_ptrs.size());

    nixlDeviceMem device_memview_mem;
    if (allocator_.allocDeviceMem(allocation_size, device_memview_mem) != NIXL_SUCCESS) {
        NIXL_ERROR << "nixlProxyMemViewRegistry: failed to allocate device memview";
        return NIXL_ERR_BACKEND;
    }
    auto *device_memview = device_memview_mem.as<nixlProxyDeviceMemView>();

    const nixlProxyDeviceMemView host_memview{
        entry->id, static_cast<uint32_t>(direct_ptrs.size()), device_context_};
    nixl_status_t copy_status =
        allocator_.copyHostToDevice(device_memview, &host_memview, sizeof(host_memview));
    if (copy_status == NIXL_SUCCESS && !direct_ptrs.empty()) {
        copy_status = allocator_.copyHostToDevice(nixlProxyDeviceMemViewDirectPtrs(device_memview),
                                                  direct_ptrs.data(),
                                                  direct_ptr_bytes);
    }
    if (copy_status != NIXL_SUCCESS) {
        NIXL_ERROR << "nixlProxyMemViewRegistry: failed to initialize device memview";
        return NIXL_ERR_BACKEND;
    }

    entry->proxy_memview = device_memview;
    entry->proxy_memview_mem = std::move(device_memview_mem);
    handle_to_id_.emplace(entry->proxy_memview, entry->id);

    // Nothing before this point consumed an id, so a failed prep leaves the
    // registry exactly as it was.
    out = entry.get();
    owned_.push_back(std::move(entry));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepLocal(const nixl_meta_dlist_t &dlist, nixlMemViewH &out) {
    const std::lock_guard<std::mutex> lock(ctrl_mutex_);

    RegistryEntry *entry = nullptr;
    const nixl_status_t status = createEntryLocked({}, entry);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    entry->remote = false;
    entry->mem_type = dlist.getType();
    fillDescs(dlist, entry->descs);
    // Publishes the entry: everything above happens-before the acquire load in
    // entryForId().
    by_id_[entry->id - 1].store(entry, std::memory_order_release);

    out = entry->proxy_memview;
    NIXL_DEBUG << "nixlProxyMemViewRegistry::prepLocal: proxy_id=" << entry->id
               << " descs=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepRemote(const nixl_remote_meta_dlist_t &dlist,
                                     const std::vector<void *> &direct_ptrs,
                                     nixlMemViewH &out) {
    // Validated before anything is allocated, so a rejected prep leaves no
    // entry to roll back.
    if (dlist.getType() != VRAM_SEG) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::prepRemote: unsupported mem type "
                   << dlist.getType();
        return NIXL_ERR_INVALID_PARAM;
    }

    const std::lock_guard<std::mutex> lock(ctrl_mutex_);

    RegistryEntry *entry = nullptr;
    const nixl_status_t status = createEntryLocked(direct_ptrs, entry);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    entry->remote = true;
    entry->mem_type = dlist.getType();
    fillDescs(dlist, entry->descs);
    by_id_[entry->id - 1].store(entry, std::memory_order_release);

    out = entry->proxy_memview;
    NIXL_DEBUG << "nixlProxyMemViewRegistry::prepRemote: proxy_id=" << entry->id
               << " descs=" << dlist.descCount() << " direct_ptrs=" << direct_ptrs.size();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::unregister(nixlMemViewH proxy_memview) {
    const std::lock_guard<std::mutex> lock(ctrl_mutex_);

    const auto it = handle_to_id_.find(proxy_memview);
    if (it == handle_to_id_.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    RegistryEntry *entry = owned_[it->second - 1].get();
    handle_to_id_.erase(it);

    // Retire first: a worker that already resolved this id still holds a valid
    // entry, but no new submission can reach it.
    by_id_[entry->id - 1].store(nullptr, std::memory_order_release);

    // Then free the device allocation. Only the GPU ever reads it, and the
    // caller is expected to have quiesced the GPU - the same contract
    // ucp_device_mem_list_release() relies on for the direct path.
    entry->proxy_memview_mem.reset();
    entry->proxy_memview = nullptr;

    NIXL_DEBUG << "nixlProxyMemViewRegistry::unregister: proxy_id=" << entry->id;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::resolve(nixlMemViewH proxy_memview, nixlMemViewH &backend_out) const {
    const std::lock_guard<std::mutex> lock(ctrl_mutex_);

    // Only live handles are in the map; unregister() erases them.
    const auto it = handle_to_id_.find(proxy_memview);
    if (it == handle_to_id_.end()) {
        return false;
    }

    backend_out = owned_[it->second - 1]->backend_memview;
    return true;
}

nixl_status_t
nixlProxyMemViewRegistry::prepareSubmission(const nixlProxySubmission &submission,
                                            nixlBackendProxySubmission &prepared_submission) const {
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
        NIXL_ERROR << "nixlProxyMemViewRegistry::prepareSubmission: unsupported opcode: "
                   << static_cast<uint32_t>(submission.opcode);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const RegistryEntry *dst_entry = nullptr;
    const StoredDesc *dst_desc = nullptr;
    nixl_status_t status = lookupDesc(submission.dst_proxy_memview_id,
                                      submission.dst_index,
                                      submission.dst_offset,
                                      transfer_size,
                                      /*want_remote=*/true,
                                      dst_entry,
                                      dst_desc);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    nixlBackendProxySubmission prepared{};
    prepared.op_idx = submission.op_idx;
    prepared.opcode = submission.opcode;
    prepared.channel_id = submission.channel_id;
    prepared.flags = submission.flags;
    prepared.size = transfer_size;
    prepared.value = submission.value;
    prepared.remote_agent = dst_desc->remote_agent;
    prepared.remote.mem_type = dst_entry->mem_type;
    prepared.remote.desc = nixlMetaDesc(dst_desc->base_addr + submission.dst_offset,
                                        transfer_size,
                                        dst_desc->dev_id,
                                        dst_desc->metadata);

    if (needs_source) {
        const RegistryEntry *src_entry = nullptr;
        const StoredDesc *src_desc = nullptr;
        status = lookupDesc(submission.src_proxy_memview_id,
                            submission.src_index,
                            submission.src_offset,
                            transfer_size,
                            /*want_remote=*/false,
                            src_entry,
                            src_desc);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        prepared.local.mem_type = src_entry->mem_type;
        prepared.local.desc = nixlMetaDesc(src_desc->base_addr + submission.src_offset,
                                           transfer_size,
                                           src_desc->dev_id,
                                           src_desc->metadata);
    }

    prepared_submission = prepared;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::lookupDesc(uint64_t proxy_memview_id,
                                     size_t index,
                                     size_t offset,
                                     size_t size,
                                     bool want_remote,
                                     const RegistryEntry *&entry_out,
                                     const StoredDesc *&desc_out) const {
    entry_out = nullptr;
    desc_out = nullptr;

    const char *const role = want_remote ? "dst" : "src";
    const RegistryEntry *entry = entryForId(proxy_memview_id);
    if (entry == nullptr) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: " << role
                   << " not ready, proxy_id=" << proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (entry->remote != want_remote) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: " << role
                   << " has the wrong role, proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }
    if (index >= entry->descs.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const StoredDesc &desc = entry->descs[index];
    if (!rangeFits(desc, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (want_remote && (desc.remote_agent.empty() || desc.remote_agent == nixl_null_agent)) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst remote agent invalid"
                   << " proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    entry_out = entry;
    desc_out = &desc;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::rangeFits(const StoredDesc &desc, size_t offset, size_t size) {
    return offset <= desc.len && size <= desc.len - offset;
}

template<typename DlistT>
void
nixlProxyMemViewRegistry::fillDescs(const DlistT &dlist, std::vector<StoredDesc> &out) {
    out.clear();
    out.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        StoredDesc stored;
        stored.base_addr = desc.addr;
        stored.len = desc.len;
        stored.dev_id = desc.devId;
        stored.metadata = desc.metadataP;
        if constexpr (std::is_same_v<DlistT, nixl_remote_meta_dlist_t>) {
            stored.remote_agent = desc.remoteAgent;
        }
        out.push_back(std::move(stored));
    }
}
