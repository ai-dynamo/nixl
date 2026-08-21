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
#include "proxy_runtime.h"
#include "backend_adapter.h"
#include "nixl_types.h"
#include "proxy_worker.h"
#include "nixl_log.h"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <cuda_runtime.h>

nixlProxyMemViewRegistry::~nixlProxyMemViewRegistry() {
    clear();
}

nixl_status_t
nixlProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                               nixlMemViewH *proxy_memview) {
    return registerProxyMemView(backend_memview, {}, proxy_memview);
}

nixl_status_t
nixlProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                               const std::vector<void *> &direct_ptrs,
                                               nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    registryEntry entry;
    entry.proxyMemViewId = static_cast<uint32_t>(next_proxy_memview_id_);
    entry.backendMemView = backend_memview;

    const size_t direct_ptr_bytes = direct_ptrs.size() * sizeof(void *);
    const size_t allocation_size = sizeof(nixlProxyDeviceMemView) + direct_ptr_bytes;

    nixlProxyDeviceMemView *device_memview = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_memview), allocation_size) != cudaSuccess) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::register: failed to allocate device memview";
        return NIXL_ERR_BACKEND;
    }

    const nixlProxyDeviceMemView host_memview{
        entry.proxyMemViewId, static_cast<uint32_t>(direct_ptrs.size()), device_context_};
    cudaError_t cuda_status =
        cudaMemcpy(device_memview, &host_memview, sizeof(host_memview), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess && !direct_ptrs.empty()) {
        cuda_status = cudaMemcpy(device_memview->directPtrs,
                                 direct_ptrs.data(),
                                 direct_ptr_bytes,
                                 cudaMemcpyHostToDevice);
    }
    if (cuda_status != cudaSuccess) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::register: failed to initialize device memview";
        cudaFree(device_memview);
        return NIXL_ERR_BACKEND;
    }

    entry.proxyMemView = device_memview;
    entries_.push_back(entry);
    handle_to_id_.emplace(entry.proxyMemView, entry.proxyMemViewId);

    *proxy_memview = entry.proxyMemView;
    ++next_proxy_memview_id_;
    NIXL_DEBUG << "nixlProxyMemViewRegistry::register: backend_mvh=" << backend_memview
               << " -> proxy_id=" << (next_proxy_memview_id_ - 1);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    return prepMemView(nullptr, dlist, proxy_memview);
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                                      nixlMemViewH *proxy_memview) {
    return prepMemView(dlist, {}, proxy_memview);
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                                      const std::vector<void *> &direct_ptrs,
                                      nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlMemViewH registered_proxy_memview = nullptr;
    nixl_status_t status = registerProxyMemView(nullptr, direct_ptrs, &registered_proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = storeMetadata(registered_proxy_memview, dlist);
    if (status != NIXL_SUCCESS) {
        unregisterProxyMemView(registered_proxy_memview);
        return status;
    }

    *proxy_memview = registered_proxy_memview;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(nixlMemViewH backend_memview,
                                      const nixl_meta_dlist_t &dlist,
                                      nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlMemViewH registered_proxy_memview = nullptr;
    nixl_status_t status = registerProxyMemView(backend_memview, &registered_proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = storeMetadata(registered_proxy_memview, dlist);
    if (status != NIXL_SUCCESS) {
        unregisterProxyMemView(registered_proxy_memview);
        return status;
    }

    *proxy_memview = registered_proxy_memview;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(nixlMemViewH backend_memview,
                                      const nixl_remote_meta_dlist_t &dlist,
                                      nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlMemViewH registered_proxy_memview = nullptr;
    nixl_status_t status = registerProxyMemView(backend_memview, &registered_proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = storeMetadata(registered_proxy_memview, dlist);
    if (status != NIXL_SUCCESS) {
        unregisterProxyMemView(registered_proxy_memview);
        return status;
    }

    *proxy_memview = registered_proxy_memview;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    registryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    entry->state = proxy_memview_reg_entry_state_t::ENTRY_RETIRED;
    handle_to_id_.erase(proxy_memview);
    releaseDeviceMemView(*entry);
    NIXL_DEBUG << "nixlProxyMemViewRegistry::unregister: proxy_id=" << entry->proxyMemViewId;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::resolveProxyMemView(nixlMemViewH proxy_memview,
                                              nixlMemViewH &backend_memview) const {
    const registryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == proxy_memview_reg_entry_state_t::ENTRY_RETIRED) {
        return false;
    }
    backend_memview = entry->backendMemView;
    return true;
}

bool
nixlProxyMemViewRegistry::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                                nixlMemViewH &backend_memview) const {
    const registryEntry *entry = getEntryForId(proxy_memview_id);
    if (entry == nullptr || entry->state == proxy_memview_reg_entry_state_t::ENTRY_RETIRED) {
        return false;
    }
    backend_memview = entry->backendMemView;
    return true;
}

nixl_status_t
nixlProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                        const nixl_meta_dlist_t &dlist) {
    registryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == proxy_memview_reg_entry_state_t::ENTRY_RETIRED) {
        return NIXL_ERR_NOT_FOUND;
    }

    fillLocalMetadata(dlist, entry->localMetadata);
    entry->remoteMetadata = remoteMetadataInfo{};
    entry->metadataKind = proxy_memview_reg_metadata_kind_t::METADATA_KIND_LOCAL;
    entry->state = proxy_memview_reg_entry_state_t::ENTRY_READY;

    NIXL_DEBUG << "nixlProxyMemViewRegistry::storeMetadata(local): proxy_id="
               << entry->proxyMemViewId << " entries=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                        const nixl_remote_meta_dlist_t &dlist) {
    registryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == proxy_memview_reg_entry_state_t::ENTRY_RETIRED) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (dlist.getType() != VRAM_SEG) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::storeMetadata(remote): unsupported mem type "
                   << dlist.getType();
        return NIXL_ERR_INVALID_PARAM;
    }

    fillRemoteMetadata(dlist, entry->remoteMetadata);
    entry->localMetadata = localMetadataInfo{};
    entry->metadataKind = proxy_memview_reg_metadata_kind_t::METADATA_KIND_REMOTE;
    entry->state = proxy_memview_reg_entry_state_t::ENTRY_READY;

    NIXL_DEBUG << "nixlProxyMemViewRegistry::storeMetadata(remote): proxy_id="
               << entry->proxyMemViewId << " entries=" << dlist.descCount();
    return NIXL_SUCCESS;
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

    const proxyMemViewRegStoredEntry *dst_metadata = nullptr;
    nixl_status_t status = getRemoteEntryForSubmission(submission.dstProxyMemViewId,
                                                       submission.dstIndex,
                                                       submission.dstOffset,
                                                       transfer_size,
                                                       dst_metadata);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    nixlBackendProxySubmission prepared{};
    prepared.opIdx = submission.opIdx;
    prepared.opcode = submission.opcode;
    prepared.channelId = submission.channelId;
    prepared.flags = submission.flags;
    prepared.size = transfer_size;
    prepared.value = submission.value;
    prepared.remoteAgent = dst_metadata->remoteAgent;
    prepared.remote.memType = VRAM_SEG;
    prepared.remote.desc = nixlMetaDesc(dst_metadata->baseAddr + submission.dstOffset,
                                        transfer_size,
                                        dst_metadata->devId,
                                        dst_metadata->metadata);

    if (needs_source) {
        const localMetadataInfo *local_metadata = nullptr;
        const proxyMemViewRegStoredEntry *src_metadata = nullptr;
        status = getLocalEntryForSubmission(submission.srcProxyMemViewId,
                                            submission.srcIndex,
                                            submission.srcOffset,
                                            transfer_size,
                                            local_metadata,
                                            src_metadata);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        prepared.local.memType = local_metadata->memType;
        prepared.local.desc = nixlMetaDesc(src_metadata->baseAddr + submission.srcOffset,
                                           transfer_size,
                                           src_metadata->devId,
                                           src_metadata->metadata);
    }

    prepared_submission = prepared;
    return NIXL_SUCCESS;
}

void
nixlProxyMemViewRegistry::clear() noexcept {
    for (auto &entry : entries_) {
        entry.state = proxy_memview_reg_entry_state_t::ENTRY_RETIRED;
        releaseDeviceMemView(entry);
    }
    handle_to_id_.clear();
}

void
nixlProxyMemViewRegistry::releaseDeviceMemView(registryEntry &entry) noexcept {
    if (entry.proxyMemView != nullptr) {
        cudaFree(entry.proxyMemView);
        entry.proxyMemView = nullptr;
    }
}

nixlProxyMemViewRegistry::registryEntry *
nixlProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) {
    const auto it = handle_to_id_.find(proxy_memview);
    return it == handle_to_id_.end() ? nullptr : getEntryForId(it->second);
}

const nixlProxyMemViewRegistry::registryEntry *
nixlProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) const {
    const auto it = handle_to_id_.find(proxy_memview);
    return it == handle_to_id_.end() ? nullptr : getEntryForId(it->second);
}

nixlProxyMemViewRegistry::registryEntry *
nixlProxyMemViewRegistry::getEntryForId(uint64_t proxy_memview_id) {
    if (proxy_memview_id < 1 || proxy_memview_id >= next_proxy_memview_id_ ||
        proxy_memview_id > entries_.size()) {
        return nullptr;
    }
    return &entries_[proxy_memview_id - 1];
}

const nixlProxyMemViewRegistry::registryEntry *
nixlProxyMemViewRegistry::getEntryForId(uint64_t proxy_memview_id) const {
    if (proxy_memview_id < 1 || proxy_memview_id >= next_proxy_memview_id_ ||
        proxy_memview_id > entries_.size()) {
        return nullptr;
    }
    return &entries_[proxy_memview_id - 1];
}

nixl_status_t
nixlProxyMemViewRegistry::getRemoteEntryForSubmission(
    uint64_t proxy_memview_id,
    size_t index,
    size_t offset,
    size_t size,
    const proxyMemViewRegStoredEntry *&entry) const {
    entry = nullptr;

    const registryEntry *registry_entry = getEntryForId(proxy_memview_id);
    if (registry_entry == nullptr ||
        registry_entry->state != proxy_memview_reg_entry_state_t::ENTRY_READY) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst not ready"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (registry_entry->metadataKind != proxy_memview_reg_metadata_kind_t::METADATA_KIND_REMOTE) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst metadata kind invalid"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &remote_metadata = registry_entry->remoteMetadata;
    if (index >= remote_metadata.entries.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const proxyMemViewRegStoredEntry &remote_entry = remote_metadata.entries[index];
    if (!rangeFits(remote_entry, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_entry.remoteAgent.empty() || remote_entry.remoteAgent == nixl_null_agent) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst remote agent invalid"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    entry = &remote_entry;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::getLocalEntryForSubmission(
    uint64_t proxy_memview_id,
    size_t index,
    size_t offset,
    size_t size,
    const localMetadataInfo *&metadata,
    const proxyMemViewRegStoredEntry *&entry) const {
    metadata = nullptr;
    entry = nullptr;

    const registryEntry *registry_entry = getEntryForId(proxy_memview_id);
    if (registry_entry == nullptr ||
        registry_entry->state != proxy_memview_reg_entry_state_t::ENTRY_READY) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: src not ready"
                   << " src_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (registry_entry->metadataKind != proxy_memview_reg_metadata_kind_t::METADATA_KIND_LOCAL) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: src metadata kind invalid"
                   << " src_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &local_metadata = registry_entry->localMetadata;
    if (index >= local_metadata.entries.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const proxyMemViewRegStoredEntry &local_entry = local_metadata.entries[index];
    if (!rangeFits(local_entry, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    metadata = &local_metadata;
    entry = &local_entry;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::rangeFits(const proxyMemViewRegStoredEntry &entry,
                                    size_t offset,
                                    size_t size) {
    return offset <= entry.len && size <= entry.len - offset;
}

void
nixlProxyMemViewRegistry::fillLocalMetadata(const nixl_meta_dlist_t &dlist,
                                            localMetadataInfo &out) {
    out = localMetadataInfo{};
    out.memType = dlist.getType();
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        out.entries.push_back(
            proxyMemViewRegStoredEntry{desc.addr, desc.len, desc.devId, desc.metadataP});
    }
}

void
nixlProxyMemViewRegistry::fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist,
                                             remoteMetadataInfo &out) {
    out = remoteMetadataInfo{};
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        out.entries.push_back(proxyMemViewRegStoredEntry{
            desc.addr, desc.len, desc.devId, desc.metadataP, desc.remoteAgent});
    }
}

nixl_status_t
nixlProxyChannelState::allocate(uint32_t depth,
                                nixlProxyControlBuffer *control_slots,
                                size_t control_slot_index) {
    NIXL_INFO << "nixlProxyChannelState::allocate: depth=" << depth
              << " control_slot_index=" << control_slot_index;
    if (depth == 0 || control_slots == nullptr ||
        control_slots->devicePtr(control_slot_index) == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    ringDepth = depth;
    controlSlots = control_slots;
    controlSlotIndex = control_slot_index;
    consumerIdxDev = controlSlots->devicePtr(controlSlotIndex);
    consumerIdxShadow = 0;

    if (cudaMalloc(reinterpret_cast<void **>(&workRingDev), sizeof(nixlProxyWorkRing)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&producerIdxDev), sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&consumerIdxCacheDev), sizeof(uint64_t)) !=
            cudaSuccess ||
        cudaMallocHost(&recordsHost, sizeof(nixlProxySubmission) * depth) != cudaSuccess ||
        cudaMallocHost(&completionSlotHost, sizeof(nixlProxyCompletionSlot)) != cudaSuccess) {
        NIXL_ERROR << "nixlProxyChannelState::allocate: CUDA allocation failed";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    void *records_dev = nullptr;
    if (cudaHostGetDevicePointer(&records_dev, recordsHost, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    auto *records_dev_ptr = static_cast<nixlProxySubmission *>(records_dev);

    void *completion_dev = nullptr;
    if (cudaHostGetDevicePointer(&completion_dev, completionSlotHost, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    completionSlotDev = static_cast<nixlProxyCompletionSlot *>(completion_dev);

    for (uint32_t i = 0; i < depth; ++i) {
        recordsHost[i] = nixlProxySubmission{};
    }
    if (cudaMemset(producerIdxDev, 0, sizeof(*producerIdxDev)) != cudaSuccess ||
        cudaMemset(consumerIdxCacheDev, 0, sizeof(*consumerIdxCacheDev)) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    if (publishConsumerIdx(0) != NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    submitIdx = 0;
    completionSlotHost->nextStatus = NIXL_IN_PROG;
    __atomic_store_n(&completionSlotHost->completedIdx, uint64_t{0}, __ATOMIC_RELEASE);
    nixlProxyWorkRing work_ring{
        records_dev_ptr,
        producerIdxDev,
        consumerIdxDev,
        consumerIdxCacheDev,
        depth,
    };
    if (cudaMemcpy(workRingDev, &work_ring, sizeof(work_ring), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    deviceView = nixlProxyChannelView{workRingDev, completionSlotDev};

    inflightSlots.assign(depth, nixlProxyRequestState{});
    NIXL_INFO << "nixlProxyChannelState::allocate: ready"
              << " work_ring(dev)=" << workRingDev << " records=" << recordsHost
              << " records(dev)=" << records_dev_ptr << " producer_idx(dev)=" << producerIdxDev
              << " consumer_idx(shadow)=" << consumerIdxShadow
              << " consumer_idx(dev)=" << consumerIdxDev
              << " consumer_idx_cache(dev)=" << consumerIdxCacheDev
              << " completion_slot(host)=" << completionSlotHost
              << " completion_slot(dev)=" << completionSlotDev;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyChannelState::publishConsumerIdx(uint64_t value) noexcept {
    if (controlSlots == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const nixl_status_t status = controlSlots->writeSlot(controlSlotIndex, value);
    if (status == NIXL_SUCCESS) {
        consumerIdxShadow = value;
    }
    return status;
}

void
nixlProxyChannelState::deallocate() noexcept {
    if (completionSlotHost) {
        cudaFreeHost(completionSlotHost);
        completionSlotHost = nullptr;
        completionSlotDev = nullptr;
    }
    if (producerIdxDev) {
        cudaFree(producerIdxDev);
        producerIdxDev = nullptr;
    }
    if (consumerIdxCacheDev) {
        cudaFree(consumerIdxCacheDev);
        consumerIdxCacheDev = nullptr;
    }
    consumerIdxDev = nullptr;
    controlSlots = nullptr;
    controlSlotIndex = 0;
    consumerIdxShadow = 0;
    if (recordsHost) {
        cudaFreeHost(recordsHost);
        recordsHost = nullptr;
    }
    if (workRingDev) {
        cudaFree(workRingDev);
        workRingDev = nullptr;
    }
    inflightSlots.clear();
    submitIdx = 0;
    ringDepth = 0;
    deviceView = nixlProxyChannelView{};
}

nixlProxyChannelState::~nixlProxyChannelState() {
    deallocate();
}

nixlProxyChannelState::nixlProxyChannelState(nixlProxyChannelState &&other) noexcept {
    *this = std::move(other);
}

nixlProxyChannelState &
nixlProxyChannelState::operator=(nixlProxyChannelState &&other) noexcept {
    if (this != &other) {
        deallocate();
        deviceView = other.deviceView;
        inflightSlots = std::move(other.inflightSlots);
        submitIdx = other.submitIdx;
        workRingDev = other.workRingDev;
        recordsHost = other.recordsHost;
        producerIdxDev = other.producerIdxDev;
        consumerIdxDev = other.consumerIdxDev;
        consumerIdxCacheDev = other.consumerIdxCacheDev;
        controlSlots = other.controlSlots;
        controlSlotIndex = other.controlSlotIndex;
        consumerIdxShadow = other.consumerIdxShadow;
        ringDepth = other.ringDepth;
        completionSlotHost = other.completionSlotHost;
        completionSlotDev = other.completionSlotDev;
        other.workRingDev = nullptr;
        other.recordsHost = nullptr;
        other.producerIdxDev = nullptr;
        other.consumerIdxDev = nullptr;
        other.consumerIdxCacheDev = nullptr;
        other.controlSlots = nullptr;
        other.controlSlotIndex = 0;
        other.consumerIdxShadow = 0;
        other.ringDepth = 0;
        other.submitIdx = 0;
        other.completionSlotHost = nullptr;
        other.completionSlotDev = nullptr;
        other.deviceView = nixlProxyChannelView{};
    }
    return *this;
}

nixlProxyRuntime::nixlProxyRuntime() = default;

nixlProxyRuntime::~nixlProxyRuntime() {
    if (backend_) {
        shutdown();
    }
}

nixl_status_t
nixlProxyRuntime::init(std::unique_ptr<nixlDeviceProxyBackendAdapter> backend,
                       uint32_t max_peers,
                       uint32_t channel_count,
                       uint32_t worker_count,
                       uint64_t pthr_delay_us) {
    NIXL_INFO << "ProxyRuntime::init: max_peers=" << max_peers << " channel_count=" << channel_count
              << " worker_count=" << worker_count << " pthr_delay_us=" << pthr_delay_us
              << " backend=" << backend.get();
    if (backend == nullptr || max_peers == 0 || channel_count == 0 || worker_count == 0) {
        NIXL_ERROR << "ProxyRuntime::init: invalid params";
        return NIXL_ERR_INVALID_PARAM;
    }

    backend_ = std::move(backend);
    memview_registry_.clear();

    const uint32_t effective_worker_count = std::min(worker_count, channel_count);
    NIXL_INFO << "ProxyRuntime::init: effective worker_count=" << effective_worker_count
              << " (clamped to channel_count)";

    nixl_status_t rc = backend_->init(effective_worker_count, channel_count, max_peers);
    if ((rc != NIXL_SUCCESS) && (rc != NIXL_ERR_NOT_SUPPORTED)) {
        NIXL_ERROR << "ProxyRuntime::init: backend init failed: " << rc;
        backend_.reset();
        return rc;
    }
    if (rc == NIXL_ERR_NOT_SUPPORTED) {
        NIXL_INFO << "ProxyRuntime::init: backend init hook not supported; continuing";
    }

    const size_t channel_slots = static_cast<size_t>(max_peers) * channel_count;
    rc = control_slots_.allocate(proxy_ci_slot_base + channel_slots);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::init: failed to create GPU-visible control slab";
        shutdown();
        return rc;
    }
    shutdown_word_dev_ = control_slots_.devicePtr(proxy_shutdown_slot);
    channels_.resize(channel_slots);
    device_channel_views_.resize(channel_slots);
    for (uint32_t channel_idx = 0; channel_idx < channel_count; channel_idx++) {
        for (uint32_t peer_idx = 0; peer_idx < max_peers; peer_idx++) {
            const size_t slot = static_cast<size_t>(channel_idx) * max_peers + peer_idx;
            rc = channels_[slot].allocate(ring_depth_, &control_slots_, proxy_ci_slot_base + slot);
            if (rc != NIXL_SUCCESS) {
                shutdown();
                return rc;
            }
            device_channel_views_[slot] = channels_[slot].deviceView;
        }
    }

    if (cudaMalloc(reinterpret_cast<void **>(&device_channel_views_dev_),
                   sizeof(nixlProxyChannelView) * channel_slots) != cudaSuccess ||
        cudaMemcpy(device_channel_views_dev_,
                   device_channel_views_.data(),
                   sizeof(nixlProxyChannelView) * channel_slots,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        shutdown();
        return NIXL_ERR_BACKEND;
    }

    nixlProxyDeviceContextData device_context{
        device_channel_views_dev_, max_peers, channel_count, shutdown_word_dev_};
    if (cudaMalloc(reinterpret_cast<void **>(&device_context_),
                   sizeof(nixlProxyDeviceContextData)) != cudaSuccess ||
        cudaMemcpy(
            device_context_, &device_context, sizeof(device_context), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        if (device_context_) {
            cudaFree(device_context_);
            device_context_ = nullptr;
        }
        shutdown();
        return NIXL_ERR_BACKEND;
    }
    memview_registry_.setDeviceContext(device_context_);

    workers_.clear();
    workers_.reserve(effective_worker_count);
    workers_started_ = false;

    for (uint32_t worker_idx = 0; worker_idx < effective_worker_count; worker_idx++) {
        NIXL_INFO << "ProxyRuntime::init: worker " << worker_idx
                  << " owns channel(s) where channel_id % " << effective_worker_count
                  << " == " << worker_idx << "; handles all dest rings of those channels";
        workers_.push_back(std::make_unique<nixlProxyWorker>(backend_.get(),
                                                             &memview_registry_,
                                                             &shutdown_state_,
                                                             channels_.data(),
                                                             max_peers,
                                                             channel_count,
                                                             worker_idx,
                                                             effective_worker_count,
                                                             pthr_delay_us));
    }

    NIXL_INFO << "ProxyRuntime::init: complete — " << max_peers << " peers, " << channel_count
              << " channels (rings per dest), " << effective_worker_count
              << " workers, device_context(dev)=" << device_context_;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::loadRemoteConnInfo(const std::string &remote_name, const nixl_blob_t &conn_info) {
    NIXL_INFO << "ProxyRuntime::loadRemoteConnInfo: remote='" << remote_name
              << "' conn_info_size=" << conn_info.size();
    if (backend_ == nullptr) {
        NIXL_ERROR << "ProxyRuntime::loadRemoteConnInfo: no backend";
        return NIXL_ERR_NOT_SUPPORTED;
    }
    nixl_status_t rc = backend_->loadRemoteConnInfo(remote_name, conn_info);
    NIXL_INFO << "ProxyRuntime::loadRemoteConnInfo: result=" << rc;
    return rc;
}

nixl_status_t
nixlProxyRuntime::registerProxyMemView(nixlMemViewH backend_memview, nixlMemViewH *proxy_memview) {
    return memview_registry_.registerProxyMemView(backend_memview, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    return memview_registry_.prepMemView(dlist, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    std::vector<void *> direct_ptrs;
    if (backend_ != nullptr) {
        const nixl_status_t resolve_status = backend_->resolveDirectPointers(dlist, direct_ptrs);
        if (resolve_status == NIXL_ERR_NOT_SUPPORTED) {
            direct_ptrs.clear();
        } else if (resolve_status != NIXL_SUCCESS) {
            return resolve_status;
        }
    }

    return memview_registry_.prepMemView(dlist, direct_ptrs, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(nixlMemViewH backend_memview,
                              const nixl_meta_dlist_t &dlist,
                              nixlMemViewH *proxy_memview) {
    return memview_registry_.prepMemView(backend_memview, dlist, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(nixlMemViewH backend_memview,
                              const nixl_remote_meta_dlist_t &dlist,
                              nixlMemViewH *proxy_memview) {
    return memview_registry_.prepMemView(backend_memview, dlist, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    return memview_registry_.unregisterProxyMemView(proxy_memview);
}

nixl_status_t
nixlProxyRuntime::storeMetadata(nixlMemViewH proxy_memview, const nixl_meta_dlist_t &dlist) {
    return memview_registry_.storeMetadata(proxy_memview, dlist);
}

nixl_status_t
nixlProxyRuntime::storeMetadata(nixlMemViewH proxy_memview, const nixl_remote_meta_dlist_t &dlist) {
    return memview_registry_.storeMetadata(proxy_memview, dlist);
}

bool
nixlProxyRuntime::resolveProxyMemView(nixlMemViewH proxy_memview,
                                      nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemView(proxy_memview, backend_memview);
}

bool
nixlProxyRuntime::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                        nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemViewId(proxy_memview_id, backend_memview);
}

nixl_status_t
nixlProxyRuntime::startWorkers() {
    NIXL_INFO << "ProxyRuntime::startWorkers: launching " << workers_.size() << " worker thread(s)";
    if (!control_slots_.allocated()) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: runtime not initialized";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (workers_started_) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: workers already started";
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixl_status_t publish_status = control_slots_.writeSlot(
        proxy_shutdown_slot, static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING));
    if (publish_status != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: failed to publish RUNNING state";
        return publish_status;
    }
    shutdown_state_.store(static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING),
                          std::memory_order_release);

    for (auto &worker : workers_) {
        worker->start();
    }
    workers_started_ = true;

    NIXL_INFO << "ProxyRuntime::startWorkers: all threads launched";
    return NIXL_SUCCESS;
}

void
nixlProxyRuntime::joinWorkerThreads() noexcept {
    for (auto &worker : workers_) {
        worker->join();
    }
}

nixl_status_t
nixlProxyRuntime::shutdown() {
    NIXL_INFO << "ProxyRuntime::shutdown: signalling workers to stop";
    nixl_status_t shutdown_signal_status = NIXL_SUCCESS;
    if (control_slots_.allocated()) {
        shutdown_signal_status = control_slots_.writeSlot(
            proxy_shutdown_slot, static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN));
        if (shutdown_signal_status != NIXL_SUCCESS) {
            NIXL_ERROR << "ProxyRuntime::shutdown: failed to publish SHUTDOWN state";
        }
    }
    shutdown_state_.store(static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN),
                          std::memory_order_release);

    joinWorkerThreads();
    workers_started_ = false;
    NIXL_INFO << "ProxyRuntime::shutdown: all worker threads joined";

    if (backend_ != nullptr) {
        size_t released = 0;
        for (auto &channel : channels_) {
            if (channel.ringDepth == 0 || channel.consumerIdxDev == nullptr) {
                continue;
            }

            const uint64_t consumer_idx = channel.consumerIdxShadow;
            for (uint64_t idx = consumer_idx; idx < channel.submitIdx; ++idx) {
                nixlProxyRequestState &inflight = channel.inflightSlots[idx % channel.ringDepth];
                if (inflight.status == NIXL_IN_PROG && inflight.backendRequest) {
                    backend_->releaseRequest(inflight.backendRequest);
                    ++released;
                }
                inflight = nixlProxyRequestState{};
            }
            channel.submitIdx = consumer_idx;
        }
        if (released != 0) {
            NIXL_INFO << "ProxyRuntime::shutdown: released " << released
                      << " pending backend request(s)";
        }
    }

    nixl_status_t backend_status = NIXL_SUCCESS;
    if (backend_ != nullptr) {
        NIXL_INFO << "ProxyRuntime::shutdown: shutting down backend";
        backend_status = backend_->shutdown();
        NIXL_INFO << "ProxyRuntime::shutdown: backend shutdown status=" << backend_status;
        if (backend_status == NIXL_ERR_NOT_SUPPORTED) {
            backend_status = NIXL_SUCCESS;
        }
    }

    workers_.clear();
    memview_registry_.clear();
    memview_registry_.setDeviceContext(nullptr);

    if (device_context_) {
        cudaFree(device_context_);
        device_context_ = nullptr;
    }
    shutdown_word_dev_ = nullptr;
    if (device_channel_views_dev_) {
        cudaFree(device_channel_views_dev_);
        device_channel_views_dev_ = nullptr;
    }
    device_channel_views_.clear();

    channels_.clear();
    control_slots_.deallocate();
    backend_.reset();
    NIXL_INFO << "ProxyRuntime::shutdown: complete";
    if (backend_status != NIXL_SUCCESS) {
        return backend_status;
    }
    if (shutdown_signal_status != NIXL_SUCCESS) {
        return shutdown_signal_status;
    }
    return NIXL_SUCCESS;
}
