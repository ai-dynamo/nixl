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
#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <cuda_runtime.h>

nixl_status_t
nixlProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                               nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (next_proxy_memview_id_ > std::numeric_limits<uint32_t>::max()) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::register: proxy memview ID space exhausted";
        return NIXL_ERR_NOT_ALLOWED;
    }

    RegistryEntry entry;
    entry.proxy_memview_id = static_cast<uint32_t>(next_proxy_memview_id_);
    entry.backend_memview = backend_memview;
    entries_.push_back(entry);

    *proxy_memview = reinterpret_cast<nixlMemViewH>(static_cast<uintptr_t>(entry.proxy_memview_id));
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
    return prepMemView(nullptr, dlist, proxy_memview);
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
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    entry->state = ProxyMemViewRegEntryState::ENTRY_RETIRED;
    NIXL_DEBUG << "nixlProxyMemViewRegistry::unregister: proxy_id=" << entry->proxy_memview_id;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::resolveProxyMemView(nixlMemViewH proxy_memview,
                                              nixlMemViewH &backend_memview) const {
    const RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return false;
    }
    backend_memview = entry->backend_memview;
    return true;
}

bool
nixlProxyMemViewRegistry::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                                nixlMemViewH &backend_memview) const {
    const RegistryEntry *entry = getEntryForId(proxy_memview_id);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return false;
    }
    backend_memview = entry->backend_memview;
    return true;
}

nixl_status_t
nixlProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                        const nixl_meta_dlist_t &dlist) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return NIXL_ERR_NOT_FOUND;
    }

    fillLocalMetadata(dlist, entry->local_metadata);
    entry->remote_metadata = RemoteMetadata{};
    entry->metadata_kind = ProxyMemViewRegMetadataKind::METADATA_KIND_LOCAL;
    entry->state = ProxyMemViewRegEntryState::ENTRY_READY;

    NIXL_DEBUG << "nixlProxyMemViewRegistry::storeMetadata(local): proxy_id="
               << entry->proxy_memview_id << " entries=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                        const nixl_remote_meta_dlist_t &dlist) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return NIXL_ERR_NOT_FOUND;
    }

    fillRemoteMetadata(dlist, entry->remote_metadata);
    entry->local_metadata = LocalMetadata{};
    entry->metadata_kind = ProxyMemViewRegMetadataKind::METADATA_KIND_REMOTE;
    entry->state = ProxyMemViewRegEntryState::ENTRY_READY;

    NIXL_DEBUG << "nixlProxyMemViewRegistry::storeMetadata(remote): proxy_id="
               << entry->proxy_memview_id << " entries=" << dlist.descCount();
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

    const RemoteMetadata *remote_metadata = nullptr;
    const ProxyMemViewRegStoredEntry *dst_metadata = nullptr;
    nixl_status_t status = getRemoteEntryForSubmission(submission.dst_proxy_memview_id,
                                                       submission.dst_index,
                                                       submission.dst_offset,
                                                       transfer_size,
                                                       remote_metadata,
                                                       dst_metadata);
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
    // Use the element's own agent (the aggregate memview spans many peers; the
    // view-level remote_agent is only the first one).
    prepared.remote_agent = dst_metadata->remote_agent.empty() ? remote_metadata->remote_agent :
                                                                 dst_metadata->remote_agent;
    prepared.remote.mem_type = remote_metadata->mem_type;
    prepared.remote.desc = nixlMetaDesc(dst_metadata->base_addr + submission.dst_offset,
                                        transfer_size,
                                        dst_metadata->dev_id,
                                        dst_metadata->metadata);

    if (needs_source) {
        const LocalMetadata *local_metadata = nullptr;
        const ProxyMemViewRegStoredEntry *src_metadata = nullptr;
        status = getLocalEntryForSubmission(submission.src_proxy_memview_id,
                                            submission.src_index,
                                            submission.src_offset,
                                            transfer_size,
                                            local_metadata,
                                            src_metadata);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        prepared.local.mem_type = local_metadata->mem_type;
        prepared.local.desc = nixlMetaDesc(src_metadata->base_addr + submission.src_offset,
                                           transfer_size,
                                           src_metadata->dev_id,
                                           src_metadata->metadata);
    }

    prepared_submission = prepared;
    return NIXL_SUCCESS;
}

void
nixlProxyMemViewRegistry::clear() noexcept {
    for (auto &entry : entries_) {
        entry.state = ProxyMemViewRegEntryState::ENTRY_RETIRED;
    }
}

nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) {
    return getEntryForId(reinterpret_cast<uint64_t>(proxy_memview));
}

const nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) const {
    return getEntryForId(reinterpret_cast<uint64_t>(proxy_memview));
}

nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::getEntryForId(uint64_t proxy_memview_id) {
    if (proxy_memview_id < 1 || proxy_memview_id >= next_proxy_memview_id_ ||
        proxy_memview_id > entries_.size()) {
        return nullptr;
    }
    return &entries_[proxy_memview_id - 1];
}

const nixlProxyMemViewRegistry::RegistryEntry *
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
    const RemoteMetadata *&metadata,
    const ProxyMemViewRegStoredEntry *&entry) const {
    metadata = nullptr;
    entry = nullptr;

    const RegistryEntry *registry_entry = getEntryForId(proxy_memview_id);
    if (registry_entry == nullptr ||
        registry_entry->state != ProxyMemViewRegEntryState::ENTRY_READY) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst not ready"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (registry_entry->metadata_kind != ProxyMemViewRegMetadataKind::METADATA_KIND_REMOTE) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst metadata kind invalid"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &remote_metadata = registry_entry->remote_metadata;
    if (index >= remote_metadata.entries.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const ProxyMemViewRegStoredEntry &remote_entry = remote_metadata.entries[index];
    if (!rangeFits(remote_entry, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    metadata = &remote_metadata;
    entry = &remote_entry;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::getLocalEntryForSubmission(
    uint64_t proxy_memview_id,
    size_t index,
    size_t offset,
    size_t size,
    const LocalMetadata *&metadata,
    const ProxyMemViewRegStoredEntry *&entry) const {
    metadata = nullptr;
    entry = nullptr;

    const RegistryEntry *registry_entry = getEntryForId(proxy_memview_id);
    if (registry_entry == nullptr ||
        registry_entry->state != ProxyMemViewRegEntryState::ENTRY_READY) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: src not ready"
                   << " src_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (registry_entry->metadata_kind != ProxyMemViewRegMetadataKind::METADATA_KIND_LOCAL) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: src metadata kind invalid"
                   << " src_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &local_metadata = registry_entry->local_metadata;
    if (index >= local_metadata.entries.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const ProxyMemViewRegStoredEntry &local_entry = local_metadata.entries[index];
    if (!rangeFits(local_entry, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    metadata = &local_metadata;
    entry = &local_entry;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::rangeFits(const ProxyMemViewRegStoredEntry &entry,
                                    size_t offset,
                                    size_t size) {
    return offset <= entry.len && size <= entry.len - offset;
}

void
nixlProxyMemViewRegistry::fillLocalMetadata(const nixl_meta_dlist_t &dlist, LocalMetadata &out) {
    out = LocalMetadata{};
    out.mem_type = dlist.getType();
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        out.entries.push_back(
            ProxyMemViewRegStoredEntry{desc.addr, desc.len, desc.devId, desc.metadataP});
    }
}

void
nixlProxyMemViewRegistry::fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist,
                                             RemoteMetadata &out) {
    out = RemoteMetadata{};
    out.mem_type = dlist.getType();
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        if (out.remote_agent.empty() && desc.remoteAgent != nixl_null_agent) {
            out.remote_agent = desc.remoteAgent;
        }
        out.entries.push_back(ProxyMemViewRegStoredEntry{
            desc.addr, desc.len, desc.devId, desc.metadataP, desc.remoteAgent});
    }
}

nixl_status_t
nixlProxyChannelState::allocate(uint32_t peer_index, uint32_t channel_id, uint32_t depth) {
    NIXL_INFO << "nixlProxyChannelState::allocate: peer=" << peer_index << " channel=" << channel_id
              << " depth=" << depth;
    if (depth == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    try {
        inflight_slots_.assign(depth, nixlProxyRequestState{});
    }
    catch (const std::bad_alloc &) {
        return NIXL_ERR_BACKEND;
    }
    ring_depth_ = depth;
    if (cudaMalloc(reinterpret_cast<void **>(&work_ring_dev_), sizeof(nixlProxyWorkRing)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&producer_idx_dev_), sizeof(uint64_t)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&consumer_idx_cache_dev_), sizeof(uint64_t)) !=
            cudaSuccess ||
        cudaMallocHost(&records_host_, sizeof(nixlProxySubmission) * depth) != cudaSuccess ||
        cudaMallocHost(reinterpret_cast<void **>(&consumer_idx_host_), sizeof(uint64_t)) !=
            cudaSuccess ||
        cudaMallocHost(&completion_slot_host_, sizeof(nixlProxyCompletionSlot)) != cudaSuccess) {
        NIXL_ERROR << "nixlProxyChannelState::allocate: CUDA allocation failed for channel "
                   << channel_id;
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    void *records_dev = nullptr;
    if (cudaHostGetDevicePointer(&records_dev, records_host_, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    auto *records_dev_ptr = static_cast<nixlProxySubmission *>(records_dev);

    void *consumer_dev = nullptr;
    if (cudaHostGetDevicePointer(&consumer_dev, consumer_idx_host_, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    auto *consumer_idx_dev = static_cast<uint64_t *>(consumer_dev);

    void *completion_dev = nullptr;
    if (cudaHostGetDevicePointer(&completion_dev, completion_slot_host_, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    completion_slot_dev_ = static_cast<nixlProxyCompletionSlot *>(completion_dev);

    for (uint32_t i = 0; i < depth; ++i) {
        records_host_[i] = nixlProxySubmission{};
    }
    if (cudaMemset(producer_idx_dev_, 0, sizeof(*producer_idx_dev_)) != cudaSuccess ||
        cudaMemset(consumer_idx_cache_dev_, 0, sizeof(*consumer_idx_cache_dev_)) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    __atomic_store_n(consumer_idx_host_, uint64_t{0}, __ATOMIC_RELEASE);
    completion_slot_host_->next_status = NIXL_IN_PROG;
    __atomic_store_n(&completion_slot_host_->completed_idx, uint64_t{0}, __ATOMIC_RELEASE);
    nixlProxyWorkRing work_ring{
        records_dev_ptr,
        producer_idx_dev_,
        consumer_idx_dev,
        consumer_idx_cache_dev_,
        depth,
    };
    if (cudaMemcpy(work_ring_dev_, &work_ring, sizeof(work_ring), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    device_view =
        nixlProxyChannelView{work_ring_dev_, completion_slot_dev_, peer_index, channel_id};

    submit_idx_ = 0;
    error_latched = false;
    NIXL_INFO << "nixlProxyChannelState::allocate: peer " << peer_index << " channel " << channel_id
              << " ready" << " work_ring(dev)=" << work_ring_dev_ << " records=" << records_host_
              << " records(dev)=" << records_dev_ptr << " producer_idx(dev)=" << producer_idx_dev_
              << " consumer_idx(host)=" << consumer_idx_host_
              << " consumer_idx(dev)=" << consumer_idx_dev
              << " consumer_idx_cache(dev)=" << consumer_idx_cache_dev_
              << " completion_slot(host)=" << completion_slot_host_
              << " completion_slot(dev)=" << completion_slot_dev_;
    return NIXL_SUCCESS;
}

void
nixlProxyChannelState::deallocate() noexcept {
    if (completion_slot_host_) {
        cudaFreeHost(completion_slot_host_);
        completion_slot_host_ = nullptr;
        completion_slot_dev_ = nullptr;
    }
    if (producer_idx_dev_) {
        cudaFree(producer_idx_dev_);
        producer_idx_dev_ = nullptr;
    }
    if (consumer_idx_cache_dev_) {
        cudaFree(consumer_idx_cache_dev_);
        consumer_idx_cache_dev_ = nullptr;
    }
    if (consumer_idx_host_) {
        cudaFreeHost(consumer_idx_host_);
        consumer_idx_host_ = nullptr;
    }
    if (records_host_) {
        cudaFreeHost(records_host_);
        records_host_ = nullptr;
    }
    if (work_ring_dev_) {
        cudaFree(work_ring_dev_);
        work_ring_dev_ = nullptr;
    }
    ring_depth_ = 0;
    inflight_slots_.clear();
    submit_idx_ = 0;
    error_latched = false;
    device_view = nixlProxyChannelView{};
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
        device_view = other.device_view;
        inflight_slots_ = std::move(other.inflight_slots_);
        submit_idx_ = other.submit_idx_;
        error_latched = other.error_latched;
        work_ring_dev_ = other.work_ring_dev_;
        records_host_ = other.records_host_;
        producer_idx_dev_ = other.producer_idx_dev_;
        consumer_idx_host_ = other.consumer_idx_host_;
        consumer_idx_cache_dev_ = other.consumer_idx_cache_dev_;
        ring_depth_ = other.ring_depth_;
        completion_slot_host_ = other.completion_slot_host_;
        completion_slot_dev_ = other.completion_slot_dev_;
        other.work_ring_dev_ = nullptr;
        other.records_host_ = nullptr;
        other.producer_idx_dev_ = nullptr;
        other.consumer_idx_host_ = nullptr;
        other.consumer_idx_cache_dev_ = nullptr;
        other.ring_depth_ = 0;
        other.submit_idx_ = 0;
        other.completion_slot_host_ = nullptr;
        other.completion_slot_dev_ = nullptr;
        other.error_latched = false;
        other.device_view = nixlProxyChannelView{};
    }
    return *this;
}

void
nixlProxyChannelState::resetLocalState() noexcept {
    if (!allocated()) {
        return;
    }

    uint64_t producer_idx = 0;
    if (cudaMemcpy(
            &producer_idx, producer_idx_dev_, sizeof(producer_idx), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        NIXL_ERROR << "nixlProxyChannelState::resetLocalState: failed to read producer_idx";
        producer_idx = submit_idx_;
    }

    for (uint32_t slot = 0; slot < ring_depth_; ++slot) {
        records_host_[slot] = nixlProxySubmission{};
        __atomic_store_n(&records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
    }
    std::fill(inflight_slots_.begin(), inflight_slots_.end(), nixlProxyRequestState{});

    submit_idx_ = producer_idx;
    __atomic_store_n(consumer_idx_host_, producer_idx, __ATOMIC_RELEASE);
    if (cudaMemcpy(
            consumer_idx_cache_dev_, &producer_idx, sizeof(producer_idx), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        NIXL_ERROR << "nixlProxyChannelState::resetLocalState: failed to update consumer cache";
    }

    error_latched = false;
    completion_slot_host_->next_status = NIXL_IN_PROG;
    __atomic_store_n(&completion_slot_host_->completed_idx, producer_idx, __ATOMIC_RELEASE);
}

nixlProxyRuntime::nixlProxyRuntime() = default;

nixlProxyRuntime::~nixlProxyRuntime() {
    if (backend_) {
        shutdown();
    }
}

nixl_status_t
nixlProxyRuntime::init(std::unique_ptr<nixlDeviceProxyBackendAdapter> backend,
                       uint32_t peer_capacity,
                       uint32_t channel_count,
                       uint32_t worker_count,
                       uint64_t pthr_delay_us) {
    NIXL_INFO << "ProxyRuntime::init: peer_capacity=" << peer_capacity
              << " channel_count=" << channel_count << " worker_count=" << worker_count
              << " pthr_delay_us=" << pthr_delay_us << " backend=" << backend.get();
    if (backend == nullptr || peer_capacity == 0 || channel_count == 0 || worker_count == 0 ||
        static_cast<size_t>(peer_capacity) > std::numeric_limits<size_t>::max() / channel_count) {
        NIXL_ERROR << "ProxyRuntime::init: invalid params";
        return NIXL_ERR_INVALID_PARAM;
    }

    backend_ = std::move(backend);
    peer_capacity_ = peer_capacity;
    channel_count_ = channel_count;
    memview_registry_.clear();

    if (cudaMallocHost(reinterpret_cast<void **>(&shutdown_word_host_), sizeof(uint32_t)) !=
        cudaSuccess) {
        NIXL_ERROR << "ProxyRuntime::init: failed to allocate shutdown_word";
        shutdown_word_host_ = nullptr;
        backend_.reset();
        return NIXL_ERR_BACKEND;
    }
    void *shutdown_dev = nullptr;
    if (cudaHostGetDevicePointer(&shutdown_dev, shutdown_word_host_, 0) != cudaSuccess) {
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        backend_.reset();
        return NIXL_ERR_BACKEND;
    }
    shutdown_word_dev_ = static_cast<uint32_t *>(shutdown_dev);
    __atomic_store_n(shutdown_word_host_,
                     static_cast<uint32_t>(nixl_proxy_control_state_t::RUNNING),
                     __ATOMIC_RELEASE);

    worker_count_ = std::min(worker_count, channel_count);
    NIXL_INFO << "ProxyRuntime::init: effective worker_count=" << worker_count_
              << " (clamped to channel_count)";

    nixl_status_t rc = backend_->init(worker_count_, channel_count_, peer_capacity_);
    if ((rc != NIXL_SUCCESS) && (rc != NIXL_ERR_NOT_SUPPORTED)) {
        NIXL_ERROR << "ProxyRuntime::init: backend init failed: " << rc;
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_ = nullptr;
        backend_.reset();
        return rc;
    }
    if (rc == NIXL_ERR_NOT_SUPPORTED) {
        NIXL_INFO << "ProxyRuntime::init: backend init hook not supported; continuing";
    }

    const size_t channel_slots = static_cast<size_t>(peer_capacity_) * channel_count_;
    channels_.resize(channel_slots);
    device_channel_views_.resize(channel_slots);
    for (uint32_t channel = 0; channel < channel_count_; ++channel) {
        for (uint32_t peer = 0; peer < peer_capacity_; ++peer) {
            const size_t slot = channelSlot(peer, channel);
            device_channel_views_[slot].peer_index = peer;
            device_channel_views_[slot].channel_id = channel;
        }
    }

    if (cudaMalloc(reinterpret_cast<void **>(&device_channel_views_dev_),
                   sizeof(nixlProxyChannelView) * channel_slots) != cudaSuccess ||
        cudaMemset(device_channel_views_dev_, 0, sizeof(nixlProxyChannelView) * channel_slots) !=
            cudaSuccess) {
        shutdown();
        return NIXL_ERR_BACKEND;
    }

    nixlProxyDeviceContextData device_context{
        device_channel_views_dev_, peer_capacity_, channel_count_, shutdown_word_dev_};
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

    channel_lifecycle_ =
        std::make_unique<std::atomic<nixl_proxy_channel_lifecycle_t>[]>(channel_slots);
    for (size_t slot = 0; slot < channel_slots; ++slot) {
        channel_lifecycle_[slot].store(nixl_proxy_channel_lifecycle_t::UNALLOCATED,
                                       std::memory_order_relaxed);
    }
    active_agents_.assign(peer_capacity_, nixl_null_agent);

    workers_.clear();
    workers_.reserve(worker_count_);
    workers_started_ = false;

    for (uint32_t worker = 0; worker < worker_count_; ++worker) {
        NIXL_INFO << "ProxyRuntime::init: worker " << worker
                  << " owns channel(s) where channel_id % " << worker_count_ << " == " << worker
                  << "; handles all dest rings of those channels";
        workers_.push_back(std::make_unique<ProxyWorker>(backend_.get(),
                                                         &memview_registry_,
                                                         shutdown_word_host_,
                                                         channels_.data(),
                                                         channel_lifecycle_.get(),
                                                         peer_capacity_,
                                                         channel_count_,
                                                         worker,
                                                         worker_count_,
                                                         pthr_delay_us));
    }

    NIXL_INFO << "ProxyRuntime::init: complete — " << peer_capacity_ << " peers, " << channel_count_
              << " channels (rings per dest), " << worker_count_
              << " workers, device_context(dev)=" << device_context_;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::allocatePeerRow(uint32_t peer_index) {
    if (peer_index >= peer_capacity_) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (channels_[channelSlot(peer_index, 0)].allocated()) {
        return NIXL_SUCCESS;
    }

    uint32_t allocated_channels = 0;
    for (; allocated_channels < channel_count_; ++allocated_channels) {
        const size_t slot = channelSlot(peer_index, allocated_channels);
        nixl_status_t status =
            channels_[slot].allocate(peer_index, allocated_channels, ring_depth_);
        if (status != NIXL_SUCCESS) {
            for (uint32_t channel = 0; channel < allocated_channels; ++channel) {
                const size_t rollback_slot = channelSlot(peer_index, channel);
                channels_[rollback_slot].deallocate();
                device_channel_views_[rollback_slot] =
                    nixlProxyChannelView{nullptr, nullptr, peer_index, channel};
                channel_lifecycle_[rollback_slot].store(nixl_proxy_channel_lifecycle_t::UNALLOCATED,
                                                        std::memory_order_release);
            }
            return status;
        }
        device_channel_views_[slot] = channels_[slot].device_view;
        channel_lifecycle_[slot].store(nixl_proxy_channel_lifecycle_t::INACTIVE,
                                       std::memory_order_release);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::publishPeerRow(uint32_t peer_index, bool active) {
    if (peer_index >= peer_capacity_ || device_channel_views_dev_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Channel-major layout: a peer's rings are one column, not a contiguous row.
    for (uint32_t channel_id = 0; channel_id < channel_count_; ++channel_id) {
        const size_t slot = channelSlot(peer_index, channel_id);
        cudaError_t status;
        if (active) {
            status = cudaMemcpy(device_channel_views_dev_ + slot,
                                device_channel_views_.data() + slot,
                                sizeof(nixlProxyChannelView),
                                cudaMemcpyHostToDevice);
        } else {
            status = cudaMemset(device_channel_views_dev_ + slot, 0, sizeof(nixlProxyChannelView));
        }
        if (status != cudaSuccess) {
            NIXL_ERROR << "ProxyRuntime: failed to publish peer " << peer_index
                       << " channel=" << channel_id << " active=" << active << ": "
                       << cudaGetErrorString(status);
            return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::waitPeerChannelsInactive(uint32_t peer_index) {
    if (!workers_started_) {
        for (uint32_t channel_id = 0; channel_id < channel_count_; ++channel_id) {
            const size_t slot = channelSlot(peer_index, channel_id);
            if (channel_lifecycle_[slot].load(std::memory_order_acquire) !=
                nixl_proxy_channel_lifecycle_t::RESET_PENDING) {
                continue;
            }
            channels_[slot].resetLocalState();
            channel_lifecycle_[slot].store(nixl_proxy_channel_lifecycle_t::INACTIVE,
                                           std::memory_order_release);
        }
        return NIXL_SUCCESS;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        bool all_inactive = true;
        for (uint32_t channel_id = 0; channel_id < channel_count_; ++channel_id) {
            const nixl_proxy_channel_lifecycle_t lifecycle =
                channel_lifecycle_[channelSlot(peer_index, channel_id)].load(
                    std::memory_order_acquire);
            if (lifecycle == nixl_proxy_channel_lifecycle_t::RESET_PENDING ||
                lifecycle == nixl_proxy_channel_lifecycle_t::ACTIVE) {
                all_inactive = false;
                break;
            }
        }
        if (all_inactive) {
            return NIXL_SUCCESS;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            NIXL_ERROR << "ProxyRuntime::waitPeerChannelsInactive: timed out waiting for peer "
                       << peer_index << " channel reset acknowledgements";
            return NIXL_ERR_BACKEND;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

nixl_status_t
nixlProxyRuntime::deactivatePeer(uint32_t peer_index) {
    if (peer_index >= peer_capacity_) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixl_status_t unpublish_status = publishPeerRow(peer_index, false);
    if (unpublish_status != NIXL_SUCCESS) {
        return unpublish_status;
    }

    if (!channels_[channelSlot(peer_index, 0)].allocated()) {
        active_agents_[peer_index] = nixl_null_agent;
        return NIXL_SUCCESS;
    }

    for (uint32_t channel_id = 0; channel_id < channel_count_; ++channel_id) {
        channel_lifecycle_[channelSlot(peer_index, channel_id)].store(
            nixl_proxy_channel_lifecycle_t::RESET_PENDING, std::memory_order_release);
    }

    const nixl_status_t wait_status = waitPeerChannelsInactive(peer_index);
    if (wait_status != NIXL_SUCCESS) {
        return wait_status;
    }

    active_agents_[peer_index] = nixl_null_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::activatePeer(uint32_t peer_index, const std::string &remote_agent) {
    if (peer_index >= peer_capacity_ || remote_agent == nixl_null_agent) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t status = allocatePeerRow(peer_index);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = waitPeerChannelsInactive(peer_index);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    for (uint32_t channel_id = 0; channel_id < channel_count_; ++channel_id) {
        channel_lifecycle_[channelSlot(peer_index, channel_id)].store(
            nixl_proxy_channel_lifecycle_t::ACTIVE, std::memory_order_release);
    }

    status = publishPeerRow(peer_index, true);
    if (status != NIXL_SUCCESS) {
        for (uint32_t channel_id = 0; channel_id < channel_count_; ++channel_id) {
            channel_lifecycle_[channelSlot(peer_index, channel_id)].store(
                nixl_proxy_channel_lifecycle_t::INACTIVE, std::memory_order_release);
        }
        return status;
    }

    active_agents_[peer_index] = remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::reconcilePeer(uint32_t peer_index, const std::string &remote_agent) {
    if (peer_index >= peer_capacity_) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_agent == active_agents_[peer_index]) {
        return NIXL_SUCCESS;
    }

    if (active_agents_[peer_index] != nixl_null_agent) {
        const nixl_status_t status = deactivatePeer(peer_index);
        if (status != NIXL_SUCCESS) {
            return status;
        }
    }

    if (remote_agent == nixl_null_agent) {
        return NIXL_SUCCESS;
    }
    return activatePeer(peer_index, remote_agent);
}

nixl_status_t
nixlProxyRuntime::reconcileRemotePeers(const nixl_remote_meta_dlist_t &dlist) {
    const uint32_t peer_count = static_cast<uint32_t>(dlist.descCount());
    if (peer_count > peer_capacity_) {
        NIXL_ERROR << "ProxyRuntime: remote memory view has " << peer_count
                   << " entries, exceeding peer capacity " << peer_capacity_;
        return NIXL_ERR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(peer_reconcile_mutex_);
    uint32_t peer = 0;
    for (const auto &desc : dlist) {
        nixl_status_t status = reconcilePeer(peer, desc.remoteAgent);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        ++peer;
    }
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
    nixl_status_t status = memview_registry_.prepMemView(dlist, proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    status = reconcileRemotePeers(dlist);
    if (status != NIXL_SUCCESS) {
        memview_registry_.unregisterProxyMemView(*proxy_memview);
        *proxy_memview = nullptr;
    }
    return status;
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
    nixl_status_t status = memview_registry_.prepMemView(backend_memview, dlist, proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    status = reconcileRemotePeers(dlist);
    if (status != NIXL_SUCCESS) {
        memview_registry_.unregisterProxyMemView(*proxy_memview);
        *proxy_memview = nullptr;
    }
    return status;
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

nixl_proxy_channel_lifecycle_t
nixlProxyRuntime::channelLifecycle(uint32_t peer_index, uint32_t channel_id) const {
    if (peer_index >= peer_capacity_ || channel_id >= channel_count_ ||
        channel_lifecycle_ == nullptr) {
        return nixl_proxy_channel_lifecycle_t::UNALLOCATED;
    }
    return channel_lifecycle_[channelSlot(peer_index, channel_id)].load(std::memory_order_acquire);
}

nixl_status_t
nixlProxyRuntime::startWorkers() {
    NIXL_INFO << "ProxyRuntime::startWorkers: launching " << workers_.size() << " worker thread(s)";
    if (shutdown_word_host_ == nullptr) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: runtime not initialized";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (workers_started_) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: workers already started";
        return NIXL_ERR_INVALID_PARAM;
    }

    for (auto &channel : channels_) {
        channel.submit_idx_ =
            channel.allocated() ? __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED) : 0;
        std::fill(channel.inflight_slots_.begin(),
                  channel.inflight_slots_.end(),
                  nixlProxyRequestState{});
        channel.error_latched = false;
    }

    __atomic_store_n(shutdown_word_host_,
                     static_cast<uint32_t>(nixl_proxy_control_state_t::RUNNING),
                     __ATOMIC_RELEASE);

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
    if (shutdown_word_host_ != nullptr) {
        __atomic_store_n(shutdown_word_host_,
                         static_cast<uint32_t>(nixl_proxy_control_state_t::SHUTDOWN),
                         __ATOMIC_RELEASE);
    }

    joinWorkerThreads();
    workers_started_ = false;
    NIXL_INFO << "ProxyRuntime::shutdown: all worker threads joined";

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

    if (device_context_) {
        cudaFree(device_context_);
        device_context_ = nullptr;
    }
    if (shutdown_word_host_) {
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_ = nullptr;
    }
    if (device_channel_views_dev_) {
        cudaFree(device_channel_views_dev_);
        device_channel_views_dev_ = nullptr;
    }
    device_channel_views_.clear();

    channels_.clear();
    channel_lifecycle_.reset();
    active_agents_.clear();
    peer_capacity_ = 0;
    channel_count_ = 0;
    worker_count_ = 0;
    backend_.reset();
    NIXL_INFO << "ProxyRuntime::shutdown: complete";
    return backend_status;
}
