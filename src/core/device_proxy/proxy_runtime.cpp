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

nixl_status_t
nixlProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                               nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    RegistryEntry entry;
    entry.proxy_memview_id = next_proxy_memview_id_;
    entry.backend_memview = backend_memview;
    entries_.push_back(entry);

    *proxy_memview = reinterpret_cast<nixlMemViewH>(entry.proxy_memview_id);
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
    prepared.remote_agent = remote_metadata->remote_agent;
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
        out.entries.push_back(
            ProxyMemViewRegStoredEntry{desc.addr, desc.len, desc.devId, desc.metadataP});
    }
}

nixl_status_t
nixlProxyChannelState::allocate(uint32_t channel_id, uint32_t depth) {
    NIXL_INFO << "nixlProxyChannelState::allocate: channel_id=" << channel_id << " depth=" << depth;
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
    device_view = nixlProxyChannelView{work_ring_dev_, completion_slot_dev_, channel_id};

    inflight_requests.clear();
    NIXL_INFO << "nixlProxyChannelState::allocate: channel " << channel_id << " ready"
              << " work_ring(dev)=" << work_ring_dev_ << " records=" << records_host_
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
        inflight_requests = std::move(other.inflight_requests);
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
        other.completion_slot_host_ = nullptr;
        other.completion_slot_dev_ = nullptr;
        other.device_view = nixlProxyChannelView{};
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
                       uint32_t channel_count,
                       uint32_t worker_count,
                       uint64_t pthr_delay_us) {
    NIXL_INFO << "ProxyRuntime::init: channel_count=" << channel_count
              << " worker_count=" << worker_count << " pthr_delay_us=" << pthr_delay_us
              << " backend=" << backend.get();
    if (backend == nullptr || channel_count == 0 || worker_count == 0) {
        NIXL_ERROR << "ProxyRuntime::init: invalid params";
        return NIXL_ERR_INVALID_PARAM;
    }

    backend_ = std::move(backend);
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

    worker_count = std::min(worker_count, channel_count);
    NIXL_INFO << "ProxyRuntime::init: effective worker_count=" << worker_count
              << " (clamped to channel_count)";

    nixl_status_t rc = backend_->init(worker_count, channel_count);
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

    channels_.resize(channel_count);
    for (uint32_t channel_id = 0; channel_id < channel_count; ++channel_id) {
        rc = channels_[channel_id].allocate(channel_id, ring_depth_);
        if (rc != NIXL_SUCCESS) {
            channels_.clear();
            backend_->shutdown();
            cudaFreeHost(shutdown_word_host_);
            shutdown_word_host_ = nullptr;
            shutdown_word_dev_ = nullptr;
            backend_.reset();
            return rc;
        }
    }

    device_channel_views_.resize(channel_count);
    if (cudaMalloc(reinterpret_cast<void **>(&device_channel_views_dev_),
                   sizeof(nixlProxyChannelView) * channel_count) != cudaSuccess) {
        channels_.clear();
        backend_->shutdown();
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_ = nullptr;
        backend_.reset();
        return NIXL_ERR_BACKEND;
    }
    for (uint32_t channel_id = 0; channel_id < channel_count; ++channel_id) {
        device_channel_views_[channel_id] = channels_[channel_id].device_view;
    }
    if (cudaMemcpy(device_channel_views_dev_,
                   device_channel_views_.data(),
                   sizeof(nixlProxyChannelView) * channel_count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device_channel_views_dev_);
        device_channel_views_dev_ = nullptr;
        device_channel_views_.clear();
        channels_.clear();
        backend_->shutdown();
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_ = nullptr;
        backend_.reset();
        return NIXL_ERR_BACKEND;
    }
    nixlProxyDeviceContextData device_context{
        device_channel_views_dev_, channel_count, shutdown_word_dev_};
    if (cudaMalloc(reinterpret_cast<void **>(&device_context_),
                   sizeof(nixlProxyDeviceContextData)) != cudaSuccess ||
        cudaMemcpy(
            device_context_, &device_context, sizeof(device_context), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        if (device_context_) {
            cudaFree(device_context_);
            device_context_ = nullptr;
        }
        cudaFree(device_channel_views_dev_);
        device_channel_views_dev_ = nullptr;
        device_channel_views_.clear();
        channels_.clear();
        backend_->shutdown();
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_ = nullptr;
        backend_.reset();
        return NIXL_ERR_BACKEND;
    }

    workers_.clear();
    workers_.reserve(worker_count);
    workers_started_ = false;

    for (uint32_t w = 0; w < worker_count; ++w) {
        uint32_t first_ch = (w * channel_count) / worker_count;
        uint32_t end_ch = ((w + 1) * channel_count) / worker_count;
        uint32_t n_ch = end_ch - first_ch;

        NIXL_INFO << "ProxyRuntime::init: worker " << w << " assigned channels [" << first_ch
                  << ", " << end_ch << ")";
        workers_.push_back(std::make_unique<ProxyWorker>(backend_.get(),
                                                         &memview_registry_,
                                                         shutdown_word_host_,
                                                         &channels_[first_ch],
                                                         n_ch,
                                                         pthr_delay_us));
    }

    NIXL_INFO << "ProxyRuntime::init: complete — " << channel_count << " channels, " << worker_count
              << " workers, "
              << "device_context(dev)=" << device_context_;
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
    return memview_registry_.prepMemView(dlist, proxy_memview);
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
    if (shutdown_word_host_ == nullptr) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: runtime not initialized";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (workers_started_) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: workers already started";
        return NIXL_ERR_INVALID_PARAM;
    }

    for (auto &channel : channels_) {
        channel.inflight_requests.clear();
        channel.error_latched = false;
    }

    __atomic_store_n(shutdown_word_host_,
                     static_cast<uint32_t>(nixl_proxy_control_state_t::RUNNING),
                     __ATOMIC_RELEASE);

    uint32_t idx = 0;
    for (auto &worker : workers_) {
        worker->start(idx);
        ++idx;
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
    backend_.reset();
    NIXL_INFO << "ProxyRuntime::shutdown: complete";
    return backend_status;
}
