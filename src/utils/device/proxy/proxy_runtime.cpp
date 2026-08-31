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
#include "nixl_types.h"
#include "proxy_worker.h"
#include "nixl_log.h"
#include <algorithm>
#include <cstdint>
#include <thread>
#include <utility>

#include "device/device_allocator.h"

nixl_status_t
nixlProxyChannelState::allocate(nixlDeviceAllocator &allocator,
                                uint32_t depth,
                                nixlProxyControlBuffer *control_slots,
                                size_t control_slot_index) {
    NIXL_INFO << "nixlProxyChannelState::allocate: depth=" << depth
              << " control_slot_index=" << control_slot_index;
    if (depth == 0 || control_slots == nullptr ||
        control_slots->devicePtr(control_slot_index) == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    ring_depth_ = depth;
    control_slots_ = control_slots;
    allocator_ = &allocator;
    control_slot_index_ = control_slot_index;
    consumer_idx_dev_ = control_slots_->devicePtr(control_slot_index_);
    consumer_idx_shadow_ = 0;

    if (allocator.allocDeviceMem(sizeof(nixlProxyWorkRing), work_ring_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), producer_idx_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), consumer_idx_cache_mem_) != NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxySubmission) * depth, records_mem_) !=
            NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxyCompletionSlot), completion_slot_mem_) !=
            NIXL_SUCCESS) {
        NIXL_ERROR << "nixlProxyChannelState::allocate: device allocation failed";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    if (rearm() != NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    nixlProxyWorkRing work_ring{
        records_mem_.asDev<nixlProxySubmission>(),
        producer_idx_mem_.as<uint64_t>(),
        consumer_idx_dev_,
        consumer_idx_cache_mem_.as<uint64_t>(),
        depth,
    };
    if (allocator.copyHostToDevice(work_ring_mem_.get(), &work_ring, sizeof(work_ring)) !=
        NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    device_view = nixlProxyChannelView{work_ring_mem_.as<nixlProxyWorkRing>(),
                                       completion_slot_mem_.asDev<nixlProxyCompletionSlot>()};

    NIXL_INFO << "nixlProxyChannelState::allocate: ready"
              << " work_ring(dev)=" << work_ring_mem_.get() << " records=" << recordsHost()
              << " records(dev)=" << records_mem_.devPtr()
              << " producer_idx(dev)=" << producer_idx_mem_.get()
              << " consumer_idx(shadow)=" << consumer_idx_shadow_
              << " consumer_idx(dev)=" << consumer_idx_dev_
              << " consumer_idx_cache(dev)=" << consumer_idx_cache_mem_.get()
              << " completion_slot(host)=" << completionSlotHost()
              << " completion_slot(dev)=" << completion_slot_mem_.devPtr();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyChannelState::rearm() noexcept {
    if (allocator_ == nullptr || ring_depth_ == 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixlProxySubmission *records_host = recordsHost();
    for (uint32_t i = 0; i < ring_depth_; ++i) {
        records_host[i] = nixlProxySubmission{};
    }
    if (allocator_->memsetDeviceMem(producer_idx_mem_.get(), 0, sizeof(uint64_t)) != NIXL_SUCCESS ||
        allocator_->memsetDeviceMem(consumer_idx_cache_mem_.get(), 0, sizeof(uint64_t)) !=
            NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    if (publishConsumerIdx(0) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    submit_idx_ = 0;
    inflight_slots_.assign(ring_depth_, nixlProxyRequestState{});
    // Clearing the latch last: until it goes back to NIXL_IN_PROG the GPU
    // would read a stale terminal status for the next generation of work.
    completionSlotHost()->next_status = NIXL_IN_PROG;
    __atomic_store_n(&completionSlotHost()->completed_idx, uint64_t{0}, __ATOMIC_RELEASE);
    return NIXL_SUCCESS;
}

size_t
nixlProxyChannelState::releaseInflightRequests(const nixlProxyBackendOps &backend_ops) noexcept {
    if (ring_depth_ == 0 || consumer_idx_dev_ == nullptr || !backend_ops.release_request) {
        return 0;
    }

    size_t released = 0;
    for (uint64_t idx = consumer_idx_shadow_; idx < submit_idx_; ++idx) {
        nixlProxyRequestState &inflight = inflight_slots_[idx % ring_depth_];
        if (inflight.status == NIXL_IN_PROG && inflight.backend_request) {
            backend_ops.release_request(inflight.backend_request);
            ++released;
        }
        inflight = nixlProxyRequestState{};
    }
    submit_idx_ = consumer_idx_shadow_;
    return released;
}

nixl_status_t
nixlProxyChannelState::publishConsumerIdx(uint64_t value) noexcept {
    if (control_slots_ == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const nixl_status_t status = control_slots_->writeSlot(control_slot_index_, value);
    if (status == NIXL_SUCCESS) {
        consumer_idx_shadow_ = value;
    }
    return status;
}

void
nixlProxyChannelState::deallocate() noexcept {
    completion_slot_mem_.reset();
    records_mem_.reset();
    producer_idx_mem_.reset();
    consumer_idx_cache_mem_.reset();
    work_ring_mem_.reset();
    consumer_idx_dev_ = nullptr;
    control_slots_ = nullptr;
    allocator_ = nullptr;
    control_slot_index_ = 0;
    consumer_idx_shadow_ = 0;
    inflight_slots_.clear();
    submit_idx_ = 0;
    ring_depth_ = 0;
    device_view = nixlProxyChannelView{};
}

nixlProxyRuntime::nixlProxyRuntime(nixlProxyBackendOps backend_ops,
                                   const nixlProxyConfig &config,
                                   nixlDeviceAllocator &allocator) noexcept
    : allocator_(allocator),
      backend_ops_(std::move(backend_ops)),
      config_(config) {}

nixlProxyRuntime::~nixlProxyRuntime() {
    shutdown();
}

nixl_status_t
nixlProxyRuntime::create(nixlProxyBackendOps backend_ops,
                         const nixlProxyConfig &config,
                         std::unique_ptr<nixlProxyRuntime> &out,
                         nixlDeviceAllocator &allocator) {
    NIXL_INFO << "ProxyRuntime::create: max_peers=" << config.max_peers
              << " channel_count=" << config.channel_count
              << " thread_count=" << config.effectiveThreadCount()
              << " pthr_delay_us=" << config.pthr_delay_us << " ring_depth=" << config.ring_depth;

    if (!backend_ops.complete()) {
        NIXL_ERROR << "ProxyRuntime::create: incomplete backend callbacks";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (config.max_peers == 0 || config.channel_count == 0 || config.effectiveThreadCount() == 0 ||
        config.ring_depth == 0) {
        NIXL_ERROR << "ProxyRuntime::create: invalid config";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Held locally so a failed build tears itself down; `out` is only written
    // once the runtime is complete.
    std::unique_ptr<nixlProxyRuntime> runtime(
        new nixlProxyRuntime(std::move(backend_ops), config, allocator));
    const nixl_status_t status = runtime->build();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = std::move(runtime);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::build() {
    const uint32_t max_peers = config_.max_peers;
    const uint32_t channel_count = config_.channel_count;
    const uint32_t worker_count = config_.effectiveThreadCount();

    nixl_status_t rc = backend_ops_.init(config_);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::build: backend init failed: " << rc;
        return rc;
    }

    const size_t channel_slots = config_.ucxWorkerCount();
    rc = control_slots_.allocate(allocator_, kProxyCiSlotBase + channel_slots);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::build: failed to create GPU-visible control slab";
        return rc;
    }
    shutdown_word_dev_ = control_slots_.devicePtr(kProxyShutdownSlot);
    channels_.resize(channel_slots);
    device_channel_views_.resize(channel_slots);
    for (uint32_t channel_idx = 0; channel_idx < channel_count; channel_idx++) {
        for (uint32_t peer_idx = 0; peer_idx < max_peers; peer_idx++) {
            const size_t slot = static_cast<size_t>(channel_idx) * max_peers + peer_idx;
            rc = channels_[slot].allocate(allocator_,
                                          config_.ring_depth,
                                          &control_slots_,
                                          kProxyCiSlotBase + slot);
            if (rc != NIXL_SUCCESS) {
                return rc;
            }
            device_channel_views_[slot] = channels_[slot].device_view;
        }
    }

    if (allocator_.allocDeviceMem(sizeof(nixlProxyChannelView) * channel_slots,
                                  device_channel_views_mem_) != NIXL_SUCCESS ||
        allocator_.copyHostToDevice(device_channel_views_mem_.get(),
                                    device_channel_views_.data(),
                                    sizeof(nixlProxyChannelView) * channel_slots) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    const nixlProxyDeviceContextData device_context{
        .channels = device_channel_views_mem_.as<nixlProxyChannelView>(),
        .max_peers = max_peers,
        .num_channels = channel_count,
        .shutdown_word = shutdown_word_dev_,
        .protocol_version = kProxyProtocolVersion};
    if (allocator_.allocDeviceMem(sizeof(nixlProxyDeviceContextData), device_context_mem_) !=
            NIXL_SUCCESS ||
        allocator_.copyHostToDevice(
            device_context_mem_.get(), &device_context, sizeof(device_context)) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    memview_registry_ = std::make_unique<nixlProxyMemViewRegistry>(allocator_, deviceContext());

    workers_.reserve(worker_count);
    for (uint32_t worker_idx = 0; worker_idx < worker_count; worker_idx++) {
        NIXL_INFO << "ProxyRuntime::build: worker " << worker_idx
                  << " owns channel(s) where channel_id % " << worker_count << " == " << worker_idx
                  << "; handles all dest rings of those channels";
        workers_.push_back(std::make_unique<ProxyWorker>(&backend_ops_,
                                                         memview_registry_.get(),
                                                         &shutdown_state_,
                                                         channels_.data(),
                                                         max_peers,
                                                         channel_count,
                                                         worker_idx,
                                                         worker_count,
                                                         config_.pthr_delay_us));
    }

    NIXL_INFO << "ProxyRuntime::build: complete - " << max_peers << " peers, " << channel_count
              << " channels (rings per dest), " << worker_count
              << " workers, device_context(dev)=" << deviceContext();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::loadRemoteConnInfo(const std::string &remote_name, const nixl_blob_t &conn_info) {
    NIXL_INFO << "ProxyRuntime::loadRemoteConnInfo: remote='" << remote_name
              << "' conn_info_size=" << conn_info.size();

    if (!backend_ops_.on_remote_loaded) {
        return NIXL_SUCCESS;
    }
    const nixl_status_t rc = backend_ops_.on_remote_loaded(remote_name, conn_info);
    NIXL_INFO << "ProxyRuntime::loadRemoteConnInfo: result=" << rc;
    return rc;
}

nixl_status_t
nixlProxyRuntime::remoteDisconnected(const std::string &remote_name) {
    NIXL_INFO << "ProxyRuntime::remoteDisconnected: remote='" << remote_name << "'";
    if (!backend_ops_.on_remote_disconnected) {
        return NIXL_SUCCESS;
    }
    return backend_ops_.on_remote_disconnected(remote_name);
}

nixl_status_t
nixlProxyRuntime::prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr || memview_registry_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return memview_registry_->prepLocal(dlist, *proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr || memview_registry_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    std::vector<void *> direct_ptrs;
    if (backend_ops_.resolve_direct_ptrs) {
        const nixl_status_t resolve_status = backend_ops_.resolve_direct_ptrs(dlist, direct_ptrs);
        if (resolve_status != NIXL_SUCCESS) {
            return resolve_status;
        }
    }

    return memview_registry_->prepRemote(dlist, direct_ptrs, *proxy_memview);
}

nixl_status_t
nixlProxyRuntime::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    if (memview_registry_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    return memview_registry_->unregister(proxy_memview);
}

bool
nixlProxyRuntime::resolveProxyMemView(nixlMemViewH proxy_memview,
                                      nixlMemViewH &backend_memview) const {
    if (memview_registry_ == nullptr) {
        return false;
    }
    return memview_registry_->resolve(proxy_memview, backend_memview);
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
        kProxyShutdownSlot, static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING));
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
            kProxyShutdownSlot, static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN));
        if (shutdown_signal_status != NIXL_SUCCESS) {
            NIXL_ERROR << "ProxyRuntime::shutdown: failed to publish SHUTDOWN state";
        }
    }
    shutdown_state_.store(static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN),
                          std::memory_order_release);

    joinWorkerThreads();
    workers_started_ = false;
    NIXL_INFO << "ProxyRuntime::shutdown: all worker threads joined";

    size_t released = 0;
    for (auto &channel : channels_) {
        released += channel.releaseInflightRequests(backend_ops_);
    }
    if (released != 0) {
        NIXL_INFO << "ProxyRuntime::shutdown: released " << released
                  << " pending backend request(s)";
    }

    nixl_status_t backend_status = NIXL_SUCCESS;
    if (backend_ops_.shutdown) {
        NIXL_INFO << "ProxyRuntime::shutdown: shutting down backend";
        backend_status = backend_ops_.shutdown();
        NIXL_INFO << "ProxyRuntime::shutdown: backend shutdown status=" << backend_status;
    }

    workers_.clear();
    // Workers are joined, so nothing can be resolving a memview any more; the
    // registry takes every device memview still alive down with it.
    memview_registry_.reset();

    device_context_mem_.reset();
    shutdown_word_dev_ = nullptr;
    device_channel_views_mem_.reset();
    device_channel_views_.clear();
    channels_.clear();
    control_slots_.deallocate();
    // Drop the callbacks last: they are what makes a second shutdown a no-op,
    // and they may own state belonging to the backend that is going away.
    backend_ops_ = nixlProxyBackendOps{};
    NIXL_INFO << "ProxyRuntime::shutdown: complete";
    if (backend_status != NIXL_SUCCESS) {
        return backend_status;
    }
    if (shutdown_signal_status != NIXL_SUCCESS) {
        return shutdown_signal_status;
    }
    return NIXL_SUCCESS;
}
