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

#include "device/device_ops.h"

namespace nixl {

nixl_status_t
proxyChannelState::allocate(deviceOps &allocator,
                            uint32_t depth,
                            proxyControlBuffer *control_slots,
                            size_t control_slot_index) {
    if (depth == 0 || control_slots == nullptr ||
        control_slots->devicePointer(control_slot_index) == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    ring_depth_ = depth;
    control_slots_ = control_slots;
    allocator_ = &allocator;
    control_slot_index_ = control_slot_index;
    consumer_idx_dev_ = control_slots_->devicePointer(control_slot_index_);
    consumer_idx_shadow_ = 0;

    if (allocator.allocDeviceMem(sizeof(nixlProxyWorkRing), work_ring_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), producer_idx_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), consumer_idx_cache_mem_) != NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxySubmission) * depth, records_mem_) !=
            NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxyCompletionSlot), completion_slot_mem_) !=
            NIXL_SUCCESS) {
        NIXL_ERROR << "proxyChannelState::allocate: device allocation failed";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    if (rearm() != NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    nixlProxyWorkRing work_ring{
        records_mem_.devicePointer<nixlProxySubmission>(),
        static_cast<uint64_t *>(producer_idx_mem_.get()),
        consumer_idx_dev_,
        static_cast<uint64_t *>(consumer_idx_cache_mem_.get()),
        depth,
    };
    if (allocator.copy(work_ring_mem_.get(),
                       &work_ring,
                       sizeof(work_ring),
                       deviceOps::copyDirection::HostToDevice) != NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    device_view =
        nixlProxyChannelView{static_cast<nixlProxyWorkRing *>(work_ring_mem_.get()),
                             completion_slot_mem_.devicePointer<nixlProxyCompletionSlot>()};

    NIXL_DEBUG << "Proxy ring: depth=" << depth << " control_slot=" << control_slot_index
               << " records(dev)=" << records_mem_.devicePointer();
    return NIXL_SUCCESS;
}

nixl_status_t
proxyChannelState::rearm() noexcept {
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
    // Non-blocking producer streams must see the reset before reuse.
    if (allocator_->synchronize() != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    if (publishConsumerIdx(0) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    submit_idx_ = 0;
    inflight_slots_.assign(ring_depth_, proxyRequestState{});
    // Clear the old completion latch before reusing the ring.
    completionSlotHost()->completion_status = NIXL_IN_PROG;
    __atomic_store_n(&completionSlotHost()->completed_idx, uint64_t{0}, __ATOMIC_RELEASE);
    return NIXL_SUCCESS;
}

void
proxyChannelState::verifyDrained() const noexcept {
    if (!allocated()) {
        return;
    }
    uint64_t produced = 0;
    if (!drained() ||
        allocator_->copy(&produced,
                         producer_idx_mem_.get(),
                         sizeof(produced),
                         deviceOps::copyDirection::DeviceToHost) != NIXL_SUCCESS ||
        produced != submit_idx_) {
        NIXL_FATAL << "Proxy ring has unfinished or unpublished producer tickets";
    }
}

bool
proxyChannelState::drained() const noexcept {
    if (ring_depth_ == 0) {
        return true;
    }
    if (consumer_idx_shadow_ != submit_idx_) {
        return false;
    }
    // Include published records that have not been submitted yet.
    const uint32_t slot = static_cast<uint32_t>(submit_idx_ % ring_depth_);
    return __atomic_load_n(&recordsHost()[slot].op_idx, __ATOMIC_ACQUIRE) == 0;
}

nixl_status_t
proxyChannelState::publishConsumerIdx(uint64_t value) noexcept {
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
proxyChannelState::deallocate() noexcept {
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

proxyRuntime::proxyRuntime(proxyBackendOps backend_ops,
                           const proxyConfig &config,
                           deviceOps &allocator) noexcept
    : allocator_(allocator),
      backend_ops_(std::move(backend_ops)),
      config_(config) {}

proxyRuntime::~proxyRuntime() {
    shutdown();
}

nixl_status_t
proxyRuntime::create(proxyBackendOps backend_ops,
                     const proxyConfig &config,
                     std::unique_ptr<proxyRuntime> &out) {
    deviceOps *ops = getDeviceOps();
    if (ops == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    return create(std::move(backend_ops), config, out, *ops);
}

nixl_status_t
proxyRuntime::create(proxyBackendOps backend_ops,
                     const proxyConfig &config,
                     std::unique_ptr<proxyRuntime> &out,
                     deviceOps &allocator) {
    NIXL_INFO << "ProxyRuntime::create: max_peers=" << config.max_peers
              << " channel_count=" << config.channel_count
              << " thread_count=" << config.effectiveThreadCount()
              << " ring_depth=" << config.ring_depth;

    if (!backend_ops.complete()) {
        NIXL_ERROR << "ProxyRuntime::create: incomplete backend callbacks";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (config.max_peers == 0 || config.channel_count == 0 || config.effectiveThreadCount() == 0 ||
        config.ring_depth == 0) {
        NIXL_ERROR << "ProxyRuntime::create: invalid config";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::unique_ptr<proxyRuntime> runtime(
        new proxyRuntime(std::move(backend_ops), config, allocator));
    const nixl_status_t status = runtime->build();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = std::move(runtime);
    return NIXL_SUCCESS;
}

nixl_status_t
proxyRuntime::build() {
    const uint32_t max_peers = config_.max_peers;
    const uint32_t channel_count = config_.channel_count;
    const uint32_t worker_count = config_.effectiveThreadCount();

    nixl_status_t rc = backend_ops_.init(config_);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::build: backend init failed: " << rc;
        return rc;
    }

    const size_t channel_slots = config_.ringCount();
    rc = proxyControlBuffer::create(allocator_, kProxyCiSlotBase + channel_slots, control_slots_);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::build: failed to create GPU-visible control slab";
        return rc;
    }
    shutdown_word_dev_ = control_slots_->devicePointer(kProxyShutdownSlot);
    channels_.resize(channel_slots);
    device_channel_views_.resize(channel_slots);
    for (uint32_t channel_idx = 0; channel_idx < channel_count; channel_idx++) {
        for (uint32_t peer_idx = 0; peer_idx < max_peers; peer_idx++) {
            const size_t slot = static_cast<size_t>(channel_idx) * max_peers + peer_idx;
            rc = channels_[slot].allocate(
                allocator_, config_.ring_depth, control_slots_.get(), kProxyCiSlotBase + slot);
            if (rc != NIXL_SUCCESS) {
                return rc;
            }
            device_channel_views_[slot] = channels_[slot].device_view;
        }
    }

    if (allocator_.allocDeviceMem(sizeof(nixlProxyChannelView) * channel_slots,
                                  device_channel_views_mem_) != NIXL_SUCCESS ||
        allocator_.copy(device_channel_views_mem_.get(),
                        device_channel_views_.data(),
                        sizeof(nixlProxyChannelView) * channel_slots,
                        deviceOps::copyDirection::HostToDevice) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    const nixlProxyDeviceContextData device_context{
        .channels = static_cast<nixlProxyChannelView *>(device_channel_views_mem_.get()),
        .max_peers = max_peers,
        .num_channels = channel_count,
        .shutdown_word = shutdown_word_dev_};
    if (allocator_.allocDeviceMem(sizeof(nixlProxyDeviceContextData), device_context_mem_) !=
            NIXL_SUCCESS ||
        allocator_.copy(device_context_mem_.get(),
                        &device_context,
                        sizeof(device_context),
                        deviceOps::copyDirection::HostToDevice) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    memview_registry_ = std::make_unique<proxyMemViewRegistry>(allocator_, deviceContext());

    workers_.reserve(worker_count);
    for (uint32_t worker_idx = 0; worker_idx < worker_count; worker_idx++) {
        workers_.push_back(std::make_unique<proxyWorker>(&backend_ops_,
                                                         memview_registry_.get(),
                                                         &shutdown_state_,
                                                         channels_.data(),
                                                         max_peers,
                                                         channel_count,
                                                         worker_idx,
                                                         worker_count,
                                                         &drain_requested_));
    }

    return NIXL_SUCCESS;
}

nixl_status_t
proxyRuntime::prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    const std::lock_guard lock(control_mutex_);
    if (proxy_memview == nullptr || memview_registry_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return memview_registry_->prepLocal(dlist, *proxy_memview);
}

nixl_status_t
proxyRuntime::prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    const std::lock_guard lock(control_mutex_);
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
proxyRuntime::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    const std::lock_guard lock(control_mutex_);
    if (memview_registry_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Queued records still borrow the view being retired.
    drainChannels();
    return memview_registry_->unregister(proxy_memview);
}

void
proxyRuntime::drainChannels() noexcept {
    if (!workers_started_) {
        for (const auto &channel : channels_) {
            channel.verifyDrained();
        }
        return;
    }

    const uint64_t requested = drain_requested_.fetch_add(1, std::memory_order_acq_rel) + 1;
    for (const auto &worker : workers_) {
        while (worker->drainAcked() < requested) {
            if (shutdown_state_.load(std::memory_order_acquire) !=
                static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)) {
                NIXL_FATAL << "Proxy runtime stopped during drain";
            }
            std::this_thread::yield();
        }
    }
}

nixl_status_t
proxyRuntime::startWorkers() {
    const std::lock_guard lock(control_mutex_);
    NIXL_INFO << "ProxyRuntime::startWorkers: launching " << workers_.size() << " worker thread(s)";
    if (!control_slots_) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: runtime not initialized";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (workers_started_) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: workers already started";
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixl_status_t publish_status = control_slots_->writeSlot(
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

    return NIXL_SUCCESS;
}

void
proxyRuntime::joinWorkerThreads() noexcept {
    for (auto &worker : workers_) {
        worker->join();
    }
}

nixl_status_t
proxyRuntime::shutdown() {
    const std::lock_guard lock(control_mutex_);
    if (!backend_ops_.shutdown) {
        return NIXL_SUCCESS;
    }
    if (workers_started_ &&
        control_slots_->writeSlot(kProxyShutdownSlot,
                                  static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN)) !=
            NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to publish proxy shutdown";
    }
    // A failed build has never handed a context to a producer.
    if (memview_registry_) {
        drainChannels();
    }
    shutdown_state_.store(static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN),
                          std::memory_order_release);
    joinWorkerThreads();
    workers_started_ = false;
    workers_.clear();
    memview_registry_.reset();
    device_context_mem_.reset();
    shutdown_word_dev_ = nullptr;
    device_channel_views_mem_.reset();
    device_channel_views_.clear();
    channels_.clear();
    control_slots_.reset();
    const nixl_status_t status = backend_ops_.shutdown();
    backend_ops_ = proxyBackendOps{};
    return status;
}

} // namespace nixl
