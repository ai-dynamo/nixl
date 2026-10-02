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

proxyRuntime::proxyRuntime(std::unique_ptr<proxyTransport> transport,
                           const proxyConfig &config,
                           deviceOps &allocator) noexcept
    : transport_(std::move(transport)),
      allocator_(allocator),
      config_(config) {}

proxyRuntime::~proxyRuntime() {
    shutdown();
}

nixl_status_t
proxyRuntime::create(std::unique_ptr<proxyTransport> transport,
                     const proxyConfig &config,
                     std::unique_ptr<proxyRuntime> &out) {
    deviceOps *ops = getDeviceOps();
    if (ops == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    return create(std::move(transport), config, out, *ops);
}

nixl_status_t
proxyRuntime::create(std::unique_ptr<proxyTransport> transport,
                     const proxyConfig &config,
                     std::unique_ptr<proxyRuntime> &out,
                     deviceOps &allocator) {
    NIXL_INFO << "ProxyRuntime::create: max_peers=" << config.max_peers
              << " channel_count=" << config.channel_count
              << " thread_count=" << config.effectiveThreadCount()
              << " ring_depth=" << config.ring_depth;

    if (!transport) {
        NIXL_ERROR << "ProxyRuntime::create: no transport";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (config.max_peers == 0 || config.channel_count == 0 || config.effectiveThreadCount() == 0 ||
        config.ring_depth == 0) {
        NIXL_ERROR << "ProxyRuntime::create: invalid config";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::unique_ptr<proxyRuntime> runtime(
        new proxyRuntime(std::move(transport), config, allocator));
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

    nixl_status_t rc = transport_->init(config_);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::build: transport init failed: " << rc;
        return rc;
    }
    state_ = state::initialized;

    const size_t channel_slots = config_.ringCount();
    rc = proxyControlBuffer::create(allocator_, channel_slots, control_slots_);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::build: failed to create GPU-visible control slab";
        return rc;
    }
    shutdown_word_dev_ = control_slots_->devicePointer(proxyControlBuffer::shutdownSlot());
    channels_.reserve(channel_count);
    device_channel_views_.reserve(channel_slots);
    for (uint32_t channel_idx = 0; channel_idx < channel_count; channel_idx++) {
        channels_.emplace_back(channel_idx, max_peers);
        rc = channels_.back().allocate(allocator_, config_.ring_depth, *control_slots_);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        // Channel-major, peer-minor: the device indexes rings[channel * max_peers + peer].
        channels_.back().appendDeviceViews(device_channel_views_);
    }

    if (allocator_.allocDeviceMem(sizeof(nixlProxyRingDesc) * channel_slots,
                                  device_channel_views_mem_) != NIXL_SUCCESS ||
        allocator_.copy(device_channel_views_mem_.get(),
                        device_channel_views_.data(),
                        sizeof(nixlProxyRingDesc) * channel_slots,
                        deviceOps::copyDirection::HostToDevice) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    const nixlProxyDeviceContextData device_context{
        .rings = static_cast<nixlProxyRingDesc *>(device_channel_views_mem_.get()),
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

    worker_context_ = std::make_unique<const proxyWorkerContext>(proxyWorkerContext{
        *transport_, channels_, worker_count, drain_requested_, stop_source_.get_token()});
    workers_.reserve(worker_count);
    for (uint32_t worker_idx = 0; worker_idx < worker_count; worker_idx++) {
        workers_.push_back(std::make_unique<proxyWorker>(*worker_context_, worker_idx));
    }

    state_ = state::built;
    return NIXL_SUCCESS;
}

nixl_status_t
proxyRuntime::prepMemView(const nixl_meta_dlist_t &dlist, proxyViewHandle *proxy_memview) {
    const std::lock_guard lock(control_mutex_);
    if (proxy_memview == nullptr || memview_registry_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return memview_registry_->prepLocal(dlist, *proxy_memview);
}

nixl_status_t
proxyRuntime::prepMemView(const nixl_remote_meta_dlist_t &dlist, proxyViewHandle *proxy_memview) {
    const std::lock_guard lock(control_mutex_);
    if (proxy_memview == nullptr || memview_registry_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    std::vector<void *> direct_ptrs;
    const nixl_status_t resolve_status = transport_->resolveDirectPtrs(dlist, direct_ptrs);
    if (resolve_status != NIXL_SUCCESS) {
        return resolve_status;
    }
    if (!direct_ptrs.empty() && direct_ptrs.size() != static_cast<size_t>(dlist.descCount())) {
        NIXL_ERROR << "ProxyRuntime::prepMemView: transport resolved " << direct_ptrs.size()
                   << " direct pointer(s) for " << dlist.descCount() << " descriptor(s)";
        return NIXL_ERR_INVALID_PARAM;
    }

    return memview_registry_->prepRemote(dlist, direct_ptrs, *proxy_memview);
}

nixl_status_t
proxyRuntime::unregisterProxyMemView(proxyViewHandle proxy_memview) {
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
    if (state_ != state::running) {
        for (auto &channel : channels_) {
            for (const proxyRing &ring : channel.rings()) {
                if (!ring.assertDrained()) {
                    NIXL_FATAL << "Proxy ring has unfinished or unpublished producer tickets";
                }
            }
        }
        return;
    }

    const uint64_t requested = drain_requested_.fetch_add(1, std::memory_order_acq_rel) + 1;
    for (const auto &worker : workers_) {
        while (worker->drainAcked() < requested) {
            if (stop_source_.stop_requested()) {
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
    if (state_ == state::running) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: workers already started";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (state_ != state::built) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: runtime not initialized";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const nixl_status_t publish_status =
        control_slots_->publishShutdown(nixl_proxy_control_state_t::RUNNING);
    if (publish_status != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: failed to publish RUNNING state";
        return publish_status;
    }

    for (auto &worker : workers_) {
        worker->start();
    }
    state_ = state::running;

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
    if (state_ == state::stopped) {
        return NIXL_SUCCESS;
    }
    if (state_ == state::running &&
        control_slots_->publishShutdown(nixl_proxy_control_state_t::SHUTDOWN) != NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to publish proxy shutdown";
    }
    // A failed build has never handed a context to a producer.
    if (state_ == state::built || state_ == state::running) {
        drainChannels();
    }
    const bool transport_initialized = state_ != state::created;
    stop_source_.request_stop();
    joinWorkerThreads();
    workers_.clear();
    worker_context_.reset();
    memview_registry_.reset();
    device_context_mem_.reset();
    shutdown_word_dev_ = nullptr;
    device_channel_views_mem_.reset();
    device_channel_views_.clear();
    channels_.clear();
    control_slots_.reset();
    state_ = state::stopped;
    return transport_initialized ? transport_->shutdown() : NIXL_SUCCESS;
}

} // namespace nixl
