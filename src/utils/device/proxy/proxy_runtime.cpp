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
#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>

#include "common/exception.h"
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

std::unique_ptr<proxyRuntime>
proxyRuntime::create(std::unique_ptr<proxyTransport> transport,
                     const proxyConfig &config,
                     deviceOps &allocator) {
    NIXL_INFO << "ProxyRuntime::create: max_peers=" << config.max_peers
              << " channel_count=" << config.channel_count
              << " thread_count=" << config.thread_count << " ring_depth=" << config.ring_depth;

    if (!transport) {
        throwRuntimeError("ProxyRuntime::create: no transport");
    }
    if (config.max_peers == 0 || config.channel_count == 0 || !config.evenThreadSplit() ||
        config.ring_depth == 0) {
        throwRuntimeError("ProxyRuntime::create: invalid config");
    }
    // The GPU masks ring indices.
    if ((config.ring_depth & (config.ring_depth - 1)) != 0) {
        throwRuntimeError("ProxyRuntime::create: ring_depth must be a power of two: ",
                          config.ring_depth);
    }

    std::unique_ptr<proxyRuntime> runtime(
        new proxyRuntime(std::move(transport), config, allocator));
    nixl_status_t status = runtime->build();
    if (status != NIXL_SUCCESS) {
        throwRuntimeError("ProxyRuntime::create: build failed: status=", status);
    }
    status = runtime->start();
    if (status != NIXL_SUCCESS) {
        throwRuntimeError("ProxyRuntime::create: failed to start workers: status=", status);
    }
    return runtime;
}

nixl_status_t
proxyRuntime::build() {
    const uint32_t max_peers = config_.max_peers;
    const uint32_t channel_count = config_.channel_count;
    const uint32_t worker_count = config_.thread_count;

    const size_t channel_slots = config_.ringCount();
    if (channel_slots > std::numeric_limits<size_t>::max() / sizeof(nixlProxyRingDesc)) {
        NIXL_ERROR << "ProxyRuntime::build: invalid ring-view size: " << channel_slots
                   << " ring(s)";
        return NIXL_ERR_INVALID_PARAM;
    }
    const size_t device_channel_views_bytes = sizeof(nixlProxyRingDesc) * channel_slots;
    nixl_status_t rc = proxyControlBuffer::create(allocator_, channel_slots, control_slots_);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::build: failed to create GPU-visible control slab";
        return rc;
    }
    uint64_t *const shutdown_word_dev =
        control_slots_->devicePointer(proxyControlBuffer::shutdownSlot());
    channels_.reserve(channel_count);
    std::vector<nixlProxyRingDesc> device_channel_views;
    device_channel_views.reserve(channel_slots);
    for (uint32_t channel_idx = 0; channel_idx < channel_count; channel_idx++) {
        channels_.emplace_back(channel_idx, max_peers);
        rc = channels_.back().allocate(allocator_, config_.ring_depth, *control_slots_);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
        // Channel-major, peer-minor: the device indexes rings[channel * max_peers + peer].
        channels_.back().appendDeviceViews(device_channel_views);
    }

    if (allocator_.allocDeviceMem(device_channel_views_bytes, device_channel_views_mem_) !=
            NIXL_SUCCESS ||
        allocator_.copy(device_channel_views_mem_.get(),
                        device_channel_views.data(),
                        device_channel_views_bytes,
                        deviceOps::copyDirection::HostToDevice) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    const nixlProxyDeviceContextData device_context{
        .rings = static_cast<nixlProxyRingDesc *>(device_channel_views_mem_.get()),
        .max_peers = max_peers,
        .num_channels = channel_count,
        .shutdown_word = shutdown_word_dev};
    if (allocator_.allocDeviceMem(sizeof(nixlProxyDeviceContextData), device_context_mem_) !=
            NIXL_SUCCESS ||
        allocator_.copy(device_context_mem_.get(),
                        &device_context,
                        sizeof(device_context),
                        deviceOps::copyDirection::HostToDevice) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    memview_manager_ = std::make_unique<proxyMemViewManager>(allocator_, deviceContext());

    worker_context_ = std::make_unique<const proxyWorkerContext>(proxyWorkerContext{
        *transport_, drain_requested_, shutting_down_, stop_source_.get_token()});
    workers_.reserve(worker_count);
    for (uint32_t worker_idx = 0; worker_idx < worker_count; worker_idx++) {
        const proxyThreadShare share = config_.threadShare(worker_idx);
        workers_.push_back(std::make_unique<proxyWorker>(
            *worker_context_,
            worker_idx,
            std::span<proxyChannel>(channels_).subspan(share.first_channel, share.channel_count),
            share.peers));
    }

    state_ = state_t::BUILT;
    return NIXL_SUCCESS;
}

nixl_status_t
proxyRuntime::prepMemView(const nixl_meta_dlist_t &dlist, proxy_view_handle_t *proxy_memview) {
    const std::lock_guard lock(control_mutex_);
    if (proxy_memview == nullptr || memview_manager_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return memview_manager_->prepLocal(dlist, *proxy_memview);
}

nixl_status_t
proxyRuntime::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                          proxy_view_handle_t *proxy_memview) {
    const std::lock_guard lock(control_mutex_);
    if (proxy_memview == nullptr || memview_manager_ == nullptr) {
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

    std::vector<uint32_t> peers;
    peers.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        // A hole's command fails on the host, so any ring will do.
        if (desc.metadataP == nullptr) {
            peers.push_back(0);
            continue;
        }
        if ((peers_.count(desc.remoteAgent) == 0) && (peers_.size() == config_.max_peers)) {
            NIXL_ERROR << "ProxyRuntime::prepMemView: remote agent " << desc.remoteAgent
                       << " exceeds proxy_max_peers=" << config_.max_peers;
            return NIXL_ERR_INVALID_PARAM;
        }
        peers.push_back(peers_.try_emplace(desc.remoteAgent, peers_.size()).first->second);
    }

    return memview_manager_->prepRemote(dlist, direct_ptrs, peers, *proxy_memview);
}

void
proxyRuntime::releaseMemView(proxy_view_handle_t proxy_memview) {
    const std::lock_guard lock(control_mutex_);
    // Queued records still borrow the view being retired.
    drainChannels();
    const nixl_status_t status = memview_manager_->release(proxy_memview);
    if (status != NIXL_SUCCESS) {
        NIXL_WARN << "ProxyRuntime::releaseMemView: failed to release " << proxy_memview
                  << ": status=" << status;
    }
}

void
proxyRuntime::drainChannels() noexcept {
    if (state_ != state_t::RUNNING) {
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
    const auto start = std::chrono::steady_clock::now();
    const std::chrono::milliseconds timeout(config_.drain_timeout_ms);
    auto next_warning = start + proxy_drain_warning_interval;
    for (size_t worker_idx = 0; worker_idx < workers_.size(); ++worker_idx) {
        while (workers_[worker_idx]->drainAcked() < requested) {
            if (stop_source_.stop_requested()) {
                NIXL_FATAL << "Proxy runtime stopped during drain";
            }
            const auto now = std::chrono::steady_clock::now();
            // Freeing views the transport may still use is unsafe, so abort instead of returning.
            if (timeout.count() != 0 && now - start >= timeout) {
                NIXL_FATAL << "Proxy drain did not finish within " << timeout.count()
                           << " ms; waiting for worker " << worker_idx;
            }
            if (now >= next_warning) {
                NIXL_WARN << "Proxy drain waiting for worker " << worker_idx;
                next_warning += proxy_drain_warning_interval;
            }
            std::this_thread::yield();
        }
    }
}

nixl_status_t
proxyRuntime::start() {
    const std::lock_guard lock(control_mutex_);
    NIXL_INFO << "ProxyRuntime::start: launching " << workers_.size() << " worker thread(s)";
    const nixl_status_t publish_status =
        control_slots_->publishShutdown(nixl_proxy_control_state_t::RUNNING);
    if (publish_status != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::start: failed to publish RUNNING state";
        return publish_status;
    }

    for (auto &worker : workers_) {
        if (worker->start() != NIXL_SUCCESS) {
            // Stop the workers already started; shutdown() then releases the runtime as built.
            NIXL_ERROR << "ProxyRuntime::start: failed to start a worker thread";
            if (control_slots_->publishShutdown(nixl_proxy_control_state_t::SHUTDOWN) !=
                NIXL_SUCCESS) {
                NIXL_FATAL << "Failed to publish proxy shutdown";
            }
            stop_source_.request_stop();
            joinWorkerThreads();
            return NIXL_ERR_BACKEND;
        }
    }
    state_ = state_t::RUNNING;

    return NIXL_SUCCESS;
}

void
proxyRuntime::joinWorkerThreads() noexcept {
    for (auto &worker : workers_) {
        worker->join();
    }
}

void
proxyRuntime::shutdown() noexcept {
    const std::lock_guard lock(control_mutex_);
    shutting_down_.store(true, std::memory_order_release);
    if (state_ == state_t::RUNNING &&
        control_slots_->publishShutdown(nixl_proxy_control_state_t::SHUTDOWN) != NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to publish proxy shutdown";
    }
    // A failed build has never handed a context to a producer.
    if (state_ == state_t::BUILT || state_ == state_t::RUNNING) {
        drainChannels();
    }
    stop_source_.request_stop();
    joinWorkerThreads();
    workers_.clear();
    worker_context_.reset();
    memview_manager_.reset();
    device_context_mem_.reset();
    device_channel_views_mem_.reset();
    channels_.clear();
    control_slots_.reset();
}

} // namespace nixl
