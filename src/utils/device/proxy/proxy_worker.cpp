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
#include "proxy_worker.h"
#include "nixl_log.h"

namespace nixl {

proxyWorker::proxyWorker(const proxyWorkerContext &ctx, uint32_t worker_index) noexcept
    : ctx_(ctx),
      index_(worker_index) {}

proxyWorker::~proxyWorker() {
    join();
}

void
proxyWorker::start() {
    thread_ = std::jthread([this]() {
        selectOwnedDevice();
        while (!ctx_.stop.stop_requested()) {
            runOnce();
        }
    });
}

void
proxyWorker::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void
proxyWorker::selectOwnedDevice() noexcept {
    if (index_ >= ctx_.channels.size()) {
        return;
    }
    const proxyRing &ring = ctx_.channels[index_].ring(0);
    if (ring.allocated() && ring.selectDevice() != NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to select proxy ring device";
    }
}

template<typename Fn>
void
proxyWorker::forEachOwnedChannel(Fn &&fn) {
    for (size_t channel_id = index_; channel_id < ctx_.channels.size();
         channel_id += ctx_.worker_count) {
        fn(ctx_.channels[channel_id]);
    }
}

void
proxyWorker::passOwnedChannels() {
    forEachOwnedChannel([this](proxyChannel &channel) { channel.submitReady(ctx_.transport); });
    forEachOwnedChannel([this](proxyChannel &channel) { channel.progress(ctx_.transport); });
    forEachOwnedChannel(
        [this](proxyChannel &channel) { channel.publishCompletions(ctx_.transport); });
}

void
proxyWorker::runOnce() {
    // Retirement is acknowledged only after quiescence and reset.
    const uint64_t requested = ctx_.drain_requested.load(std::memory_order_acquire);
    if (requested != drain_acked_.load(std::memory_order_relaxed)) {
        drainOwnedChannels();
        drain_acked_.store(requested, std::memory_order_release);
    }

    passOwnedChannels();
}

bool
proxyWorker::ownedChannelsDrained() {
    bool all_drained = true;
    forEachOwnedChannel(
        [&](proxyChannel &channel) { all_drained = all_drained && channel.drained(); });
    return all_drained;
}

void
proxyWorker::drainOwnedChannels() {
    while (!ownedChannelsDrained()) {
        passOwnedChannels();
    }

    forEachOwnedChannel([this](proxyChannel &channel) { channel.drainAndRearm(ctx_.transport); });
}

} // namespace nixl
