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

#include <exception>

namespace nixl {

proxyWorker::proxyWorker(const proxyWorkerContext &ctx, uint32_t worker_index) noexcept
    : ctx_(ctx),
      index_(worker_index) {}

proxyWorker::~proxyWorker() {
    join();
}

nixl_status_t
proxyWorker::start() noexcept {
    try {
        thread_ = std::jthread([this]() {
            selectOwnedDevice();
            while (!ctx_.stop.stop_requested()) {
                runOnce();
            }
        });
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "proxyWorker::start: " << e.what();
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
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
proxyWorker::logUndrainedRings() {
    forEachOwnedChannel([this](proxyChannel &channel) {
        auto rings = channel.rings();
        for (size_t peer = 0; peer < rings.size(); ++peer) {
            if (rings[peer].drained()) {
                continue;
            }
            const proxyRequestState *oldest = rings[peer].oldestInflight();
            if (oldest != nullptr) {
                NIXL_WARN << "Proxy worker " << index_ << " still draining channel " << channel.id()
                          << " peer " << peer << ": oldest outstanding op_idx=" << oldest->op_idx;
            } else {
                NIXL_WARN << "Proxy worker " << index_ << " still draining channel " << channel.id()
                          << " peer " << peer << ": a command is not yet submitted";
            }
        }
    });
}

void
proxyWorker::drainOwnedChannels() {
    auto next_warning = std::chrono::steady_clock::now() + proxy_drain_warning_interval;
    while (!ownedChannelsDrained()) {
        passOwnedChannels();
        if (std::chrono::steady_clock::now() >= next_warning) {
            logUndrainedRings();
            next_warning += proxy_drain_warning_interval;
        }
    }

    forEachOwnedChannel([this](proxyChannel &channel) { channel.drainAndRearm(ctx_.transport); });
}

} // namespace nixl
