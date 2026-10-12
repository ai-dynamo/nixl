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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_WORKER_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_WORKER_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stop_token>
#include <thread>

#include "proxy_transport.h"
#include "proxy_channel.h"

namespace nixl {

/** How often a drain that has not finished logs what it is waiting for. */
inline constexpr std::chrono::seconds proxy_drain_warning_interval{10};

/** State every worker shares; owned by the runtime and immutable while workers exist. */
struct proxyWorkerContext {
    proxyTransport &transport;
    /** Channel-major; a worker owns channels[i] for i = index, index + worker_count, ... */
    std::span<proxyChannel> channels;
    uint32_t worker_count;
    /** Bumped by an application thread per drain; each worker acknowledges it. */
    const std::atomic<uint64_t> &drain_requested;
    /** Set before the shutdown drain. */
    const std::atomic<bool> &shutting_down;
    /** From the runtime's stop source; requested once, after the final drain. */
    std::stop_token stop;
};

class proxyWorker {
public:
    proxyWorker(const proxyWorkerContext &ctx, uint32_t worker_index) noexcept;
    ~proxyWorker();

    /** @return NIXL_ERR_BACKEND, having started nothing, if the thread cannot be created. */
    [[nodiscard]] nixl_status_t
    start() noexcept;

    void
    join() noexcept;

    /** The drain generation this worker has fully applied. */
    [[nodiscard]] uint64_t
    drainAcked() const noexcept {
        return drain_acked_.load(std::memory_order_acquire);
    }

private:
    /** Make the first owned ring's device current, so every transport call runs on it. */
    void
    selectOwnedDevice() noexcept;

    void
    runOnce();

    /** Visit the channels striped to this worker: index, index + worker_count, ... */
    template<typename Fn>
    void
    forEachOwnedChannel(Fn &&fn);

    /** One pass: submit on every owned channel, then progress them, then publish completions. */
    void
    passOwnedChannels();

    [[nodiscard]] bool
    ownedChannelsDrained();

    /** Log each owned ring that still has work, for a drain that is taking long. */
    void
    logUndrainedRings();

    [[nodiscard]] bool
    skipOwnedAbandonedTickets();

    /** Drain, quiesce and reset on the rings' owning thread. */
    void
    drainOwnedChannels();

    const proxyWorkerContext &ctx_;
    const uint32_t index_;
    alignas(64) std::atomic<uint64_t> drain_acked_{0};
    std::jthread thread_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_WORKER_H
