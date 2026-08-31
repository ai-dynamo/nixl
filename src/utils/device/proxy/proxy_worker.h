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
#include <cstddef>
#include <cstdint>
#include <thread>
#include "proxy_protocol.h"

struct nixlProxyBackendOps;
class nixlProxyMemViewRegistry;
struct nixlProxyChannelState;

class ProxyWorker {
    public:
        ProxyWorker(const nixlProxyBackendOps *backend_ops,
                    const nixlProxyMemViewRegistry *proxy_memview_registry,
                    std::atomic<uint64_t> *shutdown_state,
                    nixlProxyChannelState *channels,
                    uint32_t max_peers,
                    uint32_t channel_count,
                    uint32_t worker_index,
                    uint32_t worker_count,
                    uint64_t pthr_delay_us,
                    const std::atomic<uint64_t> *drain_requested) noexcept;
        ~ProxyWorker();

        void
        start();

        void join() noexcept;

        /** The drain generation this worker has fully applied. */
        [[nodiscard]] uint64_t
        drainAcked() const noexcept {
            return drain_acked_.load(std::memory_order_acquire);
        }

    private:
        void
        runOnce();

        nixlProxyChannelState *
        getChannelState(uint32_t peer, uint32_t channel_id);

        /** Visit all peers of the channels striped to this worker. */
        template<typename Fn>
        void
        forEachOwnedChannel(Fn &&fn);

        [[nodiscard]] bool
        ownedChannelsDrained();

        /** Drain, quiesce and reset on the ring's owning thread. */
        void
        drainOwnedChannels();

        void
        publishOwnedChannels();

        void
        submitOwnedChannels();

        void
        submitReady(nixlProxyChannelState &channel, uint32_t peer);

        void
        submitToBackend(nixlProxyChannelState &channel,
                        uint32_t peer,
                        uint32_t slot,
                        const nixlProxySubmission &submission);

        void
        driveBackendProgress();

        void
        publishCompletions(nixlProxyChannelState &channel);

        const nixlProxyBackendOps *backend_ops_ = nullptr;
        const nixlProxyMemViewRegistry *proxy_memview_registry_ = nullptr;
        std::atomic<uint64_t> *shutdown_state_ = nullptr;
        nixlProxyChannelState *channels_ = nullptr;
        uint32_t max_peers_ = 0;
        uint32_t channel_count_ = 0;
        uint32_t worker_index_ = 0;
        uint32_t worker_count_ = 0;
        uint64_t pthr_delay_us_ = 0;
        /** Owned by the runtime; bumped by an application thread per drain. */
        const std::atomic<uint64_t> *drain_requested_ = nullptr;
        alignas(64) std::atomic<uint64_t> drain_acked_{0};
        std::thread thread_;
};

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_WORKER_H
