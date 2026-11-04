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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CHANNEL_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CHANNEL_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "device/device_ops.h"
#include "nixl_types.h"
#include "proxy_config.h"
#include "proxy_transport.h"
#include "proxy_control_buffer.h"
#include "proxy_protocol.h"
#include "proxy_ring.h"

namespace nixl {

/**
 * One proxy channel: the ring of every peer that shares this channel id.
 *
 * Each ring is driven by exactly one worker thread, the only caller of the transport
 * for it; a channel's peers may be split across threads.
 */
class proxyChannel {
public:
    proxyChannel(uint32_t channel_id, uint32_t max_peers);

    /** Allocate every peer's ring; ring (id, peer) takes control-buffer ring id * max_peers + peer.
     */
    [[nodiscard]] nixl_status_t
    allocate(deviceOps &allocator, uint32_t depth, proxyControlBuffer &control);

    [[nodiscard]] uint32_t
    id() const noexcept {
        return id_;
    }

    [[nodiscard]] proxyRing &
    ring(uint32_t peer) noexcept {
        return rings_[peer];
    }

    /** Contiguous, in peer order, which is also the device order. */
    [[nodiscard]] std::span<proxyRing>
    rings() noexcept {
        return rings_;
    }

    // The thread owning `peers` only: each call is one pass over those peers' rings.

    /** Take at most one published record per ring, resolve it and submit it. */
    void
    submitReady(proxyTransport &transport, proxyPeerSlice peers);

    void
    progress(proxyTransport &transport, proxyPeerSlice peers);

    /** Publish completions in submission order, stopping at a ring's first pending request. */
    void
    publishCompletions(proxyTransport &transport, proxyPeerSlice peers);

    /** No ring has submitted or published work left. */
    [[nodiscard]] bool
    drained(proxyPeerSlice peers) const noexcept;

    [[nodiscard]] bool
    skipAbandonedTickets(proxyPeerSlice peers) noexcept;

    /** Once drained: check each ring, quiesce the transport for it, then reset it; fatal on
     * failure. */
    void
    drainAndRearm(proxyTransport &transport, proxyPeerSlice peers) noexcept;

    /** Device-side views of this channel's rings, in peer order, for the runtime to flatten. */
    void
    appendDeviceViews(std::vector<nixlProxyRingDesc> &out) const;

private:
    void
    submitRecord(proxyTransport &transport,
                 uint32_t peer,
                 uint32_t slot,
                 const nixlProxyCommand &submission);

    uint32_t id_;
    std::vector<proxyRing> rings_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CHANNEL_H
