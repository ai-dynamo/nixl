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
#include "proxy_channel.h"

#include <optional>

#include "nixl_log.h"
#include "proxy_submission.h"

namespace nixl {

proxyChannel::proxyChannel(uint32_t channel_id, uint32_t max_peers)
    : id_(channel_id),
      rings_(max_peers) {}

nixl_status_t
proxyChannel::allocate(deviceOps &allocator, uint32_t depth, proxyControlBuffer &control) {
    for (uint32_t peer = 0; peer < rings_.size(); ++peer) {
        const size_t ring_index = static_cast<size_t>(id_) * rings_.size() + peer;
        const nixl_status_t rc = rings_[peer].allocate(allocator, depth, control, ring_index);
        if (rc != NIXL_SUCCESS) {
            return rc;
        }
    }
    return NIXL_SUCCESS;
}

void
proxyChannel::deallocate() noexcept {
    for (proxyRing &ring : rings_) {
        ring.deallocate();
    }
}

void
proxyChannel::submitReady(proxyTransport &transport) {
    for (uint32_t peer = 0; peer < rings_.size(); ++peer) {
        uint32_t slot = 0;
        const std::optional<nixlProxyCommand> submission = rings_[peer].tryDequeue(slot);
        if (submission) {
            NIXL_DEBUG << "proxyChannel::submitReady: channel=" << id_ << " slot=" << slot
                       << " opcode=" << static_cast<int>(submission->opcode)
                       << " op_idx=" << submission->op_idx << " size=" << submission->size;
            submitRecord(transport, peer, slot, *submission);
        }
    }
}

void
proxyChannel::submitRecord(proxyTransport &transport,
                           uint32_t peer,
                           uint32_t slot,
                           const nixlProxyCommand &submission) {
    proxyRing &ring = rings_[peer];
    proxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;

    // dst_index is also the peer slot; this ring's quiesce covers no other peer.
    if (submission.peerIndex() != peer) {
        NIXL_ERROR << "proxyChannel::submitRecord: command for peer " << submission.peerIndex()
                   << " on the ring of peer " << peer << " op_idx=" << submission.op_idx;
        inflight.status = NIXL_ERR_INVALID_PARAM;
        ring.recordInflight(slot, inflight);
        return;
    }

    proxyBackendSubmission prepared_submission;
    nixl_status_t status = resolveSubmission(submission, id_, peer, prepared_submission);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "proxyChannel::submitRecord: submission preparation failed"
                   << " op_idx=" << submission.op_idx << " status=" << status;
        inflight.status = status;
        ring.recordInflight(slot, inflight);
        return;
    }

    // An atomic add has no source, so its local descriptor is not set.
    if (prepared_submission.opcode == nixl_proxy_opcode_t::PUT) {
        NIXL_DEBUG << "proxyChannel::submitRecord: PUT op_idx=" << submission.op_idx
                   << " channel=" << id_ << " peer=" << peer << " local_addr=" << std::hex
                   << prepared_submission.local.addr
                   << " remote_addr=" << prepared_submission.remote.addr << std::dec
                   << " size=" << prepared_submission.size;
    } else {
        NIXL_DEBUG << "proxyChannel::submitRecord: ATOMIC_ADD op_idx=" << submission.op_idx
                   << " channel=" << id_ << " peer=" << peer << " remote_addr=" << std::hex
                   << prepared_submission.remote.addr << std::dec
                   << " value=" << prepared_submission.value;
    }

    status = transport.submit(prepared_submission, inflight.backend_request);
    inflight.status = status;
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_ERROR << "proxyChannel::submitRecord: backend submit failed"
                   << " status=" << status << " op_idx=" << submission.op_idx
                   << " request_token=" << inflight.backend_request.token;
    }

    NIXL_DEBUG << "proxyChannel::submitRecord: submitted op_idx=" << submission.op_idx
               << " request_token=" << inflight.backend_request.token << " status=" << status;
    ring.recordInflight(slot, inflight);
}

void
proxyChannel::progress(proxyTransport &transport) {
    for (uint32_t peer = 0; peer < rings_.size(); ++peer) {
        transport.progress(id_, peer);
    }
}

void
proxyChannel::publishCompletions(proxyTransport &transport) {
    for (uint32_t peer = 0; peer < rings_.size(); ++peer) {
        proxyRing &ring = rings_[peer];
        while (proxyRequestState *front = ring.oldestInflight()) {
            nixl_status_t st;
            if (front->status != NIXL_IN_PROG) {
                st = front->status;
            } else {
                st = transport.checkCompletion(id_, peer, front->backend_request);
                if (st == NIXL_IN_PROG) {
                    break;
                }
                front->status = st;
            }
            NIXL_DEBUG << "proxyChannel::publishCompletions: op_idx=" << front->op_idx
                       << " status=" << st << " token=" << front->backend_request.token;
            ring.retireOldest(st);
        }
    }
}

bool
proxyChannel::drained() const noexcept {
    for (const proxyRing &ring : rings_) {
        if (!ring.drained()) {
            return false;
        }
    }
    return true;
}

bool
proxyChannel::skipAbandonedTickets() noexcept {
    bool skipped = false;
    for (proxyRing &ring : rings_) {
        skipped = ring.skipAbandonedTickets() || skipped;
    }
    return skipped;
}

void
proxyChannel::drainAndRearm(proxyTransport &transport) noexcept {
    for (uint32_t peer = 0; peer < rings_.size(); ++peer) {
        proxyRing &ring = rings_[peer];
        if (!ring.allocated()) {
            continue;
        }
        if (ring.selectDevice() != NIXL_SUCCESS) {
            NIXL_FATAL << "Failed to select proxy ring device";
        }
        if (!ring.assertDrained()) {
            NIXL_FATAL << "Proxy ring has unfinished or unpublished producer tickets";
        }
        if (transport.quiesce(id_, peer) != NIXL_SUCCESS) {
            NIXL_FATAL << "Failed to quiesce proxy backend";
        }
        if (ring.rearm() != NIXL_SUCCESS) {
            NIXL_FATAL << "Failed to reset drained proxy ring";
        }
    }
}

void
proxyChannel::appendDeviceViews(std::vector<nixlProxyRingDesc> &out) const {
    for (const proxyRing &ring : rings_) {
        out.push_back(ring.deviceView());
    }
}

} // namespace nixl
