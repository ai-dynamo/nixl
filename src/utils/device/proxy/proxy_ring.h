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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RING_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RING_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "device/device_ops.h"
#include "nixl_types.h"
#include "proxy_transport.h"
#include "proxy_control_buffer.h"
#include "proxy_protocol.h"

namespace nixl {

/** One in-flight record, retained until the completion frontier passes its ring slot. */
struct proxyRequestState {
    uint64_t op_idx = 0;
    proxyBackendRequest backend_request{};
    nixl_status_t status = NIXL_IN_PROG;
};

/**
 * One (channel, peer) work ring: the device-resident descriptor, the records in
 * mapped host memory, the indices, the completion slot, and the host side of the
 * single-producer/single-consumer protocol.
 *
 * Everything but allocation and the device view runs on the owning worker thread
 * only, so no member needs a lock.
 */
class alignas(64) proxyRing {
public:
    proxyRing() = default;
    ~proxyRing() = default;
    proxyRing(proxyRing &&) noexcept = default;
    proxyRing &
    operator=(proxyRing &&) noexcept = default;
    proxyRing(const proxyRing &) = delete;
    proxyRing &
    operator=(const proxyRing &) = delete;

    /** Publishes its consumer index through control-buffer ring `ring_index`. */
    [[nodiscard]] nixl_status_t
    allocate(deviceOps &allocator, uint32_t depth, proxyControlBuffer &control, size_t ring_index);

    void
    deallocate() noexcept;

    /**
     * Take the next published command, if the ring has room in flight for it.
     * @param[out] slot The ring slot the command occupied, for recordInflight().
     */
    [[nodiscard]] std::optional<nixlProxyCommand>
    tryDequeue(uint32_t &slot) noexcept {
        const uint64_t consumer_idx = consumer_idx_shadow_;
        const uint64_t submit_idx = submit_idx_;

        if (submit_idx - consumer_idx >= depth_) {
            return std::nullopt;
        }

        slot = static_cast<uint32_t>(submit_idx % depth_);
        const uint64_t op_idx = __atomic_load_n(&commandsHost()[slot].op_idx, __ATOMIC_ACQUIRE);
        if (op_idx == 0) {
            return std::nullopt;
        }

        nixlProxyCommand submission = commandsHost()[slot];
        submission.op_idx = op_idx;

        __atomic_store_n(&commandsHost()[slot].op_idx, 0, __ATOMIC_RELAXED);
        submit_idx_ = submit_idx + 1;
        return submission;
    }

    /** Remember a dequeued command's request until its completion is published. */
    void
    recordInflight(uint32_t slot, const proxyRequestState &state) noexcept {
        inflight_[slot] = state;
    }

    /** The oldest request whose completion is not yet published, or nullptr. */
    [[nodiscard]] proxyRequestState *
    oldestInflight() noexcept {
        const uint64_t consumer_idx = consumer_idx_shadow_;
        if (consumer_idx == submit_idx_) {
            return nullptr;
        }
        return &inflight_[consumer_idx % depth_];
    }

    /** Publish the oldest request's terminal status, then advance the consumer index. */
    void
    retireOldest(nixl_status_t status) noexcept;

    /** No submitted or published work remains. */
    [[nodiscard]] bool
    drained() const noexcept;

    /**
     * Whether every producer ticket has reached terminal completion, as reclamation requires.
     * The caller treats false as fatal.
     */
    [[nodiscard]] bool
    assertDrained() const noexcept;

    /** Reset an allocated ring after producers stop and backend work is quiescent. */
    [[nodiscard]] nixl_status_t
    rearm() noexcept;

    [[nodiscard]] const nixlProxyRingDesc &
    deviceView() const noexcept {
        return device_view_;
    }

    [[nodiscard]] bool
    allocated() const noexcept {
        return static_cast<bool>(work_ring_mem_);
    }

private:
    [[nodiscard]] nixl_status_t
    publishConsumerIdx(uint64_t value) noexcept;

    nixlProxyCommand *
    commandsHost() const noexcept {
        return records_mem_.hostPointer<nixlProxyCommand>();
    }

    nixlProxyCompletionSlot *
    completionSlotHost() const noexcept {
        return completion_slot_mem_.hostPointer<nixlProxyCompletionSlot>();
    }

    /** Host-only submit frontier; consumer_idx_shadow_ remains the completion frontier. */
    uint64_t submit_idx_ = 0;
    /** Host shadow of the authoritative GPU-visible consumer index. */
    uint64_t consumer_idx_shadow_ = 0;
    /** In-flight state retained until the completion frontier passes its ring slot. */
    std::vector<proxyRequestState> inflight_;

    /** Device-resident ring descriptor. */
    deviceMem work_ring_mem_;
    /** Device-resident producer index; only the GPU updates it. */
    deviceMem producer_idx_mem_;
    /** Device-resident cache of the consumer index used by GPU enqueue backpressure. */
    deviceMem consumer_idx_cache_mem_;
    /** Mapped pinned host commands; GPU writes via device alias, worker reads host alias. */
    mappedHostMem records_mem_;
    /** Mapped pinned host completion slot; worker writes host alias, GPU polls device alias. */
    mappedHostMem completion_slot_mem_;

    /** Publishes the authoritative consumer index through host-published device memory. */
    proxyControlBuffer *control_ = nullptr;
    size_t ring_index_ = 0;
    /** Remembered from allocate() so rearm() needs no arguments. */
    deviceOps *ops_ = nullptr;
    /** Host-side ring depth for the CPU worker; nixlProxyWorkRing itself is device-only. */
    uint32_t depth_ = 0;
    nixlProxyRingDesc device_view_{};
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_RING_H
