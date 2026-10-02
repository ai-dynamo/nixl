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
#include "proxy_ring.h"

#include "nixl_log.h"

namespace nixl {

nixl_status_t
proxyRing::allocate(deviceOps &allocator,
                    uint32_t depth,
                    proxyControlBuffer &control,
                    size_t ring_index) {
    uint64_t *const consumer_idx_dev =
        control.devicePointer(proxyControlBuffer::ringSlot(ring_index));
    if (depth == 0 || consumer_idx_dev == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    depth_ = depth;
    control_ = &control;
    ops_ = &allocator;
    if (allocator.getActiveDevice(device_id_) != NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    ring_index_ = ring_index;
    consumer_idx_shadow_ = 0;

    if (allocator.allocDeviceMem(sizeof(nixlProxyWorkRing), work_ring_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), producer_idx_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), consumer_idx_cache_mem_) != NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxyCommand) * depth, records_mem_) !=
            NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxyCompletionSlot), completion_slot_mem_) !=
            NIXL_SUCCESS) {
        NIXL_ERROR << "proxyRing::allocate: device allocation failed";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    if (rearm() != NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    nixlProxyWorkRing work_ring{
        records_mem_.devicePointer<nixlProxyCommand>(),
        static_cast<uint64_t *>(producer_idx_mem_.get()),
        consumer_idx_dev,
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
    device_view_ = nixlProxyRingDesc{static_cast<nixlProxyWorkRing *>(work_ring_mem_.get()),
                                     completion_slot_mem_.devicePointer<nixlProxyCompletionSlot>()};

    NIXL_DEBUG << "Proxy ring: depth=" << depth << " ring=" << ring_index
               << " commands(dev)=" << records_mem_.devicePointer();
    return NIXL_SUCCESS;
}

nixl_status_t
proxyRing::rearm() noexcept {
    if (ops_ == nullptr || depth_ == 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixlProxyCommand *commands_host = commandsHost();
    for (uint32_t i = 0; i < depth_; ++i) {
        commands_host[i] = nixlProxyCommand{};
    }
    if (ops_->memsetDeviceMem(producer_idx_mem_.get(), 0, sizeof(uint64_t)) != NIXL_SUCCESS ||
        ops_->memsetDeviceMem(consumer_idx_cache_mem_.get(), 0, sizeof(uint64_t)) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    // Non-blocking producer streams must see the reset before reuse.
    if (ops_->synchronize() != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    if (publishConsumerIdx(0) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }

    submit_idx_ = 0;
    inflight_.assign(depth_, proxyRequestState{});
    // Clear the old completion latch before reusing the ring.
    completionSlotHost()->completion_status = NIXL_IN_PROG;
    __atomic_store_n(&completionSlotHost()->completed_idx, uint64_t{0}, __ATOMIC_RELEASE);
    return NIXL_SUCCESS;
}

void
proxyRing::retireOldest(nixl_status_t status) noexcept {
    const uint64_t consumer_idx = consumer_idx_shadow_;
    proxyRequestState &front = inflight_[consumer_idx % depth_];
    if (completionSlotHost()->completion_status >= 0) {
        completionSlotHost()->completion_status = status;
        __atomic_store_n(&completionSlotHost()->completed_idx, front.op_idx, __ATOMIC_RELEASE);
    }
    if (publishConsumerIdx(consumer_idx + 1) != NIXL_SUCCESS) {
        NIXL_FATAL << "proxyRing::retireOldest: failed to publish CI"
                   << " consumer_idx=" << consumer_idx + 1;
    }
    front = proxyRequestState{};
}

bool
proxyRing::assertDrained() const noexcept {
    if (!allocated()) {
        return true;
    }
    uint64_t produced = 0;
    return drained() &&
        ops_->copy(&produced,
                   producer_idx_mem_.get(),
                   sizeof(produced),
                   deviceOps::copyDirection::DeviceToHost) == NIXL_SUCCESS &&
        produced == submit_idx_;
}

bool
proxyRing::drained() const noexcept {
    if (depth_ == 0) {
        return true;
    }
    if (consumer_idx_shadow_ != submit_idx_) {
        return false;
    }
    // Include published records that have not been submitted yet.
    const uint32_t slot = static_cast<uint32_t>(submit_idx_ % depth_);
    return __atomic_load_n(&commandsHost()[slot].op_idx, __ATOMIC_ACQUIRE) == 0;
}

nixl_status_t
proxyRing::publishConsumerIdx(uint64_t value) noexcept {
    if (control_ == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const nixl_status_t status = control_->publishConsumerIdx(ring_index_, value);
    if (status == NIXL_SUCCESS) {
        consumer_idx_shadow_ = value;
    }
    return status;
}

void
proxyRing::deallocate() noexcept {
    completion_slot_mem_.reset();
    records_mem_.reset();
    producer_idx_mem_.reset();
    consumer_idx_cache_mem_.reset();
    work_ring_mem_.reset();
    control_ = nullptr;
    ops_ = nullptr;
    ring_index_ = 0;
    consumer_idx_shadow_ = 0;
    inflight_.clear();
    submit_idx_ = 0;
    depth_ = 0;
    device_view_ = nixlProxyRingDesc{};
}

} // namespace nixl
