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
#include "proxy_control_buffer.h"

#include <limits>
#include <utility>

#include "nixl_log.h"

namespace nixl {

proxyControlBuffer::proxyControlBuffer(std::unique_ptr<hostPublishedDeviceMem> mem) noexcept
    : mem_(std::move(mem)) {}

nixl_status_t
proxyControlBuffer::create(deviceOps &ops,
                           size_t ring_count,
                           std::unique_ptr<proxyControlBuffer> &out) {
    if (ring_count > std::numeric_limits<size_t>::max() / sizeof(uint64_t) - ringSlot(0)) {
        NIXL_ERROR << "Invalid proxy control buffer size: " << ring_count << " ring(s)";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::unique_ptr<hostPublishedDeviceMem> mem;
    const nixl_status_t status =
        ops.allocHostPublishedMem(sizeof(uint64_t) * ringSlot(ring_count), mem);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    out.reset(new proxyControlBuffer(std::move(mem)));
    return NIXL_SUCCESS;
}

nixl_status_t
proxyControlBuffer::publishShutdown(nixl_proxy_control_state_t state) noexcept {
    return mem_->publish(sizeof(uint64_t) * shutdownSlot(), static_cast<uint64_t>(state));
}

nixl_status_t
proxyControlBuffer::publishConsumerIdx(size_t ring_index, uint64_t value) noexcept {
    if (ring_index >= mem_->size() / sizeof(uint64_t) - ringSlot(0)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return mem_->publish(sizeof(uint64_t) * ringSlot(ring_index), value);
}

uint64_t *
proxyControlBuffer::devicePointer(size_t slot) const noexcept {
    return slot < mem_->size() / sizeof(uint64_t) ?
        static_cast<uint64_t *>(mem_->devicePointer()) + slot :
        nullptr;
}

} // namespace nixl
