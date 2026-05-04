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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

enum class nixl_proxy_opcode_t : uint32_t {
    PUT = 0,
    ATOMIC_ADD = 1,
};

enum class nixl_proxy_control_state_t : uint32_t {
    RUNNING = 0,
    SHUTDOWN = 1,
};

struct alignas(64) nixlProxySubmission {
    uint64_t op_idx = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint32_t channel_id = 0;
    uint32_t flags = 0;

    uint32_t src_index = 0;
    uint32_t src_offset = 0;
    uint32_t dst_index = 0;
    uint32_t dst_offset = 0;
    uint32_t size = 0;

    uint64_t src_proxy_memview_id = 0;
    uint64_t dst_proxy_memview_id = 0;
    uint64_t value = 0;
};

static_assert(sizeof(nixlProxySubmission) == 64, "nixlProxySubmission must be 64 bytes");
static_assert(offsetof(nixlProxySubmission, op_idx) == 0,
              "op_idx must be the first word because it publishes record readiness");

struct nixlProxyWorkRing {
    /** Mapped host records: GPU writes via device alias; CPU worker reads host alias. */
    nixlProxySubmission *records = nullptr;
    /** Device-resident producer index; only the GPU updates it. */
    uint64_t *producer_idx = nullptr;
    /** Mapped pinned consumer; host proxy uses __atomic_* on host alias (nixlProxyChannelState). */
    uint64_t *consumer_idx = nullptr;
    /** The depth of the work ring. */
    uint32_t depth = 0;
};

struct alignas(16) nixlProxyCompletionSlot {
    uint64_t completed_idx = 0;
    nixl_status_t next_status = NIXL_IN_PROG;
};

struct nixlProxyChannelView {
    nixlProxyWorkRing *work_ring = nullptr;
    /** Mapped pinned host memory (device alias); host writes via host pointer with atomics. */
    nixlProxyCompletionSlot *completion_slot = nullptr;
    uint32_t channel_id = 0;
};

struct nixlProxyDeviceContextData {
    nixlProxyChannelView *channels = nullptr;
    uint32_t num_channels = 0;
    uint32_t *shutdown_word = nullptr;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
